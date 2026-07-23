from __future__ import annotations

from typing import Any, Dict, List, Optional
import time

import requests


class DialogueGeneratorAgent:
    """
    Generates the final first-person NPC reply with an LLM.

    Responsibilities:
    - receive a structured decision from DialogueAgent;
    - build a grounded prompt from NPC state, personality and memory;
    - call Ollama;
    - validate the generated reply;
    - preserve deterministic state updates;
    - return a safe fallback if generation fails.

    This agent must not:
    - calculate combat;
    - change inventory;
    - invent completed actions;
    - directly modify game state;
    - decide whether an action is allowed.
    """

    DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
    DEFAULT_MODEL = "mistral"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        ollama_url: str = DEFAULT_OLLAMA_URL,
        timeout: int = 60,
        max_memory_items: int = 6,
        debug: bool = True
    ) -> None:
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.timeout = timeout
        self.max_memory_items = max_memory_items
        self.debug = debug

    def execute(
        self,
        dialogue_result: Dict[str, Any],
        game_state: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Converts a structured DialogueAgent result into a final NPC reply.

        Parameters
        ----------
        dialogue_result:
            Result returned by DialogueAgent.

        game_state:
            Current global game state. It is used only as read-only
            context for the prompt.

        Returns
        -------
        Dict[str, Any]
            Final result containing player_response, response_source,
            model metadata and the original deterministic state updates.
        """
        game_state = game_state or {}

        if not isinstance(dialogue_result, dict):
            return self._build_error_result(
                dialogue_result={},
                error="DialogueGeneratorAgent received an invalid result."
            )

        success = bool(dialogue_result.get("success", False))
        data = dialogue_result.get("data", {})

        if not isinstance(data, dict):
            data = {}

        if not success:
            return self._handle_unsuccessful_decision(
                dialogue_result=dialogue_result,
                data=data
            )

        generate_dialogue = bool(
            data.get("generate_dialogue", True)
        )

        if not generate_dialogue:
            return self._handle_non_generated_result(
                dialogue_result=dialogue_result,
                data=data
            )

        dialogue_context = data.get(
            "dialogue_context",
            {}
        )

        if not isinstance(dialogue_context, dict):
            dialogue_context = {}

        prompt = self._build_prompt(
            dialogue_result=dialogue_result,
            data=data,
            dialogue_context=dialogue_context,
            game_state=game_state
        )

        relevant_memories = self._collect_relevant_memories(
            dialogue_context
        )

        response_source = (
            "llm_with_memory"
            if relevant_memories
            else "llm"
        )

        start_time = time.perf_counter()

        try:
            generated_reply = self._call_ollama(prompt)

            generation_time = round(
                time.perf_counter() - start_time,
                3
            )

            validation = self._validate_reply(
                generated_reply=generated_reply,
                data=data,
                dialogue_context=dialogue_context
            )

            if not validation["valid"]:
                if self.debug:
                    print(
                        "[DEBUG] Dialogue validation failed: "
                        f"{validation['reason']}"
                    )

                generated_reply = self._generate_fallback_reply(
                    data=data,
                    dialogue_context=dialogue_context
                )

                response_source = "deterministic_fallback"

            return self._build_success_result(
                dialogue_result=dialogue_result,
                generated_reply=generated_reply,
                response_source=response_source,
                generation_time=generation_time,
                prompt=prompt,
                relevant_memories=relevant_memories,
                validation=validation
            )

        except Exception as error:
            generation_time = round(
                time.perf_counter() - start_time,
                3
            )

            fallback_reply = self._generate_fallback_reply(
                data=data,
                dialogue_context=dialogue_context
            )

            if self.debug:
                print(
                    "[DEBUG] Dialogue generation failed: "
                    f"{error}"
                )

            return self._build_success_result(
                dialogue_result=dialogue_result,
                generated_reply=fallback_reply,
                response_source="deterministic_fallback",
                generation_time=generation_time,
                prompt=prompt,
                relevant_memories=relevant_memories,
                validation={
                    "valid": False,
                    "reason": str(error)
                },
                generation_error=str(error)
            )

    # =========================================================
    # Prompt construction
    # =========================================================

    def _build_prompt(
        self,
        dialogue_result: Dict[str, Any],
        data: Dict[str, Any],
        dialogue_context: Dict[str, Any],
        game_state: Dict[str, Any]
    ) -> str:
        target = str(
            dialogue_context.get(
                "target",
                data.get("target", "unknown NPC")
            )
        )

        player_input = str(
            dialogue_context.get(
                "player_input",
                ""
            )
        ).strip()

        profile = dialogue_context.get(
            "profile",
            {}
        )

        if not isinstance(profile, dict):
            profile = {}

        npc_state = dialogue_context.get(
            "npc_state",
            {}
        )

        if not isinstance(npc_state, dict):
            npc_state = {}

        personality = dialogue_context.get(
            "personality",
            profile.get("personality", {})
        )

        goals = dialogue_context.get(
            "goals",
            profile.get("goals", [])
        )

        npc_intent = str(
            data.get(
                "npc_intent",
                "answer_player"
            )
        )

        emotion = str(
            data.get(
                "emotion",
                dialogue_context.get(
                    "emotion",
                    "neutral"
                )
            )
        )

        tone = str(
            data.get(
                "tone",
                "conversational"
            )
        )

        topic = str(
            data.get(
                "topic",
                "player_request"
            )
        )

        reaction_style = str(
            data.get(
                "reaction_style",
                "neutral"
            )
        )

        response_length = str(
            data.get(
                "response_length",
                "medium"
            )
        )

        current_goal = str(
            data.get(
                "current_goal",
                dialogue_context.get(
                    "current_goal",
                    "respond appropriately"
                )
            )
        )

        allow_trade = bool(
            data.get(
                "allow_trade",
                False
            )
        )

        refuses_conversation = bool(
            data.get(
                "refuses_conversation",
                False
            )
        )

        memories = self._collect_relevant_memories(
            dialogue_context
        )

        memory_text = self._format_memories(memories)
        personality_text = self._format_value(personality)
        goals_text = self._format_value(goals)

        world_context = self._build_world_context(
            game_state
        )

        length_instruction = self._get_length_instruction(
            response_length
        )

        return f"""
You generate dialogue for a fantasy role-playing game.

Write only the NPC's spoken reply.
The reply must be in the first person.
Do not write narration, action descriptions, labels, quotation marks,
speaker names, system notes or explanations.

NPC
Name or role: {target}
Personality: {personality_text}
Long-term goals: {goals_text}
Current goal: {current_goal}

Current NPC state
Emotion: {emotion}
Tone: {tone}
Reaction style: {reaction_style}
Trust: {dialogue_context.get("trust", "unknown")}
Fear: {dialogue_context.get("fear", "unknown")}
Anger: {dialogue_context.get("anger", "unknown")}
Stress: {dialogue_context.get("stress", "unknown")}
Hostile: {dialogue_context.get("hostile", False)}

Dialogue decision
NPC intent: {npc_intent}
Topic: {topic}
Trading allowed: {allow_trade}
Conversation refused: {refuses_conversation}

Relevant verified memories
{memory_text}

Limited world context
{world_context}

Player says
{player_input}

Rules
1. Stay in character.
2. Respond directly to the player's actual words.
3. Follow the selected NPC intent, emotion, tone and reaction style.
4. Never contradict verified memories.
5. Do not claim that an event happened unless it appears in the memories,
   NPC state, world context or player's current message.
6. Do not invent inventory changes, rewards, damage, purchases,
   completed quests or state changes.
7. Do not reveal information the NPC could not reasonably know.
8. If memories do not contain the requested factual information,
   answer with uncertainty or a personal opinion instead of inventing facts.
9. If conversation is refused, give a brief in-character refusal.
10. {length_instruction}

Return only the final spoken reply.
""".strip()

    def _build_world_context(
        self,
        game_state: Dict[str, Any]
    ) -> str:
        """
        Exposes only a limited set of safe world-state fields.
        """
        allowed_keys = [
            "current_location",
            "active_location",
            "world_mood",
            "time_of_day",
            "weather",
            "player_reputation",
            "crime_level",
            "guard_alert_level",
            "tavern_reputation"
        ]

        context_lines: List[str] = []

        for key in allowed_keys:
            if key not in game_state:
                continue

            value = game_state.get(key)

            context_lines.append(
                f"- {key}: {self._format_value(value)}"
            )

        if not context_lines:
            return "- No additional verified world facts."

        return "\n".join(context_lines)

    # =========================================================
    # Ollama
    # =========================================================

    def _call_ollama(
        self,
        prompt: str
    ) -> str:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "keep_alive": "15m",
            "options": {
                "temperature": 0.65,
                "top_p": 0.9,
                "repeat_penalty": 1.1,

                # NPC does not need to generate a long essay.
                "num_predict": 90,

                # Prevent an unnecessarily large context window.
                "num_ctx": 2048
            }
        }

        request_started = time.perf_counter()

        response = requests.post(
            self.ollama_url,
            json=payload,
            timeout=self.timeout
        )

        response.raise_for_status()

        request_seconds = time.perf_counter() - request_started
        result = response.json()

        generated_reply = str(
            result.get("response", "")
        ).strip()

        if not generated_reply:
            raise ValueError(
                "Ollama returned an empty dialogue response."
            )

        def ns_to_seconds(value: Any) -> float:
            try:
                return round(float(value) / 1_000_000_000, 3)
            except (TypeError, ValueError):
                return 0.0

        load_seconds = ns_to_seconds(
            result.get("load_duration")
        )
        prompt_seconds = ns_to_seconds(
            result.get("prompt_eval_duration")
        )
        generation_seconds = ns_to_seconds(
            result.get("eval_duration")
        )
        total_seconds = ns_to_seconds(
            result.get("total_duration")
        )

        prompt_tokens = int(
            result.get("prompt_eval_count") or 0
        )
        generated_tokens = int(
            result.get("eval_count") or 0
        )

        tokens_per_second = (
            round(generated_tokens / generation_seconds, 2)
            if generation_seconds > 0
            else 0.0
        )

        if self.debug:
            print("\n[OLLAMA PROFILE]")
            print(f"Model load:          {load_seconds:.3f}s")
            print(f"Prompt evaluation:   {prompt_seconds:.3f}s")
            print(f"Token generation:    {generation_seconds:.3f}s")
            print(f"Ollama total:        {total_seconds:.3f}s")
            print(f"HTTP request total:  {request_seconds:.3f}s")
            print(f"Prompt tokens:       {prompt_tokens}")
            print(f"Generated tokens:    {generated_tokens}")
            print(f"Generation speed:    {tokens_per_second} tokens/s")
            print("[/OLLAMA PROFILE]\n")

        return self._clean_reply(generated_reply)

    # =========================================================
    # Memory
    # =========================================================

    def _collect_relevant_memories(
        self,
        dialogue_context: Dict[str, Any]
    ) -> List[str]:
        collected: List[str] = []

        semantic_memories = dialogue_context.get(
            "semantic_memories",
            []
        )

        personal_memories = dialogue_context.get(
            "personal_memories",
            []
        )

        for memory in semantic_memories:
            formatted = self._format_memory_item(memory)

            if formatted:
                collected.append(formatted)

        for memory in personal_memories:
            formatted = self._format_memory_item(memory)

            if formatted:
                collected.append(formatted)

        unique_memories: List[str] = []
        seen = set()

        for memory in collected:
            normalized = memory.strip().lower()

            if not normalized:
                continue

            if normalized in seen:
                continue

            seen.add(normalized)
            unique_memories.append(memory.strip())

            if len(unique_memories) >= self.max_memory_items:
                break

        return unique_memories

    def _format_memory_item(
        self,
        memory: Any
    ) -> str:
        if memory is None:
            return ""

        if isinstance(memory, str):
            return memory.strip()

        if isinstance(memory, dict):
            preferred_keys = [
                "fact",
                "event",
                "content",
                "text",
                "description",
                "summary",
                "memory"
            ]

            for key in preferred_keys:
                value = memory.get(key)

                if value:
                    return str(value).strip()

            return self._format_value(memory)

        return str(memory).strip()

    @staticmethod
    def _format_memories(
        memories: List[str]
    ) -> str:
        if not memories:
            return "- No relevant verified memories were retrieved."

        return "\n".join(
            f"- {memory}"
            for memory in memories
        )

    # =========================================================
    # Validation
    # =========================================================

    def _validate_reply(
        self,
        generated_reply: str,
        data: Dict[str, Any],
        dialogue_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        if not generated_reply:
            return {
                "valid": False,
                "reason": "The generated reply is empty."
            }

        normalized = generated_reply.lower().strip()

        forbidden_markers = [
            "as an ai",
            "language model",
            "system prompt",
            "npc intent:",
            "emotion:",
            "tone:",
            "player says:",
            "dialogue decision:",
            "relevant verified memories:"
        ]

        for marker in forbidden_markers:
            if marker in normalized:
                return {
                    "valid": False,
                    "reason": (
                        "The generated reply contains a forbidden "
                        f"system marker: {marker}"
                    )
                }

        if len(generated_reply) > 1200:
            return {
                "valid": False,
                "reason": "The generated reply is too long."
            }

        target = str(
            dialogue_context.get(
                "target",
                ""
            )
        ).lower()

        if target and normalized.startswith(
            f"{target}:"
        ):
            return {
                "valid": False,
                "reason": "The reply contains an unwanted speaker label."
            }

        refuses_conversation = bool(
            data.get(
                "refuses_conversation",
                False
            )
        )

        allow_trade = bool(
            data.get(
                "allow_trade",
                False
            )
        )

        if refuses_conversation and allow_trade:
            return {
                "valid": False,
                "reason": (
                    "The dialogue decision contains contradictory "
                    "conversation and trade permissions."
                )
            }

        return {
            "valid": True,
            "reason": ""
        }

    @staticmethod
    def _clean_reply(
        reply: str
    ) -> str:
        cleaned = reply.strip()

        if (
            len(cleaned) >= 2
            and cleaned[0] == '"'
            and cleaned[-1] == '"'
        ):
            cleaned = cleaned[1:-1].strip()

        prefixes = [
            "NPC:",
            "Bartender:",
            "Merchant:",
            "Guard:",
            "Traveler:",
            "Traveller:",
            "Goblin:"
        ]

        for prefix in prefixes:
            if cleaned.lower().startswith(
                prefix.lower()
            ):
                cleaned = cleaned[len(prefix):].strip()
                break

        return cleaned

    # =========================================================
    # Fallbacks
    # =========================================================

    def _generate_fallback_reply(
        self,
        data: Dict[str, Any],
        dialogue_context: Dict[str, Any]
    ) -> str:
        npc_intent = str(
            data.get(
                "npc_intent",
                "answer_player"
            )
        ).lower()

        emotion = str(
            data.get(
                "emotion",
                "neutral"
            )
        ).lower()

        if npc_intent in {
            "order_player_to_leave",
            "refuse_trade",
            "refuse_conversation",
            "avoid_player"
        }:
            if emotion == "afraid":
                return (
                    "Stay back. I have nothing more to say to you."
                )

            return (
                "Leave me alone. I will not speak with you."
            )

        if npc_intent in {
            "consider_apology",
            "accept_apology"
        }:
            return (
                "I heard your apology, but trust will take time "
                "and more than words."
            )

        if npc_intent in {
            "share_information",
            "share_secret"
        }:
            return (
                "I may know something, but I need you to be more "
                "specific about what you want to hear."
            )

        if npc_intent in {
            "welcome",
            "greet"
        }:
            return (
                "Welcome. Tell me what brings you here."
            )

        return (
            "I am listening. What exactly do you want to know?"
        )

    # =========================================================
    # Result building
    # =========================================================

    def _build_success_result(
        self,
        dialogue_result: Dict[str, Any],
        generated_reply: str,
        response_source: str,
        generation_time: float,
        prompt: str,
        relevant_memories: List[str],
        validation: Dict[str, Any],
        generation_error: Optional[str] = None
    ) -> Dict[str, Any]:
        original_data = dialogue_result.get(
            "data",
            {}
        )

        if not isinstance(original_data, dict):
            original_data = {}

        result_data = dict(original_data)

        result_data.update({
            "npc_reply": generated_reply,
            "response_source": response_source,
            "model_used": (
                self.model_name
                if response_source in {
                    "llm",
                    "llm_with_memory"
                }
                else None
            ),
            "generation_time_seconds": generation_time,
            "retrieved_memory_count": len(
                relevant_memories
            ),
            "dialogue_validation": validation
        })

        if generation_error:
            result_data["generation_error"] = generation_error

        if self.debug:
            result_data["generation_prompt"] = prompt

            print(
                f"[DEBUG] Response source: {response_source}"
            )
            print(
                f"[DEBUG] Model used: "
                f"{result_data.get('model_used')}"
            )
            print(
                f"[DEBUG] Retrieved memories: "
                f"{len(relevant_memories)}"
            )
            print(
                f"[DEBUG] Generation time: "
                f"{generation_time:.3f}s"
            )

        return {
            "success": True,
            "message": generated_reply,
            "player_response": generated_reply,
            "state_updates": dialogue_result.get(
                "state_updates",
                {}
            ),
            "data": result_data
        }

    def _handle_unsuccessful_decision(
        self,
        dialogue_result: Dict[str, Any],
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        existing_message = str(
            dialogue_result.get(
                "message",
                ""
            )
        ).strip()

        if not existing_message:
            existing_message = (
                "The character cannot respond."
            )

        result_data = dict(data)

        result_data.update({
            "npc_reply": existing_message,
            "response_source": "deterministic",
            "model_used": None
        })

        return {
            "success": False,
            "message": existing_message,
            "player_response": existing_message,
            "state_updates": dialogue_result.get(
                "state_updates",
                {}
            ),
            "data": result_data
        }

    def _handle_non_generated_result(
        self,
        dialogue_result: Dict[str, Any],
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        existing_message = str(
            dialogue_result.get(
                "message",
                ""
            )
        ).strip()

        if not existing_message:
            existing_message = (
                "The character does not respond."
            )

        result_data = dict(data)

        result_data.update({
            "npc_reply": existing_message,
            "response_source": "deterministic",
            "model_used": None
        })

        return {
            "success": bool(
                dialogue_result.get(
                    "success",
                    True
                )
            ),
            "message": existing_message,
            "player_response": existing_message,
            "state_updates": dialogue_result.get(
                "state_updates",
                {}
            ),
            "data": result_data
        }

    def _build_error_result(
        self,
        dialogue_result: Dict[str, Any],
        error: str
    ) -> Dict[str, Any]:
        return {
            "success": False,
            "message": error,
            "player_response": error,
            "state_updates": dialogue_result.get(
                "state_updates",
                {}
            ),
            "data": {
                "response_source": "error",
                "model_used": None,
                "generation_error": error
            }
        }

    # =========================================================
    # Formatting
    # =========================================================

    @staticmethod
    def _get_length_instruction(
        response_length: str
    ) -> str:
        normalized = response_length.lower()

        if normalized == "short":
            return "Use one or two short sentences."

        if normalized == "long":
            return "Use three to five concise sentences."

        return "Use two or three concise sentences."

    def _format_value(
        self,
        value: Any
    ) -> str:
        if value is None:
            return "unknown"

        if isinstance(value, dict):
            if not value:
                return "unknown"

            return ", ".join(
                f"{key}={self._format_value(item)}"
                for key, item in value.items()
            )

        if isinstance(value, (list, tuple, set)):
            if not value:
                return "unknown"

            return ", ".join(
                self._format_value(item)
                for item in value
            )

        return str(value)

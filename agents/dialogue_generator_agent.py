from __future__ import annotations

from typing import Any, Dict, List, Optional
import time
import re

import requests

from agents.dialogue_validator_agent import DialogueValidatorAgent


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
        timeout: int = 180,
        max_memory_items: int = 3,
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

        has_personal_beliefs = bool(
            dialogue_context.get("relevant_semantic_beliefs", [])
        )

        response_source = (
            "llm_with_memory"
            if relevant_memories or has_personal_beliefs
            else "llm"
        )

        # A claim about a past sale is a factual question, not an invitation
        # to invent a shared memory. When there is no supporting NPC-specific
        # record, answer deterministically instead of trusting an LLM to deny it.
        # This guard is deliberately limited to past-sale claims.
        if self._is_unverified_personal_sale(dialogue_context):
            if self.debug:
                print("[NPC MEMORY GUARD] Unverified personal sale; using grounded reply.")
            return self._build_success_result(
                dialogue_result=dialogue_result,
                generated_reply=self._generate_fallback_reply(data, dialogue_context),
                response_source="deterministic_fallback",
                generation_time=0.0,
                prompt=prompt,
                relevant_memories=relevant_memories,
                validation={
                    "valid": True,
                    "reason": "Unverified past sale handled without LLM generation.",
                },
            )

        start_time = time.perf_counter()

        try:
            generated_reply = self._call_ollama(prompt)

            # Keep a complete portion of the model's answer when possible.
            # Do not replace ordinary unpunctuated dialogue unnecessarily.
            if self._reply_looks_truncated(generated_reply):
                salvaged = self._salvage_complete_reply(generated_reply)
                if salvaged and self._validate_reply(
                    salvaged, data, dialogue_context
                )["valid"]:
                    if self.debug:
                        print("[NPC REPLY] Removed incomplete trailing text.")
                    generated_reply = salvaged
                    response_source = "llm_salvaged"
                else:
                    if self.debug:
                        print("[NPC REPLY] No complete valid reply; using contextual fallback.")
                    generated_reply = self._truncation_fallback(data, dialogue_context)
                    response_source = "deterministic_fallback"

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

                generated_reply = self._truncation_fallback(
                    data=data,
                    dialogue_context=dialogue_context
                )
                response_source = "deterministic_fallback"
                validation = self._validate_reply(
                    generated_reply, data, dialogue_context
                )

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

        semantic_beliefs = dialogue_context.get(
            "relevant_semantic_beliefs",
            []
        )
        belief_text = self._format_semantic_beliefs(
            semantic_beliefs
        )

        conversation_history = dialogue_context.get(
            "conversation_history",
            []
        )
        conversation_text = self._format_conversation_history(
            conversation_history
        )
        personality_text = self._format_value(personality)
        goals_text = self._format_value(goals)

        world_context = self._build_world_context(
            game_state
        )

        role_guidance = self._role_guidance(target)

        # Keep the prompt compact. Past event records can contain full game-state
        # snapshots; those belong to the memory system, not the LLM's dialogue.
        follow_up = bool(conversation_history)
        opening_rule = (
            "Continue the conversation directly. Do NOT greet the player again, "
            "say welcome back, or repeat an introduction."
            if follow_up else
            "A brief greeting is allowed only if the player greets you."
        )
        return f"""You are the {target}, speaking directly to the player in a fantasy RPG.
Role boundaries: {role_guidance}
Personality: {personality_text}
Goals: {goals_text}
Current emotion: {emotion}; tone: {tone}; attitude: trust={dialogue_context.get('trust', 'unknown')}, hostile={dialogue_context.get('hostile', False)}.
Decision: {npc_intent}; topic={topic}; trade_allowed={allow_trade}; refusing={refuses_conversation}.
Verified world context:
{world_context}
Recent dialogue:
{conversation_text}
Relevant past events (may be incomplete; do not assume they are current facts):
{memory_text}
Personal beliefs (opinions, not objective facts):
{belief_text}

Rules:
- Answer the player's ACTUAL latest question first. Stay in character and speak in first person only.
- {opening_rule}
- Usually 1-2 concise sentences; up to 3 only if the question needs detail. No speeches or repeated sales pitches.
- Keep natural personality, but do not repeat wording from recent NPC replies.
- Do not invent a confirmed inventory item, fixed price, completed sale, action, or event.
- If no verified price or stock data is supplied, say you cannot confirm it; do not invent a price.
- Do not confuse your occupation with another NPC's. Do not claim to remember a past event without supporting evidence.
- No narration, role labels, system explanations, or commentary.

Player: {player_input}
NPC:""".strip()

    @staticmethod
    def _role_guidance(target: str) -> str:
        role = str(target or "").strip().lower()
        guidance = {
            "merchant": (
                "You sell goods and negotiate prices. You are not the bartender "
                "or innkeeper; you do not serve drinks or invent customers "
                "waiting for drinks. You may discuss the tavern if relevant."
            ),
            "bartender": (
                "You serve drinks and converse with patrons. You are not "
                "the merchant; do not claim to sell the merchant's goods."
            ),
        }
        return guidance.get(role, "Follow the supplied NPC profile and established role.")

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
            "keep_alive": "30m",
            "options": {
                "temperature": 0.85,
                "top_p": 0.95,
                "repeat_penalty": 1.15,

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

        # Ollama reports "length" when generation reaches num_predict.
        # A successful HTTP response does not guarantee a complete sentence.
        if result.get("done_reason") == "length":
            if self.debug:
                print("[OLLAMA] Token limit reached (done_reason=length).")
            generated_reply = generated_reply.rstrip() + " [TRUNCATED]"

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
        cached_prompt_tokens = int(
            result.get("prompt_eval_cached_count") or 0
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
            print(f"Cached prompt tokens:{cached_prompt_tokens:>8}")
            print(f"Generated tokens:    {generated_tokens}")
            print(f"Stop reason:         {result.get('done_reason', 'unknown')}")
            print(f"Generation speed:    {tokens_per_second} tokens/s")
            print("[/OLLAMA PROFILE]\n")

        return self._clean_reply(generated_reply)

    @staticmethod
    def _reply_looks_truncated(reply: str) -> bool:
        text = reply.strip()
        if not text:
            return True
        if text.endswith("[TRUNCATED]"):
            return True
        # A dangling list item, e.g. "; 3" or "3.", is incomplete.
        if re.search(r"(?:^|[;\s])\d+[.)]?\s*$", text):
            return True
        # Ordinary dialogue can legitimately end without terminal punctuation.
        # Do not reject it unless the model reports a length stop above.
        return False

    @staticmethod
    def _salvage_complete_reply(reply: str) -> str:
        text = reply.replace("[TRUNCATED]", "").strip()
        # Preserve the longest prefix ending in a complete sentence. A period
        # within a numbered item ("1.") is not a sentence boundary.
        boundaries = list(re.finditer(r"[!?](?=\s|$)|(?<!\d)\.(?=\s|$)", text))
        for boundary in reversed(boundaries):
            candidate = text[:boundary.end()].strip()
            if candidate and not re.search(r"(?:^|[;\s])\d+[.)]?\s*$", candidate):
                return candidate
        return ""

    def _truncation_fallback(
        self,
        data: Dict[str, Any],
        dialogue_context: Dict[str, Any]
    ) -> str:
        # Preserve refusals, threats, apologies and other structured decisions.
        npc_intent = str(data.get("npc_intent", "")).lower()
        if npc_intent in {
            "order_player_to_leave", "refuse_trade", "refuse_conversation",
            "avoid_player", "consider_apology", "accept_apology"
        } or self._is_unverified_personal_sale(dialogue_context):
            return self._generate_fallback_reply(data, dialogue_context)

        target = str(dialogue_context.get("target", "")).strip().lower()
        player_input = str(dialogue_context.get("player_input", "")).lower()
        if target == "merchant" and re.search(
            r"\b(?:sell|selling|goods|wares|stock|items)\b", player_input
        ):
            return (
                "I trade in supplies and equipment for travelers. "
                "What kind of goods are you looking for?"
            )
        return self._generate_fallback_reply(data, dialogue_context)

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

    @staticmethod
    def _format_semantic_beliefs(
        beliefs: Any
    ) -> str:
        """
        Format personal NPC beliefs separately from event memories.

        Negative scores are rendered as opposite beliefs rather than as
        confusing negative confidence values.
        """
        if not isinstance(beliefs, list) or not beliefs:
            return "- No stable personal beliefs about the player yet."

        opposites = {
            "dangerous": "not dangerous",
            "trustworthy": "untrustworthy",
            "generous": "not generous",
        }

        lines: List[str] = []

        for belief in beliefs[:5]:
            if not isinstance(belief, dict):
                continue

            subject = str(belief.get("subject", "player"))
            predicate = str(belief.get("predicate", "unknown"))

            try:
                score = float(
                    belief.get(
                        "score",
                        belief.get("confidence", 0.0),
                    )
                )
            except (TypeError, ValueError):
                continue

            score = max(-1.0, min(1.0, score))
            strength = abs(score)

            if score >= 0:
                meaning = predicate
            else:
                meaning = opposites.get(
                    predicate,
                    f"not {predicate}",
                )

            if strength >= 0.75:
                qualifier = "strongly believes"
            elif strength >= 0.40:
                qualifier = "believes"
            else:
                qualifier = "somewhat believes"

            lines.append(
                f"- {subject}: {qualifier} {meaning} "
                f"(belief score {score:+.2f})"
            )

        if not lines:
            return "- No stable personal beliefs about the player yet."

        return "\n".join(lines)

    @staticmethod
    def _format_conversation_history(
        history: Any
    ) -> str:
        """
        Format short NPC-specific working memory separately from semantic
        memory. This is intentionally chronological and explicit about
        speakers so follow-up references remain resolvable.
        """
        if not isinstance(history, list) or not history:
            return "- No previous turns with this NPC in the current conversation."

        lines: List[str] = []

        for turn in history[-2:]:
            if not isinstance(turn, dict):
                continue

            player_text = str(turn.get("player") or "").strip()
            npc_text = str(turn.get("npc") or "").strip()

            if player_text:
                lines.append(f"Player: {player_text[:180]}")
            if npc_text:
                lines.append(f"NPC: {npc_text[:260]}")

        if not lines:
            return "- No previous turns with this NPC in the current conversation."

        return "\n".join(lines)

    def _format_memory_item(
        self,
        memory: Any
    ) -> str:
        if memory is None:
            return ""

        if isinstance(memory, str):
            text = memory.strip()
            # Semantic retrieval sometimes returns complete event dumps, including
            # irrelevant state snapshots. Retain only the actual interaction.
            if "Player input:" in text and "System result:" in text:
                player_match = re.search(r"Player input:\s*(.*?)(?:\n(?:Intent|Target|System result):|$)", text, re.S)
                reply_match = re.search(r"System result:\s*(.*?)(?:\n(?:State changes|Important state):|$)", text, re.S)
                event_match = re.search(r"Event type:\s*([^\n]+)", text)
                player = " ".join(player_match.group(1).split())[:130] if player_match else ""
                reply = " ".join(reply_match.group(1).split())[:170] if reply_match else ""
                event_type = event_match.group(1).strip().lower() if event_match else ""
                # Old logs may mislabel ordinary questions as attacks. Trust the
                # action intent rather than a contradictory legacy event label.
                intent_match = re.search(r"Intent:\s*([^\n]+)", text)
                intent = intent_match.group(1).strip().lower() if intent_match else ""
                if event_type == "attack" and intent == "dialogue_action":
                    event_type = "dialogue"
                return f"{event_type or 'interaction'}: player asked {player!r}; NPC replied {reply!r}"
            return " ".join(text.split())[:320]

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
            f"- {memory[:360]}"
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
            "relevant verified memories:",
            "here's what the bartender might say",
            "here is what the bartender might say",
            "here's what the npc might say",
            "here is what the npc might say",
            "the bartender might say",
            "the npc might say",
            "a possible response is"
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

        # Narrow, deterministic factual check. An NPC must not confirm a
        # personal sale merely because the player suggested that it happened.
        if self._is_unverified_personal_sale(dialogue_context):
            affirmative = re.search(
                r"\b(?:yes|indeed|certainly|of course|absolutely|"
                r"i (?:do |certainly )?remember|i recall|i sold|"
                r"i served|i remember you|familiar face|welcome back)\b",
                normalized,
            )
            denial = re.search(
                r"\b(?:don't|do not|cannot|can't|not|never|no recollection|"
                r"no memory|unsure|uncertain|perhaps|maybe)\b",
                normalized,
            )
            if affirmative and not denial:
                return {
                    "valid": False,
                    "reason": "NPC affirmed a personal sale without supporting memory.",
                }

        role_conflict = DialogueValidatorAgent.find_role_conflict(
            reply=normalized,
            target=target,
            player_input=str(dialogue_context.get("player_input") or ""),
        )
        if role_conflict:
            return {"valid": False, "reason": role_conflict}

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
    # Personal-event grounding
    # =========================================================

    def _is_unverified_personal_sale(
        self,
        dialogue_context: Dict[str, Any]
    ) -> bool:
        """Identify unsupported player claims that this NPC sold them something.

        This intentionally targets only explicit past-sale questions; it is not
        a general-purpose factual verifier. The existing memory systems remain
        the authority on what an NPC knows.
        """
        player_input = str(dialogue_context.get("player_input") or "").lower()
        # Only explicit claims/questions about a completed transaction qualify.
        # Present-tense inventory questions such as "What do you sell?" must
        # reach the dialogue model instead of triggering the memory fallback.
        mentions_sale = bool(re.search(
            r"\b(?:you|your)\b(?:\W+\w+){0,5}\W+"
            r"(?:sold|gave|served)\b|"
            r"\b(?:did you|have you|had you)\s+(?:ever\s+)?"
            r"(?:sell|sold|give|given|serve|served)\s+(?:me|us)\b|"
            r"\b(?:sold|gave|served)\s+(?:me|us)\b|"
            r"\b(?:sold|gave|served)\s+to\s+(?:me|us)\b",
            player_input,
        ))
        if not mentions_sale:
            return False

        memories = self._collect_relevant_memories(dialogue_context)
        # Presence of memory alone is not proof: require explicit sale evidence.
        verified_sale = any(re.search(
            r"\b(?:you order|you bought|you purchased|sold (?:you|the player)|"
            r"sold to (?:you|the player)|served (?:you|the player)|"
            r"player (?:bought|purchased|ordered))\b",
            memory.lower(),
        ) for memory in memories)
        return not verified_sale

    # =========================================================
    # Fallbacks
    # =========================================================

    def _generate_fallback_reply(
        self,
        data: Dict[str, Any],
        dialogue_context: Dict[str, Any]
    ) -> str:
        if self._is_unverified_personal_sale(dialogue_context):
            return (
                "I don't recall selling you anything myself, traveler. "
                "Perhaps you're thinking of someone else?"
            )

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
                    "llm_with_memory",
                    "llm_salvaged"
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
            return "Use about three natural sentences; stay vivid and in character."

        if normalized == "long":
            return "Use four to five natural sentences with atmosphere, personality and detail."

        return "Use three to five natural sentences with personality and useful detail."

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

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


class DialogueValidatorAgent:
    """
    Validates a generated NPC reply.

    This agent does not call an LLM.

    Responsibilities:
    - remove speaker labels and quotation marks;
    - reject system-prompt leakage;
    - reject empty or excessively long replies;
    - detect Cyrillic text;
    - detect narration instead of spoken dialogue;
    - detect suspicious claims about state-changing actions;
    - provide a deterministic fallback when validation fails.
    """

    MAX_REPLY_LENGTH = 700
    MAX_SENTENCES = 5

    FORBIDDEN_SYSTEM_MARKERS = (
        "as an ai",
        "as a language model",
        "system prompt",
        "system message",
        "developer message",
        "npc intent:",
        "dialogue decision:",
        "relevant memories:",
        "verified memories:",
        "player input:",
        "player says:",
        "current game state:",
        "instructions:",
        "output rules:"
    )

    STATE_CHANGE_PATTERNS = (
        r"\bi gave you\b",
        r"\bi have given you\b",
        r"\byou received\b",
        r"\byou gained\b",
        r"\byou lost\b",
        r"\bi added\b",
        r"\bi removed\b",
        r"\bi healed you\b",
        r"\bi damaged you\b",
        r"\bi completed\b",
        r"\bquest completed\b",
        r"\byour gold is now\b",
        r"\byour health is now\b",
        r"\bi sold you\b",
        r"\bi bought\b"
    )

    NARRATION_PREFIXES = (
        "the bartender ",
        "the merchant ",
        "the guard ",
        "the traveler ",
        "the traveller ",
        "the goblin ",
        "the npc ",
        "he says",
        "she says",
        "they say",
        "he replies",
        "she replies",
        "they reply"
    )

    def __init__(
        self,
        max_reply_length: int = MAX_REPLY_LENGTH,
        debug: bool = True
    ) -> None:
        self.max_reply_length = max_reply_length
        self.debug = debug

    def validate(
        self,
        generated_result: Dict[str, Any],
        dialogue_decision: Optional[Dict[str, Any]] = None,
        game_state: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Validates the output returned by DialogueGeneratorAgent.
        """

        dialogue_decision = dialogue_decision or {}
        game_state = game_state or {}

        if not isinstance(generated_result, dict):
            return self._invalid_result(
                reason="DialogueGeneratorAgent returned an invalid object.",
                original_result={}
            )

        raw_reply = self._extract_reply(generated_result)
        cleaned_reply = self._clean_reply(raw_reply)

        validation_errors: List[str] = []
        validation_warnings: List[str] = []

        if not cleaned_reply:
            validation_errors.append(
                "The generated reply is empty."
            )

        if len(cleaned_reply) > self.max_reply_length:
            validation_errors.append(
                "The generated reply exceeds the maximum length."
            )

        if self._contains_cyrillic(cleaned_reply):
            validation_errors.append(
                "The generated reply contains Cyrillic characters."
            )

        system_marker = self._find_system_marker(
            cleaned_reply
        )

        if system_marker:
            validation_errors.append(
                f"System prompt leakage detected: {system_marker}"
            )

        if self._looks_like_narration(cleaned_reply):
            validation_warnings.append(
                "The generated reply may contain third-person narration."
            )

        state_change_claim = self._find_state_change_claim(
            cleaned_reply
        )

        if state_change_claim:
            validation_errors.append(
                "The generated dialogue claims an unverified state change: "
                f"{state_change_claim}"
            )

        sentence_count = self._count_sentences(
            cleaned_reply
        )

        if sentence_count > self.MAX_SENTENCES:
            validation_warnings.append(
                f"The reply contains {sentence_count} sentences."
            )

        decision_data = dialogue_decision.get(
            "data",
            {}
        )

        if not isinstance(decision_data, dict):
            decision_data = {}

        refuses_conversation = bool(
            decision_data.get(
                "refuses_conversation",
                False
            )
        )

        allow_trade = bool(
            decision_data.get(
                "allow_trade",
                False
            )
        )

        if refuses_conversation and allow_trade:
            validation_errors.append(
                "The dialogue decision simultaneously refuses conversation "
                "and permits trading."
            )

        if validation_errors:
            fallback_reply = self._build_fallback_reply(
                dialogue_decision=dialogue_decision,
                game_state=game_state
            )

            return self._invalid_result(
                reason="; ".join(validation_errors),
                original_result=generated_result,
                fallback_reply=fallback_reply,
                warnings=validation_warnings
            )

        final_result = dict(generated_result)

        result_data = final_result.get(
            "data",
            {}
        )

        if not isinstance(result_data, dict):
            result_data = {}

        result_data = dict(result_data)

        result_data.update({
            "dialogue_validation": {
                "valid": True,
                "errors": [],
                "warnings": validation_warnings,
                "original_reply": raw_reply,
                "cleaned_reply": cleaned_reply
            },
            "validator_used": True
        })

        final_result["success"] = True
        final_result["message"] = cleaned_reply
        final_result["player_response"] = cleaned_reply
        final_result["data"] = result_data

        if self.debug:
            print("[DIALOGUE VALIDATOR]")
            print("Valid: True")
            print(f"Warnings: {validation_warnings}")
            print("[/DIALOGUE VALIDATOR]")

        return final_result

    def _invalid_result(
        self,
        reason: str,
        original_result: Dict[str, Any],
        fallback_reply: Optional[str] = None,
        warnings: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        fallback_reply = (
            fallback_reply
            or "I have nothing more to say right now."
        )

        warnings = warnings or []

        original_data = original_result.get(
            "data",
            {}
        )

        if not isinstance(original_data, dict):
            original_data = {}

        result_data = dict(original_data)

        result_data.update({
            "response_source": "validator_fallback",
            "model_used": None,
            "validator_used": True,
            "dialogue_validation": {
                "valid": False,
                "errors": [reason],
                "warnings": warnings,
                "original_reply": self._extract_reply(
                    original_result
                ),
                "cleaned_reply": fallback_reply
            }
        })

        if self.debug:
            print("[DIALOGUE VALIDATOR]")
            print("Valid: False")
            print(f"Reason: {reason}")
            print(f"Fallback: {fallback_reply}")
            print("[/DIALOGUE VALIDATOR]")

        return {
            "success": True,
            "message": fallback_reply,
            "player_response": fallback_reply,
            "state_updates": original_result.get(
                "state_updates",
                {}
            ),
            "data": result_data
        }

    @staticmethod
    def _extract_reply(
        generated_result: Dict[str, Any]
    ) -> str:
        if not isinstance(generated_result, dict):
            return ""

        possible_values = [
            generated_result.get("player_response"),
            generated_result.get("message")
        ]

        data = generated_result.get(
            "data",
            {}
        )

        if isinstance(data, dict):
            possible_values.extend([
                data.get("npc_reply"),
                data.get("generated_reply")
            ])

        for value in possible_values:
            if value is not None and str(value).strip():
                return str(value).strip()

        return ""

    @staticmethod
    def _contains_cyrillic(text: str) -> bool:
        return any(
            "\u0400" <= character <= "\u04FF"
            for character in text
        )

    def _find_system_marker(
        self,
        text: str
    ) -> Optional[str]:
        normalized = text.lower()

        for marker in self.FORBIDDEN_SYSTEM_MARKERS:
            if marker in normalized:
                return marker

        return None

    def _find_state_change_claim(
        self,
        text: str
    ) -> Optional[str]:
        normalized = text.lower()

        for pattern in self.STATE_CHANGE_PATTERNS:
            match = re.search(
                pattern,
                normalized
            )

            if match:
                return match.group(0)

        return None

    def _looks_like_narration(
        self,
        text: str
    ) -> bool:
        normalized = text.lower().strip()

        return normalized.startswith(
            self.NARRATION_PREFIXES
        )

    @staticmethod
    def _count_sentences(text: str) -> int:
        sentences = re.split(
            r"[.!?]+",
            text
        )

        return len([
            sentence
            for sentence in sentences
            if sentence.strip()
        ])

    @staticmethod
    def _clean_reply(text: str) -> str:
        cleaned = str(text or "").strip()

        if (
            len(cleaned) >= 2
            and cleaned[0] in {'"', "“", "«"}
            and cleaned[-1] in {'"', "”", "»"}
        ):
            cleaned = cleaned[1:-1].strip()

        speaker_pattern = re.compile(
            r"^(bartender|barmaid|barman|innkeeper|merchant|guard|"
            r"traveler|traveller|goblin|npc)\s*:\s*",
            flags=re.IGNORECASE
        )

        cleaned = speaker_pattern.sub(
            "",
            cleaned
        ).strip()

        cleaned = re.sub(
            r"\s+",
            " ",
            cleaned
        )

        return cleaned

    def _build_fallback_reply(
        self,
        dialogue_decision: Dict[str, Any],
        game_state: Dict[str, Any]
    ) -> str:
        data = dialogue_decision.get(
            "data",
            {}
        )

        if not isinstance(data, dict):
            data = {}

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

        refuses_conversation = bool(
            data.get(
                "refuses_conversation",
                False
            )
        )

        if refuses_conversation:
            if emotion in {
                "afraid",
                "fearful",
                "nervous"
            }:
                return (
                    "Stay back. I do not wish to speak with you."
                )

            return (
                "I have nothing to say to you. Leave me alone."
            )

        if npc_intent in {
            "order_player_to_leave",
            "threaten_player",
            "warn_player"
        }:
            return (
                "You should leave before this situation becomes worse."
            )

        if npc_intent in {
            "accept_apology",
            "consider_apology"
        }:
            return (
                "I heard your apology, but rebuilding trust will take time."
            )

        if npc_intent in {
            "refuse_trade",
            "deny_service"
        }:
            return (
                "I will not do business with you."
            )

        if npc_intent in {
            "greet",
            "welcome"
        }:
            return (
                "Welcome. What brings you here?"
            )

        if npc_intent in {
            "share_information",
            "answer_question",
            "answer_player"
        }:
            return (
                "I am not certain how to answer that."
            )

        return (
            "I am listening. Speak plainly."
        )

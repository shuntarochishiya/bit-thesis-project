from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from agents.world_facts import (
    grounded_reply, grounded_followup_reply, confirmed_shipments,
    public_events, _verified_stock, is_news_question, is_news_followup, news_followup_anchor,
)


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

    @staticmethod
    def find_role_conflict(reply: str, target: str, player_input: str = "") -> Optional[str]:
        """Catch clear occupational role confusion, not ordinary tavern references.

        This is deliberately narrow: a merchant may mention drinks or taverns,
        but should not act as the bartender or divert an unrelated exchange
        into serving a drink. Contextual mentions remain permissible.
        """
        role = str(target or "").strip().lower()
        text = str(reply or "").lower()
        request = str(player_input or "").lower()

        if role != "merchant":
            return None

        # Merchant explicitly claiming to perform bartender duties.
        bartender_claims = (
            r"\bi\s+(?:am|work as)\s+(?:the\s+|a\s+)?(?:bartender|barman|barmaid|innkeeper)\b",
            r"\bi\s+(?:will|can|shall|could|would|\x27ll)\s+(?:pour|serve)\s+(?:you\s+)?(?:a\s+|some\s+)?(?:beer|ale|wine|mead|drink|liquor)\b",
            r"\b(?:let me|i\x27ll|i will)\s+(?:pour|fetch|bring)\s+(?:you\s+)?(?:a\s+|some\s+)?(?:beer|ale|wine|mead|drink)\b",
        )
        for pattern in bartender_claims:
            if re.search(pattern, text):
                return "Merchant assumes bartender duties."

        # Do not treat an on-topic discussion of drinks as a mistake.
        if re.search(r"\b(?:drink|wine|ale|beer|mead|tavern|inn|bar|thirst)\b", request):
            return None

        # Specific observed failure: a merchant unexpectedly dismisses the
        # player because a different customer supposedly needs a drink.
        if re.search(
            r"\b(?:customer|patron|someone|traveler|traveller)\b.{0,65}"
            r"\b(?:need|needs|want|wants|order|orders|waiting for)\b.{0,35}"
            r"\b(?:a\s+)?(?:drink|beer|ale|wine|mead)\b",
            text,
        ):
            return "Merchant introduces unrelated bartender service."
        return None

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

        # Creative worldbuilding is intentionally permitted: goods, places,
        # rumors, legends and fictional events are narrative content, not
        # authoritative state mutations. Mechanical changes are still checked
        # by STATE_CHANGE_PATTERNS and the execution engine above.

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

        # Validate occupational grounding against the NPC selected by Unity.
        context = decision_data.get("dialogue_context", {})
        if not isinstance(context, dict):
            context = {}
        target = str(context.get("target") or decision_data.get("target") or "")
        player_input = str(context.get("player_input") or "")
        role_conflict = self.find_role_conflict(cleaned_reply, target, player_input)
        if role_conflict:
            validation_errors.append(role_conflict)

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

    @staticmethod
    def _find_unverified_world_claim(
        reply: str,
        game_state: Dict[str, Any],
        target: str,
        player_input: str = "",
        conversation_history: Any = None,
    ) -> Optional[str]:
        text = str(reply or "").lower()
        if target == "merchant" and not confirmed_shipments(game_state, target):
            if re.search(
                r"\b(?:we|i)\s+(?:have\s+)?(?:just\s+|recently\s+)?"
                r"(?:received|restocked|got\s+in)\s+(?:some|new|a\s+|an\s+|the\s+)", text
            ) or re.search(
                r"\b(?:new|fresh|recent)\s+(?:shipment|delivery|stock|arrivals)\b", text
            ):
                return "Merchant claimed an unverified delivery or restock."

        # When the merchant has no authoritative stock record, do not let
        # a spontaneous LLM reply assert specific products are available.
        if target == "merchant" and _verified_stock(game_state, target) is None:
            if re.search(
                r"\b(?:we|i)\s+(?:have|carry|stock|sell|offer)\s+"
                r"(?:(?:some|a|an|new|durable|enchanted|fresh)\s+){0,4}"
                r"(?:backpacks?|lanterns?|potions?|weapons?|swords?|"
                r"armou?r|shields?|maps?|torches?)\b", text
            ):
                return "Merchant asserted unverified specific stock."

        if target == "merchant" and _verified_stock(game_state, target) is None:
            # Capture positive first-person inventory claims regardless of item
            # noun; generic role statements ("I sell supplies") are okay.
            claim = re.search(
                r"\b(?:i|we)\s+(?:currently\s+)?(?:have|carry|stock|sell|offer)\s+"
                r"(?:(?:some|a|an|new|fresh|fine|enchanted)\s+){0,3}"
                r"([a-z][a-z -]{1,48})", text
            )
            if claim:
                object_text = claim.group(1).strip().split(".")[0].split(",")[0].strip()
                generic = {"goods", "supplies", "wares", "equipment", "items", "things",
                           "nothing", "no idea", "no stock", "no goods", "no supplies",
                           "a shop", "a business", "heard", "been", "time", "to"}
                if not (object_text.startswith(("no ", "not ", "nothing", "never "))
                        or any(object_text == word or object_text.startswith(word + " ")
                               for word in generic)
                        or object_text.startswith(("a feeling", "a question", "a story",
                                                   "a suspicion", "a thought", "my doubts",
                                                   "an idea", "some doubts", "some news"))):
                    return "Merchant asserted specific goods without verified stock."

        # Rumors and atmospheric dialogue are not world-state mutations.
        # Do not reject a harmless "I heard whispers" phrase merely because
        # the player's reply lacks a fixed keyword. Only reject actual
        # contradictions, fabricated named locations, or a different rumor
        # substituted for the topic the player is actively following up on.
        allowed_news_context = (
            is_news_question(player_input)
            or is_news_followup(player_input, conversation_history)
        )
        if re.search(r"\b(?:dawnbreak lake|old keep)\b", text):
            return "NPC invented a named location not present in world state."
        if is_news_followup(player_input, conversation_history):
            previous = str(news_followup_anchor(player_input, conversation_history).get("npc") or "").lower()
            from agents.world_facts import RUMOR_TOPICS
            old_topics = {i for i, (pattern, _, _) in enumerate(RUMOR_TOPICS)
                          if pattern.search(previous)}
            new_topics = {i for i, (pattern, _, _) in enumerate(RUMOR_TOPICS)
                          if pattern.search(text)}
            if old_topics and new_topics and not (old_topics & new_topics):
                return "NPC changed the subject to a different unsupported rumor."

        # Event-specific evidence, rather than merely checking that some
        # unrelated public event exists in the journal.
        observed = {item["event"] for item in public_events(game_state, target)}
        event_claims = (
            (r"\bbandits?\s+(?:have\s+)?(?:blocked|raided|attacked|ambushed)\b", "bandit_roadblock"),
            (r"\b(?:a|an|the)\s+(?:bandit\s+)?roadblock\s+(?:was|has been)\b", "bandit_roadblock"),
            (r"\b(?:an?\s+)?enemy\s+(?:ambushed|attacked)\b", "enemy_ambush"),
            (r"\b(?:a|the)\s+guard\s+patrol\s+(?:arrived|passed|came|was seen)\b", "guard_patrol"),
            (r"\b(?:a|the)\s+merchant\s+cart\s+(?:arrived|passed|came|was seen)\b", "merchant_cart"),
        )
        for pattern, event_id in event_claims:
            match = re.search(pattern, text)
            if match and event_id not in observed:
                # A report clearly introduced as hearsay is not a claim that
                # the event actually occurred. It remains a rumor in dialogue.
                preceding = text[max(0, match.start() - 65):match.start()]
                hedged = bool(re.search(
                    r"(?:some say|people say|rumou?r(?:ed|s)?(?: that)?|"
                    r"i(?:'ve| have) heard(?: that)?|supposedly|allegedly|"
                    r"there is talk(?: that)?|whispers(?: that)?)\s*$", preceding
                ))
                if not (hedged and allowed_news_context):
                    return f"NPC asserted unverified world event: {event_id}."
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

        context = data.get("dialogue_context", {})
        if isinstance(context, dict) and not context.get("hostile"):
            factual = grounded_reply(
                game_state,
                str(context.get("target", "")),
                str(context.get("player_input", "")),
            )
            if factual is None:
                factual = grounded_followup_reply(
                    game_state,
                    str(context.get("target", "")),
                    str(context.get("player_input", "")),
                    context.get("conversation_history", []),
                )
            if factual is not None:
                return factual

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

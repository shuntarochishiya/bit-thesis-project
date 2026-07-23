from __future__ import annotations

from typing import Any, Dict, Iterable, Optional


class IntentRecognitionAgent:
    """
    Deterministic intent recognizer for the RPG backend.

    Responsibilities:
    - classify player input into a high-level action intent;
    - identify the most likely target;
    - keep tavern services separate from NPC dialogue;
    - route persuasion involving both merchant and bartender;
    - avoid unnecessary LLM calls.

    Supported intents:
    - combat_action
    - persuasion_action
    - dialogue_action
    - tavern_action
    - exploration_action
    - general_action
    """

    def __init__(self) -> None:
        # ---------------------------------------------------------
        # Combat
        # ---------------------------------------------------------
        self.combat_words = {
            "attack",
            "hit",
            "strike",
            "stab",
            "slash",
            "shoot",
            "punch",
            "kick",
            "kill",
            "fight",
            "assault",
            "smash",
            "hurt",
            "wound",
            "destroy",
            "ambush",
            "execute",
        }

        # ---------------------------------------------------------
        # Persuasion
        # ---------------------------------------------------------
        self.persuasion_words = {
            "persuade",
            "convince",
            "negotiate",
            "bargain",
            "bribe",
            "threaten",
            "intimidate",
            "pressure",
            "charm",
            "plead",
            "beg",
            "reason",
            "influence",
            "talk into",
            "talk him into",
            "talk her into",
            "talk them into",
            "lower the price",
            "give me a discount",
            "offer a discount",
            "make a deal",
            "let me pass",
            "tell the truth",
            "reveal",
            "cooperate",
        }

        # ---------------------------------------------------------
        # Dialogue
        # ---------------------------------------------------------
        self.dialogue_words = {
            "talk",
            "speak",
            "ask",
            "say",
            "tell",
            "greet",
            "hello",
            "hi",
            "question",
            "chat",
            "conversation",
            "reply",
            "answer",
            "explain",
            "who are you",
            "what is your name",
            "how are you",
            "what happened",
            "tell me more",
            "continue",
            "go on",
            "rumor",
            "rumors",
            "news",
            "information",
            "details",
            "anything interesting",
            "heard anything",
            "know anything",
            "what do you know",
        }

        self.dialogue_continuation_phrases = {
            "tell me more",
            "continue",
            "go on",
            "what else",
            "and then",
            "why",
            "how",
            "really",
            "explain",
            "more details",
        }

        # ---------------------------------------------------------
        # Tavern services only
        # ---------------------------------------------------------
        self.tavern_service_words = {
            "drink",
            "ale",
            "beer",
            "wine",
            "mead",
            "water",
            "food",
            "meal",
            "eat",
            "dinner",
            "lunch",
            "room",
            "rent",
            "sleep",
            "rest",
            "enter tavern",
            "enter the tavern",
            "enter inn",
            "enter the inn",
            "go to the tavern",
            "go inside",
            "walk inside",
            "walk into the tavern",
            "leave tavern",
            "leave the tavern",
            "exit tavern",
            "exit the tavern",
        }

        self.tavern_location_words = {
            "tavern",
            "inn",
            "pub",
            "bar",
            "alehouse",
        }

        # ---------------------------------------------------------
        # Exploration
        # ---------------------------------------------------------
        self.exploration_words = {
            "explore",
            "search",
            "look around",
            "investigate",
            "inspect",
            "travel",
            "move",
            "walk",
            "go",
            "head",
            "leave",
            "enter",
            "approach",
            "visit",
            "follow",
            "climb",
            "descend",
            "cross",
            "return",
            "continue forward",
            "look for",
            "find a path",
            "scout",
        }

        self.location_words = {
            "forest",
            "woods",
            "village",
            "town",
            "city",
            "road",
            "path",
            "river",
            "bridge",
            "mountain",
            "cave",
            "castle",
            "ruins",
            "market",
            "square",
            "gate",
            "field",
            "camp",
            "harbor",
            "dock",
            "tavern",
            "inn",
        }

        # ---------------------------------------------------------
        # Targets
        # ---------------------------------------------------------
        self.target_aliases: Dict[str, set[str]] = {
            "enemy": {
                "enemy",
                "bandit",
                "goblin",
                "orc",
                "monster",
                "guard",
                "attacker",
                "opponent",
                "creature",
                "wolf",
                "skeleton",
                "thief",
            },
            "merchant": {
                "merchant",
                "trader",
                "shopkeeper",
                "vendor",
                "seller",
                "dealer",
            },
            "bartender": {
                "bartender",
                "barman",
                "barmaid",
                "innkeeper",
                "tavern keeper",
                "tavernkeeper",
                "barkeep",
                "tender",
            },
        }

    # =============================================================
    # Public API
    # =============================================================

    def recognize_intent(
        self,
        player_input: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Classify the player's input and return a normalized result.

        Example:
        {
            "intent": "dialogue_action",
            "target": "bartender",
            "confidence": 0.95,
            "reason": "bartender dialogue keyword"
        }
        """
        context = context or {}
        text = self._normalize(player_input)
        target = self.detect_target(text, context)

        if not text:
            return self._build_result(
                intent="general_action",
                target=target,
                confidence=0.2,
                reason="empty input",
            )

        # 1. Combat always has highest priority.
        if self._contains_any(text, self.combat_words):
            return self._build_result(
                intent="combat_action",
                target=target or self._context_target(context),
                confidence=0.99,
                reason="combat keyword",
            )

        # 2. Persuasion must be checked before dialogue.
        if self._contains_any(text, self.persuasion_words):
            persuasion_target = target or self._context_target(context)

            if persuasion_target in {"merchant", "bartender", "enemy"}:
                return self._build_result(
                    intent="persuasion_action",
                    target=persuasion_target,
                    confidence=0.97,
                    reason="persuasion keyword with supported target",
                )

            return self._build_result(
                intent="persuasion_action",
                target=persuasion_target,
                confidence=0.85,
                reason="persuasion keyword",
            )

        # 3. Explicit tavern service requests.
        if self._is_tavern_service(text):
            return self._build_result(
                intent="tavern_action",
                target="bartender",
                confidence=0.96,
                reason="tavern service request",
            )

        # 4. Explicit NPC dialogue.
        if self._is_dialogue(text, target, context):
            return self._build_result(
                intent="dialogue_action",
                target=target or self._context_target(context),
                confidence=0.94,
                reason="dialogue request",
            )

        # 5. Contextual dialogue continuation.
        if self._is_dialogue_continuation(text, context):
            return self._build_result(
                intent="dialogue_action",
                target=target or self._context_target(context),
                confidence=0.9,
                reason="dialogue continuation",
            )

        # 6. Exploration and movement.
        if self._is_exploration(text):
            return self._build_result(
                intent="exploration_action",
                target=target,
                confidence=0.9,
                reason="movement or exploration keyword",
            )

        # 7. Tavern context fallback:
        # only service commands become tavern_action.
        # Generic interaction with the bartender becomes dialogue.
        active_location = self._active_location(context)
        if active_location == "tavern":
            if target == "bartender":
                return self._build_result(
                    intent="dialogue_action",
                    target="bartender",
                    confidence=0.78,
                    reason="bartender interaction inside tavern",
                )

            if self._contains_any(text, self.tavern_location_words):
                return self._build_result(
                    intent="tavern_action",
                    target="bartender",
                    confidence=0.7,
                    reason="tavern context fallback",
                )

        return self._build_result(
            intent="general_action",
            target=target,
            confidence=0.5,
            reason="no specialized rule matched",
        )

    def classify(
        self,
        player_input: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Compatibility alias."""
        return self.recognize_intent(player_input, context)

    def execute(
        self,
        player_input: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Compatibility alias used by orchestration code."""
        return self.recognize_intent(player_input, context)

    # =============================================================
    # Target detection
    # =============================================================

    def detect_target(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        normalized = self._normalize(text)

        for canonical_target, aliases in self.target_aliases.items():
            if self._contains_any(normalized, aliases):
                return canonical_target

        context_target = self._context_target(context or {})
        if context_target:
            return context_target

        return None

    # =============================================================
    # Intent helpers
    # =============================================================

    def _is_tavern_service(self, text: str) -> bool:
        if self._contains_any(text, self.tavern_service_words):
            return True

        # Tavern location by itself is a service/location action only when
        # the command clearly indicates movement.
        has_tavern_location = self._contains_any(
            text,
            self.tavern_location_words,
        )
        has_enter_or_leave = self._contains_any(
            text,
            {
                "enter",
                "go to",
                "walk into",
                "step into",
                "leave",
                "exit",
                "go inside",
                "go outside",
            },
        )

        return has_tavern_location and has_enter_or_leave

    def _is_dialogue(
        self,
        text: str,
        target: Optional[str],
        context: Dict[str, Any],
    ) -> bool:
        if self._contains_any(text, self.dialogue_words):
            return True

        if target in {"merchant", "bartender", "enemy"}:
            conversation_verbs = {
                "talk",
                "speak",
                "ask",
                "say",
                "greet",
                "question",
                "chat",
                "address",
            }
            if self._contains_any(text, conversation_verbs):
                return True

        # Questions aimed at an NPC should be treated as dialogue.
        if target and text.endswith("?"):
            return True

        if target and self._looks_like_question(text):
            return True

        active_target = self._context_target(context)
        if active_target and self._looks_like_question(text):
            return True

        return False

    def _is_dialogue_continuation(
        self,
        text: str,
        context: Dict[str, Any],
    ) -> bool:
        if not self._contains_any(text, self.dialogue_continuation_phrases):
            return False

        previous_intent = str(
            context.get("previous_intent")
            or context.get("last_intent")
            or ""
        ).lower()

        active_target = self._context_target(context)

        return (
            previous_intent in {"dialogue_action", "persuasion_action"}
            or active_target in {"merchant", "bartender", "enemy"}
        )

    def _is_exploration(self, text: str) -> bool:
        if self._contains_any(text, self.exploration_words):
            return True

        has_location = self._contains_any(text, self.location_words)
        has_movement = self._contains_any(
            text,
            {
                "go",
                "move",
                "walk",
                "travel",
                "head",
                "enter",
                "leave",
                "return",
                "approach",
                "visit",
            },
        )

        return has_location and has_movement

    # =============================================================
    # Context helpers
    # =============================================================

    def _active_location(self, context: Dict[str, Any]) -> Optional[str]:
        value = (
            context.get("active_location")
            or context.get("location")
            or context.get("current_location")
        )

        if value is None:
            game_state = context.get("game_state")
            if isinstance(game_state, dict):
                value = (
                    game_state.get("active_location")
                    or game_state.get("location")
                    or game_state.get("current_location")
                )

        return str(value).lower() if value is not None else None

    def _context_target(self, context: Dict[str, Any]) -> Optional[str]:
        value = (
            context.get("active_target")
            or context.get("target")
            or context.get("current_target")
            or context.get("last_target")
            or context.get("dialogue_target")
        )

        if value is None:
            game_state = context.get("game_state")
            if isinstance(game_state, dict):
                value = (
                    game_state.get("active_target")
                    or game_state.get("target")
                    or game_state.get("current_target")
                    or game_state.get("last_target")
                    or game_state.get("dialogue_target")
                )

        if value is None:
            return None

        normalized = self._normalize(str(value))

        for canonical_target, aliases in self.target_aliases.items():
            if normalized == canonical_target:
                return canonical_target
            if normalized in aliases:
                return canonical_target

        return normalized or None

    # =============================================================
    # Generic helpers
    # =============================================================

    def _normalize(self, text: str) -> str:
        return " ".join(str(text).strip().lower().split())

    def _contains_any(
        self,
        text: str,
        candidates: Iterable[str],
    ) -> bool:
        return any(candidate in text for candidate in candidates)

    def _looks_like_question(self, text: str) -> bool:
        question_starters = (
            "who ",
            "what ",
            "where ",
            "when ",
            "why ",
            "how ",
            "can ",
            "could ",
            "would ",
            "will ",
            "do ",
            "does ",
            "did ",
            "is ",
            "are ",
            "have ",
            "has ",
            "tell me ",
        )
        return text.startswith(question_starters)

    def _build_result(
        self,
        *,
        intent: str,
        target: Optional[str],
        confidence: float,
        reason: str,
    ) -> Dict[str, Any]:
        return {
            "intent": intent,
            "target": target,
            "confidence": confidence,
            "reason": reason,
        }

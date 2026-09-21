from __future__ import annotations

import re
from typing import Any, Dict, Iterable, Optional


class IntentRecognitionAgent:
    """
    Deterministic intent recognizer for the RPG backend.

    Responsibilities:
    - classify player input into a high-level action intent;
    - identify NPC targets separately from location targets;
    - keep tavern services separate from NPC dialogue;
    - route explicit movement into exploration;
    - route persuasion involving merchant, bartender and enemy;
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
        # =========================================================
        # Combat
        # =========================================================
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

        # =========================================================
        # Social threat / intimidation
        # =========================================================
        # These are threats, not completed attacks. Physical menace such as
        # pointing or holding a weapon at an NPC must stay in the social
        # pipeline until the player actually strikes/stabs/shoots/etc.
        self.social_threat_words = {
            "threaten",
            "intimidate",
            "menace",
            "blackmail",
            "hold at knifepoint",
            "hold at gunpoint",
            "knife to his throat",
            "knife to her throat",
            "knife at his throat",
            "knife at her throat",
            "blade to his throat",
            "blade to her throat",
            "blade at his throat",
            "blade at her throat",
            "sword to his throat",
            "sword to her throat",
            "sword at his throat",
            "sword at her throat",
            "gun to his head",
            "gun to her head",
            "gun at his head",
            "gun at her head",
            "point a knife",
            "point the knife",
            "point my knife",
            "point a blade",
            "point the blade",
            "point my blade",
            "point a sword",
            "point the sword",
            "point my sword",
            "point a gun",
            "point the gun",
            "point my gun",
            "draw a knife on",
            "draw my knife on",
            "pull a knife on",
            "pull my knife on",
            "draw a gun on",
            "draw my gun on",
            "pull a gun on",
            "pull my gun on",
            "or else",
            "you'll regret it",
            "you will regret it",
            "make you regret it",
        }

        # =========================================================
        # Persuasion
        # =========================================================
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

        # =========================================================
        # Dialogue
        # =========================================================
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

        # =========================================================
        # Tavern services only
        # =========================================================
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
            "order a drink",
            "order food",
            "order a meal",
            "rent a room",
            "buy a drink",
            "buy food",
            "get a drink",
            "get some food",
            "have a drink",
            "have a meal",
        }

        self.tavern_location_words = {
            "tavern",
            "inn",
            "pub",
            "bar",
            "alehouse",
        }

        # =========================================================
        # Exploration
        # =========================================================
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
            "go deeper",
            "move deeper",
            "walk deeper",
            "head deeper",
            "look for",
            "find a path",
            "scout",
        }

        # Canonical location names are aligned with the rest of the project.
        self.location_aliases: Dict[str, set[str]] = {
            "old forest": {
                "old forest",
                "forest",
                "woods",
            },
            "forest road": {
                "forest road",
                "road through the forest",
                "road in the forest",
            },
            "old ruins": {
                "old ruins",
                "ruins",
            },
            "village": {
                "village",
            },
            "valley": {
                "valley",
            },
            "riverbank": {
                "riverbank",
                "river bank",
            },
            "tavern": {
                "tavern",
                "inn",
                "pub",
                "bar",
                "alehouse",
            },
            "town": {
                "town",
            },
            "city": {
                "city",
            },
            "road": {
                "road",
            },
            "path": {
                "path",
                "trail",
            },
            "river": {
                "river",
            },
            "bridge": {
                "bridge",
            },
            "mountain": {
                "mountain",
            },
            "cave": {
                "cave",
            },
            "castle": {
                "castle",
            },
            "market": {
                "market",
            },
            "square": {
                "square",
            },
            "gate": {
                "gate",
            },
            "field": {
                "field",
            },
            "camp": {
                "camp",
            },
            "harbor": {
                "harbor",
                "harbour",
            },
            "dock": {
                "dock",
                "docks",
            },
        }

        # =========================================================
        # NPC targets
        # =========================================================
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
            "reason": "dialogue request"
        }
        """

        context = context or {}
        text = self._normalize(player_input)

        explicit_npc_target = self._detect_explicit_npc_target(text)
        context_npc_target = self._context_target(context)

        location_target = self._detect_location_target(text)

        if not text:
            return self._build_result(
                intent="general_action",
                target=None,
                confidence=0.2,
                reason="empty input",
            )

        # =========================================================
        # 0. Social threat / intimidation
        # =========================================================
        if self._is_social_threat(
            text=text,
            explicit_target=explicit_npc_target,
            context=context,
        ):
            threat_target = (
                explicit_npc_target
                or context_npc_target
            )

            return self._build_result(
                intent="persuasion_action",
                target=threat_target,
                confidence=0.98,
                reason="social threat or intimidation",
            )

        # =========================================================
        # 1. Combat
        # =========================================================
        if self._contains_any(
            text,
            self.combat_words
        ):
            combat_target = (
                explicit_npc_target
                or context_npc_target
                or "enemy"
            )

            return self._build_result(
                intent="combat_action",
                target=combat_target,
                confidence=0.99,
                reason="combat keyword",
            )

        # =========================================================
        # 2. Persuasion
        # =========================================================
        if self._contains_any(
            text,
            self.persuasion_words
        ):
            persuasion_target = (
                explicit_npc_target
                or context_npc_target
            )

            if persuasion_target in {
                "merchant",
                "bartender",
                "enemy",
            }:
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

        # =========================================================
        # 3. Tavern service
        #
        # Service requests must be checked before generic dialogue.
        # Otherwise phrases such as "I ask for ale" are captured by
        # the dialogue keyword "ask" and lose the implicit bartender.
        # =========================================================
        if self._is_tavern_service(text, context):
            return self._build_result(
                intent="tavern_action",
                target="bartender",
                confidence=0.97,
                reason="tavern service request",
            )

        # =========================================================
        # 4. Explicit NPC dialogue
        # =========================================================
        if self._is_dialogue(
            text=text,
            explicit_target=explicit_npc_target,
            context=context
        ):
            dialogue_target = (
                explicit_npc_target
                or context_npc_target
            )

            return self._build_result(
                intent="dialogue_action",
                target=dialogue_target,
                confidence=0.94,
                reason="dialogue request",
            )

        # =========================================================
        # 5. Explicit movement / exploration
        #
        # Checked BEFORE tavern services so phrases such as:
        #
        # "I go deeper into the forest"
        # "I leave the tavern"
        # "I enter the ruins"
        #
        # are not mistaken for tavern actions.
        # =========================================================
        if self._is_exploration(
            text=text,
            location_target=location_target
        ):
            exploration_target = (
                location_target
                or self._infer_contextual_exploration_target(
                    text=text,
                    context=context
                )
            )

            return self._build_result(
                intent="exploration_action",
                target=exploration_target,
                confidence=0.96,
                reason="explicit movement or exploration request",
            )

        # =========================================================
        # 6. Dialogue continuation
        # =========================================================
        if self._is_dialogue_continuation(
            text,
            context
        ):
            return self._build_result(
                intent="dialogue_action",
                target=context_npc_target,
                confidence=0.9,
                reason="dialogue continuation",
            )

        # =========================================================
        # 7. Tavern context fallback
        # =========================================================
        active_location = self._active_location(
            context
        )

        if (
            active_location == "tavern"
            and explicit_npc_target == "bartender"
        ):
            return self._build_result(
                intent="dialogue_action",
                target="bartender",
                confidence=0.78,
                reason="bartender interaction inside tavern",
            )

        # =========================================================
        # 8. General fallback
        # =========================================================
        return self._build_result(
            intent="general_action",
            target=explicit_npc_target or location_target,
            confidence=0.5,
            reason="no specialized rule matched",
        )

    def classify(
        self,
        player_input: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Compatibility alias."""
        return self.recognize_intent(
            player_input,
            context
        )

    def execute(
        self,
        player_input: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Compatibility alias used by orchestration code."""
        return self.recognize_intent(
            player_input,
            context
        )

    # =============================================================
    # NPC target detection
    # =============================================================

    def detect_target(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Compatibility method.

        Explicit NPC target has priority.
        Context target is used only when no explicit target exists.
        """

        explicit_target = self._detect_explicit_npc_target(
            text
        )

        if explicit_target:
            return explicit_target

        return self._context_target(
            context or {}
        )

    def _detect_explicit_npc_target(
        self,
        text: str,
    ) -> Optional[str]:
        """
        Detect an NPC mentioned in the current player input only.

        This prevents old context such as "bartender" from leaking into
        unrelated exploration commands.
        """

        normalized = self._normalize(
            text
        )

        for canonical_target, aliases in self.target_aliases.items():
            if self._contains_any(
                normalized,
                aliases
            ):
                return canonical_target

        return None

    # =============================================================
    # Location target detection
    # =============================================================

    def _detect_location_target(
        self,
        text: str,
    ) -> Optional[str]:
        """
        Detect a location independently from NPC targets.

        Examples:
        "go deeper into the forest" -> old forest
        "enter the old ruins" -> old ruins
        "walk along the forest road" -> forest road
        """

        normalized = self._normalize(
            text
        )

        aliases_with_canonical = []

        for canonical_location, aliases in self.location_aliases.items():
            for alias in aliases:
                aliases_with_canonical.append(
                    (
                        canonical_location,
                        alias
                    )
                )

        # Longest aliases first so "forest road" wins over "forest"
        # and "old ruins" wins over "ruins".
        aliases_with_canonical.sort(
            key=lambda item: len(item[1]),
            reverse=True
        )

        for canonical_location, alias in aliases_with_canonical:
            if self._contains_phrase(
                normalized,
                alias
            ):
                return canonical_location

        return None

    def _infer_contextual_exploration_target(
        self,
        text: str,
        context: Dict[str, Any],
    ) -> Optional[str]:
        """
        Resolve movement commands that do not repeat the location.

        Example:
        Current location = old forest
        Player = "I go deeper"
        -> old forest
        """

        active_location = self._active_location(
            context
        )

        if not active_location:
            return None

        normalized_location = self._canonicalize_location(
            active_location
        )

        directional_phrases = {
            "go deeper",
            "move deeper",
            "walk deeper",
            "head deeper",
            "continue",
            "continue forward",
            "go forward",
            "move forward",
            "walk forward",
            "keep going",
        }

        if self._contains_any(
            text,
            directional_phrases
        ):
            return normalized_location

        return None

    def _canonicalize_location(
        self,
        value: str,
    ) -> str:
        """
        Convert aliases stored in context into canonical project names.
        """

        normalized = self._normalize(
            value
        )

        for canonical_location, aliases in self.location_aliases.items():
            if normalized == canonical_location:
                return canonical_location

            for alias in aliases:
                if normalized == self._normalize(alias):
                    return canonical_location

        return normalized

    # =============================================================
    # Intent helpers
    # =============================================================

    def _is_social_threat(
        self,
        text: str,
        explicit_target: Optional[str],
        context: Dict[str, Any],
    ) -> bool:
        """
        Detect intimidation, including physical gestures with weapons.

        A displayed/pointed weapon is a social threat. A completed harmful
        action (stab, slash, punch, shoot, attack...) remains combat.
        """
        target = explicit_target or self._context_target(context)

        if target not in {"merchant", "bartender", "enemy"}:
            return False

        # Direct intimidation vocabulary and common weapon-menace phrases.
        if self._contains_any(text, self.social_threat_words):
            # Explicit completed attack phrasing still wins unless the attack
            # word is embedded in a conditional verbal threat.
            if self._is_completed_attack(text):
                return False
            return True

        # Flexible construction:
        # "I pull out a knife and point at his throat"
        weapon_words = {
            "knife", "blade", "dagger", "sword", "gun", "pistol", "weapon"
        }
        menace_verbs = {
            "point", "aim", "brandish", "draw", "pull out", "hold", "press"
        }
        vulnerable_targets = {
            "throat", "neck", "head", "face", "chest"
        }

        if (
            self._contains_any(text, weapon_words)
            and self._contains_any(text, menace_verbs)
            and (
                self._contains_any(text, vulnerable_targets)
                or explicit_target is not None
                or self._context_target(context) is not None
            )
        ):
            return not self._is_completed_attack(text)

        # Conditional violence is intimidation rather than an executed attack.
        conditional_threat_patterns = (
            r"\b(?:i(?:'ll| will)|i am going to|i'm going to)\s+"
            r"(?:kill|hurt|stab|shoot|cut|beat)\b.*\bif\b",
            r"\bif\b.*\b(?:don't|do not|won't|will not)\b.*"
            r"\b(?:kill|hurt|stab|shoot|cut|beat)\b",
        )

        return any(
            re.search(pattern, text) is not None
            for pattern in conditional_threat_patterns
        )

    def _is_completed_attack(
        self,
        text: str,
    ) -> bool:
        """
        Return True only for wording that describes an executed attack.

        This prevents weapon display from becoming combat merely because a
        weapon is present in the sentence.
        """
        attack_patterns = (
            r"\b(?:i\s+)?(?:attack|hit|strike|stab|slash|shoot|punch|kick|"
            r"assault|smash|wound|execute)\b",
            r"\b(?:i\s+)?kill\s+(?:him|her|them|the\s+\w+|merchant|"
            r"bartender|enemy|bandit|guard)\b",
        )

        return any(
            re.search(pattern, text) is not None
            for pattern in attack_patterns
        )

    def _is_tavern_service(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Detect requests for tavern goods/services.

        Inside a tavern, requests such as "I ask for a glass of good ale"
        implicitly target the bartender even when the player does not name
        the bartender explicitly.

        Movement to/from the tavern is still handled as exploration.
        """
        context = context or {}
        active_location = self._active_location(context)

        service_request_phrases = {
            "order",
            "ask for",
            "buy",
            "get me",
            "give me",
            "bring me",
            "can i have",
            "could i have",
            "i want",
            "i'd like",
            "i would like",
            "have a",
            "have some",
            "rent",
        }

        service_object_words = {
            "drink", "ale", "beer", "wine", "mead", "water",
            "food", "meal", "dinner", "lunch", "bread", "stew",
            "room", "bed",
        }

        explicit_service = self._contains_any(text, self.tavern_service_words)

        # Strong service phrases can resolve the bartender implicitly,
        # but only when a tavern service object is also mentioned.
        phrased_service = (
            self._contains_any(text, service_request_phrases)
            and self._contains_any(text, service_object_words)
        )

        # Bare service nouns such as "ale" should only become tavern actions
        # when the player is actually in the tavern.
        if active_location == "tavern":
            return explicit_service or phrased_service

        # Outside the tavern require a clear request, rather than treating
        # any mention of food/drink as a service action.
        return phrased_service

    def _is_dialogue(
        self,
        text: str,
        explicit_target: Optional[str],
        context: Dict[str, Any],
    ) -> bool:
        """
        Detect direct NPC conversation.
        """

        if self._contains_any(
            text,
            self.dialogue_words
        ):
            # An explicit dialogue verb is sufficient.
            return True

        if explicit_target in {
            "merchant",
            "bartender",
            "enemy",
        }:
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

            if self._contains_any(
                text,
                conversation_verbs
            ):
                return True

            if text.endswith("?"):
                return True

            if self._looks_like_question(
                text
            ):
                return True

        active_target = self._context_target(
            context
        )

        # A question with no explicit NPC still counts as dialogue when
        # there is an active conversation target from the previous turn.
        if (
            active_target in {
                "merchant",
                "bartender",
                "enemy",
            }
            and self._looks_like_question(text)
        ):
            return True

        return False

    def _is_dialogue_continuation(
        self,
        text: str,
        context: Dict[str, Any],
    ) -> bool:
        """
        Detect short dialogue follow-ups.

        Examples:
        - why?
        - really?
        - tell me more
        - go on
        """

        if not self._contains_any(
            text,
            self.dialogue_continuation_phrases
        ):
            return False

        previous_intent = str(
            context.get("previous_intent")
            or context.get("last_intent")
            or ""
        ).lower()

        active_target = self._context_target(
            context
        )

        return (
            previous_intent in {
                "dialogue_action",
                "persuasion_action",
            }
            or active_target in {
                "merchant",
                "bartender",
                "enemy",
            }
        )

    def _is_exploration(
        self,
        text: str,
        location_target: Optional[str],
    ) -> bool:
        """
        Detect world movement and exploration.

        Examples:
        - I go deeper into the forest
        - enter the old ruins
        - walk toward the village
        - leave the tavern
        - explore the cave
        """

        exploration_verbs = {
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
            "scout",
        }

        if (
            location_target
            and self._contains_any(
                text,
                exploration_verbs
            )
        ):
            return True

        explicit_movement_phrases = {
            "go deeper",
            "move deeper",
            "walk deeper",
            "head deeper",
            "continue forward",
            "go forward",
            "move forward",
            "walk forward",
            "keep going",
        }

        if self._contains_any(
            text,
            explicit_movement_phrases
        ):
            return True

        if self._contains_any(
            text,
            {
                "explore",
                "look around",
                "investigate",
                "scout",
            }
        ):
            return True

        return False

    # =============================================================
    # Context helpers
    # =============================================================

    def _active_location(
        self,
        context: Dict[str, Any]
    ) -> Optional[str]:
        value = (
            context.get("active_location")
            or context.get("location")
            or context.get("current_location")
        )

        if value is None:
            game_state = context.get(
                "game_state"
            )

            if isinstance(
                game_state,
                dict
            ):
                value = (
                    game_state.get("active_location")
                    or game_state.get("location")
                    or game_state.get("current_location")
                )

        if value is None:
            return None

        return self._canonicalize_location(
            str(value)
        )

    def _context_target(
        self,
        context: Dict[str, Any]
    ) -> Optional[str]:
        """
        Return an active NPC target from context.

        Unknown values and locations are deliberately rejected.
        """

        value = (
            context.get("active_target")
            or context.get("target")
            or context.get("current_target")
            or context.get("last_target")
            or context.get("dialogue_target")
        )

        if value is None:
            game_state = context.get(
                "game_state"
            )

            if isinstance(
                game_state,
                dict
            ):
                value = (
                    game_state.get("active_target")
                    or game_state.get("target")
                    or game_state.get("current_target")
                    or game_state.get("last_target")
                    or game_state.get("dialogue_target")
                )

        if value is None:
            return None

        normalized = self._normalize(
            str(value)
        )

        for canonical_target, aliases in self.target_aliases.items():
            if normalized == canonical_target:
                return canonical_target

            for alias in aliases:
                if normalized == self._normalize(alias):
                    return canonical_target

        return None

    # =============================================================
    # Generic helpers
    # =============================================================

    def _normalize(
        self,
        text: str
    ) -> str:
        """
        Normalize whitespace and lowercase text.
        """

        return " ".join(
            str(text)
            .strip()
            .lower()
            .split()
        )

    def _contains_phrase(
        self,
        text: str,
        phrase: str,
    ) -> bool:
        """
        Match a complete word or phrase instead of an arbitrary substring.
        """

        normalized_text = self._normalize(
            text
        )

        normalized_phrase = self._normalize(
            phrase
        )

        if not normalized_phrase:
            return False

        pattern = (
            r"(?<!\w)"
            + re.escape(normalized_phrase)
            + r"(?!\w)"
        )

        return re.search(
            pattern,
            normalized_text
        ) is not None

    def _contains_any(
        self,
        text: str,
        candidates: Iterable[str],
    ) -> bool:
        """
        Match complete words or phrases.

        Longer candidates are checked first.

        This avoids accidental substring matches from the old implementation:
            candidate in text
        """

        ordered_candidates = sorted(
            candidates,
            key=len,
            reverse=True
        )

        for candidate in ordered_candidates:
            if self._contains_phrase(
                text,
                candidate
            ):
                return True

        return False

    def _looks_like_question(
        self,
        text: str
    ) -> bool:
        normalized = self._normalize(
            text
        )

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

        if normalized.endswith("?"):
            return True

        return normalized.startswith(
            question_starters
        )

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

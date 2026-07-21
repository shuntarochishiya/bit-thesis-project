from typing import Dict, Any, List, Optional

from state.npc_profiles import NPCProfiles
from state.npc_state_manager import NPCStateManager


class ConsequenceAgent:
    """
    Checks previous actions and the current world state before an action
    is executed.

    The agent uses:
    - current player input
    - recognized intent
    - interaction target
    - current game state
    - semantic memory
    - dynamic NPC state

    It can:
    - allow or block an action;
    - define the NPC reaction style;
    - update legacy GameState fields;
    - update NPC emotions, trust, hostility and personal memory.
    """

    def __init__(
        self,
        npc_state_manager: Optional[NPCStateManager] = None
    ):
        self.npc_state_manager = npc_state_manager

    def evaluate(
        self,
        player_input: str,
        intent: str,
        target: str,
        game_state: Dict[str, Any],
        relevant_memory: List[str]
    ) -> Dict[str, Any]:

        memory_text = " ".join(
            str(item) for item in relevant_memory
        ).lower()

        input_text = player_input.lower()

        normalized_target = self._normalize_target(target)

        npc_state = self._get_npc_state(normalized_target)

        result: Dict[str, Any] = {
            "allow_action": True,
            "reason": "No blocking consequence was found.",
            "reaction_modifier": "neutral",
            "state_updates": {},
            "npc_state_updates": {},
            "npc_context": self._get_npc_context(normalized_target),
            "system_note": ""
        }

        # =====================================================
        # 1. Bartender consequences
        # =====================================================

        if normalized_target == "bartender":
            bartender_was_attacked = (
                "target: bartender" in memory_text
                and (
                    "event type: attack" in memory_text
                    or "attacks the bartender" in memory_text
                    or "attack the bartender" in memory_text
                    or "attacks the barmaid" in memory_text
                    or "attack the barmaid" in memory_text
                    or "damage" in memory_text
                )
            )

            asking_for_service = (
                intent == "tavern_action"
                and any(
                    word in input_text
                    for word in [
                        "drink",
                        "ale",
                        "beer",
                        "wine",
                        "mead",
                        "room",
                        "rent",
                        "food",
                        "meal",
                        "rest",
                        "sleep",
                        "order",
                        "buy",
                        "purchase"
                    ]
                )
            )

            asking_for_information = (
                intent == "tavern_action"
                and any(
                    word in input_text
                    for word in [
                        "rumor",
                        "rumour",
                        "information",
                        "news",
                        "odd",
                        "strange",
                        "weird",
                        "nearby",
                        "recently",
                        "details",
                        "what happened"
                    ]
                )
            )

            bartender_is_hostile = (
                game_state.get("bartender_hostile", False)
                or bartender_was_attacked
                or npc_state.get("hostile", False)
            )

            bartender_trust = npc_state.get(
                "trust",
                game_state.get(
                    "relationship_with_bartender",
                    50
                )
            )

            if bartender_is_hostile:
                if asking_for_service:
                    result["allow_action"] = False

                    result["reason"] = (
                        "The bartender refuses service because the player "
                        "previously attacked or seriously threatened them."
                    )

                    result["reaction_modifier"] = "hostile"

                    result["state_updates"] = {
                        "bartender_hostile": True,
                        "bartender_mood": "angry",
                        "relationship_with_bartender": max(
                            game_state.get(
                                "relationship_with_bartender",
                                50
                            ) - 3,
                            0
                        ),
                        "tavern_reputation": max(
                            game_state.get(
                                "tavern_reputation",
                                50
                            ) - 2,
                            0
                        )
                    }

                    result["npc_state_updates"] = {
                        "emotion": "hostile",
                        "hostile": True,
                        "trust": max(
                            bartender_trust - 3,
                            0
                        ),
                        "anger": min(
                            npc_state.get("anger", 0) + 5,
                            100
                        ),
                        "stress": min(
                            npc_state.get("stress", 0) + 3,
                            100
                        )
                    }

                    result["system_note"] = (
                        "The current tavern action is blocked by past "
                        "violence against the bartender."
                    )

                    self._apply_npc_updates(
                        npc_id="bartender",
                        updates=result["npc_state_updates"],
                        reason=result["reason"]
                    )

                    return result

                if asking_for_information:
                    result["allow_action"] = False

                    result["reason"] = (
                        "The bartender refuses to share information "
                        "because the player is not trusted."
                    )

                    result["reaction_modifier"] = "distrustful"

                    result["state_updates"] = {
                        "bartender_mood": "angry",
                        "relationship_with_bartender": max(
                            game_state.get(
                                "relationship_with_bartender",
                                50
                            ) - 2,
                            0
                        )
                    }

                    result["npc_state_updates"] = {
                        "emotion": "suspicious",
                        "trust": max(
                            bartender_trust - 2,
                            0
                        ),
                        "stress": min(
                            npc_state.get("stress", 0) + 2,
                            100
                        )
                    }

                    result["system_note"] = (
                        "The bartender withholds rumors and useful "
                        "information due to past hostility."
                    )

                    self._apply_npc_updates(
                        npc_id="bartender",
                        updates=result["npc_state_updates"],
                        reason=result["reason"]
                    )

                    return result

            expensive_drink_history = (
                "event type: drink_order" in memory_text
                and (
                    "finest" in memory_text
                    or "royal" in memory_text
                    or "expensive" in memory_text
                    or "premium" in memory_text
                )
            )

            npc_remembers_generosity = self._npc_memory_contains(
                npc_id="bartender",
                phrases=[
                    "bought expensive drink",
                    "ordered premium drink",
                    "spent generously",
                    "bought goods",
                    "bought_drink",
                    "left_tip"
                ]
            )

            if (
                asking_for_information
                and (
                    expensive_drink_history
                    or npc_remembers_generosity
                )
                and not bartender_is_hostile
            ):
                result["allow_action"] = True

                result["reason"] = (
                    "The bartender remembers that the player "
                    "spent generously before."
                )

                result["reaction_modifier"] = "friendly"

                result["state_updates"] = {
                    "bartender_mood": "friendly",
                    "relationship_with_bartender": min(
                        game_state.get(
                            "relationship_with_bartender",
                            50
                        ) + 3,
                        100
                    )
                }

                result["npc_state_updates"] = {
                    "emotion": "grateful",
                    "trust": min(
                        bartender_trust + 3,
                        100
                    ),
                    "stress": max(
                        npc_state.get("stress", 0) - 2,
                        0
                    )
                }

                result["system_note"] = (
                    "Past generous drink orders make the bartender "
                    "more willing to share information."
                )

                self._apply_npc_updates(
                    npc_id="bartender",
                    updates=result["npc_state_updates"],
                    reason=result["reason"]
                )

                return result

        # =====================================================
        # 2. Merchant consequences
        # =====================================================

        if normalized_target == "merchant":
            merchant_was_attacked = (
                "target: merchant" in memory_text
                and (
                    "event type: attack" in memory_text
                    or "attacks the merchant" in memory_text
                    or "attack the merchant" in memory_text
                    or "attacks the vendor" in memory_text
                    or "attack the vendor" in memory_text
                    or "damage" in memory_text
                )
            )

            merchant_remembers_attack = self._npc_memory_contains(
                npc_id="merchant",
                phrases=[
                    "attacked this npc",
                    "robbed this npc",
                    "threatened this npc"
                ]
            )

            asking_for_trade_or_discount = (
                intent in [
                    "persuasion_action",
                    "dialogue_action",
                    "trade_action"
                ]
                and any(
                    word in input_text
                    for word in [
                        "discount",
                        "free",
                        "price",
                        "artifact",
                        "buy",
                        "sell",
                        "trade",
                        "cheaper",
                        "give me"
                    ]
                )
            )

            merchant_is_hostile = (
                game_state.get("merchant_hostile", False)
                or merchant_was_attacked
                or merchant_remembers_attack
                or npc_state.get("hostile", False)
            )

            merchant_trust = npc_state.get(
                "trust",
                game_state.get(
                    "relationship_with_merchant",
                    50
                )
            )

            if merchant_is_hostile and asking_for_trade_or_discount:
                result["allow_action"] = False

                result["reason"] = (
                    "The merchant refuses because the player previously "
                    "attacked, robbed or threatened them."
                )

                result["reaction_modifier"] = "hostile"

                result["state_updates"] = {
                    "merchant_hostile": True,
                    "relationship_with_merchant": max(
                        game_state.get(
                            "relationship_with_merchant",
                            50
                        ) - 3,
                        0
                    )
                }

                result["npc_state_updates"] = {
                    "hostile": True,
                    "emotion": "hostile",
                    "trust": max(
                        merchant_trust - 3,
                        0
                    ),
                    "anger": min(
                        npc_state.get("anger", 0) + 5,
                        100
                    ),
                    "stress": min(
                        npc_state.get("stress", 0) + 4,
                        100
                    )
                }

                result["system_note"] = (
                    "The merchant-related action is blocked by "
                    "previous violence."
                )

                self._apply_npc_updates(
                    npc_id="merchant",
                    updates=result["npc_state_updates"],
                    reason=result["reason"]
                )

                return result

            asking_for_apology = any(
                phrase in input_text
                for phrase in [
                    "sorry",
                    "apologize",
                    "apologise",
                    "forgive me",
                    "my mistake"
                ]
            )

            if asking_for_apology and merchant_is_hostile:
                result["allow_action"] = True

                result["reason"] = (
                    "The merchant listens to the apology, but remains cautious."
                )

                result["reaction_modifier"] = "cautious"

                result["state_updates"] = {
                    "relationship_with_merchant": min(
                        game_state.get(
                            "relationship_with_merchant",
                            50
                        ) + 5,
                        100
                    )
                }

                result["npc_state_updates"] = {
                    "emotion": "suspicious",
                    "trust": min(
                        merchant_trust + 5,
                        100
                    ),
                    "anger": max(
                        npc_state.get("anger", 0) - 10,
                        0
                    ),
                    "stress": max(
                        npc_state.get("stress", 0) - 5,
                        0
                    )
                }

                if result["npc_state_updates"]["trust"] >= 35:
                    result["npc_state_updates"]["hostile"] = False
                    result["state_updates"]["merchant_hostile"] = False

                result["system_note"] = (
                    "The apology slightly improves the merchant's "
                    "attitude, but does not erase the previous event."
                )

                self._apply_npc_updates(
                    npc_id="merchant",
                    updates=result["npc_state_updates"],
                    reason=result["reason"]
                )

                return result

        # =====================================================
        # 3. Enemy consequences
        # =====================================================

        if normalized_target == "forest_goblin":
            if intent == "dialogue_action":
                enemy_health = npc_state.get(
                    "health",
                    game_state.get("enemy_health", 60)
                )

                enemy_fear = npc_state.get("fear", 10)
                enemy_anger = npc_state.get("anger", 40)

                if enemy_health <= 0:
                    result["allow_action"] = False

                    result["reason"] = (
                        "The enemy cannot respond because it is no longer alive."
                    )

                    result["reaction_modifier"] = "none"

                    result["system_note"] = (
                        "Do not generate dialogue for a dead NPC."
                    )

                    return result

                if enemy_health <= 15 or enemy_fear >= 70:
                    result["allow_action"] = True

                    result["reason"] = (
                        "The enemy is badly wounded or frightened and "
                        "may be willing to talk."
                    )

                    result["reaction_modifier"] = "fearful"

                    result["state_updates"] = {
                        "world_mood": "tense"
                    }

                    result["npc_state_updates"] = {
                        "emotion": "afraid",
                        "fear": min(
                            enemy_fear + 5,
                            100
                        ),
                        "stress": min(
                            npc_state.get("stress", 0) + 5,
                            100
                        ),
                        "current_goal": "survive"
                    }

                    result["system_note"] = (
                        "Enemy dialogue should sound fearful, cautious "
                        "or desperate."
                    )

                    self._apply_npc_updates(
                        npc_id="forest_goblin",
                        updates=result["npc_state_updates"],
                        reason=result["reason"]
                    )

                    return result

                if enemy_health >= 45 and enemy_anger >= 40:
                    result["allow_action"] = True

                    result["reason"] = (
                        "The enemy is still strong and remains hostile "
                        "during conversation."
                    )

                    result["reaction_modifier"] = "hostile"

                    result["state_updates"] = {
                        "world_mood": "hostile"
                    }

                    result["npc_state_updates"] = {
                        "emotion": "hostile",
                        "hostile": True,
                        "current_goal": "protect_territory"
                    }

                    result["system_note"] = (
                        "Enemy dialogue should sound threatening "
                        "and aggressive."
                    )

                    self._apply_npc_updates(
                        npc_id="forest_goblin",
                        updates=result["npc_state_updates"],
                        reason=result["reason"]
                    )

                    return result

        # =====================================================
        # 4. Guard consequences
        # =====================================================

        if normalized_target == "guard":
            crime_level = int(game_state.get("crime_level", 0))
            guard_alert = int(
                game_state.get("guard_alert_level", 10)
            )

            player_reputation = int(
                game_state.get("player_reputation", 50)
            )

            guard_trust = npc_state.get("trust", 45)

            asking_guard_for_help = any(
                phrase in input_text
                for phrase in [
                    "help me",
                    "protect me",
                    "bandits",
                    "danger",
                    "report",
                    "crime"
                ]
            )

            if (
                crime_level >= 40
                or guard_alert >= 60
                or player_reputation <= 20
            ):
                result["reaction_modifier"] = "suspicious"

                result["reason"] = (
                    "The guard treats the player as a possible suspect "
                    "because of the current crime and alert levels."
                )

                result["npc_state_updates"] = {
                    "emotion": "suspicious",
                    "stress": min(
                        npc_state.get("stress", 0) + 5,
                        100
                    ),
                    "trust": max(
                        guard_trust - 5,
                        0
                    ),
                    "current_goal": "investigate_player"
                }

                result["system_note"] = (
                    "Guard dialogue should be formal, suspicious "
                    "and authoritative."
                )

                self._apply_npc_updates(
                    npc_id="guard",
                    updates=result["npc_state_updates"],
                    reason=result["reason"]
                )

                return result

            if asking_guard_for_help and player_reputation >= 60:
                result["reaction_modifier"] = "cooperative"

                result["reason"] = (
                    "The guard considers the player trustworthy enough "
                    "to provide assistance."
                )

                result["npc_state_updates"] = {
                    "emotion": "calm",
                    "trust": min(
                        guard_trust + 3,
                        100
                    ),
                    "current_goal": "protect_civilians"
                }

                result["system_note"] = (
                    "The guard should respond professionally and "
                    "offer limited assistance."
                )

                self._apply_npc_updates(
                    npc_id="guard",
                    updates=result["npc_state_updates"],
                    reason=result["reason"]
                )

                return result

        # =====================================================
        # 5. Traveler consequences
        # =====================================================

        if normalized_target == "traveler":
            traveler_trust = npc_state.get("trust", 50)

            traveler_remembers_help = self._npc_memory_contains(
                npc_id="traveler",
                phrases=[
                    "helped this npc",
                    "saved this npc",
                    "assisted this npc"
                ]
            )

            asking_for_information = any(
                word in input_text
                for word in [
                    "road",
                    "path",
                    "direction",
                    "where",
                    "rumor",
                    "information",
                    "danger"
                ]
            )

            if traveler_remembers_help and asking_for_information:
                result["reaction_modifier"] = "grateful"

                result["reason"] = (
                    "The traveler remembers the player's help and "
                    "is willing to share useful information."
                )

                result["npc_state_updates"] = {
                    "emotion": "grateful",
                    "trust": min(
                        traveler_trust + 3,
                        100
                    ),
                    "current_goal": "repay_player"
                }

                result["system_note"] = (
                    "The traveler should respond warmly and provide "
                    "helpful information."
                )

                self._apply_npc_updates(
                    npc_id="traveler",
                    updates=result["npc_state_updates"],
                    reason=result["reason"]
                )

                return result

        return result

    # =========================================================
    # Helper methods
    # =========================================================

    def _normalize_target(self, target: str) -> str:
        """
        Converts aliases such as enemy, barmaid and vendor
        into internal NPC IDs.
        """
        if not target:
            return ""

        try:
            return NPCProfiles.normalize_npc_id(target)
        except Exception:
            return str(target).strip().lower()

    def _get_npc_state(self, npc_id: str) -> Dict[str, Any]:
        """
        Returns the current dynamic NPC state.

        If NPCStateManager is not connected or the target is unknown,
        an empty dictionary is returned.
        """
        if self.npc_state_manager is None:
            return {}

        if not npc_id:
            return {}

        try:
            return self.npc_state_manager.get_state(npc_id)
        except KeyError:
            return {}

    def _get_npc_context(self, npc_id: str) -> Dict[str, Any]:
        """
        Returns the static profile together with the dynamic state.
        """
        if self.npc_state_manager is None:
            return {}

        if not npc_id:
            return {}

        try:
            return self.npc_state_manager.build_simulation_context(
                npc_id
            )
        except KeyError:
            return {}

    def _apply_npc_updates(
        self,
        npc_id: str,
        updates: Dict[str, Any],
        reason: str
    ):
        """
        Applies dynamic NPC changes through the shared NPCStateManager.
        """
        if self.npc_state_manager is None:
            return

        if not updates:
            return

        try:
            self.npc_state_manager.update_state(
                npc_id=npc_id,
                updates=updates,
                source="ConsequenceAgent",
                reason=reason
            )
        except (KeyError, ValueError) as error:
            print(
                f"[ConsequenceAgent] Failed to update NPC "
                f"'{npc_id}': {error}"
            )

    def _npc_memory_contains(
        self,
        npc_id: str,
        phrases: List[str]
    ) -> bool:
        """
        Searches the NPC's personal memory for one of the supplied phrases.
        """
        if self.npc_state_manager is None:
            return False

        try:
            memories = self.npc_state_manager.retrieve_memories(
                npc_id=npc_id,
                limit=20
            )
        except KeyError:
            return False

        memory_text = " ".join(
            str(memory.get("event", ""))
            for memory in memories
        ).lower()

        return any(
            phrase.lower() in memory_text
            for phrase in phrases
        )

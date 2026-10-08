from typing import Dict, Any, Optional


class PreconditionAgent:
    """
    Checks preconditions before the main action is executed.

    Besides ordinary location/resource/health checks, this agent now enforces
    NPC presence. A player may only interact with an NPC that is physically
    available at the current location.
    """

    # Current prototype world placement.
    # This is intentionally deterministic and can later be moved to world data
    # or Unity scene metadata without changing the validation API.
    NPC_LOCATIONS = {
        "merchant": {"village", "town", "market", "prototype_hub"},
        "bartender": {"tavern", "prototype_hub"},
    }

    @staticmethod
    def _normalized_location(game_state: Dict[str, Any]) -> str:
        return str(
            game_state.get("location")
            or game_state.get("active_location")
            or "unknown"
        ).strip().lower()

    @classmethod
    def _npc_present(
        cls,
        target: Optional[str],
        game_state: Dict[str, Any],
    ) -> bool:
        target = str(target or "").strip().lower()

        # Enemies are governed by combat/current-enemy state rather than a
        # fixed settlement location in this prototype.
        if target in {"", "enemy"}:
            return True

        allowed_locations = cls.NPC_LOCATIONS.get(target)
        if not allowed_locations:
            # Unknown/dynamic NPCs are not rejected here; Unity/world data can
            # later provide their presence explicitly.
            return True

        location = cls._normalized_location(game_state)
        return location in allowed_locations

    def validate_location(
        self,
        intent: str,
        target: str,
        game_state: Dict[str, Any],
        player_input: str
    ) -> Dict[str, Any]:
        text = player_input.lower()
        location = self._normalized_location(game_state)

        if intent == "tavern_action":
            tavern_words = [
                "tavern", "inn", "pub", "bar", "alehouse",
                "bartender", "barmaid", "barman", "innkeeper", "barkeep"
            ]

            player_is_entering_tavern = any(word in text for word in tavern_words)

            if location != "tavern" and not player_is_entering_tavern:
                return {
                    "success": False,
                    "message": "The player is not in a tavern, so this tavern action cannot be performed.",
                    "state_updates": {},
                    "precondition_type": "location"
                }

        return {
            "success": True,
            "message": "Location precondition passed.",
            "state_updates": {},
            "precondition_type": "location"
        }

    def check_player_resources(
        self,
        intent: str,
        target: str,
        game_state: Dict[str, Any],
        player_input: str
    ) -> Dict[str, Any]:
        text = player_input.lower()
        gold = game_state.get("gold", 0)

        if intent == "tavern_action":
            paid_action_words = [
                "drink", "ale", "beer", "wine", "mead",
                "glass", "cup", "bottle",
                "room", "rent", "food", "meal",
                "order", "buy", "purchase"
            ]

            if any(word in text for word in paid_action_words) and gold <= 0:
                return {
                    "success": False,
                    "message": "The player has no gold, so paid tavern actions are not possible.",
                    "state_updates": {},
                    "precondition_type": "resources"
                }

        return {
            "success": True,
            "message": "Resource precondition passed.",
            "state_updates": {},
            "precondition_type": "resources"
        }

    def check_target_status(
        self,
        intent: str,
        target: str,
        game_state: Dict[str, Any],
        player_input: str
    ) -> Dict[str, Any]:
        target = str(target or "").strip().lower()

        # Presence is checked here because dialogue/combat DAGs already call
        # check_target_status. No new DAG node is required.
        interaction_intents = {
            "dialogue_action",
            "persuasion_action",
            "combat_action",
            "trade_action",
            "tavern_action",
        }

        if (
            target in self.NPC_LOCATIONS
            and intent in interaction_intents
            and not self._npc_present(target, game_state)
        ):
            location = self._normalized_location(game_state)
            return {
                "success": False,
                "message": (
                    f"The {target} is not present at the current location "
                    f"({location}), so this interaction cannot be performed."
                ),
                "state_updates": {},
                "precondition_type": "target_presence"
            }

        if target == "enemy" and game_state.get("enemy_health", 0) <= 0:
            return {
                "success": False,
                "message": "The enemy is already defeated.",
                "state_updates": {},
                "precondition_type": "target_status"
            }

        if target == "merchant" and game_state.get("merchant_health", 0) <= 0:
            return {
                "success": False,
                "message": "The merchant is unable to respond.",
                "state_updates": {},
                "precondition_type": "target_status"
            }

        if target == "bartender" and game_state.get("bartender_health", 0) <= 0:
            return {
                "success": False,
                "message": "The bartender is unable to respond.",
                "state_updates": {},
                "precondition_type": "target_status"
            }

        return {
            "success": True,
            "message": "Target status precondition passed.",
            "state_updates": {},
            "precondition_type": "target_status"
        }

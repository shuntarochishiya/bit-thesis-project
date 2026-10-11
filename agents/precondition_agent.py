
from typing import Dict, Any, Optional


class PreconditionAgent:
    """
    Checks preconditions before the main action is executed.

    Besides ordinary location/resource/health checks, this agent enforces
    NPC presence. The prototype_hub location is a Unity integration/demo
    location where both current NPCs are physically present.
    """

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

        if target in {"", "enemy"}:
            return True

        allowed_locations = cls.NPC_LOCATIONS.get(target)
        if not allowed_locations:
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
        target = str(target or "").strip().lower()

        if intent == "tavern_action":
            # Explicit scene travel is separate from ordering a drink.
            entering = any(phrase in text for phrase in (
                "enter tavern", "enter the tavern", "enter inn",
                "enter the inn", "go to the tavern", "walk into the tavern",
            ))
            leaving = any(phrase in text for phrase in (
                "leave tavern", "leave the tavern", "exit tavern",
                "exit the tavern", "leave the inn", "exit the inn",
            ))

            if not entering and not leaving:
                if location not in {"tavern", "prototype_hub"}:
                    return {
                        "success": False,
                        "message": "Tavern services are not available at this location.",
                        "state_updates": {},
                        "precondition_type": "location",
                    }

                if target != "bartender":
                    return {
                        "success": False,
                        "message": "Tavern services must be requested from the bartender.",
                        "state_updates": {},
                        "precondition_type": "target",
                    }

        return {
            "success": True,
            "message": "Location precondition passed.",
            "state_updates": {},
            "precondition_type": "location",
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

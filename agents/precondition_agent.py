from typing import Dict, Any


class PreconditionAgent:
    """
    Checks preconditions before the main action is executed.

    These checks are designed as independent DAG nodes.
    Later they can be executed in parallel with semantic memory retrieval.
    """

    def validate_location(
        self,
        intent: str,
        target: str,
        game_state: Dict[str, Any],
        player_input: str
    ) -> Dict[str, Any]:
        text = player_input.lower()
        location = game_state.get("location", "unknown")

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

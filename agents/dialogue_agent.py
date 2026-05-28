from typing import Dict, Any


class DialogueAgent:
    """
    Composite agent.
    It handles conversations with NPCs such as enemies and merchants.
    The reaction depends on the current game state and previous player actions.
    """

    def execute(self, game_state: Dict[str, Any], target: str = "unknown") -> Dict[str, Any]:
        if target == "enemy":
            enemy_health = game_state["enemy_health"]

            if enemy_health <= 0:
                return {
                    "success": False,
                    "message": "The enemy is defeated and cannot respond.",
                    "state_updates": {}
                }

            if enemy_health <= 15:
                return {
                    "success": True,
                    "message": "The wounded enemy hesitates and seems willing to speak, possibly out of fear.",
                    "state_updates": {
                        "world_mood": "tense"
                    }
                }

            if enemy_health <= 35:
                return {
                    "success": True,
                    "message": "The enemy growls but listens for a moment, unsure whether to continue fighting.",
                    "state_updates": {
                        "world_mood": "uneasy"
                    }
                }

            return {
                "success": True,
                "message": "The enemy refuses peaceful conversation and responds with hostility.",
                "state_updates": {
                    "world_mood": "hostile"
                }
            }

        if target == "merchant":
            if game_state["merchant_health"] <= 0:
                return {
                    "success": False,
                    "message": "The merchant is unable to respond.",
                    "state_updates": {}
                }

            if game_state["merchant_hostile"]:
                return {
                    "success": True,
                    "message": "The merchant refuses to talk because the player previously attacked or threatened them.",
                    "state_updates": {
                        "relationship_with_merchant": max(game_state["relationship_with_merchant"] - 2, 0)
                    }
                }

            return {
                "success": True,
                "message": "The merchant listens carefully and waits for the player to explain what they want.",
                "state_updates": {}
            }

        if target == "bartender":
            if game_state["bartender_health"] <= 0:
                return {
                    "success": False,
                    "message": "The bartender is unable to respond.",
                    "state_updates": {}
                }

            if game_state["bartender_hostile"]:
                return {
                    "success": True,
                    "message": "The bartender refuses to talk because the player caused trouble in the tavern.",
                    "state_updates": {
                        "relationship_with_bartender": max(game_state["relationship_with_bartender"] - 2, 0),
                        "bartender_mood": "angry"
                    }
                }

            if game_state["tavern_reputation"] < 25:
                return {
                    "success": True,
                    "message": "The bartender answers coldly. The player's poor reputation in the tavern makes the conversation tense.",
                    "state_updates": {
                        "bartender_mood": "suspicious"
                    }
                }

            if game_state["relationship_with_bartender"] >= 70:
                return {
                    "success": True,
                    "message": "The bartender greets the player warmly and seems willing to share useful information.",
                    "state_updates": {
                        "bartender_mood": "friendly"
                    }
                }

            return {
                "success": True,
                "message": "The bartender wipes a wooden mug and waits to hear what the player wants.",
                "state_updates": {
                    "bartender_mood": "neutral"
                }
            }

        return {
            "success": False,
            "message": "There is no clear character to talk to.",
            "state_updates": {}
        }


from typing import Dict, Any
import random


class TavernAgent:
    """
    Composite agent.
    Handles tavern-specific actions such as buying drinks,
    asking for rumors, renting a room, or getting food.
    """

    def execute(self, game_state: Dict[str, Any], player_input: str) -> Dict[str, Any]:
        text = player_input.lower()

        if game_state["bartender_hostile"]:
            return {
                "success": False,
                "message": "The bartender refuses to serve the player because of previous hostile behavior.",
                "state_updates": {
                    "bartender_mood": "angry",
                    "relationship_with_bartender": max(game_state["relationship_with_bartender"] - 2, 0)
                }
            }

        if (
            "rumor" in text
            or "rumour" in text
            or "information" in text
            or "news" in text
            or "odd" in text
            or "strange" in text
            or "spotted" in text
            or "nearby" in text
            or "recently" in text
            or "heard" in text
            or "seen" in text
            or "anything" in text
        ):
            rumors = [
                "The bartender shares a rumor about strange lights near the old ruins.",
                "The bartender says a wounded traveler recently saw goblins gathering near the forest road.",
                "The bartender mentions that the merchant has been hiding something valuable.",
                "The bartender quietly warns that not everyone in the tavern can be trusted."
            ]

            relationship_bonus = 3 if game_state["player_reputation"] >= 50 else 0

            return {
                "success": True,
                "message": random.choice(rumors),
                "state_updates": {
                    "relationship_with_bartender": min(
                        game_state["relationship_with_bartender"] + relationship_bonus,
                        100
                    ),
                    "bartender_mood": "talkative"
                }
            }

        if "drink" in text or "ale" in text or "beer" in text or "wine" in text:
            if game_state["gold"] < 2:
                return {
                    "success": False,
                    "message": "The player does not have enough gold to buy a drink.",
                    "state_updates": {
                        "bartender_mood": "unimpressed"
                    }
                }

            return {
                "success": True,
                "message": "The bartender serves a simple drink. The tavern feels a little warmer and safer.",
                "state_updates": {
                    "gold": game_state["gold"] - 2,
                    "relationship_with_bartender": min(game_state["relationship_with_bartender"] + 2, 100),
                    "bartender_mood": "calm",
                    "world_mood": "warm"
                }
            }

        if "room" in text or "sleep" in text or "rest" in text:
            if game_state["gold"] < 10:
                return {
                    "success": False,
                    "message": "The player does not have enough gold to rent a room.",
                    "state_updates": {
                        "bartender_mood": "unimpressed"
                    }
                }

            return {
                "success": True,
                "message": "The player rents a small room and takes time to recover.",
                "state_updates": {
                    "gold": game_state["gold"] - 10,
                    "player_health": min(game_state["player_health"] + 20, 100),
                    "relationship_with_bartender": min(game_state["relationship_with_bartender"] + 3, 100),
                    "world_mood": "restful"
                }
            }

        if "food" in text or "meal" in text:
            if game_state["gold"] < 5:
                return {
                    "success": False,
                    "message": "The player does not have enough gold to buy a meal.",
                    "state_updates": {
                        "bartender_mood": "unimpressed"
                    }
                }

            return {
                "success": True,
                "message": "The bartender brings a hot meal. The player feels slightly better.",
                "state_updates": {
                    "gold": game_state["gold"] - 5,
                    "player_health": min(game_state["player_health"] + 5, 100),
                    "relationship_with_bartender": min(game_state["relationship_with_bartender"] + 2, 100),
                    "bartender_mood": "calm"
                }
            }

        return {
            "success": True,
            "message": "The tavern is noisy, warm, and full of tired travelers. The bartender waits behind the counter.",
            "state_updates": {}
        }

import random
from typing import Dict, Any


class ExplorationAgent:
    """
    Composite agent.
    It handles exploration actions.
    """

    def execute(self, game_state: Dict[str, Any], player_input: str = "") -> Dict[str, Any]:
        text = player_input.lower()

        if "tavern" in text or "inn" in text or "pub" in text or "bar" in text:
            return {
                "success": True,
                "message": "The player enters a warm tavern filled with candlelight, smoke, and quiet conversations.",
                "state_updates": {
                    "location": "tavern",
                    "world_mood": "warm"
                }
            }

        if "forest" in text or "woods" in text:
            return {
                "success": True,
                "message": "The player moves deeper into the old forest, where the trees grow darker and the air becomes colder.",
                "state_updates": {
                    "location": "old forest",
                    "world_mood": "mysterious"
                }
            }

        possible_events = [
            "The player finds old footprints near the trees.",
            "The player discovers a hidden path covered by leaves.",
            "The player hears strange sounds deeper in the forest.",
            "The player finds a small silver coin on the ground."
        ]

        event = random.choice(possible_events)
        updates = {}

        if "coin" in event:
            updates["gold"] = game_state["gold"] + 1

        return {
            "success": True,
            "message": event,
            "state_updates": updates
        }

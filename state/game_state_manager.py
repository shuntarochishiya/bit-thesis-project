from typing import Dict, Any


class GameStateManager:
    """
    This agent manages the current game state.
    In the supervisor's architecture, this corresponds to the Game State Manager Agent.
    """

    def __init__(self):
        self.state = {
            "location": "old forest",

            "player_health": 100,
            "gold": 50,
            "inventory": ["small knife", "map"],

            "enemy_health": 60,
            "current_enemy": "forest goblin",

            "merchant_health": 100,
            "relationship_with_merchant": 50,
            "merchant_hostile": False,

            "bartender_health": 100,
            "relationship_with_bartender": 50,
            "bartender_hostile": False,
            "bartender_mood": "neutral",
            "bartender_role": "bartender",
            "bartender_gender": "unknown",
            "bartender_pronouns": "they/them",

            "tavern_reputation": 50,
            "player_reputation": 50,

            "world_mood": "mysterious"
        }

    def get_state(self) -> Dict[str, Any]:
        return self.state.copy()

    def update_state(self, updates: Dict[str, Any]):
        for key, value in updates.items():
            self.state[key] = value

    def display_state(self):
        print("\n--- Current Game State ---")
        for key, value in self.state.items():
            print(f"{key}: {value}")
        print("--------------------------\n")

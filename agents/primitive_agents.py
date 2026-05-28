import random
from typing import Dict, Any


class AttributeCalculationAgent:
    """
    Primitive agent.
    Calculates basic values such as hit chance and damage.
    """

    def calculate_hit(self) -> bool:
        hit_chance = random.randint(1, 100)
        return hit_chance <= 75

    def calculate_damage(self) -> int:
        return random.randint(8, 20)

    def calculate_persuasion_success(self, relationship_score: int) -> bool:
        base_chance = 40
        bonus = relationship_score // 2
        final_chance = min(base_chance + bonus, 90)
        roll = random.randint(1, 100)
        return roll <= final_chance


class ValidationAgent:
    """
    Primitive agent.
    Checks whether the player's action is possible.
    """

    def validate_combat(self, game_state: Dict[str, Any]) -> bool:
        return game_state["enemy_health"] > 0

    def validate_persuasion(self, game_state: Dict[str, Any]) -> bool:
        return game_state["relationship_with_merchant"] > 0


from typing import Dict, Any
from agents.primitive_agents import AttributeCalculationAgent, ValidationAgent


class CombatAgent:
    """
    Composite agent.
    It uses primitive agents to process combat.
    """

    def __init__(self, attribute_agent: AttributeCalculationAgent, validation_agent: ValidationAgent):
        self.attribute_agent = attribute_agent
        self.validation_agent = validation_agent

    def execute(self, game_state: Dict[str, Any], target: str = "enemy") -> Dict[str, Any]:
        """
        Executes combat logic.
        The target can be either 'enemy' or 'merchant'.
        This allows the system to react differently when the player attacks an NPC.
        """

        # =========================
        # Case 1: Player attacks merchant/vendor/trader
        # =========================
        if target == "merchant":
            if game_state["merchant_health"] <= 0:
                return {
                    "success": False,
                    "message": "The merchant is already unable to respond.",
                    "state_updates": {}
                }

            hit = self.attribute_agent.calculate_hit()

            if not hit:
                return {
                    "success": True,
                    "message": "The player attacks the merchant, but misses. The merchant becomes frightened and hostile.",
                    "state_updates": {
                        "merchant_hostile": True,
                        "relationship_with_merchant": max(game_state["relationship_with_merchant"] - 30, 0),
                        "player_reputation": max(game_state["player_reputation"] - 15, 0),
                        "world_mood": "tense"
                    }
                }

            damage = self.attribute_agent.calculate_damage()
            new_merchant_health = max(game_state["merchant_health"] - damage, 0)

            return {
                "success": True,
                "message": (
                    f"The player attacks the merchant and deals {damage} damage. "
                    "The merchant becomes hostile and will no longer trust the player."
                ),
                "state_updates": {
                    "merchant_health": new_merchant_health,
                    "merchant_hostile": True,
                    "relationship_with_merchant": max(game_state["relationship_with_merchant"] - 50, 0),
                    "player_reputation": max(game_state["player_reputation"] - 25, 0),
                    "world_mood": "dangerous"
                }
            }

        if target == "bartender":
            if game_state["bartender_health"] <= 0:
                return {
                    "success": False,
                    "message": "The bartender is already unable to respond.",
                    "state_updates": {}
                }

            hit = self.attribute_agent.calculate_hit()

            if not hit:
                return {
                    "success": True,
                    "message": "The player attacks the bartender, but misses. The tavern falls silent, and the bartender becomes hostile.",
                    "state_updates": {
                        "bartender_hostile": True,
                        "relationship_with_bartender": max(game_state["relationship_with_bartender"] - 35, 0),
                        "tavern_reputation": max(game_state["tavern_reputation"] - 25, 0),
                        "player_reputation": max(game_state["player_reputation"] - 20, 0),
                        "bartender_mood": "angry",
                        "world_mood": "dangerous"
                    }
                }

            damage = self.attribute_agent.calculate_damage()
            new_bartender_health = max(game_state["bartender_health"] - damage, 0)

            return {
                "success": True,
                "message": (
                    f"The player attacks the bartender and deals {damage} damage. "
                    "The bartender becomes hostile, and the tavern turns against the player."
                ),
                "state_updates": {
                    "bartender_health": new_bartender_health,
                    "bartender_hostile": True,
                    "relationship_with_bartender": max(game_state["relationship_with_bartender"] - 60, 0),
                    "tavern_reputation": max(game_state["tavern_reputation"] - 40, 0),
                    "player_reputation": max(game_state["player_reputation"] - 30, 0),
                    "bartender_mood": "furious",
                    "world_mood": "dangerous"
                }
            }

        # =========================
        # Case 2: Player attacks normal enemy
        # =========================
        if not self.validation_agent.validate_combat(game_state):
            return {
                "success": False,
                "message": "There is no enemy to attack.",
                "state_updates": {}
            }

        hit = self.attribute_agent.calculate_hit()

        if not hit:
            return {
                "success": True,
                "message": "The player attacks, but misses the enemy.",
                "state_updates": {}
            }

        damage = self.attribute_agent.calculate_damage()
        new_enemy_health = max(game_state["enemy_health"] - damage, 0)

        return {
            "success": True,
            "message": f"The player hits the enemy and deals {damage} damage.",
            "state_updates": {
                "enemy_health": new_enemy_health
            }
        }

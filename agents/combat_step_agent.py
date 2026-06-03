import random
from typing import Dict, Any


class CombatStepAgent:
    """
    Handles separate combat calculation steps.

    This makes combat_action more suitable for DAG execution:
    hit calculation, damage calculation, death check, result application,
    and enemy reaction are separate nodes.
    """

    def calculate_hit(self, action_blocked: bool = False) -> Dict[str, Any]:
        if action_blocked:
            return {
                "success": False,
                "message": "Hit calculation skipped because the action was blocked.",
                "state_updates": {},
                "data": {
                    "hit": False
                }
            }

        roll = random.randint(1, 100)
        hit = roll <= 75

        return {
            "success": True,
            "message": f"Hit calculation completed. Roll: {roll}. Hit: {hit}.",
            "state_updates": {},
            "data": {
                "hit": hit,
                "hit_roll": roll
            }
        }

    def calculate_damage(self, hit: bool, action_blocked: bool = False) -> Dict[str, Any]:
        if action_blocked:
            return {
                "success": False,
                "message": "Damage calculation skipped because the action was blocked.",
                "state_updates": {},
                "data": {
                    "damage": 0
                }
            }

        if not hit:
            return {
                "success": True,
                "message": "No damage was calculated because the attack missed.",
                "state_updates": {},
                "data": {
                    "damage": 0
                }
            }

        damage = random.randint(8, 20)

        return {
            "success": True,
            "message": f"Damage calculation completed. Damage: {damage}.",
            "state_updates": {},
            "data": {
                "damage": damage
            }
        }

    def check_death(
        self,
        game_state: Dict[str, Any],
        target: str,
        hit: bool,
        damage: int,
        action_blocked: bool = False
    ) -> Dict[str, Any]:
        if action_blocked:
            return {
                "success": False,
                "message": "Death check skipped because the action was blocked.",
                "state_updates": {},
                "data": {
                    "target_defeated": False
                }
            }

        if not hit:
            return {
                "success": True,
                "message": "Death check completed. Target was not hit.",
                "state_updates": {},
                "data": {
                    "target_defeated": False
                }
            }

        health_key = None

        if target == "enemy":
            health_key = "enemy_health"
        elif target == "merchant":
            health_key = "merchant_health"
        elif target == "bartender":
            health_key = "bartender_health"

        if health_key is None:
            return {
                "success": False,
                "message": "Death check failed because the target is unknown.",
                "state_updates": {},
                "data": {
                    "target_defeated": False
                }
            }

        current_health = game_state.get(health_key, 0)
        projected_health = max(current_health - damage, 0)
        target_defeated = projected_health <= 0

        return {
            "success": True,
            "message": (
                f"Death check completed. "
                f"Target health would change from {current_health} to {projected_health}."
            ),
            "state_updates": {},
            "data": {
                "target_defeated": target_defeated,
                "projected_health": projected_health,
                "health_key": health_key
            }
        }

    def apply_combat_result(
        self,
        game_state: Dict[str, Any],
        target: str,
        hit: bool,
        damage: int,
        target_defeated: bool,
        action_blocked: bool = False,
        blocked_reason: str = ""
    ) -> Dict[str, Any]:
        if action_blocked:
            return {
                "success": False,
                "message": blocked_reason or "The combat action was blocked.",
                "state_updates": {}
            }

        if target == "enemy":
            if not hit:
                return {
                    "success": True,
                    "message": "The player attacks, but misses the enemy.",
                    "state_updates": {}
                }

            new_enemy_health = max(game_state["enemy_health"] - damage, 0)

            if target_defeated:
                message = f"The player hits the enemy for {damage} damage. The enemy is defeated."
            else:
                message = f"The player hits the enemy and deals {damage} damage."

            return {
                "success": True,
                "message": message,
                "state_updates": {
                    "enemy_health": new_enemy_health,
                    "world_mood": "tense"
                }
            }

        if target == "merchant":
            if not hit:
                return {
                    "success": True,
                    "message": (
                        "The player attacks the merchant, but misses. "
                        "The merchant becomes frightened and hostile."
                    ),
                    "state_updates": {
                        "merchant_hostile": True,
                        "relationship_with_merchant": max(
                            game_state["relationship_with_merchant"] - 30,
                            0
                        ),
                        "player_reputation": max(
                            game_state["player_reputation"] - 15,
                            0
                        ),
                        "world_mood": "tense"
                    }
                }

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
                    "relationship_with_merchant": max(
                        game_state["relationship_with_merchant"] - 50,
                        0
                    ),
                    "player_reputation": max(
                        game_state["player_reputation"] - 25,
                        0
                    ),
                    "world_mood": "dangerous"
                }
            }

        if target == "bartender":
            if not hit:
                return {
                    "success": True,
                    "message": (
                        "The player attacks the bartender, but misses. "
                        "The tavern falls silent, and the bartender becomes hostile."
                    ),
                    "state_updates": {
                        "bartender_hostile": True,
                        "relationship_with_bartender": max(
                            game_state["relationship_with_bartender"] - 35,
                            0
                        ),
                        "tavern_reputation": max(
                            game_state["tavern_reputation"] - 25,
                            0
                        ),
                        "player_reputation": max(
                            game_state["player_reputation"] - 20,
                            0
                        ),
                        "bartender_mood": "angry",
                        "world_mood": "dangerous"
                    }
                }

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
                    "relationship_with_bartender": max(
                        game_state["relationship_with_bartender"] - 60,
                        0
                    ),
                    "tavern_reputation": max(
                        game_state["tavern_reputation"] - 40,
                        0
                    ),
                    "player_reputation": max(
                        game_state["player_reputation"] - 30,
                        0
                    ),
                    "bartender_mood": "furious",
                    "world_mood": "dangerous"
                }
            }

        return {
            "success": False,
            "message": "Combat result could not be applied because the target is unknown.",
            "state_updates": {}
        }

    def enemy_reaction(
        self,
        game_state: Dict[str, Any],
        target: str,
        target_defeated: bool,
        action_blocked: bool = False
    ) -> Dict[str, Any]:
        if action_blocked:
            return {
                "success": False,
                "message": "Enemy reaction skipped because the action was blocked.",
                "state_updates": {}
            }

        if target != "enemy":
            return {
                "success": True,
                "message": "No enemy reaction is needed for this target.",
                "state_updates": {}
            }

        if target_defeated or game_state.get("enemy_health", 0) <= 0:
            return {
                "success": True,
                "message": "The enemy cannot react because it has been defeated.",
                "state_updates": {}
            }

        retaliation_damage = random.randint(3, 10)
        new_player_health = max(game_state["player_health"] - retaliation_damage, 0)

        return {
            "success": True,
            "message": (
                f"The enemy retaliates and deals {retaliation_damage} damage to the player."
            ),
            "state_updates": {
                "player_health": new_player_health,
                "world_mood": "hostile"
            }
        }

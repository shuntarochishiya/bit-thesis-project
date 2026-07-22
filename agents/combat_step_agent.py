import random
from typing import Dict, Any


class CombatStepAgent:
    """
    Handles deterministic combat calculation steps for DAG execution.

    The agent returns structured data for every step. Player-facing text is
    assembled by ExecutionEngine after the target reaction has been applied.
    """

    TARGET_HEALTH_KEYS = {
        "enemy": "enemy_health",
        "merchant": "merchant_health",
        "bartender": "bartender_health"
    }

    def calculate_hit(self, action_blocked: bool = False) -> Dict[str, Any]:
        if action_blocked:
            return {
                "success": False,
                "message": "Hit calculation skipped because the action was blocked.",
                "state_updates": {},
                "data": {"hit": False, "hit_roll": None}
            }

        roll = random.randint(1, 100)
        hit = roll <= 75

        return {
            "success": True,
            "message": "Hit calculation completed.",
            "state_updates": {},
            "data": {"hit": hit, "hit_roll": roll}
        }

    def calculate_damage(self, hit: bool, action_blocked: bool = False) -> Dict[str, Any]:
        if action_blocked:
            return {
                "success": False,
                "message": "Damage calculation skipped because the action was blocked.",
                "state_updates": {},
                "data": {"damage": 0}
            }

        damage = random.randint(8, 20) if hit else 0

        return {
            "success": True,
            "message": "Damage calculation completed.",
            "state_updates": {},
            "data": {"damage": damage}
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
                    "target_defeated": False,
                    "projected_health": None,
                    "health_key": None,
                    "target_health_before": None
                }
            }

        health_key = self.TARGET_HEALTH_KEYS.get(target)
        if health_key is None:
            return {
                "success": False,
                "message": "Death check failed because the target is unknown.",
                "state_updates": {},
                "data": {
                    "target_defeated": False,
                    "projected_health": None,
                    "health_key": None,
                    "target_health_before": None
                }
            }

        current_health = int(game_state.get(health_key, 0))
        projected_health = max(current_health - damage, 0) if hit else current_health

        return {
            "success": True,
            "message": "Death check completed.",
            "state_updates": {},
            "data": {
                "target_defeated": projected_health <= 0,
                "projected_health": projected_health,
                "health_key": health_key,
                "target_health_before": current_health
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
                "state_updates": {},
                "data": {"attack_applied": False}
            }

        health_key = self.TARGET_HEALTH_KEYS.get(target)
        if health_key is None:
            return {
                "success": False,
                "message": "Combat result could not be applied because the target is unknown.",
                "state_updates": {},
                "data": {"attack_applied": False}
            }

        updates: Dict[str, Any] = {"world_mood": "tense"}
        target_health_after = int(game_state.get(health_key, 0))

        if hit:
            target_health_after = max(target_health_after - damage, 0)
            updates[health_key] = target_health_after

        if target == "merchant":
            updates.update({
                "merchant_hostile": True,
                "relationship_with_merchant": max(
                    int(game_state.get("relationship_with_merchant", 0)) - (50 if hit else 30),
                    0
                ),
                "player_reputation": max(
                    int(game_state.get("player_reputation", 0)) - (25 if hit else 15),
                    0
                ),
                "world_mood": "dangerous" if hit else "tense"
            })

        elif target == "bartender":
            updates.update({
                "bartender_hostile": True,
                "relationship_with_bartender": max(
                    int(game_state.get("relationship_with_bartender", 0)) - (60 if hit else 35),
                    0
                ),
                "tavern_reputation": max(
                    int(game_state.get("tavern_reputation", 0)) - (40 if hit else 25),
                    0
                ),
                "player_reputation": max(
                    int(game_state.get("player_reputation", 0)) - (30 if hit else 20),
                    0
                ),
                "bartender_mood": "furious" if hit else "angry",
                "world_mood": "dangerous"
            })

        return {
            "success": True,
            "message": "Combat result applied.",
            "state_updates": updates,
            "data": {
                "attack_applied": True,
                "target_health_after": target_health_after,
                "target_defeated": target_defeated
            }
        }

    def calculate_target_reaction(
        self,
        game_state: Dict[str, Any],
        target: str,
        hit: bool,
        target_defeated: bool,
        action_blocked: bool = False
    ) -> Dict[str, Any]:
        if action_blocked:
            return {
                "success": False,
                "message": "Target reaction calculation skipped because the action was blocked.",
                "state_updates": {},
                "data": {
                    "reaction_type": "blocked",
                    "retaliation_damage": 0
                }
            }

        if target_defeated:
            return {
                "success": True,
                "message": "The defeated target cannot react.",
                "state_updates": {},
                "data": {
                    "reaction_type": "defeated",
                    "retaliation_damage": 0
                }
            }

        if target == "enemy":
            retaliation_damage = random.randint(3, 10)
            return {
                "success": True,
                "message": "Enemy counterattack selected.",
                "state_updates": {},
                "data": {
                    "reaction_type": "counterattack",
                    "retaliation_damage": retaliation_damage
                }
            }

        if target == "merchant":
            health = int(game_state.get("merchant_health", 0))
            reaction_type = "flee" if hit and health <= 15 else "call_for_help"
            return {
                "success": True,
                "message": "Merchant reaction selected.",
                "state_updates": {},
                "data": {
                    "reaction_type": reaction_type,
                    "retaliation_damage": 0
                }
            }

        if target == "bartender":
            return {
                "success": True,
                "message": "Bartender reaction selected.",
                "state_updates": {},
                "data": {
                    "reaction_type": "turn_tavern_hostile",
                    "retaliation_damage": 0
                }
            }

        return {
            "success": False,
            "message": "Target reaction could not be calculated because the target is unknown.",
            "state_updates": {},
            "data": {
                "reaction_type": "none",
                "retaliation_damage": 0
            }
        }

    def apply_target_reaction(
        self,
        game_state: Dict[str, Any],
        target: str,
        reaction_type: str,
        retaliation_damage: int,
        action_blocked: bool = False
    ) -> Dict[str, Any]:
        if action_blocked:
            return {
                "success": False,
                "message": "Target reaction skipped because the action was blocked.",
                "state_updates": {},
                "data": {
                    "player_health_after": game_state.get("player_health"),
                    "reaction_applied": False
                }
            }

        updates: Dict[str, Any] = {}
        player_health_after = int(game_state.get("player_health", 0))

        if reaction_type == "counterattack":
            player_health_after = max(player_health_after - retaliation_damage, 0)
            updates = {
                "player_health": player_health_after,
                "world_mood": "hostile"
            }

        elif reaction_type == "call_for_help":
            updates = {"world_mood": "dangerous"}

        elif reaction_type == "flee":
            updates = {"world_mood": "dangerous"}

        elif reaction_type == "turn_tavern_hostile":
            updates = {
                "bartender_hostile": True,
                "bartender_mood": "furious",
                "world_mood": "dangerous"
            }

        elif reaction_type in {"defeated", "none"}:
            updates = {}

        else:
            return {
                "success": False,
                "message": f"Unknown target reaction type: {reaction_type}",
                "state_updates": {},
                "data": {
                    "player_health_after": player_health_after,
                    "reaction_applied": False
                }
            }

        return {
            "success": True,
            "message": "Target reaction applied.",
            "state_updates": updates,
            "data": {
                "player_health_after": player_health_after,
                "reaction_applied": True
            }
        }

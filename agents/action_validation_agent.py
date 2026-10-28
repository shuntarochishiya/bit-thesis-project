from typing import Any, Dict


class ActionValidationAgent:
    """Validate the basic action contract without changing game state.

    This agent checks the *shape* of a requested action. It deliberately does
    not decide whether an NPC is alive, present, hostile, or affordable: those
    are world-state preconditions handled by PreconditionAgent.
    """

    SUPPORTED_INTENTS = frozenset({"combat_action", "persuasion_action"})
    NPC_TARGETS = frozenset({"enemy", "merchant", "bartender"})

    def execute(
        self,
        player_input: str,
        intent: str,
        target: str,
        game_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        def rejected(reason: str) -> Dict[str, Any]:
            return {"success": False, "message": reason, "state_updates": {}}

        if not isinstance(game_state, dict):
            return rejected("Action validation failed: game state must be a dictionary.")
        if not isinstance(intent, str) or intent not in self.SUPPORTED_INTENTS:
            return rejected("Action validation failed: unsupported action intent.")
        if not isinstance(player_input, str) or not player_input.strip():
            return rejected("Action validation failed: player input is empty.")
        if not isinstance(target, str) or target not in self.NPC_TARGETS:
            return rejected("Action validation failed: target is not a recognized NPC.")

        # The following are structural consistency checks, not availability checks.
        # Missing optional fields remain the responsibility of existing state and
        # precondition components; do not introduce a new schema here.
        health_key = f"{target}_health"
        if health_key in game_state:
            health = game_state[health_key]
            if isinstance(health, bool) or not isinstance(health, (int, float)):
                return rejected(f"Action validation failed: invalid {health_key} type.")
        hostile_key = f"{target}_hostile"
        if hostile_key in game_state and not isinstance(game_state[hostile_key], bool):
            return rejected(f"Action validation failed: invalid {hostile_key} type.")

        return {
            "success": True,
            "message": "Action contract validated.",
            "state_updates": {},
        }

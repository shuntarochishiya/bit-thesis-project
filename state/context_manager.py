from typing import Dict, Any, Optional


class ContextManager:
    """
    Stores short-term interaction context.

    This helps the system understand that the next player input may continue
    the previous scene or conversation.
    """

    def __init__(self):
        self.context: Dict[str, Optional[str]] = {
            "active_location": "old forest",
            "active_target": None,
            "active_intent": None,
            "active_conversation": None,
            "last_player_input": None,
            "last_system_result": None
        }

    def get_context(self) -> Dict[str, Optional[str]]:
        return self.context.copy()

    def update_after_turn(
        self,
        player_input: str,
        intent: str,
        target: str,
        system_result: str,
        game_state: Dict[str, Any]
    ):
        self.context["active_location"] = str(game_state.get("location", "unknown"))
        self.context["active_intent"] = intent
        self.context["last_player_input"] = player_input
        self.context["last_system_result"] = system_result

        if target != "unknown":
            self.context["active_target"] = target

        if target in ["bartender", "merchant", "enemy"]:
            self.context["active_conversation"] = target

        if intent == "exploration_action" and target == "environment":
            self.context["active_conversation"] = None

    def resolve_target_from_context(self, target: str) -> str:
        """
        If the current input has no clear target, use the active conversation target.
        """

        if target != "unknown":
            return target

        if self.context.get("active_conversation"):
            return str(self.context["active_conversation"])

        if self.context.get("active_target"):
            return str(self.context["active_target"])

        return "unknown"

    def display_context(self):
        print("\n--- Interaction Context ---")
        for key, value in self.context.items():
            print(f"{key}: {value}")
        print("---------------------------\n")

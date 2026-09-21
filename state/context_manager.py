from typing import Dict, Any, Optional


class ContextManager:
    """Short-term scene and NPC interaction context."""

    CONVERSATION_INTENTS = {"dialogue_action", "persuasion_action", "tavern_action"}

    def __init__(self):
        self.context: Dict[str, Optional[str]] = {
            "active_location": "old forest",
            "active_target": None,
            "active_intent": None,
            "active_conversation": None,
            "last_player_input": None,
            "last_system_result": None,
        }

    def get_context(self) -> Dict[str, Optional[str]]:
        return self.context.copy()

    def update_after_turn(self, player_input: str, intent: str,
                          target: Optional[str], system_result: str,
                          game_state: Dict[str, Any]):
        location = game_state.get("location")
        if location not in (None, "", "unknown"):
            self.context["active_location"] = str(location)

        self.context["active_intent"] = intent
        self.context["last_player_input"] = player_input
        self.context["last_system_result"] = system_result

        target_norm = self._normalize_target(target)

        if intent == "exploration_action":
            self.context["active_target"] = None
            self.context["active_conversation"] = None
            return

        if intent in self.CONVERSATION_INTENTS and target_norm:
            self.context["active_target"] = target_norm
            self.context["active_conversation"] = target_norm
            return

        if intent == "combat_action":
            self.context["active_target"] = target_norm
            self.context["active_conversation"] = None
            return

        self.context["active_target"] = target_norm

    def resolve_target_from_context(self, target: Optional[str]) -> str:
        target_norm = self._normalize_target(target)
        if target_norm:
            return target_norm

        active_conversation = self._normalize_target(
            self.context.get("active_conversation")
        )
        if active_conversation:
            return active_conversation

        active_target = self._normalize_target(self.context.get("active_target"))
        if active_target:
            return active_target

        return "unknown"

    @staticmethod
    def _normalize_target(target: Optional[str]) -> Optional[str]:
        if target is None:
            return None
        value = str(target).strip()
        if not value or value.lower() in {"unknown", "none", "environment"}:
            return None
        return value

    def display_context(self):
        print("\n--- Interaction Context ---")
        for key, value in self.context.items():
            print(f"{key}: {value}")
        print("---------------------------\n")

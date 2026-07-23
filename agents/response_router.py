class ResponseRouter:
    DETERMINISTIC_INTENTS = {
        "move_action",
        "combat_action",
        "take_action",
        "use_item_action",
        "buy_action",
        "sell_action",
        "enter_location_action",
        "leave_location_action"
    }

    MEMORY_INTENTS = {
        "recall_action",
        "context_action"
    }

    LLM_INTENTS = {
        "dialogue_action",
        "persuasion_action"
    }

    def choose_source(
        self,
        intent: str,
        relevant_memory: list,
        requires_dialogue: bool = False
    ) -> str:
        if requires_dialogue or intent in self.LLM_INTENTS:
            return (
                "llm_with_memory"
                if relevant_memory
                else "llm"
            )

        if intent in self.MEMORY_INTENTS and relevant_memory:
            return "memory"

        return "deterministic"

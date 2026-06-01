import json
from datetime import datetime
from typing import Dict, Any, List


class MemorySystem:
    """
    Persistent event memory system.

    This memory stores structured game events in memory.json.
    It keeps player actions, recognized intent, target, system result,
    and state changes across different program runs.

    This represents a simplified version of long-term event memory.
    Later, it can be upgraded to vector-based semantic memory.
    """

    def __init__(self, memory_path: str = "memory.json"):
        self.memory_path = memory_path
        self.events: List[Dict[str, Any]] = []
        self.load_memory()

    def load_memory(self):
        try:
            with open(self.memory_path, "r", encoding="utf-8") as file:
                data = json.load(file)

            if isinstance(data, list):
                self.events = data
            else:
                self.events = []

        except FileNotFoundError:
            self.events = []
            self.save_memory()

        except json.JSONDecodeError:
            print("Warning: memory.json is corrupted. Starting with empty memory.")
            self.events = []
            self.save_memory()

    def save_memory(self):
        with open(self.memory_path, "w", encoding="utf-8") as file:
            json.dump(self.events, file, indent=2, ensure_ascii=False)

    def add_event(
        self,
        player_input: str,
        intent: str,
        target: str,
        system_result: str,
        state_before: Dict[str, Any],
        state_after: Dict[str, Any]
    ):
        """
        Saves one structured event into persistent memory.
        """

        state_changes = self.calculate_state_changes(state_before, state_after)

        memory_item = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "player_input": player_input,
            "intent": intent,
            "target": target,
            "system_result": system_result,
            "state_changes": state_changes,
            "important_state": {
                "merchant_health": state_after.get("merchant_health"),
                "merchant_hostile": state_after.get("merchant_hostile"),
                "relationship_with_merchant": state_after.get("relationship_with_merchant"),

                "bartender_health": state_after.get("bartender_health"),
                "bartender_hostile": state_after.get("bartender_hostile"),
                "relationship_with_bartender": state_after.get("relationship_with_bartender"),
                "bartender_mood": state_after.get("bartender_mood"),
                "bartender_role": state_after.get("bartender_role"),
                "bartender_gender": state_after.get("bartender_gender"),
                "bartender_pronouns": state_after.get("bartender_pronouns"),
                "tavern_reputation": state_after.get("tavern_reputation"),

                "player_reputation": state_after.get("player_reputation"),
                "enemy_health": state_after.get("enemy_health"),
                "location": state_after.get("location"),
                "world_mood": state_after.get("world_mood")
            }
        }

        self.events.append(memory_item)
        self.save_memory()

        return memory_item

    def calculate_state_changes(
        self,
        state_before: Dict[str, Any],
        state_after: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compares two game states and returns only changed values.
        """

        changes = {}

        all_keys = set(state_before.keys()) | set(state_after.keys())

        for key in all_keys:
            before_value = state_before.get(key)
            after_value = state_after.get(key)

            if before_value != after_value:
                changes[key] = {
                    "before": before_value,
                    "after": after_value
                }

        return changes

    def retrieve_recent_events(self, limit: int = 5) -> List[str]:
        """
        Returns recent events in a readable text format for the LLM prompt.
        This method is also compatible with older memory formats.
        """

        recent_events = self.events[-limit:]
        formatted_events = []

        for item in recent_events:
            # New structured format
            if isinstance(item, dict) and "player_input" in item:
                formatted_events.append(
                    f"[{item.get('timestamp')}] "
                    f"Input: {item.get('player_input')} | "
                    f"Intent: {item.get('intent')} | "
                    f"Target: {item.get('target')} | "
                    f"Result: {item.get('system_result')} | "
                    f"State changes: {item.get('state_changes')}"
                )

            # Old format: {"timestamp": "...", "event": "..."}
            elif isinstance(item, dict) and "event" in item:
                formatted_events.append(
                    f"[{item.get('timestamp')}] {item.get('event')}"
                )

            # Very old or unexpected format
            else:
                formatted_events.append(str(item))

        return formatted_events

    def search_memory(self, keyword: str) -> List[str]:
        """
        Simple keyword-based search over memory.
        """

        results = []
        keyword = keyword.lower()

        for item in self.events:
            item_text = json.dumps(item, ensure_ascii=False).lower()

            if keyword in item_text:
                results.append(item_text)

        return results

    def display_memory(self, limit: int = 10):
        """
        Displays recent memory events in the terminal.
        """

        print("\n--- Persistent Memory ---")

        if not self.events:
            print("No memory events available.")
            print("-------------------------\n")
            return

        for item in self.events[-limit:]:
            if isinstance(item, dict) and "player_input" in item:
                print(f"\nTime: {item.get('timestamp')}")
                print(f"Player input: {item.get('player_input')}")
                print(f"Intent: {item.get('intent')}")
                print(f"Target: {item.get('target')}")
                print(f"System result: {item.get('system_result')}")

                print("State changes:")
                state_changes = item.get("state_changes", {})

                if state_changes:
                    for key, value in state_changes.items():
                        print(f"  - {key}: {value.get('before')} -> {value.get('after')}")
                else:
                    print("  - No state changes")

                important_state = item.get("important_state", {})
                print("Important state after action:")
                for key, value in important_state.items():
                    print(f"  - {key}: {value}")

            elif isinstance(item, dict) and "event" in item:
                print(f"[{item.get('timestamp')}] {item.get('event')}")

            else:
                print(item)

        print("-------------------------\n")

    def clear_memory(self):
        """
        Clears persistent memory.
        """

        self.events = []
        self.save_memory()
        print("\nPersistent memory has been cleared.\n")

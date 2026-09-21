import json
from datetime import datetime
from typing import Dict, Any, List, Optional


class MemorySystem:
    """Persistent structured event memory with NPC-specific working dialogue."""

    def __init__(self, memory_path: str = "memory.json"):
        self.memory_path = memory_path
        self.events: List[Dict[str, Any]] = []
        self.load_memory()

    def load_memory(self):
        try:
            with open(self.memory_path, "r", encoding="utf-8") as file:
                data = json.load(file)
            self.events = data if isinstance(data, list) else []
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

    def add_event(self, player_input: str, intent: str, target: str,
                  system_result: str, state_before: Dict[str, Any],
                  state_after: Dict[str, Any]):
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
                "world_mood": state_after.get("world_mood"),
            }
        }
        self.events.append(memory_item)
        self.save_memory()
        return memory_item

    def calculate_state_changes(self, state_before: Dict[str, Any],
                                state_after: Dict[str, Any]) -> Dict[str, Any]:
        changes = {}
        for key in set(state_before) | set(state_after):
            before_value = state_before.get(key)
            after_value = state_after.get(key)
            if before_value != after_value:
                changes[key] = {"before": before_value, "after": after_value}
        return changes

    def retrieve_dialogue_history(self, target: Optional[str],
                                  limit: int = 3) -> List[Dict[str, str]]:
        if not target:
            return []

        target_norm = str(target).strip().lower()
        dialogue_intents = {"dialogue_action", "persuasion_action", "tavern_action"}
        boundary_intents = {"exploration_action", "combat_action"}
        matched: List[Dict[str, str]] = []

        for item in reversed(self.events):
            if not isinstance(item, dict):
                continue

            item_intent = str(item.get("intent") or "").strip().lower()
            item_target = str(item.get("target") or "").strip().lower()

            # Movement/exploration/combat closes the current working conversation.
            if item_intent in boundary_intents:
                break

            # Dialogue with another NPC starts a different working conversation.
            if (item_intent in dialogue_intents and item_target
                    and item_target != target_norm):
                break

            if item_target != target_norm or item_intent not in dialogue_intents:
                continue

            player_text = str(item.get("player_input") or "").strip()
            npc_text = str(item.get("system_result") or "").strip()
            if player_text or npc_text:
                matched.append({"player": player_text, "npc": npc_text})

            if len(matched) >= max(1, limit):
                break

        matched.reverse()
        return matched

    def retrieve_recent_events(self, limit: int = 5) -> List[str]:
        formatted_events = []
        for item in self.events[-limit:]:
            if isinstance(item, dict) and "player_input" in item:
                formatted_events.append(
                    f"[{item.get('timestamp')}] Input: {item.get('player_input')} | "
                    f"Intent: {item.get('intent')} | Target: {item.get('target')} | "
                    f"Result: {item.get('system_result')} | "
                    f"State changes: {item.get('state_changes')}"
                )
            elif isinstance(item, dict) and "event" in item:
                formatted_events.append(f"[{item.get('timestamp')}] {item.get('event')}")
            else:
                formatted_events.append(str(item))
        return formatted_events

    def search_memory(self, keyword: str) -> List[str]:
        results = []
        keyword = keyword.lower()
        for item in self.events:
            item_text = json.dumps(item, ensure_ascii=False).lower()
            if keyword in item_text:
                results.append(item_text)
        return results

    def display_memory(self, limit: int = 10):
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
                print("Important state after action:")
                for key, value in item.get("important_state", {}).items():
                    print(f"  - {key}: {value}")
            elif isinstance(item, dict) and "event" in item:
                print(f"[{item.get('timestamp')}] {item.get('event')}")
            else:
                print(item)
        print("-------------------------\n")

    def clear_memory(self):
        self.events = []
        self.save_memory()
        print("\nPersistent memory has been cleared.\n")

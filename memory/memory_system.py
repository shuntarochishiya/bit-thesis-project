import json
import re
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

    def add_event(
        self,
        player_input: str,
        intent: str,
        target: str,
        system_result: str,
        state_before: Dict[str, Any],
        state_after: Dict[str, Any],
    ):
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
            },
        }
        self.events.append(memory_item)
        self.save_memory()
        return memory_item

    def calculate_state_changes(
        self,
        state_before: Dict[str, Any],
        state_after: Dict[str, Any],
    ) -> Dict[str, Any]:
        changes = {}
        for key in set(state_before) | set(state_after):
            before_value = state_before.get(key)
            after_value = state_after.get(key)
            if before_value != after_value:
                changes[key] = {"before": before_value, "after": after_value}
        return changes

    def retrieve_dialogue_history(
        self,
        target: Optional[str],
        limit: int = 3,
    ) -> List[Dict[str, str]]:
        """
        Return recent dialogue turns for one NPC only.

        Switching to another NPC does NOT erase or close this NPC's working
        dialogue. This lets the player talk to Merchant, switch to Bartender,
        then return to Merchant and continue the previous thread.

        Exploration/movement and combat remain hard conversation boundaries.
        """
        if not target:
            return []

        target_norm = str(target).strip().lower()

        dialogue_intents = {
            "dialogue_action",
            "persuasion_action",
            "tavern_action",
        }

        # These actions genuinely interrupt/close the current conversation.
        boundary_intents = {
            "exploration_action",
            "combat_action",
            "dialogue_reset",  # persistent global boundary for all NPC threads
            "scene_transition",  # Unity scene sync starts a new working session
        }

        matched: List[Dict[str, str]] = []
        latest_substantive: Optional[Dict[str, str]] = None

        for item in reversed(self.events):
            if not isinstance(item, dict):
                continue

            item_intent = str(item.get("intent") or "").strip().lower()
            item_target = str(item.get("target") or "").strip().lower()

            # Real scene/action boundaries still close working dialogue.
            if item_intent in boundary_intents:
                break

            # IMPORTANT:
            # Dialogue with another NPC is simply skipped, not treated as a
            # boundary. Each NPC therefore has an independent working thread.
            if item_target != target_norm:
                continue

            if item_intent not in dialogue_intents:
                continue

            player_text = str(item.get("player_input") or "").strip()
            npc_text = str(item.get("system_result") or "").strip()

            if player_text or npc_text:
                turn = {
                    "player": player_text,
                    "npc": npc_text,
                    "intent": item_intent,
                }
                if len(matched) < max(1, limit):
                    matched.append(turn)
                # Keep one substantive exchange as a topic anchor even after
                # multiple purchases. A newer unrelated exchange supersedes it.
                if item_intent != "tavern_action" and latest_substantive is None:
                    latest_substantive = turn
                if len(matched) >= max(1, limit) and latest_substantive is not None:
                    break

        # When the last N turns were only service actions, prepend the last
        # actual conversation so follow-ups retain their referent. Do not
        # expose other NPCs or cross movement/combat/reset boundaries.
        if latest_substantive is not None and latest_substantive not in matched:
            matched.append(latest_substantive)
        matched.reverse()
        return matched

    def retrieve_prior_session_dialogue(
        self,
        target: Optional[str],
        player_input: str,
        limit: int = 2,
    ) -> List[Dict[str, str]]:
        """Recall older NPC-specific dialogue when the player refers to the past.

        This is persistent episodic narrative context, NOT the active working
        conversation. It never alters game state or implicitly resumes a topic.
        Events are read from memory.json, so recall survives process restarts.
        """
        if not target or not player_input or limit <= 0:
            return []

        target_norm = str(target).strip().lower()
        utterance = player_input.lower()
        # Explicit reminders, or a meaningful subject referenced in the past.
        recall_requested = bool(re.search(
            r"\b(?:remember|recall|last time|earlier|previously|"
            r"we (?:spoke|talked|discussed)|you (?:told|said|mentioned)|"
            r"back (?:then|when)|again)\b", utterance,
        ))
        if not recall_requested:
            return []

        # Working conversation closes at movement, combat, explicit reset,
        # or scene synchronization. Only events before that boundary qualify.
        boundary = {"exploration_action", "combat_action", "dialogue_reset", "scene_transition"}
        cut = None
        for index in range(len(self.events) - 1, -1, -1):
            event = self.events[index]
            if isinstance(event, dict) and str(event.get("intent", "")).lower() in boundary:
                cut = index
                break
        if cut is None:
            return []  # There has not yet been a separate conversation session.

        stopwords = {
            "remember", "recall", "earlier", "previously", "again", "about", "that",
            "those", "there", "where", "when", "what", "from", "with", "have", "said",
            "told", "mentioned", "talked", "spoke", "discussed", "before", "last",
            "time", "you", "your", "our", "the", "and", "were", "was", "did",
            "we", "me", "can", "could", "some", "more", "this", "they", "them",
        }
        words = {w for w in re.findall(r"[a-z]{3,}", utterance) if w not in stopwords}
        ranked = []
        for index, item in enumerate(self.events[:cut]):
            if not isinstance(item, dict):
                continue
            if str(item.get("target", "")).strip().lower() != target_norm:
                continue
            if str(item.get("intent", "")).strip().lower() not in {"dialogue_action", "persuasion_action"}:
                continue
            player = str(item.get("player_input") or "").strip()
            npc = str(item.get("system_result") or "").strip()
            if not player or not npc:
                continue
            content = (player + " " + npc).lower()
            score = sum(1 for word in words if re.search(r"\b" + re.escape(word) + r"\b", content))
            # If the user names a topic, do not substitute a different memory.
            if words and score == 0:
                continue
            ranked.append((score, index, {"player": player[:220], "npc": npc[:420]}))

        ranked.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return [entry for _, _, entry in ranked[:limit]]

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
                formatted_events.append(
                    f"[{item.get('timestamp')}] {item.get('event')}"
                )
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
                        print(
                            f"  - {key}: "
                            f"{value.get('before')} -> {value.get('after')}"
                        )
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

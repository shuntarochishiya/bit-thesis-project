import copy
from datetime import datetime
from typing import Dict, Any, List, Optional


class GameStateManager:
    """
    Manages the current game state.

    This version supports:
    - state versioning
    - snapshots
    - rollback
    - audit log
    """

    def __init__(self):
        self.state: Dict[str, Any] = {
            "location": "old forest",

            "player_health": 100,
            "gold": 50,
            "inventory": ["small knife", "map"],

            "enemy_health": 60,
            "current_enemy": "forest goblin",

            "merchant_health": 100,
            "relationship_with_merchant": 50,
            "merchant_hostile": False,

            "bartender_health": 100,
            "relationship_with_bartender": 50,
            "bartender_hostile": False,
            "bartender_mood": "neutral",
            "bartender_role": "bartender",
            "bartender_gender": "unknown",
            "bartender_pronouns": "they/them",

            "tavern_reputation": 50,
            "player_reputation": 50,

            "world_mood": "mysterious",
            "world_danger": 20,
            "guard_alert_level": 10,
            "trade_activity": 60,
            "crime_level": 0,
            "active_conflicts": [],
            "pending_events": [],
            "recent_events": []
        }

        self.version: int = 0
        self.snapshots: List[Dict[str, Any]] = []
        self.audit_log: List[Dict[str, Any]] = []

    def get_state(self) -> Dict[str, Any]:
        return copy.deepcopy(self.state)

    def get_version(self) -> int:
        return self.version

    def create_snapshot(self, label: str = "") -> int:
        """
        Creates a snapshot of the current state.
        Returns snapshot_id.
        """

        snapshot_id = len(self.snapshots)

        snapshot = {
            "snapshot_id": snapshot_id,
            "version": self.version,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "label": label,
            "state": copy.deepcopy(self.state)
        }

        self.snapshots.append(snapshot)

        return snapshot_id

    def rollback_to_snapshot(self, snapshot_id: int) -> bool:
        """
        Restores the state from a snapshot.
        """

        if snapshot_id < 0 or snapshot_id >= len(self.snapshots):
            return False

        snapshot = self.snapshots[snapshot_id]

        old_state = copy.deepcopy(self.state)
        self.state = copy.deepcopy(snapshot["state"])
        self.version += 1

        self.audit_log.append({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "version": self.version,
            "source": "GameStateManager",
            "reason": f"Rollback to snapshot {snapshot_id}: {snapshot.get('label')}",
            "changes": self.calculate_state_changes(old_state, self.state)
        })

        return True

    def update_state(
        self,
        updates: Dict[str, Any],
        source: str = "unknown",
        reason: str = ""
    ):
        """
        Updates the game state and records changes in the audit log.
        Existing code can still call update_state(updates) because source and reason are optional.
        """

        if not updates:
            return

        old_state = copy.deepcopy(self.state)

        for key, value in updates.items():
            self.state[key] = value

        self.version += 1

        changes = self.calculate_state_changes(old_state, self.state)

        self.audit_log.append({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "version": self.version,
            "source": source,
            "reason": reason,
            "changes": changes
        })

    def calculate_state_changes(
        self,
        state_before: Dict[str, Any],
        state_after: Dict[str, Any]
    ) -> Dict[str, Any]:
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

    def validate_state(self) -> bool:
        """
        Basic state validation.
        Returns False if the state contains impossible values.
        """

        numeric_limits = {
            "player_health": (0, 100),
            "enemy_health": (0, 100),
            "merchant_health": (0, 100),
            "bartender_health": (0, 100),

            "gold": (0, 999999),

            "relationship_with_merchant": (0, 100),
            "relationship_with_bartender": (0, 100),
            "player_reputation": (0, 100),
            "tavern_reputation": (0, 100),
            "world_danger": (0, 100),
            "guard_alert_level": (0, 100),
            "trade_activity": (0, 100),
            "crime_level": (0, 100)
        }

        for key, limits in numeric_limits.items():
            if key not in self.state:
                continue

            value = self.state[key]
            min_value, max_value = limits

            if not isinstance(value, int):
                return False

            if value < min_value or value > max_value:
                return False

        list_fields = [
            "inventory",
            "active_conflicts",
            "pending_events",
            "recent_events"
        ]

        for field in list_fields:
            if not isinstance(self.state.get(field), list):
                return False

        return True

    def display_state(self):
        print("\n--- Current Game State ---")
        print(f"State version: {self.version}")
        for key, value in self.state.items():
            print(f"{key}: {value}")
        print("--------------------------\n")

    def display_audit_log(self, limit: int = 10):
        print("\n--- State Audit Log ---")

        if not self.audit_log:
            print("No audit log entries available.")
            print("-----------------------\n")
            return

        for item in self.audit_log[-limit:]:
            print(f"\nTime: {item['timestamp']}")
            print(f"Version: {item['version']}")
            print(f"Source: {item['source']}")
            print(f"Reason: {item['reason']}")

            changes = item.get("changes", {})
            if changes:
                print("Changes:")
                for key, value in changes.items():
                    print(f"  - {key}: {value['before']} -> {value['after']}")
            else:
                print("Changes: none")

        print("-----------------------\n")

    def display_snapshots(self):
        print("\n--- State Snapshots ---")

        if not self.snapshots:
            print("No snapshots available.")
            print("-----------------------\n")
            return

        for snapshot in self.snapshots:
            print(
                f"Snapshot {snapshot['snapshot_id']} | "
                f"Version: {snapshot['version']} | "
                f"Time: {snapshot['timestamp']} | "
                f"Label: {snapshot['label']}"
            )

        print("-----------------------\n")

import copy
from datetime import datetime
from threading import RLock
from typing import Dict, Any, List, Optional, Tuple
from state.sqlite_state_store import SQLiteStateStore


class GameStateManager:
    """
    Manages the current game state.

    Features:
    - state versioning
    - snapshots
    - rollback
    - audit log
    - Compare-And-Set (CAS)
    - thread-safe state operations
    - validation before committing updates
    """

    def __init__(self, persistence_path=None):
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
            "pending_npc_events": [],
            "recent_events": []
        }

        self.version: int = 0
        self.snapshots: List[Dict[str, Any]] = []
        self.audit_log: List[Dict[str, Any]] = []

        self._lock = RLock()
        self._store = SQLiteStateStore(persistence_path) if persistence_path else None
        if self._store:
            loaded = self._store.load("game")
            if loaded is not None:
                if not self._validate_state_data(loaded["state"]):
                    raise ValueError("Invalid persisted game state")
                self.state = loaded["state"]
                self.version = loaded["version"]
                self.snapshots = loaded["snapshots"]
                self.audit_log = loaded["audit_log"]
            else:
                self._persist()

    def _persist(self):
        if self._store:
            self._store.save("game", {
                "state": self.state, "version": self.version,
                "snapshots": self.snapshots, "audit_log": self.audit_log,
            })

    # ============================================================
    # STATE ACCESS
    # ============================================================

    def get_state(self) -> Dict[str, Any]:
        """
        Returns a deep copy of the current game state.
        """
        with self._lock:
            return copy.deepcopy(self.state)

    def get_version(self) -> int:
        """
        Returns the current state version.
        """
        with self._lock:
            return self.version

    def get_state_with_version(
        self
    ) -> Tuple[Dict[str, Any], int]:
        """
        Atomically retrieves the current state and its version.

        This method should be used when preparing CAS updates.
        """
        with self._lock:
            return copy.deepcopy(self.state), self.version

    # ============================================================
    # SNAPSHOTS
    # ============================================================

    def create_snapshot(self, label: str = "") -> int:
        """
        Creates a snapshot of the current state.

        Returns:
            snapshot_id
        """
        with self._lock:
            snapshot_id = len(self.snapshots)

            snapshot = {
                "snapshot_id": snapshot_id,
                "version": self.version,
                "timestamp": datetime.now().isoformat(
                    timespec="seconds"
                ),
                "label": label,
                "state": copy.deepcopy(self.state)
            }

            self.snapshots.append(snapshot)
            try:
                self._persist()
            except Exception:
                self.snapshots.pop()
                raise
            return snapshot_id

    # ============================================================
    # ROLLBACK
    # ============================================================

    def rollback_to_snapshot(self, snapshot_id: int) -> bool:
        """
        Restores the game state from an existing snapshot.

        Rollback is recorded as a new state version.
        The version number never decreases.

        Returns:
            True if rollback succeeded.
            False if snapshot does not exist or is invalid.
        """
        with self._lock:
            if (
                not isinstance(snapshot_id, int)
                or isinstance(snapshot_id, bool)
                or snapshot_id < 0
                or snapshot_id >= len(self.snapshots)
            ):
                return False

            snapshot = self.snapshots[snapshot_id]

            restored_state = copy.deepcopy(snapshot["state"])

            if not self._validate_state_data(restored_state):
                return False

            old_state = copy.deepcopy(self.state)

            changes = self.calculate_state_changes(
                old_state,
                restored_state
            )

            self.state = restored_state
            self.version += 1

            self.audit_log.append({
                "timestamp": datetime.now().isoformat(
                    timespec="seconds"
                ),
                "version": self.version,
                "source": "GameStateManager",
                "reason": (
                    f"Rollback to snapshot {snapshot_id}: "
                    f"{snapshot.get('label', '')}"
                ),
                "changes": copy.deepcopy(changes)
            })

            try:
                self._persist()
            except Exception:
                self.state = old_state
                self.version -= 1
                self.audit_log.pop()
                raise
            return True

    # ============================================================
    # STATE UPDATES + CAS
    # ============================================================

    def update_state(
        self,
        updates: Dict[str, Any],
        source: str = "unknown",
        reason: str = "",
        expected_version: Optional[int] = None
    ) -> bool:
        """
        Atomically updates the game state.

        Supports Compare-And-Set (CAS).

        If expected_version is provided, the update succeeds
        only if it matches the current state version.

        The candidate state is validated before committing.

        Existing calls remain compatible:

            manager.update_state({"gold": 40})

        CAS example:

            state, version = manager.get_state_with_version()

            success = manager.update_state(
                {"gold": state["gold"] - 5},
                expected_version=version
            )

        Returns:
            True  - update accepted.
            False - version conflict or invalid state.
        """
        with self._lock:
            if (
                expected_version is not None
                and expected_version != self.version
            ):
                return False

            if not isinstance(updates, dict):
                return False

            if not updates:
                return True

            old_state = copy.deepcopy(self.state)

            candidate_state = copy.deepcopy(self.state)

            candidate_state.update(copy.deepcopy(updates))

            if not self._validate_state_data(candidate_state):
                return False

            changes = self.calculate_state_changes(
                old_state,
                candidate_state
            )

            if not changes:
                return True

            self.state = candidate_state
            self.version += 1

            self.audit_log.append({
                "timestamp": datetime.now().isoformat(
                    timespec="seconds"
                ),
                "version": self.version,
                "source": source,
                "reason": reason,
                "changes": copy.deepcopy(changes)
            })

            try:
                self._persist()
            except Exception:
                self.state = old_state
                self.version -= 1
                self.audit_log.pop()
                raise
            return True

    # ============================================================
    # STATE CHANGE TRACKING
    # ============================================================

    def calculate_state_changes(
        self,
        state_before: Dict[str, Any],
        state_after: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculates differences between two game states.
        """
        changes = {}

        all_keys = (
            set(state_before.keys())
            | set(state_after.keys())
        )

        for key in all_keys:
            before_value = state_before.get(key)
            after_value = state_after.get(key)

            if before_value != after_value:
                changes[key] = {
                    "before": copy.deepcopy(before_value),
                    "after": copy.deepcopy(after_value)
                }

        return changes

    # ============================================================
    # STATE VALIDATION
    # ============================================================

    def _validate_state_data(
        self,
        state: Dict[str, Any]
    ) -> bool:
        """
        Validates a supplied game state without modifying it.

        Used before updates and rollbacks.
        """
        if not isinstance(state, dict):
            return False

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
            if key not in state:
                continue

            value = state[key]
            min_value, max_value = limits

            # bool is a subclass of int in Python.
            # Explicitly reject boolean values.
            if type(value) is not int:
                return False

            if value < min_value or value > max_value:
                return False

        list_fields = [
            "inventory",
            "active_conflicts",
            "pending_events",
            "pending_npc_events",
            "recent_events"
        ]

        for field in list_fields:
            if not isinstance(state.get(field), list):
                return False

        return True

    def validate_state(self) -> bool:
        """
        Validates the current game state.

        Returns:
            True if valid.
            False if invalid.
        """
        with self._lock:
            return self._validate_state_data(self.state)

    # ============================================================
    # DISPLAY STATE
    # ============================================================

    def display_state(self):
        with self._lock:
            state = copy.deepcopy(self.state)
            version = self.version

        print("\n--- Current Game State ---")
        print(f"State version: {version}")

        for key, value in state.items():
            print(f"{key}: {value}")

        print("--------------------------\n")

    # ============================================================
    # DISPLAY AUDIT LOG
    # ============================================================

    def display_audit_log(self, limit: int = 10):
        with self._lock:
            entries = copy.deepcopy(self.audit_log[-limit:])

        print("\n--- State Audit Log ---")

        if not entries:
            print("No audit log entries available.")
            print("-----------------------\n")
            return

        for item in entries:
            print(f"\nTime: {item['timestamp']}")
            print(f"Version: {item['version']}")
            print(f"Source: {item['source']}")
            print(f"Reason: {item['reason']}")

            changes = item.get("changes", {})

            if changes:
                print("Changes:")

                for key, value in changes.items():
                    print(
                        f"  - {key}: "
                        f"{value['before']} -> {value['after']}"
                    )
            else:
                print("Changes: none")

        print("-----------------------\n")

    # ============================================================
    # DISPLAY SNAPSHOTS
    # ============================================================

    def display_snapshots(self):
        with self._lock:
            snapshots = copy.deepcopy(self.snapshots)

        print("\n--- State Snapshots ---")

        if not snapshots:
            print("No snapshots available.")
            print("-----------------------\n")
            return

        for snapshot in snapshots:
            print(
                f"Snapshot {snapshot['snapshot_id']} | "
                f"Version: {snapshot['version']} | "
                f"Time: {snapshot['timestamp']} | "
                f"Label: {snapshot['label']}"
            )

        print("-----------------------\n")

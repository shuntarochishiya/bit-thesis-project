import copy
from datetime import datetime
from uuid import uuid4
from typing import Any, Dict, List, Optional

from state.npc_profiles import NPCProfiles


class NPCStateManager:
    """Dynamic NPC state: emotion, trust, stress, goals, hostility and personal memory."""

    VALID_EMOTIONS = {
        "neutral", "calm", "happy", "grateful", "curious", "suspicious",
        "afraid", "angry", "hostile", "sad", "stressed", "alert", "friendly",
    }

    def __init__(self):
        self.states: Dict[str, Dict[str, Any]] = {
            npc_id: self._build_default_state(npc_id)
            for npc_id in NPCProfiles.list_npc_ids()
        }
        self.audit_log: List[Dict[str, Any]] = []
        self.version = 0

    def _build_default_state(self, npc_id: str) -> Dict[str, Any]:
        profile = NPCProfiles.get_profile(npc_id)
        initial_state = copy.deepcopy(profile.get("initial_state", {}))
        goals = profile.get("goals", [])

        return {
            "npc_id": npc_id,
            "health": int(initial_state.get("health", 100)),
            "alive": bool(initial_state.get("alive", True)),
            "emotion": initial_state.get("emotion", "neutral"),
            "stress": int(initial_state.get("stress", 10)),
            "fear": int(initial_state.get("fear", 0)),
            "anger": int(initial_state.get("anger", 0)),
            "trust": int(initial_state.get("trust", 50)),
            "hostile": bool(initial_state.get("hostile", False)),
            "current_goal": goals[0] if goals else "idle",
            "current_target": "player",
            "last_action": None,
            "last_interaction": None,
            "personal_memory": [],
            "relationship_modifiers": {},
            "status_effects": [],
        }

    @staticmethod
    def _clamp(value: int) -> int:
        return max(0, min(100, int(value)))

    def get_state(self, npc_id: str) -> Dict[str, Any]:
        npc_id = NPCProfiles.normalize_npc_id(npc_id)
        if npc_id not in self.states:
            raise KeyError(f"Unknown NPC state: {npc_id}")
        return copy.deepcopy(self.states[npc_id])

    def get_all_states(self) -> Dict[str, Dict[str, Any]]:
        return copy.deepcopy(self.states)

    def infer_emotion(self, state: Dict[str, Any]) -> str:
        if not state.get("alive", True):
            return "neutral"
        anger, fear = int(state.get("anger", 0)), int(state.get("fear", 0))
        stress, trust = int(state.get("stress", 0)), int(state.get("trust", 50))
        if state.get("hostile") and anger >= 60:
            return "hostile"
        if anger >= 70:
            return "angry"
        if fear >= 70:
            return "afraid"
        if stress >= 70:
            return "stressed"
        if trust >= 75 and stress <= 35:
            return "grateful"
        if trust <= 25:
            return "suspicious"
        if stress <= 20 and anger <= 20 and fear <= 20:
            return "calm"
        return "neutral"

    def update_state(
        self,
        npc_id: str,
        updates: Dict[str, Any],
        source: str = "unknown",
        reason: str = ""
    ):
        npc_id = NPCProfiles.normalize_npc_id(npc_id)
        before = self.get_state(npc_id)
        self.states[npc_id].update(copy.deepcopy(updates))
        self._normalize_state(npc_id)
        if not self.validate_state(npc_id):
            self.states[npc_id] = before
            raise ValueError(f"Invalid NPC state update for {npc_id}; changes rolled back.")
        self.version += 1
        self.audit_log.append({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "version": self.version,
            "npc_id": npc_id,
            "source": source,
            "reason": reason,
            "changes": self._calculate_changes(before, self.states[npc_id]),
        })

    def apply_emotional_change(
        self,
        npc_id: str,
        trust_delta: int = 0,
        stress_delta: int = 0,
        fear_delta: int = 0,
        anger_delta: int = 0,
        emotion: Optional[str] = None,
        source: str = "NPCStateManager",
        reason: str = "Emotional state change"
    ):
        state = self.get_state(npc_id)
        updates = {
            "trust": state["trust"] + trust_delta,
            "stress": state["stress"] + stress_delta,
            "fear": state["fear"] + fear_delta,
            "anger": state["anger"] + anger_delta,
        }
        updates["emotion"] = emotion or self.infer_emotion({**state, **updates})
        self.update_state(npc_id, updates, source, reason)

    def set_current_goal(self, npc_id: str, goal: str, target: Optional[str] = None):
        updates: Dict[str, Any] = {"current_goal": goal}
        if target is not None:
            updates["current_target"] = target
        self.update_state(npc_id, updates, "NPCStateManager", f"Goal changed to {goal}")

    def set_hostility(self, npc_id: str, hostile: bool, reason: str = "NPC hostility changed"):
        state = self.get_state(npc_id)
        updates: Dict[str, Any] = {"hostile": hostile}
        if hostile:
            updates.update({
                "anger": max(state["anger"], 60),
                "trust": min(state["trust"], 25),
                "emotion": "hostile"
            })
        else:
            updates["emotion"] = self.infer_emotion({**state, "hostile": False})
        self.update_state(npc_id, updates, "NPCStateManager", reason)

    def apply_damage(self, npc_id: str, damage: int, source: str = "unknown"):
        state = self.get_state(npc_id)
        health = max(0, state["health"] - max(0, int(damage)))
        updates = {
            "health": health,
            "alive": health > 0,
            "stress": state["stress"] + 25,
            "fear": state["fear"] + 20,
            "anger": state["anger"] + 30,
            "hostile": health > 0,
        }
        updates["emotion"] = "neutral" if health == 0 else self.infer_emotion({**state, **updates})
        self.update_state(npc_id, updates, source, f"NPC received {damage} damage")

    def add_memory(
        self,
        npc_id: str,
        event: str,
        importance: int = 50,
        emotional_tag: str = "neutral",
        related_entity: Optional[str] = None,
        max_memories: int = 20,
        event_type: str = "interaction",
        participants: Optional[List[str]] = None,
        emotional_impact: Optional[Dict[str, int]] = None,
        source: str = "experienced",
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Store one NPC-specific episodic memory.

        The original parameters remain valid for backward compatibility.
        """
        npc_id = NPCProfiles.normalize_npc_id(npc_id)
        state = self.get_state(npc_id)
        memories = list(state["personal_memory"])

        source_norm = str(source or "experienced").strip().lower()
        if source_norm not in {"experienced", "witnessed", "heard", "inferred"}:
            source_norm = "experienced"

        try:
            confidence_value = max(0.0, min(1.0, float(confidence)))
        except (TypeError, ValueError):
            confidence_value = 1.0

        impact = emotional_impact or {}
        normalized_impact = {
            "trust": int(impact.get("trust", 0)),
            "stress": int(impact.get("stress", 0)),
            "fear": int(impact.get("fear", 0)),
            "anger": int(impact.get("anger", 0)),
        }

        participant_list: List[str] = []
        for participant in participants or []:
            value = str(participant).strip()
            if value and value not in participant_list:
                participant_list.append(value)

        if npc_id not in participant_list:
            participant_list.append(npc_id)
        if related_entity and related_entity not in participant_list:
            participant_list.append(related_entity)

        memory = {
            "id": f"evt_{uuid4().hex[:12]}",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "event": event,          # backward-compatible field
            "summary": event,
            "event_type": str(event_type or "interaction").strip().lower(),
            "participants": participant_list,
            "related_entity": related_entity,
            "importance": self._clamp(importance),
            "emotional_tag": emotional_tag,
            "emotional_impact": normalized_impact,
            "source": source_norm,
            "confidence": confidence_value,
            "metadata": copy.deepcopy(metadata or {}),
        }

        memories.append(memory)
        memories = sorted(
            memories,
            key=lambda item: (
                int(item.get("importance", 0)),
                str(item.get("timestamp", "")),
            ),
            reverse=True,
        )[:max_memories]

        self.update_state(
            npc_id,
            {"personal_memory": memories, "last_interaction": event},
            "NPCStateManager",
            "NPC episodic memory updated",
        )
        return copy.deepcopy(memory)

    def retrieve_memories(
        self,
        npc_id: str,
        related_entity: Optional[str] = None,
        emotional_tag: Optional[str] = None,
        limit: int = 5,
        event_type: Optional[str] = None,
        source: Optional[str] = None,
        min_importance: int = 0,
        min_confidence: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Retrieve personal episodic memories for one NPC."""
        memories = self.get_state(npc_id)["personal_memory"]

        if related_entity is not None:
            memories = [
                m for m in memories
                if m.get("related_entity") == related_entity
            ]

        if emotional_tag is not None:
            memories = [
                m for m in memories
                if m.get("emotional_tag") == emotional_tag
            ]

        if event_type is not None:
            wanted = str(event_type).strip().lower()
            memories = [
                m for m in memories
                if str(m.get("event_type") or "").strip().lower() == wanted
            ]

        if source is not None:
            wanted = str(source).strip().lower()
            memories = [
                m for m in memories
                if str(m.get("source") or "").strip().lower() == wanted
            ]

        min_importance_value = self._clamp(min_importance)
        try:
            min_confidence_value = max(0.0, min(1.0, float(min_confidence)))
        except (TypeError, ValueError):
            min_confidence_value = 0.0

        memories = [
            m for m in memories
            if int(m.get("importance", 0)) >= min_importance_value
            and float(m.get("confidence", 1.0)) >= min_confidence_value
        ]

        return copy.deepcopy(
            sorted(
                memories,
                key=lambda m: (
                    int(m.get("importance", 0)),
                    str(m.get("timestamp", "")),
                ),
                reverse=True,
            )[:max(1, int(limit))]
        )

    def get_memory_by_id(
        self,
        npc_id: str,
        memory_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Return one personal episodic memory by stable ID."""
        for memory in self.get_state(npc_id)["personal_memory"]:
            if memory.get("id") == memory_id:
                return copy.deepcopy(memory)
        return None

    def apply_relationship_event(
        self,
        npc_id: str,
        event_type: str,
        related_entity: str = "player",
        source: str = "experienced",
        confidence: float = 1.0,
    ):
        """
        Apply deterministic relationship effects and store the same event
        as structured episodic memory.
        """
        effects = {
            "helped": (15, -10, -5, -10, "grateful", False),
            "bought_goods": (5, -2, 0, -2, None, None),
            "bought_drink": (6, -3, 0, -2, "friendly", None),
            "left_tip": (10, -5, 0, -4, "grateful", None),
            "insulted": (-10, 10, 0, 20, "angry", None),
            "threatened": (-25, 25, 25, 25, "afraid", True),
            "attacked": (-40, 35, 30, 40, "hostile", True),
            "robbed": (-50, 40, 35, 45, "hostile", True),
            "apologized": (8, -8, -5, -10, None, None),
        }

        if event_type not in effects:
            raise ValueError(f"Unknown relationship event: {event_type}")

        state = self.get_state(npc_id)
        trust, stress, fear, anger, emotion, hostile = effects[event_type]

        emotional_impact = {
            "trust": trust,
            "stress": stress,
            "fear": fear,
            "anger": anger,
        }

        updates = {
            "trust": state["trust"] + trust,
            "stress": state["stress"] + stress,
            "fear": state["fear"] + fear,
            "anger": state["anger"] + anger,
        }

        if hostile is not None:
            updates["hostile"] = hostile

        updates["emotion"] = emotion or self.infer_emotion({**state, **updates})

        self.update_state(
            npc_id,
            updates,
            "NPCStateManager",
            f"Relationship event: {event_type}",
        )

        importance = {
            "helped": 65,
            "bought_goods": 35,
            "bought_drink": 30,
            "left_tip": 50,
            "insulted": 60,
            "threatened": 80,
            "attacked": 95,
            "robbed": 95,
            "apologized": 55,
        }.get(event_type, 55)

        return self.add_memory(
            npc_id=npc_id,
            event=f"{related_entity} {event_type} this NPC.",
            importance=importance,
            emotional_tag=updates["emotion"],
            related_entity=related_entity,
            event_type=event_type,
            participants=[related_entity, NPCProfiles.normalize_npc_id(npc_id)],
            emotional_impact=emotional_impact,
            source=source,
            confidence=confidence,
            metadata={"relationship_event": True},
        )

    def build_simulation_context(self, npc_id: str) -> Dict[str, Any]:
        npc_id = NPCProfiles.normalize_npc_id(npc_id)
        return {
            "profile": NPCProfiles.get_profile(npc_id),
            "state": self.get_state(npc_id)
        }

    def validate_state(self, npc_id: str) -> bool:
        npc_id = NPCProfiles.normalize_npc_id(npc_id)
        if npc_id not in self.states:
            return False

        state = self.states[npc_id]
        for field in ["health", "stress", "fear", "anger", "trust"]:
            if not isinstance(state.get(field), int) or not 0 <= state[field] <= 100:
                return False

        return (
            state.get("emotion") in self.VALID_EMOTIONS
            and isinstance(state.get("alive"), bool)
            and isinstance(state.get("hostile"), bool)
            and isinstance(state.get("personal_memory"), list)
            and isinstance(state.get("status_effects"), list)
        )

    def _normalize_state(self, npc_id: str):
        state = self.states[npc_id]
        for field in ["health", "stress", "fear", "anger", "trust"]:
            state[field] = self._clamp(state.get(field, 0))
        if state["health"] == 0:
            state["alive"] = False
            state["hostile"] = False

    @staticmethod
    def _calculate_changes(
        before: Dict[str, Any],
        after: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        return {
            key: {"before": before.get(key), "after": after.get(key)}
            for key in set(before) | set(after)
            if before.get(key) != after.get(key)
        }

    def display_state(self, npc_id: str):
        normalized = NPCProfiles.normalize_npc_id(npc_id)
        print(f"\n--- NPC State: {normalized} ---")
        for key, value in self.get_state(normalized).items():
            print(f"{key}: {value}")
        print("-------------------------\n")

    def display_all_states(self):
        for npc_id in NPCProfiles.list_npc_ids():
            self.display_state(npc_id)

    def display_audit_log(self, npc_id: Optional[str] = None):
        normalized = NPCProfiles.normalize_npc_id(npc_id) if npc_id else None
        entries = [
            entry for entry in self.audit_log
            if normalized is None or entry.get("npc_id") == normalized
        ]

        print("\n--- NPC Audit Log ---")
        if not entries:
            print("No NPC state changes recorded.")
        else:
            for entry in entries:
                print(
                    f"v{entry['version']} | {entry['timestamp']} | "
                    f"{entry['npc_id']} | {entry['source']}"
                )
                print(f"Reason: {entry['reason']}")
                print(f"Changes: {entry['changes']}")
                print()
        print("---------------------\n")

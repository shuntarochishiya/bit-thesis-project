"""Declarative, context-aware DAG planner.

The JSON templates remain the source of truth. A plan is always a separate
copy and is validated by the existing DAGScheduler before it is returned.
"""

from __future__ import annotations

import copy
import json
from typing import Any, Dict, List

from engine.dag_scheduler import DAGScheduler


class TaskPlanner:
    def __init__(self, template_path: str = "task_templates.json"):
        self.template_path = template_path
        self.templates = self.load_templates()
        self.scheduler = DAGScheduler()

    def load_templates(self) -> Dict[str, Any]:
        try:
            with open(self.template_path, "r", encoding="utf-8") as file:
                templates = json.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Task template file '{self.template_path}' was not found."
            ) from None
        except json.JSONDecodeError:
            raise ValueError(
                f"Task template file '{self.template_path}' contains invalid JSON."
            ) from None

        if not isinstance(templates, dict):
            raise ValueError("Task templates must be a JSON object.")
        return templates

    @staticmethod
    def _npc_is_incapacitated(target: str | None, game_state: Dict[str, Any]) -> bool:
        """Only use an explicit, authoritative health value to select this branch.

        Missing, malformed, or uncertain state must retain the full DAG.
        """
        if target not in {"merchant", "bartender"}:
            return False
        key = f"{target}_health"
        health = game_state.get(key)
        return isinstance(health, (int, float)) and not isinstance(health, bool) and health <= 0

    @staticmethod
    def _shorten_blocked_dialogue(plan: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Retain the existing blocked-dialogue path and its finalization.

        DialogueStepAgent.apply_dialogue_result accepts action_blocked=True and
        the ExecutionEngine initializes its dialogue context with defaults.
        No new task type or response contract is introduced here.
        """
        retained_ids = {
            "check_target_status",
            "evaluate_consequences",
            "apply_dialogue_result",
            "update_memory",
            "generate_narrative",
        }
        if not retained_ids.issubset({task["task_id"] for task in plan}):
            return plan  # Custom templates retain their original semantics.

        shortened = [task for task in plan if task["task_id"] in retained_ids]
        for task in shortened:
            task["depends_on"] = [
                dependency for dependency in task.get("depends_on", [])
                if dependency in retained_ids
            ]

        for task in shortened:
            if task["task_id"] == "apply_dialogue_result":
                dependencies = task.setdefault("depends_on", [])
                if "evaluate_consequences" not in dependencies:
                    dependencies.append("evaluate_consequences")
                break
        return shortened

    def build_plan(
        self,
        intent: str,
        game_state: Dict[str, Any] | None = None,
        target: str | None = None,
        npc_context: Dict[str, Any] | None = None,
    ) -> List[Dict[str, Any]]:
        """Build an isolated DAG; legacy build_plan(intent) remains supported.

        npc_context is reserved for future verified rules. We deliberately do
        not infer an NPC's availability from an optional simulation profile.
        """
        template = self.templates.get(intent, self.templates.get("general_action", []))
        if not isinstance(template, list):
            raise ValueError(f"Task template for '{intent}' must be a list.")
        plan = copy.deepcopy(template)

        if (
            intent == "dialogue_action"
            and game_state is not None
            and self._npc_is_incapacitated(target, game_state)
        ):
            plan = self._shorten_blocked_dialogue(plan)

        # Add a personal-memory retrieval node only when the addressed NPC
        # actually has episodic memories concerning the player. The ordinary
        # dialogue template remains unchanged for new relationships.
        if intent == "dialogue_action" and isinstance(npc_context, dict):
            npc_state = npc_context.get("state", {})
            if isinstance(npc_state, dict) and npc_state.get("alive", True):
                memories = npc_state.get("personal_memory", [])
                has_player_history = isinstance(memories, list) and any(
                    isinstance(item, dict)
                    and item.get("related_entity") == "player"
                    for item in memories
                )
                if has_player_history:
                    plan = self._add_npc_memory_node(plan)

        self.scheduler.validate_plan(plan)
        return plan

    @staticmethod
    def _add_npc_memory_node(plan: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Compose an NPC-specific episodic-memory branch into a dialogue DAG.

        The branch feeds attitude analysis, not the final response directly.
        All original dependencies and task identifiers are preserved.
        """
        ids = {task.get("task_id") for task in plan}
        if "retrieve_npc_episodic_memory" in ids:
            return plan
        if not {"check_target_status", "analyze_npc_attitude"}.issubset(ids):
            return plan

        new_task = {
            "task_id": "retrieve_npc_episodic_memory",
            "depends_on": ["check_target_status"],
            "agent": "npc_memory",
            "fallback": "generic_fallback",
        }
        insertion_index = next(
            i for i, task in enumerate(plan)
            if task["task_id"] == "analyze_npc_attitude"
        )
        plan.insert(insertion_index, new_task)
        for task in plan:
            if task["task_id"] == "analyze_npc_attitude":
                dependencies = task.setdefault("depends_on", [])
                if new_task["task_id"] not in dependencies:
                    dependencies.append(new_task["task_id"])
                break
        return plan

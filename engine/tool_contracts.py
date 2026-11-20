"""Declarative tool-function contracts for planning a computation DAG.

This module validates the interface exposed by agent families *at planning
 time*. It deliberately does not claim to perform runtime input/output schema
validation: that will be a separate execution-engine integration step.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ToolContract:
    agent: str
    operation_key: str | None = None
    operations: frozenset[str] = frozenset()

    def validate(self, task: Mapping[str, Any]) -> None:
        if self.operation_key is not None:
            operation = task.get(self.operation_key)
            if not isinstance(operation, str) or operation not in self.operations:
                raise ValueError(
                    f"Tool contract for agent '{self.agent}' requires "
                    f"{self.operation_key} in {sorted(self.operations)}; got {operation!r} "
                    f"for task '{task.get('task_id')}'."
                )


# This registry reflects the operations used by the existing execution engine
# and task_templates.json. Unknown agents are rejected before execution.
DEFAULT_TOOL_CONTRACTS: dict[str, ToolContract] = {
    name: ToolContract(name) for name in (
        "validation", "semantic_memory", "consequence", "memory",
        "narrative", "tavern", "npc_memory",
    )
}
DEFAULT_TOOL_CONTRACTS.update({
    "precondition": ToolContract("precondition", "check", frozenset({
        "check_target_status", "validate_location", "check_player_resources",
    })),
    "combat_step": ToolContract("combat_step", "step", frozenset({
        "calculate_hit", "calculate_damage", "check_death",
        "apply_combat_result", "calculate_target_reaction", "apply_target_reaction",
    })),
    "persuasion_step": ToolContract("persuasion_step", "step", frozenset({
        "analyze_relationship", "analyze_reputation", "calculate_persuasion_chance",
        "apply_persuasion_result",
    })),
    "dialogue_step": ToolContract("dialogue_step", "step", frozenset({
        "analyze_npc_attitude", "analyze_dialogue_context", "choose_dialogue_strategy",
        "apply_dialogue_result",
    })),
    "event_step": ToolContract("event_step", "step", frozenset({
        "calculate_event_probabilities", "sample_event", "check_event_plausibility",
        "detect_conflict", "escalate_conflict", "apply_event_result",
        "generate_followup_event",
    })),
})


class ToolContractRegistry:
    def __init__(self, contracts: Mapping[str, ToolContract] | None = None):
        self.contracts = dict(DEFAULT_TOOL_CONTRACTS if contracts is None else contracts)

    def validate_plan(self, plan: Sequence[Mapping[str, Any]]) -> None:
        for task in plan:
            if not isinstance(task, Mapping):
                raise ValueError("Task plan entries must be mappings.")
            task_id = task.get("task_id")
            agent = task.get("agent")
            if not isinstance(task_id, str) or not task_id.strip():
                raise ValueError("Every task needs a non-empty task_id.")
            if not isinstance(agent, str) or agent not in self.contracts:
                raise ValueError(f"Unknown tool agent {agent!r} in task {task_id!r}.")
            deps = task.get("depends_on", [])
            if not isinstance(deps, list) or not all(isinstance(d, str) for d in deps):
                raise ValueError(f"Task {task_id!r} requires a list of dependency IDs.")
            fallback = task.get("fallback")
            if fallback is not None and not isinstance(fallback, str):
                raise ValueError(f"Task {task_id!r} has a non-string fallback.")
            parallel_safe = task.get("parallel_safe", False)
            if not isinstance(parallel_safe, bool):
                raise ValueError(f"Task {task_id!r} has a non-boolean parallel_safe flag.")
            self.contracts[agent].validate(task)

from typing import Dict, Any, List


class ExecutionLogger:
    """
    This class records the internal execution process of the system.
    It is useful for debugging and for demonstrating that the system uses
    a structured agent-based execution pipeline instead of a direct chatbot call.
    """

    def __init__(self):
        self.logs = []

    def start_turn(self, player_input: str, state_before: Dict[str, Any]):
        log_entry = {
            "player_input": player_input,
            "recognized_intent": None,
            "target": None,
            "execution_dag": [],
            "executed_tasks": [],
            "fallbacks_used": [],
            "dag_execution_levels": [],
            "errors": [],
            "semantic_memory_results": [],
            "consequence_decision": None,
            "state_before": state_before,
            "state_after": None,
            "state_changes": {},
            "memory_event": None,
            "final_response": None
        }

        self.logs.append(log_entry)

    def set_intent(self, intent: str, target: str):
        self.logs[-1]["recognized_intent"] = intent
        self.logs[-1]["target"] = target

    def set_dag(self, plan: List[Dict[str, Any]]):
        self.logs[-1]["execution_dag"] = [
            {
                "task_id": task["task_id"],
                "depends_on": task["depends_on"],
                "agent": task["agent"],
                "fallback": task.get("fallback")
            }
            for task in plan
        ]

    def set_dag_execution_levels(self, execution_levels: List[Dict[str, Any]]):
        self.logs[-1]["dag_execution_levels"] = execution_levels

    def set_consequence_decision(self, consequence_decision: Dict[str, Any]):
        self.logs[-1]["consequence_decision"] = consequence_decision

    def add_executed_task(self, task_id: str, agent_type: str):
        self.logs[-1]["executed_tasks"].append({
            "task_id": task_id,
            "agent": agent_type
        })

    def set_memory_event(self, memory_event: str):
        self.logs[-1]["memory_event"] = memory_event

    def set_semantic_memory_results(self, semantic_memory_results: List[str]):
        self.logs[-1]["semantic_memory_results"] = semantic_memory_results

    def finish_turn(self, state_after: Dict[str, Any], final_response: str):
        self.logs[-1]["state_after"] = state_after
        self.logs[-1]["final_response"] = final_response
        self.logs[-1]["state_changes"] = self.calculate_state_changes(
            self.logs[-1]["state_before"],
            state_after
        )

    def calculate_state_changes(
        self,
        state_before: Dict[str, Any],
        state_after: Dict[str, Any]
    ) -> Dict[str, Any]:

        changes = {}

        for key in state_after:
            before_value = state_before.get(key)
            after_value = state_after.get(key)

            if before_value != after_value:
                changes[key] = {
                    "before": before_value,
                    "after": after_value
                }

        return changes

    def get_last_log(self) -> Dict[str, Any]:
        if not self.logs:
            return {}

        return self.logs[-1]

    def display_last_log(self):
        if not self.logs:
            print("No execution logs available.")
            return

        log = self.logs[-1]

        print("\n--- Execution Log ---")
        print(f"Player input: {log['player_input']}")
        print(f"Recognized intent: {log['recognized_intent']}")
        print(f"Target: {log['target']}")
        print(f"Consequence decision: {log.get('consequence_decision')}")

        print("\nExecution DAG:")
        for task in log["execution_dag"]:
            print(
                f"- {task['task_id']} | agent: {task['agent']} | "
                f"depends on: {task['depends_on']} | fallback: {task['fallback']}"
            )

        print("\nDAG Execution Levels:")
        if log.get("dag_execution_levels"):
            for level in log["dag_execution_levels"]:
                print(f"Level {level['level']}:")
                for task in level["tasks"]:
                    print(
                        f"  - {task['task_id']} | agent: {task['agent']} | "
                        f"depends on: {task['depends_on']}"
                    )
        else:
            print("- No DAG execution levels recorded")

        print("\nExecuted tasks:")
        for task in log["executed_tasks"]:
            print(f"- {task['task_id']} by {task['agent']}")

        print("\nSemantic memory results:")
        if log.get("semantic_memory_results"):
            for index, memory in enumerate(log["semantic_memory_results"], start=1):
                print(f"{index}. {memory}")
        else:
            print("- No semantic memory results recorded")

        print("\nErrors:")
        if log["errors"]:
            for error in log["errors"]:
                print(f"- {error['task_id']}: {error['error']}")
        else:
            print("- No errors")

        print("\nFallbacks used:")
        if log["fallbacks_used"]:
            for fallback in log["fallbacks_used"]:
                print(f"- {fallback['task_id']}: {fallback['fallback']}")
        else:
            print("- No fallbacks used")

        print("\nState changes:")
        if log["state_changes"]:
            for key, value in log["state_changes"].items():
                print(f"- {key}: {value['before']} -> {value['after']}")
        else:
            print("- No state changes")

        print(f"\nMemory event: {log['memory_event']}")
        print(f"\nFinal response: {log['final_response']}")
        print("---------------------\n")

    def add_error(self, task_id: str, error_message: str):
        self.logs[-1]["errors"].append({
            "task_id": task_id,
            "error": error_message
        })

    def add_fallback(self, task_id: str, fallback_name: str):
        self.logs[-1]["fallbacks_used"].append({
            "task_id": task_id,
            "fallback": fallback_name
        })



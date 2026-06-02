from typing import Dict, Any, List, Set


class DAGScheduler:
    """
    Builds and validates execution levels for a task DAG.

    The scheduler receives a task plan from task_templates.json.
    Each task must have:
    - task_id
    - depends_on
    - agent

    It checks:
    - duplicate task IDs
    - missing dependencies
    - cyclic dependencies

    Then it groups tasks into execution levels.
    Tasks in the same level have no dependencies between each other
    and may later be executed in parallel.
    """

    def validate_plan(self, plan: List[Dict[str, Any]]) -> None:
        """
        Validates the task plan before scheduling.
        """

        task_ids = [task["task_id"] for task in plan]

        if len(task_ids) != len(set(task_ids)):
            raise ValueError("DAG validation failed: duplicate task_id found.")

        task_id_set = set(task_ids)

        for task in plan:
            task_id = task["task_id"]
            dependencies = task.get("depends_on", [])

            for dependency in dependencies:
                if dependency not in task_id_set:
                    raise ValueError(
                        f"DAG validation failed: task '{task_id}' depends on missing task '{dependency}'."
                    )

        if self.has_cycle(plan):
            raise ValueError("DAG validation failed: cycle detected in task dependencies.")

    def has_cycle(self, plan: List[Dict[str, Any]]) -> bool:
        """
        Detects whether the DAG has a cycle.
        """

        dependency_map = {
            task["task_id"]: task.get("depends_on", [])
            for task in plan
        }

        visiting: Set[str] = set()
        visited: Set[str] = set()

        def visit(task_id: str) -> bool:
            if task_id in visiting:
                return True

            if task_id in visited:
                return False

            visiting.add(task_id)

            for dependency in dependency_map.get(task_id, []):
                if visit(dependency):
                    return True

            visiting.remove(task_id)
            visited.add(task_id)

            return False

        for task_id in dependency_map:
            if visit(task_id):
                return True

        return False

    def build_execution_levels(self, plan: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        Groups tasks into dependency-based execution levels.
        """

        self.validate_plan(plan)

        remaining_tasks = {
            task["task_id"]: task
            for task in plan
        }

        completed_tasks: Set[str] = set()
        execution_levels: List[List[Dict[str, Any]]] = []

        while remaining_tasks:
            current_level = []

            for task_id, task in list(remaining_tasks.items()):
                dependencies = set(task.get("depends_on", []))

                if dependencies.issubset(completed_tasks):
                    current_level.append(task)

            if not current_level:
                raise ValueError(
                    "DAG scheduling failed: no executable tasks found. "
                    "This usually means there is a dependency problem."
                )

            execution_levels.append(current_level)

            for task in current_level:
                task_id = task["task_id"]
                completed_tasks.add(task_id)
                del remaining_tasks[task_id]

        return execution_levels

    def format_execution_levels(self, execution_levels: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Converts execution levels into a log-friendly format.
        """

        formatted_levels = []

        for index, level in enumerate(execution_levels, start=1):
            formatted_levels.append({
                "level": index,
                "tasks": [
                    {
                        "task_id": task["task_id"],
                        "agent": task["agent"],
                        "depends_on": task.get("depends_on", []),
                        "fallback": task.get("fallback")
                    }
                    for task in level
                ]
            })

        return formatted_levels

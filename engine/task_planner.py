import json
from typing import Dict, Any, List


class TaskPlanner:
    """
    This component creates a simplified execution DAG.

    In this version, task plans are not hardcoded.
    They are loaded from an external declarative JSON template file.
    """

    def __init__(self, template_path: str = "task_templates.json"):
        self.template_path = template_path
        self.templates = self.load_templates()

    def load_templates(self) -> Dict[str, Any]:
        try:
            with open(self.template_path, "r", encoding="utf-8") as file:
                return json.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Task template file '{self.template_path}' was not found."
            )
        except json.JSONDecodeError:
            raise ValueError(
                f"Task template file '{self.template_path}' contains invalid JSON."
            )

    def build_plan(self, intent: str) -> List[Dict[str, Any]]:
        if intent in self.templates:
            return self.templates[intent]

        return self.templates.get("general_action", [])


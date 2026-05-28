from typing import Dict, Any


class FallbackManager:
    """
    This class handles fallback strategies when a task fails.
    It makes the system more robust and prevents the whole game loop from crashing.
    """

    def handle_fallback(
        self,
        fallback_name: str,
        task_id: str,
        agent_type: str,
        game_state: Dict[str, Any],
        error_message: str
    ) -> Dict[str, Any]:

        if fallback_name == "cancel_action":
            return {
                "success": False,
                "message": "The action could not be completed, so the system safely cancelled it.",
                "state_updates": {}
            }

        if fallback_name == "safe_combat_response":
            return {
                "success": False,
                "message": "The combat action failed, but the system recovered safely. The enemy remains alert.",
                "state_updates": {}
            }

        if fallback_name == "safe_persuasion_response":
            return {
                "success": False,
                "message": "The persuasion attempt could not be resolved, so the merchant gives no clear answer.",
                "state_updates": {}
            }

        if fallback_name == "safe_exploration_response":
            return {
                "success": False,
                "message": "The exploration action failed, but the player notices nothing dangerous nearby.",
                "state_updates": {}
            }

        if fallback_name == "skip_memory_update":
            return {
                "success": True,
                "message": "Memory update was skipped due to an internal issue.",
                "state_updates": {}
            }

        if fallback_name == "basic_narrative":
            return {
                "success": True,
                "message": "The world reacts, but the details remain unclear for now.",
                "state_updates": {}
            }

        if fallback_name == "safe_dialogue_response":
            return {
                "success": False,
                "message": "The dialogue action failed, but the system recovered safely. The character gives no clear answer.",
                "state_updates": {}
            }

        if fallback_name == "safe_tavern_response":
            return {
                "success": False,
                "message": "The tavern action failed, but the system recovered safely. The bartender gives no clear response.",
                "state_updates": {}
            }

        return {
            "success": False,
            "message": f"Task '{task_id}' failed, but the system recovered using a generic fallback.",
            "state_updates": {}
        }

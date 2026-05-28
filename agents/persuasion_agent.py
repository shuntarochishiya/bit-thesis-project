from typing import Dict, Any
from agents.primitive_agents import AttributeCalculationAgent, ValidationAgent


class PersuasionAgent:
    """
    Composite agent.
    It processes persuasion attempts with an NPC.
    """

    def __init__(self, attribute_agent: AttributeCalculationAgent, validation_agent: ValidationAgent):
        self.attribute_agent = attribute_agent
        self.validation_agent = validation_agent

    def execute(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        if not self.validation_agent.validate_persuasion(game_state):
            return {
                "success": False,
                "message": "The merchant refuses to listen.",
                "state_updates": {}
            }

        relationship = game_state["relationship_with_merchant"]
        success = self.attribute_agent.calculate_persuasion_success(relationship)

        if success:
            return {
                "success": True,
                "message": "The player successfully persuades the merchant to offer a discount.",
                "state_updates": {
                    "gold": game_state["gold"] + 10,
                    "relationship_with_merchant": min(relationship + 5, 100)
                }
            }

        return {
            "success": True,
            "message": "The persuasion attempt fails. The merchant remains unconvinced.",
            "state_updates": {
                "relationship_with_merchant": max(relationship - 5, 0)
            }
        }


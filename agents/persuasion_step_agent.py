import random
from typing import Dict, Any, List


class PersuasionStepAgent:
    """
    Handles separate persuasion calculation steps.

    This makes persuasion_action suitable for DAG execution:
    relationship analysis, reputation analysis, chance calculation,
    and persuasion result application are separate nodes.
    """

    def analyze_relationship(
        self,
        game_state: Dict[str, Any],
        target: str,
        relevant_memory: List[str],
        action_blocked: bool = False
    ) -> Dict[str, Any]:

        if action_blocked:
            return {
                "success": False,
                "message": "Relationship analysis skipped because the action was blocked.",
                "state_updates": {},
                "data": {
                    "relationship_score": 0,
                    "relationship_modifier": -50
                }
            }

        relationship_score = 50

        if target == "merchant":
            relationship_score = game_state.get("relationship_with_merchant", 50)

        elif target == "bartender":
            relationship_score = game_state.get("relationship_with_bartender", 50)

        memory_text = " ".join(relevant_memory).lower()
        relationship_modifier = 0

        positive_memory_words = [
            "helped",
            "successful persuasion",
            "discount",
            "bought",
            "paid",
            "generous",
            "friendly",
            "relationship"
        ]

        negative_memory_words = [
            "attack",
            "attacked",
            "hostile",
            "threatened",
            "damage",
            "refuses",
            "violence"
        ]

        if any(word in memory_text for word in positive_memory_words):
            relationship_modifier += 10

        if any(word in memory_text for word in negative_memory_words):
            relationship_modifier -= 25

        adjusted_relationship = max(
            min(relationship_score + relationship_modifier, 100),
            0
        )

        return {
            "success": True,
            "message": (
                f"Relationship analysis completed. "
                f"Base relationship: {relationship_score}. "
                f"Modifier: {relationship_modifier}. "
                f"Adjusted relationship: {adjusted_relationship}."
            ),
            "state_updates": {},
            "data": {
                "relationship_score": adjusted_relationship,
                "relationship_modifier": relationship_modifier
            }
        }

    def analyze_reputation(
        self,
        game_state: Dict[str, Any],
        relevant_memory: List[str],
        action_blocked: bool = False
    ) -> Dict[str, Any]:

        if action_blocked:
            return {
                "success": False,
                "message": "Reputation analysis skipped because the action was blocked.",
                "state_updates": {},
                "data": {
                    "reputation_score": 0,
                    "reputation_modifier": -50
                }
            }

        player_reputation = game_state.get("player_reputation", 50)
        memory_text = " ".join(relevant_memory).lower()
        reputation_modifier = 0

        if "helped" in memory_text or "friendly" in memory_text or "generous" in memory_text:
            reputation_modifier += 5

        if "attack" in memory_text or "hostile" in memory_text or "violence" in memory_text:
            reputation_modifier -= 15

        adjusted_reputation = max(
            min(player_reputation + reputation_modifier, 100),
            0
        )

        return {
            "success": True,
            "message": (
                f"Reputation analysis completed. "
                f"Base reputation: {player_reputation}. "
                f"Modifier: {reputation_modifier}. "
                f"Adjusted reputation: {adjusted_reputation}."
            ),
            "state_updates": {},
            "data": {
                "reputation_score": adjusted_reputation,
                "reputation_modifier": reputation_modifier
            }
        }

    def calculate_persuasion_chance(
        self,
        relationship_score: int,
        reputation_score: int,
        action_blocked: bool = False
    ) -> Dict[str, Any]:

        if action_blocked:
            return {
                "success": False,
                "message": "Persuasion chance calculation skipped because the action was blocked.",
                "state_updates": {},
                "data": {
                    "persuasion_chance": 0,
                    "roll": None,
                    "persuasion_success": False
                }
            }

        base_chance = 25
        relationship_bonus = relationship_score // 3
        reputation_bonus = reputation_score // 5

        persuasion_chance = base_chance + relationship_bonus + reputation_bonus
        persuasion_chance = max(min(persuasion_chance, 90), 5)

        roll = random.randint(1, 100)
        persuasion_success = roll <= persuasion_chance

        return {
            "success": True,
            "message": (
                f"Persuasion chance calculated. "
                f"Chance: {persuasion_chance}%. "
                f"Roll: {roll}. "
                f"Success: {persuasion_success}."
            ),
            "state_updates": {},
            "data": {
                "persuasion_chance": persuasion_chance,
                "roll": roll,
                "persuasion_success": persuasion_success
            }
        }

    def apply_persuasion_result(
        self,
        game_state: Dict[str, Any],
        target: str,
        persuasion_success: bool,
        persuasion_chance: int,
        roll: int,
        player_input: str,
        action_blocked: bool = False,
        blocked_reason: str = ""
    ) -> Dict[str, Any]:

        if action_blocked:
            return {
                "success": False,
                "message": blocked_reason or "The persuasion action was blocked.",
                "state_updates": {}
            }

        text = player_input.lower()

        if target != "merchant":
            return {
                "success": False,
                "message": "Persuasion failed because the target is not valid for this persuasion action.",
                "state_updates": {}
            }

        if game_state.get("merchant_hostile"):
            return {
                "success": False,
                "message": "The merchant refuses to listen because they are hostile toward the player.",
                "state_updates": {
                    "relationship_with_merchant": max(
                        game_state.get("relationship_with_merchant", 0) - 3,
                        0
                    )
                }
            }

        if persuasion_success:
            updates = {
                "relationship_with_merchant": min(
                    game_state.get("relationship_with_merchant", 50) + 5,
                    100
                ),
                "world_mood": "hopeful"
            }

            if "discount" in text or "cheaper" in text or "price" in text:
                message = (
                    f"The persuasion succeeds. The merchant agrees to offer a better price. "
                    f"Chance was {persuasion_chance}%, roll was {roll}."
                )

            elif "free" in text or "give me" in text or "artifact" in text:
                updates["relationship_with_merchant"] = max(
                    game_state.get("relationship_with_merchant", 50) - 5,
                    0
                )
                message = (
                    f"The persuasion partially succeeds. The merchant does not give the item for free, "
                    f"but becomes willing to negotiate. Chance was {persuasion_chance}%, roll was {roll}."
                )

            else:
                message = (
                    f"The persuasion succeeds. The merchant becomes more open to the player. "
                    f"Chance was {persuasion_chance}%, roll was {roll}."
                )

            return {
                "success": True,
                "message": message,
                "state_updates": updates
            }

        updates = {
            "relationship_with_merchant": max(
                game_state.get("relationship_with_merchant", 50) - 5,
                0
            ),
            "world_mood": "tense"
        }

        return {
            "success": True,
            "message": (
                f"The persuasion attempt fails. "
                f"The merchant remains unconvinced. "
                f"Chance was {persuasion_chance}%, roll was {roll}."
            ),
            "state_updates": updates
        }

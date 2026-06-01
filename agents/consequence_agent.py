from typing import Dict, Any, List


class ConsequenceAgent:
    """
    This agent checks previous actions and current game state before an action is executed.

    It uses:
    - current player input
    - recognized intent
    - target
    - current game state
    - relevant semantic memory

    The goal is to make NPC behavior history-aware.
    """

    def evaluate(
        self,
        player_input: str,
        intent: str,
        target: str,
        game_state: Dict[str, Any],
        relevant_memory: List[str]
    ) -> Dict[str, Any]:
        memory_text = " ".join(relevant_memory).lower()
        input_text = player_input.lower()

        result: Dict[str, Any] = {
            "allow_action": True,
            "reason": "No blocking consequence was found.",
            "reaction_modifier": "neutral",
            "state_updates": {},
            "system_note": ""
        }

        # =========================
        # 1. Bartender consequences
        # =========================

        if target == "bartender":
            bartender_was_attacked = (
                "target: bartender" in memory_text
                and (
                    "event type: attack" in memory_text
                    or "attacks the bartender" in memory_text
                    or "attack the bartender" in memory_text
                    or "attacks the barmaid" in memory_text
                    or "attack the barmaid" in memory_text
                    or "damage" in memory_text
                )
            )

            asking_for_service = (
                intent == "tavern_action"
                and any(word in input_text for word in [
                    "drink", "ale", "beer", "wine", "mead",
                    "room", "rent", "food", "meal", "rest", "sleep",
                    "order", "buy", "purchase"
                ])
            )

            asking_for_information = (
                intent == "tavern_action"
                and any(word in input_text for word in [
                    "rumor", "rumour", "information", "news",
                    "odd", "strange", "weird", "nearby", "recently",
                    "details", "what happened"
                ])
            )

            if game_state.get("bartender_hostile") or bartender_was_attacked:
                if asking_for_service:
                    result["allow_action"] = False
                    result["reason"] = (
                        "The bartender refuses service because the player previously attacked "
                        "or seriously threatened the bartender."
                    )
                    result["reaction_modifier"] = "hostile"
                    result["state_updates"] = {
                        "bartender_hostile": True,
                        "bartender_mood": "angry",
                        "relationship_with_bartender": max(
                            game_state.get("relationship_with_bartender", 0) - 3,
                            0
                        ),
                        "tavern_reputation": max(
                            game_state.get("tavern_reputation", 0) - 2,
                            0
                        )
                    }
                    result["system_note"] = (
                        "The current tavern action is blocked by past violence against the bartender."
                    )
                    return result

                if asking_for_information:
                    result["allow_action"] = False
                    result["reason"] = (
                        "The bartender refuses to share information because the player is not trusted."
                    )
                    result["reaction_modifier"] = "distrustful"
                    result["state_updates"] = {
                        "bartender_mood": "angry",
                        "relationship_with_bartender": max(
                            game_state.get("relationship_with_bartender", 0) - 2,
                            0
                        )
                    }
                    result["system_note"] = (
                        "The bartender withholds rumors and useful information due to past hostility."
                    )
                    return result

            expensive_drink_history = (
                "event type: drink_order" in memory_text
                and (
                    "finest" in memory_text
                    or "royal" in memory_text
                    or "expensive" in memory_text
                    or "premium" in memory_text
                )
            )

            if asking_for_information and expensive_drink_history:
                result["allow_action"] = True
                result["reason"] = (
                    "The bartender remembers that the player spent generously before."
                )
                result["reaction_modifier"] = "friendly"
                result["state_updates"] = {
                    "bartender_mood": "friendly",
                    "relationship_with_bartender": min(
                        game_state.get("relationship_with_bartender", 50) + 3,
                        100
                    )
                }
                result["system_note"] = (
                    "Past generous drink orders make the bartender more willing to share information."
                )
                return result

        # =========================
        # 2. Merchant consequences
        # =========================

        if target == "merchant":
            merchant_was_attacked = (
                "target: merchant" in memory_text
                and (
                    "event type: attack" in memory_text
                    or "attacks the merchant" in memory_text
                    or "attack the merchant" in memory_text
                    or "attacks the vendor" in memory_text
                    or "attack the vendor" in memory_text
                    or "damage" in memory_text
                )
            )

            asking_for_trade_or_discount = (
                intent in ["persuasion_action", "dialogue_action"]
                and any(word in input_text for word in [
                    "discount", "free", "price", "artifact", "buy",
                    "sell", "trade", "cheaper", "give me"
                ])
            )

            if game_state.get("merchant_hostile") or merchant_was_attacked:
                if asking_for_trade_or_discount:
                    result["allow_action"] = False
                    result["reason"] = (
                        "The merchant refuses because the player previously attacked or threatened them."
                    )
                    result["reaction_modifier"] = "hostile"
                    result["state_updates"] = {
                        "merchant_hostile": True,
                        "relationship_with_merchant": max(
                            game_state.get("relationship_with_merchant", 0) - 3,
                            0
                        )
                    }
                    result["system_note"] = (
                        "The merchant-related action is blocked by previous violence."
                    )
                    return result

        # =========================
        # 3. Enemy consequences
        # =========================

        if target == "enemy":
            if intent == "dialogue_action":
                enemy_health = game_state.get("enemy_health", 60)

                if enemy_health <= 15:
                    result["allow_action"] = True
                    result["reason"] = (
                        "The enemy is badly wounded and may be willing to talk out of fear."
                    )
                    result["reaction_modifier"] = "fearful"
                    result["state_updates"] = {
                        "world_mood": "tense"
                    }
                    result["system_note"] = (
                        "Enemy dialogue should sound fearful or desperate."
                    )
                    return result

                if enemy_health >= 45:
                    result["allow_action"] = True
                    result["reason"] = (
                        "The enemy is still strong and remains hostile during conversation."
                    )
                    result["reaction_modifier"] = "hostile"
                    result["state_updates"] = {
                        "world_mood": "hostile"
                    }
                    result["system_note"] = (
                        "Enemy dialogue should sound threatening."
                    )
                    return result

        return result

from typing import Dict, Any


class IntentRecognitionAgent:
    """
    Recognizes the player's intent and target using rule-based logic.
    This is a lightweight alternative to calling an LLM for every input.
    """

    def recognize_intent(self, player_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        text = player_input.lower()

        if context is None:
            context = {}

        active_location = context.get("active_location")
        active_conversation = context.get("active_conversation")

        merchant_words = [
            "merchant", "trader", "shopkeeper", "seller", "vendor",
            "merchant's", "trader's", "shop"
        ]

        bartender_words = [
            "bartender", "barman", "barmaid", "innkeeper",
            "tavern keeper", "tavernkeeper",
            "bar keeper", "barkeep",
            "bar tender", "tender"
        ]

        tavern_words = [
            "tavern", "inn", "pub", "bar", "alehouse", "drinking hall"
        ]

        enemy_words = [
            "goblin", "enemy", "monster", "creature", "beast", "orc", "ghoul"
        ]

        attack_words = [
            "attack", "hit", "fight", "strike", "beat", "kill",
            "hurt", "stab", "shoot", "punch", "kick", "slash"
        ]

        dialogue_words = [
            "talk", "speak", "say", "ask", "question", "chat",
            "negotiate", "tell", "answer", "conversation", "discuss",
            "heard", "seen", "spotted", "noticed", "know"
        ]

        tavern_action_words = [
            "drink", "ale", "beer", "wine", "mead", "liquor",
            "glass", "cup", "bottle",
            "room", "rent", "food", "meal", "rest", "sleep",
            "order", "buy", "purchase",
            "cheap", "simple", "regular", "good", "fine", "finest",
            "expensive", "best", "premium", "royal",
            "rumor", "rumour", "information", "news",
            "odd", "strange", "weird", "nearby", "recently",
            "anything", "happened", "details", "more details",
            "explain", "tell me more", "what happened",
            "enter", "go", "walk", "inside", "greet", "wink"
        ]

        information_words = [
            "rumor", "rumour", "information", "news",
            "odd", "strange", "weird", "nearby", "recently",
            "anything", "happened", "details", "more details",
            "explain", "tell me more", "what happened",
            "heard", "seen", "spotted", "noticed"
        ]

        persuasion_words = [
            "persuade", "convince", "discount", "bargain",
            "request", "free", "cheaper", "price", "lower price",
            "better price", "give me", "trade", "sell", "offer",
            "artifact", "item", "goods", "deal", "negotiate",
            "lower the price", "reduce the price"
        ]

        exploration_words = [
            "look", "explore", "search", "inspect", "walk", "go",
            "move", "enter", "leave", "travel", "continue", "follow",
            "approach", "wander", "step", "run", "climb"
        ]

        exploration_phrases = [
            "walk deeper",
            "go deeper",
            "move deeper",
            "deeper into the forest",
            "into the forest",
            "through the forest",
            "enter the forest",
            "follow the path",
            "look around",
            "search the area",
            "explore the area",
            "continue forward",
            "move forward",
            "walk forward"
        ]

        location_words = [
            "forest", "valley", "cave", "road", "path", "village",
            "mountain", "river", "castle", "ruins", "woods", "area",
            "tavern", "inn", "pub", "bar"
        ]

        is_merchant_target = any(word in text for word in merchant_words)
        is_bartender_target = any(word in text for word in bartender_words)
        is_tavern_related = any(word in text for word in tavern_words)
        is_enemy_target = any(word in text for word in enemy_words)
        has_location = any(word in text for word in location_words)

        # 1. Combat intent
        if any(word in text for word in attack_words):
            if is_merchant_target:
                return {"intent": "combat_action", "target": "merchant"}

            if is_bartender_target:
                return {"intent": "combat_action", "target": "bartender"}

            if is_enemy_target:
                return {"intent": "combat_action", "target": "enemy"}

            return {"intent": "combat_action", "target": "enemy"}

        # 2. Tavern action by explicit tavern / bartender mention
        if (is_tavern_related or is_bartender_target) and any(word in text for word in tavern_action_words):
            return {"intent": "tavern_action", "target": "bartender"}

        # 3. Tavern action by current location context
        # Example: player is already in tavern and writes:
        # "I order a glass of a good ale"
        if active_location == "tavern" and any(word in text for word in tavern_action_words):
            return {"intent": "tavern_action", "target": "bartender"}

        # 4. Dialogue with bartender
        if is_bartender_target and (
            any(word in text for word in dialogue_words)
            or any(word in text for word in information_words)
        ):
            return {"intent": "dialogue_action", "target": "bartender"}

        # 5. Dialogue by context
        if active_location == "tavern" and active_conversation == "bartender":
            if any(word in text for word in dialogue_words):
                return {"intent": "dialogue_action", "target": "bartender"}

            if any(word in text for word in information_words):
                return {"intent": "tavern_action", "target": "bartender"}

        # 6. Dialogue with enemy
        if is_enemy_target and any(word in text for word in dialogue_words):
            return {"intent": "dialogue_action", "target": "enemy"}

        # 7. Persuasion / trade intent with merchant
        # This must be checked before regular dialogue,
        # because phrases like "ask the merchant for a discount"
        # contain dialogue words but are actually persuasion/trade actions.
        if is_merchant_target and any(word in text for word in persuasion_words):
            return {"intent": "persuasion_action", "target": "merchant"}

        # 8. Dialogue with merchant
        if is_merchant_target and any(word in text for word in dialogue_words):
            return {"intent": "dialogue_action", "target": "merchant"}

        # 9. Exploration / movement
        if any(phrase in text for phrase in exploration_phrases):
            return {"intent": "exploration_action", "target": "environment"}

        if any(word in text for word in exploration_words) and has_location:
            return {"intent": "exploration_action", "target": "environment"}

        # 10. Context-based continuation
        # If the player is currently talking to the bartender in the tavern,
        # vague follow-up requests should continue as tavern actions.
        if active_location == "tavern" and active_conversation == "bartender":
            if any(word in text for word in tavern_action_words):
                return {"intent": "tavern_action", "target": "bartender"}

            if any(word in text for word in information_words):
                return {"intent": "tavern_action", "target": "bartender"}

            if any(word in text for word in dialogue_words):
                return {"intent": "dialogue_action", "target": "bartender"}

        # 11. General fallback
        return {"intent": "general_action", "target": "unknown"}

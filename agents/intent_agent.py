from typing import Dict, Any


class IntentRecognitionAgent:
    """
    This agent recognizes the player's intention.
    For the MVP, we use simple rules instead of an expensive LLM call.
    """

    def recognize_intent(self, player_input: str) -> Dict[str, Any]:
        text = player_input.lower()

        merchant_words = [
            "merchant", "trader", "shopkeeper", "seller", "vendor",
            "merchant's", "trader's", "shop"
        ]

        bartender_words = [
            "bartender", "barman", "barmaid", "innkeeper", "tavern keeper",
            "tavernkeeper", "bar keeper", "bar tender", "tender"
        ]

        tavern_words = [
            "tavern", "inn", "pub", "bar", "alehouse", "drinking hall"
        ]

        enemy_words = [
            "goblin", "enemy", "monster", "creature", "beast", "orc"
        ]

        attack_words = [
            "attack", "hit", "fight", "strike", "beat", "kill",
            "hurt", "stab", "shoot", "punch", "kick", "slash"
        ]

        dialogue_words = [
            "talk", "speak", "say", "ask", "question", "chat",
            "negotiate", "tell", "answer", "conversation", "discuss"
        ]

        persuasion_words = [
            "persuade", "convince", "discount", "bargain",
            "request", "free", "cheaper", "price",
            "give me", "trade", "buy", "sell", "offer",
            "artifact", "item", "goods"
        ]

        tavern_action_words = [
            "drink", "ale", "beer", "wine",
            "rumor", "rumour", "information", "news",
            "job", "quest", "room", "rent",
            "food", "meal", "rest", "sleep",
            "odd", "strange", "spotted", "nearby", "recently",
            "heard", "seen", "happened", "anything"
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
            "walk forward",
            "go to the tavern",
            "enter the tavern",
            "walk into the tavern",
            "go inside the tavern",
            "enter the inn",
            "go to the inn"
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

        # 2. Tavern-specific action
        if is_tavern_related and any(word in text for word in tavern_action_words):
            return {"intent": "tavern_action", "target": "bartender"}

        if is_bartender_target and any(word in text for word in tavern_action_words):
            return {"intent": "tavern_action", "target": "bartender"}

        # 3. Dialogue
        if is_enemy_target and any(word in text for word in dialogue_words):
            return {"intent": "dialogue_action", "target": "enemy"}

        if is_merchant_target and any(word in text for word in dialogue_words):
            return {"intent": "dialogue_action", "target": "merchant"}

        if is_bartender_target and any(word in text for word in dialogue_words):
            return {"intent": "dialogue_action", "target": "bartender"}

        # 4. Persuasion / trade intent with merchant
        if is_merchant_target and any(word in text for word in persuasion_words):
            return {"intent": "persuasion_action", "target": "merchant"}

        # 5. Exploration / movement
        if any(phrase in text for phrase in exploration_phrases):
            return {"intent": "exploration_action", "target": "environment"}

        if any(word in text for word in exploration_words) and has_location:
            return {"intent": "exploration_action", "target": "environment"}

        return {"intent": "general_action", "target": "unknown"}

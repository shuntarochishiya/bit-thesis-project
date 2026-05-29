from typing import Dict, Any
import random


class TavernAgent:
    """
    Composite agent.
    Handles tavern-specific actions such as buying drinks,
    asking for rumors, renting a room, or getting food.

    This version includes drink pricing based on drink type and quality.
    """

    def __init__(self):
        self.drink_menu = {
            "water": {
                "base_price": 1,
                "health_effect": 0,
                "mood": "neutral"
            },
            "ale": {
                "base_price": 2,
                "health_effect": 0,
                "mood": "warm"
            },
            "beer": {
                "base_price": 3,
                "health_effect": 0,
                "mood": "warm"
            },
            "mead": {
                "base_price": 5,
                "health_effect": 1,
                "mood": "cheerful"
            },
            "wine": {
                "base_price": 6,
                "health_effect": 1,
                "mood": "refined"
            },
            "finest drink": {
                "base_price": 12,
                "health_effect": 2,
                "mood": "impressed"
            },
            "royal wine": {
                "base_price": 20,
                "health_effect": 3,
                "mood": "luxurious"
            }
        }

        self.quality_modifiers = {
            "cheap": 0.6,
            "simple": 0.8,
            "regular": 1.0,
            "normal": 1.0,
            "good": 1.2,
            "fine": 1.5,
            "finest": 2.0,
            "expensive": 2.0,
            "best": 2.2,
            "premium": 2.5,
            "royal": 3.0
        }

    def detect_bartender_identity(self, player_input: str) -> Dict[str, Any]:
        """
        Detects the specific bartender variation mentioned by the player.
        For example: barmaid, barman, bartender, innkeeper.
        """

        text = player_input.lower()

        if "barmaid" in text:
            return {
                "bartender_role": "barmaid",
                "bartender_gender": "female",
                "bartender_pronouns": "she/her"
            }

        if "barman" in text:
            return {
                "bartender_role": "barman",
                "bartender_gender": "male",
                "bartender_pronouns": "he/him"
            }

        if "innkeeper" in text:
            return {
                "bartender_role": "innkeeper",
                "bartender_gender": "unknown",
                "bartender_pronouns": "they/them"
            }

        if "bartender" in text or "barkeep" in text or "tender" in text:
            return {
                "bartender_role": "bartender",
                "bartender_gender": "unknown",
                "bartender_pronouns": "they/them"
            }

        return {}

    def apply_base_updates(self, updates: Dict[str, Any], player_input: str) -> Dict[str, Any]:
        """
        If the player mentions tavern/inn/pub/bar/bartender/barmaid,
        update location context and bartender identity.
        """

        text = player_input.lower()

        tavern_context_words = [
            "tavern", "inn", "pub", "bar", "alehouse",
            "bartender", "barman", "barmaid", "innkeeper",
            "tavern keeper", "tavernkeeper", "barkeep", "tender"
        ]

        if any(word in text for word in tavern_context_words):
            updates["location"] = "tavern"
            updates["world_mood"] = "warm"

        bartender_identity = self.detect_bartender_identity(player_input)
        updates.update(bartender_identity)

        return updates

    def detect_drink_type(self, player_input: str) -> str:
        """
        Detects what type of drink the player ordered.
        """

        text = player_input.lower()

        if "royal wine" in text:
            return "royal wine"

        if "finest drink" in text or "best drink" in text:
            return "finest drink"

        if "wine" in text:
            return "wine"

        if "mead" in text:
            return "mead"

        if "beer" in text:
            return "beer"

        if "ale" in text:
            return "ale"

        if "water" in text:
            return "water"

        if "drink" in text or "glass" in text or "cup" in text:
            return "ale"

        return "ale"

    def detect_quality_modifier(self, player_input: str) -> float:
        """
        Detects quality level and returns a price multiplier.
        """

        text = player_input.lower()

        for quality_word, modifier in self.quality_modifiers.items():
            if quality_word in text:
                return modifier

        return 1.0

    def calculate_drink_price(self, drink_type: str, player_input: str) -> int:
        """
        Calculates final drink price based on drink type and quality.
        """

        base_price = self.drink_menu[drink_type]["base_price"]
        quality_modifier = self.detect_quality_modifier(player_input)

        final_price = round(base_price * quality_modifier)

        return max(final_price, 1)

    def handle_drink_order(self, game_state: Dict[str, Any], player_input: str) -> Dict[str, Any]:
        """
        Handles drink purchases.
        The player's gold decreases based on the drink type and quality.
        """

        drink_type = self.detect_drink_type(player_input)
        drink_price = self.calculate_drink_price(drink_type, player_input)

        if game_state["gold"] < drink_price:
            return {
                "success": False,
                "message": (
                    f"The player tries to order {drink_type}, but does not have enough gold. "
                    f"The drink costs {drink_price} coins."
                ),
                "state_updates": self.apply_base_updates({
                    "bartender_mood": "unimpressed"
                }, player_input)
            }

        drink_data = self.drink_menu[drink_type]
        new_gold = game_state["gold"] - drink_price
        new_health = min(game_state["player_health"] + drink_data["health_effect"], 100)

        relationship_bonus = 1

        if drink_price >= 10:
            relationship_bonus = 3

        if drink_price >= 20:
            relationship_bonus = 5

        return {
            "success": True,
            "message": (
                f"The bartender serves {drink_type}. "
                f"It costs {drink_price} coins. "
                f"The player's gold decreases from {game_state['gold']} to {new_gold}."
            ),
            "state_updates": self.apply_base_updates({
                "gold": new_gold,
                "player_health": new_health,
                "relationship_with_bartender": min(
                    game_state["relationship_with_bartender"] + relationship_bonus,
                    100
                ),
                "bartender_mood": drink_data["mood"],
                "world_mood": "warm"
            }, player_input)
        }

    def execute(self, game_state: Dict[str, Any], player_input: str) -> Dict[str, Any]:
        text = player_input.lower()

        if game_state["bartender_hostile"]:
            return {
                "success": False,
                "message": "The bartender refuses to serve the player because of previous hostile behavior.",
                "state_updates": self.apply_base_updates({
                    "bartender_mood": "angry",
                    "relationship_with_bartender": max(
                        game_state["relationship_with_bartender"] - 2,
                        0
                    )
                }, player_input)
            }

        # Asking for rumors / information
        if (
            "rumor" in text
            or "rumour" in text
            or "information" in text
            or "news" in text
            or "odd" in text
            or "strange" in text
            or "weird" in text
            or "spotted" in text
            or "nearby" in text
            or "recently" in text
            or "heard" in text
            or "seen" in text
            or "anything" in text
            or "details" in text
            or "more details" in text
            or "what happened" in text
        ):
            rumors = [
                "The tavern keeper lowers their voice and says that strange lights were seen near the old ruins.",
                "The tavern keeper says a wounded traveler came in last night, claiming goblins were gathering near the forest road.",
                "The tavern keeper mentions that the local merchant has been hiding something valuable from travelers.",
                "The tavern keeper quietly warns that people have disappeared near the valley after sunset."
            ]

            relationship_bonus = 3 if game_state["player_reputation"] >= 50 else 0

            return {
                "success": True,
                "message": random.choice(rumors),
                "state_updates": self.apply_base_updates({
                    "relationship_with_bartender": min(
                        game_state["relationship_with_bartender"] + relationship_bonus,
                        100
                    ),
                    "bartender_mood": "talkative"
                }, player_input)
            }

        # Buying drinks
        if (
            "drink" in text
            or "ale" in text
            or "beer" in text
            or "wine" in text
            or "mead" in text
            or "glass" in text
            or "cup" in text
            or "bottle" in text
        ):
            return self.handle_drink_order(game_state, player_input)

        # Renting a room / resting
        if "room" in text or "sleep" in text or "rest" in text:
            room_price = 10

            if game_state["gold"] < room_price:
                return {
                    "success": False,
                    "message": f"The player does not have enough gold to rent a room. A room costs {room_price} coins.",
                    "state_updates": self.apply_base_updates({
                        "bartender_mood": "unimpressed"
                    }, player_input)
                }

            return {
                "success": True,
                "message": f"The player rents a small room for {room_price} coins and takes time to recover.",
                "state_updates": self.apply_base_updates({
                    "gold": game_state["gold"] - room_price,
                    "player_health": min(game_state["player_health"] + 20, 100),
                    "relationship_with_bartender": min(
                        game_state["relationship_with_bartender"] + 3,
                        100
                    ),
                    "world_mood": "restful"
                }, player_input)
            }

        # Buying food
        if "food" in text or "meal" in text:
            meal_price = 5

            if game_state["gold"] < meal_price:
                return {
                    "success": False,
                    "message": f"The player does not have enough gold to buy a meal. A meal costs {meal_price} coins.",
                    "state_updates": self.apply_base_updates({
                        "bartender_mood": "unimpressed"
                    }, player_input)
                }

            return {
                "success": True,
                "message": f"The bartender brings a hot meal for {meal_price} coins. The player feels slightly better.",
                "state_updates": self.apply_base_updates({
                    "gold": game_state["gold"] - meal_price,
                    "player_health": min(game_state["player_health"] + 5, 100),
                    "relationship_with_bartender": min(
                        game_state["relationship_with_bartender"] + 2,
                        100
                    ),
                    "bartender_mood": "calm"
                }, player_input)
            }

        return {
            "success": True,
            "message": "The tavern is noisy, warm, and full of tired travelers. The bartender waits behind the counter.",
            "state_updates": self.apply_base_updates({}, player_input)
        }

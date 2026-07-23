from typing import Dict, Any


class TavernAgent:
    """
    Handles deterministic tavern services only.

    Responsibilities:
    - entering the tavern;
    - buying drinks;
    - buying food;
    - renting a room;
    - leaving the tavern;
    - updating player resources and tavern-related state.

    Dialogue, rumors, persuasion, and NPC conversations must be routed
    to DialogueAgent or PersuasionAgent instead.
    """

    def __init__(self) -> None:
        self.drink_menu: Dict[str, Dict[str, Any]] = {
            "water": {"base_price": 1, "health_effect": 0, "mood": "neutral"},
            "ale": {"base_price": 2, "health_effect": 0, "mood": "warm"},
            "beer": {"base_price": 3, "health_effect": 0, "mood": "warm"},
            "mead": {"base_price": 5, "health_effect": 1, "mood": "cheerful"},
            "wine": {"base_price": 6, "health_effect": 1, "mood": "refined"},
            "finest drink": {"base_price": 12, "health_effect": 2, "mood": "impressed"},
            "royal wine": {"base_price": 20, "health_effect": 3, "mood": "luxurious"},
        }

        self.quality_modifiers: Dict[str, float] = {
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
            "royal": 3.0,
        }

        self.room_price = 10
        self.meal_price = 5

    def detect_bartender_identity(self, player_input: str) -> Dict[str, Any]:
        text = player_input.lower()

        if "barmaid" in text:
            return {
                "bartender_role": "barmaid",
                "bartender_gender": "female",
                "bartender_pronouns": "she/her",
            }

        if "barman" in text:
            return {
                "bartender_role": "barman",
                "bartender_gender": "male",
                "bartender_pronouns": "he/him",
            }

        if "innkeeper" in text:
            return {
                "bartender_role": "innkeeper",
                "bartender_gender": "unknown",
                "bartender_pronouns": "they/them",
            }

        if any(word in text for word in ["bartender", "barkeep", "tender"]):
            return {
                "bartender_role": "bartender",
                "bartender_gender": "unknown",
                "bartender_pronouns": "they/them",
            }

        return {}

    def apply_base_updates(
        self,
        updates: Dict[str, Any],
        player_input: str,
    ) -> Dict[str, Any]:
        text = player_input.lower()

        tavern_context_words = [
            "tavern", "inn", "pub", "bar", "alehouse",
            "bartender", "barman", "barmaid", "innkeeper",
            "tavern keeper", "tavernkeeper", "barkeep", "tender",
        ]

        if any(word in text for word in tavern_context_words):
            updates.setdefault("location", "tavern")
            updates.setdefault("world_mood", "warm")

        updates.update(self.detect_bartender_identity(player_input))
        return updates

    def _result(
        self,
        *,
        success: bool,
        message: str,
        state_updates: Dict[str, Any],
        action_type: str,
        data: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        return {
            "success": success,
            "message": message,
            "state_updates": state_updates,
            "data": {
                "action_type": action_type,
                **(data or {}),
            },
        }

    def detect_service_action(self, player_input: str) -> str:
        text = player_input.lower()

        if any(word in text for word in ["leave", "exit", "outside"]):
            return "leave_tavern"

        if any(word in text for word in ["room", "rent", "sleep", "rest"]):
            return "rent_room"

        if any(word in text for word in ["food", "meal", "eat", "dinner", "lunch"]):
            return "buy_food"

        if any(
            word in text
            for word in [
                "drink", "ale", "beer", "wine", "mead", "water",
                "glass", "cup", "bottle",
            ]
        ):
            return "buy_drink"

        if any(
            phrase in text
            for phrase in [
                "enter tavern", "enter the tavern", "go inside",
                "walk inside", "enter inn", "enter the inn",
                "go to the tavern", "walk into the tavern",
            ]
        ):
            return "enter_tavern"

        if any(word in text for word in ["tavern", "inn", "pub", "alehouse"]):
            return "enter_tavern"

        return "observe_tavern"

    def detect_drink_type(self, player_input: str) -> str:
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

        return "ale"

    def detect_quality_modifier(self, player_input: str) -> float:
        text = player_input.lower()
        for quality_word, modifier in self.quality_modifiers.items():
            if quality_word in text:
                return modifier
        return 1.0

    def detect_quality_name(self, player_input: str) -> str:
        text = player_input.lower()
        for quality_word in self.quality_modifiers:
            if quality_word in text:
                return quality_word
        return "regular"

    def calculate_drink_price(self, drink_type: str, player_input: str) -> int:
        base_price = self.drink_menu[drink_type]["base_price"]
        quality_modifier = self.detect_quality_modifier(player_input)
        return max(round(base_price * quality_modifier), 1)

    def handle_drink_order(
        self,
        game_state: Dict[str, Any],
        player_input: str,
    ) -> Dict[str, Any]:
        drink_type = self.detect_drink_type(player_input)
        quality = self.detect_quality_name(player_input)
        price = self.calculate_drink_price(drink_type, player_input)

        current_gold = game_state.get("gold", 0)
        current_health = game_state.get("player_health", 100)

        if current_gold < price:
            return self._result(
                success=False,
                message=(
                    f"You do not have enough gold to buy {drink_type}. "
                    f"It costs {price} coins."
                ),
                state_updates=self.apply_base_updates(
                    {"bartender_mood": "unimpressed"},
                    player_input,
                ),
                action_type="buy_drink",
                data={
                    "drink_type": drink_type,
                    "quality": quality,
                    "price": price,
                    "gold_before": current_gold,
                    "gold_after": current_gold,
                    "purchased": False,
                },
            )

        drink_data = self.drink_menu[drink_type]
        new_gold = current_gold - price
        new_health = min(current_health + int(drink_data["health_effect"]), 100)

        relationship_bonus = 1
        if price >= 20:
            relationship_bonus = 5
        elif price >= 10:
            relationship_bonus = 3

        current_relationship = game_state.get("relationship_with_bartender", 50)

        return self._result(
            success=True,
            message=(
                f"You order {drink_type} for {price} coins. "
                f"You now have {new_gold} coins."
            ),
            state_updates=self.apply_base_updates(
                {
                    "gold": new_gold,
                    "player_health": new_health,
                    "relationship_with_bartender": min(
                        current_relationship + relationship_bonus,
                        100,
                    ),
                    "bartender_mood": drink_data["mood"],
                    "world_mood": "warm",
                },
                player_input,
            ),
            action_type="buy_drink",
            data={
                "drink_type": drink_type,
                "quality": quality,
                "price": price,
                "gold_before": current_gold,
                "gold_after": new_gold,
                "health_before": current_health,
                "health_after": new_health,
                "relationship_bonus": relationship_bonus,
                "purchased": True,
            },
        )

    def handle_food_order(
        self,
        game_state: Dict[str, Any],
        player_input: str,
    ) -> Dict[str, Any]:
        current_gold = game_state.get("gold", 0)
        current_health = game_state.get("player_health", 100)

        if current_gold < self.meal_price:
            return self._result(
                success=False,
                message=(
                    "You do not have enough gold to buy a meal. "
                    f"A meal costs {self.meal_price} coins."
                ),
                state_updates=self.apply_base_updates(
                    {"bartender_mood": "unimpressed"},
                    player_input,
                ),
                action_type="buy_food",
                data={
                    "price": self.meal_price,
                    "gold_before": current_gold,
                    "gold_after": current_gold,
                    "purchased": False,
                },
            )

        new_gold = current_gold - self.meal_price
        new_health = min(current_health + 5, 100)
        current_relationship = game_state.get("relationship_with_bartender", 50)

        return self._result(
            success=True,
            message=(
                f"You buy a hot meal for {self.meal_price} coins. "
                f"You now have {new_gold} coins and recover some health."
            ),
            state_updates=self.apply_base_updates(
                {
                    "gold": new_gold,
                    "player_health": new_health,
                    "relationship_with_bartender": min(
                        current_relationship + 2,
                        100,
                    ),
                    "bartender_mood": "calm",
                    "world_mood": "warm",
                },
                player_input,
            ),
            action_type="buy_food",
            data={
                "price": self.meal_price,
                "gold_before": current_gold,
                "gold_after": new_gold,
                "health_before": current_health,
                "health_after": new_health,
                "relationship_bonus": 2,
                "purchased": True,
            },
        )

    def handle_room_rental(
        self,
        game_state: Dict[str, Any],
        player_input: str,
    ) -> Dict[str, Any]:
        current_gold = game_state.get("gold", 0)
        current_health = game_state.get("player_health", 100)

        if current_gold < self.room_price:
            return self._result(
                success=False,
                message=(
                    "You do not have enough gold to rent a room. "
                    f"A room costs {self.room_price} coins."
                ),
                state_updates=self.apply_base_updates(
                    {"bartender_mood": "unimpressed"},
                    player_input,
                ),
                action_type="rent_room",
                data={
                    "price": self.room_price,
                    "gold_before": current_gold,
                    "gold_after": current_gold,
                    "rented": False,
                },
            )

        new_gold = current_gold - self.room_price
        new_health = min(current_health + 20, 100)
        current_relationship = game_state.get("relationship_with_bartender", 50)

        return self._result(
            success=True,
            message=(
                f"You rent a room for {self.room_price} coins and rest. "
                f"You now have {new_gold} coins."
            ),
            state_updates=self.apply_base_updates(
                {
                    "gold": new_gold,
                    "player_health": new_health,
                    "relationship_with_bartender": min(
                        current_relationship + 3,
                        100,
                    ),
                    "bartender_mood": "calm",
                    "world_mood": "restful",
                },
                player_input,
            ),
            action_type="rent_room",
            data={
                "price": self.room_price,
                "gold_before": current_gold,
                "gold_after": new_gold,
                "health_before": current_health,
                "health_after": new_health,
                "relationship_bonus": 3,
                "rented": True,
            },
        )

    def handle_enter_tavern(self, player_input: str) -> Dict[str, Any]:
        return self._result(
            success=True,
            message=(
                "You enter the tavern. "
                "The room is warm, noisy, and filled with tired travelers."
            ),
            state_updates=self.apply_base_updates(
                {"location": "tavern", "world_mood": "warm"},
                player_input,
            ),
            action_type="enter_tavern",
            data={"location": "tavern"},
        )

    def handle_leave_tavern(self) -> Dict[str, Any]:
        return self._result(
            success=True,
            message="You leave the tavern and step outside.",
            state_updates={"location": "village", "world_mood": "neutral"},
            action_type="leave_tavern",
            data={"location": "village"},
        )

    def handle_observe_tavern(self, player_input: str) -> Dict[str, Any]:
        return self._result(
            success=True,
            message=(
                "The tavern is warm and noisy. "
                "Travelers sit around wooden tables while the bartender waits behind the counter."
            ),
            state_updates=self.apply_base_updates({}, player_input),
            action_type="observe_tavern",
            data={"location": "tavern"},
        )

    def execute(
        self,
        game_state: Dict[str, Any],
        player_input: str,
    ) -> Dict[str, Any]:
        action_type = self.detect_service_action(player_input)

        if action_type == "enter_tavern":
            return self.handle_enter_tavern(player_input)

        if action_type == "leave_tavern":
            return self.handle_leave_tavern()

        bartender_hostile = game_state.get("bartender_hostile", False)

        if bartender_hostile and action_type in {
            "buy_drink",
            "buy_food",
            "rent_room",
        }:
            current_relationship = game_state.get("relationship_with_bartender", 50)

            return self._result(
                success=False,
                message=(
                    "The bartender refuses to serve you because of your previous hostile behavior."
                ),
                state_updates=self.apply_base_updates(
                    {
                        "bartender_mood": "angry",
                        "relationship_with_bartender": max(
                            current_relationship - 2,
                            0,
                        ),
                    },
                    player_input,
                ),
                action_type=action_type,
                data={
                    "service_refused": True,
                    "reason": "bartender_hostile",
                },
            )

        if action_type == "buy_drink":
            return self.handle_drink_order(game_state, player_input)

        if action_type == "buy_food":
            return self.handle_food_order(game_state, player_input)

        if action_type == "rent_room":
            return self.handle_room_rental(game_state, player_input)

        return self.handle_observe_tavern(player_input)

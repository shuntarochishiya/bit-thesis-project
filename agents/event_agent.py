import random
from typing import Dict, Any, List


class EventAgent:
    """
    Generates dynamic world events for exploration actions.

    This agent supports:
    - location-based event probabilities
    - semantic memory influence
    - event sampling
    - plausibility checking
    - event result application
    """

    def detect_requested_location(self, player_input: str, current_location: str) -> str:
        text = player_input.lower()

        if "forest" in text or "woods" in text:
            return "old forest"

        if "road" in text or "path" in text:
            return "forest road"

        if "ruins" in text or "ruin" in text:
            return "old ruins"

        if "village" in text or "town" in text:
            return "village"

        if "valley" in text:
            return "valley"

        if "river" in text:
            return "riverbank"

        return current_location

    def calculate_event_probabilities(
        self,
        game_state: Dict[str, Any],
        player_input: str,
        relevant_memory: List[str]
    ) -> Dict[str, Any]:
        current_location = game_state.get("location", "old forest")
        requested_location = self.detect_requested_location(player_input, current_location)

        memory_text = " ".join(relevant_memory).lower()
        world_mood = game_state.get("world_mood", "mysterious")
        player_reputation = game_state.get("player_reputation", 50)

        probabilities = {
            "find_footprints": 10,
            "find_coin": 10,
            "hidden_path": 10,
            "strange_noise": 10,
            "npc_encounter": 10,
            "enemy_ambush": 10,
            "merchant_cart": 5,
            "old_ruin_discovery": 5,
            "nothing_special": 10
        }

        if requested_location == "old forest":
            probabilities["find_footprints"] += 20
            probabilities["strange_noise"] += 15
            probabilities["hidden_path"] += 15
            probabilities["enemy_ambush"] += 10

        elif requested_location == "forest road":
            probabilities["merchant_cart"] += 25
            probabilities["npc_encounter"] += 15
            probabilities["enemy_ambush"] += 10

        elif requested_location == "old ruins":
            probabilities["old_ruin_discovery"] += 35
            probabilities["strange_noise"] += 15
            probabilities["enemy_ambush"] += 10

        elif requested_location == "village":
            probabilities["npc_encounter"] += 25
            probabilities["merchant_cart"] += 15
            probabilities["enemy_ambush"] -= 5

        elif requested_location == "valley":
            probabilities["hidden_path"] += 15
            probabilities["strange_noise"] += 15
            probabilities["old_ruin_discovery"] += 10

        if "goblin" in memory_text or "enemy" in memory_text or "attack" in memory_text:
            probabilities["enemy_ambush"] += 15

        if "strange lights" in memory_text or "old ruins" in memory_text:
            probabilities["old_ruin_discovery"] += 20
            probabilities["strange_noise"] += 10

        if "merchant" in memory_text or "trade" in memory_text:
            probabilities["merchant_cart"] += 10

        if world_mood in ["dangerous", "hostile", "tense"]:
            probabilities["enemy_ambush"] += 15
            probabilities["strange_noise"] += 10

        if player_reputation >= 70:
            probabilities["npc_encounter"] += 10
            probabilities["enemy_ambush"] -= 5

        if player_reputation <= 25:
            probabilities["enemy_ambush"] += 10
            probabilities["npc_encounter"] -= 5

        probabilities = {
            event: max(weight, 0)
            for event, weight in probabilities.items()
        }

        return {
            "success": True,
            "message": (
                f"Event probabilities calculated for location: {requested_location}."
            ),
            "state_updates": {},
            "data": {
                "event_probabilities": probabilities,
                "requested_location": requested_location
            }
        }

    def sample_event(
        self,
        event_probabilities: Dict[str, int]
    ) -> Dict[str, Any]:
        events = list(event_probabilities.keys())
        weights = list(event_probabilities.values())

        if not events or sum(weights) <= 0:
            selected_event = "nothing_special"
        else:
            selected_event = random.choices(events, weights=weights, k=1)[0]

        return {
            "success": True,
            "message": f"Event sampled: {selected_event}.",
            "state_updates": {},
            "data": {
                "selected_event": selected_event
            }
        }

    def check_event_plausibility(
        self,
        selected_event: str,
        requested_location: str,
        game_state: Dict[str, Any],
        relevant_memory: List[str]
    ) -> Dict[str, Any]:
        plausible = True
        reason = "Event is plausible."

        if selected_event == "merchant_cart" and requested_location in ["old ruins", "riverbank"]:
            plausible = False
            reason = "A merchant cart is unlikely in this location."

        if selected_event == "old_ruin_discovery" and requested_location == "village":
            plausible = False
            reason = "Old ruin discovery is unlikely inside the village."

        if selected_event == "enemy_ambush" and game_state.get("enemy_health", 60) <= 0:
            memory_text = " ".join(relevant_memory).lower()

            if "goblin gathering" not in memory_text and "enemy" not in memory_text:
                plausible = False
                reason = "Enemy ambush is less plausible because no active enemy threat is known."

        if selected_event == "npc_encounter" and game_state.get("world_mood") == "dangerous":
            reason = "NPC encounter is plausible but should feel tense because the world mood is dangerous."

        return {
            "success": plausible,
            "message": reason,
            "state_updates": {},
            "data": {
                "event_plausible": plausible,
                "plausibility_reason": reason
            }
        }

    def apply_event_result(
        self,
        selected_event: str,
        event_plausible: bool,
        requested_location: str,
        plausibility_reason: str,
        game_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        if not event_plausible:
            selected_event = "nothing_special"

        updates: Dict[str, Any] = {
            "location": requested_location
        }

        if selected_event == "find_footprints":
            updates["world_mood"] = "mysterious"
            message = "The player finds fresh footprints leading deeper into the area."

        elif selected_event == "find_coin":
            updates["gold"] = game_state.get("gold", 0) + 1
            updates["world_mood"] = "curious"
            message = "The player notices a small coin half-buried in the dirt."

        elif selected_event == "hidden_path":
            updates["world_mood"] = "mysterious"
            message = "The player discovers a narrow hidden path behind thick branches."

        elif selected_event == "strange_noise":
            updates["world_mood"] = "tense"
            message = "The player hears a strange sound somewhere nearby."

        elif selected_event == "npc_encounter":
            updates["world_mood"] = "watchful"
            message = "The player encounters a cautious traveler nearby."

        elif selected_event == "enemy_ambush":
            updates["world_mood"] = "hostile"
            message = "An enemy presence is felt nearby. The player may be in danger."

        elif selected_event == "merchant_cart":
            updates["world_mood"] = "active"
            message = "The player sees a merchant cart moving slowly along the road."

        elif selected_event == "old_ruin_discovery":
            updates["world_mood"] = "ancient"
            message = "The player finds signs of old stone ruins covered by moss and dust."

        else:
            updates["world_mood"] = "quiet"
            message = (
                "The player explores the area, but nothing important happens. "
                f"Plausibility note: {plausibility_reason}"
            )

        return {
            "success": True,
            "message": message,
            "state_updates": updates,
            "data": {
                "final_event": selected_event
            }
        }

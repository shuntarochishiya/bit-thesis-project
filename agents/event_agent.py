import random
from typing import Dict, Any, List, Optional


class EventAgent:
    """
    Generates dynamic world events and manages conflicts.

    The agent supports:
    - probability calculation from world state and semantic memory
    - weighted event sampling
    - event plausibility checks
    - conflict detection and escalation
    - state evolution
    - follow-up event generation
    """

    EVENT_PARTICIPANTS = {
        "enemy_ambush": ["player", "enemy"],
        "bandit_roadblock": ["player", "bandits"],
        "guard_patrol": ["player", "guards"],
        "merchant_cart": ["player", "merchant"],
        "npc_encounter": ["player", "traveler"],
        "traveler_in_need": ["player", "traveler"],
        "npc_returns_favor": ["player", "friendly_npc"],
        "strange_noise": ["player", "unknown_entity"],
    }

    def _clamp(self, value: int, minimum: int = 0, maximum: int = 100) -> int:
        return max(minimum, min(maximum, int(value)))

    def _memory_text(self, relevant_memory: List[str]) -> str:
        return " ".join(str(item) for item in relevant_memory).lower()

    def detect_requested_location(self, player_input: str, current_location: str) -> str:
        text = player_input.lower()

        location_keywords = {
            "old forest": ["forest", "woods"],
            "forest road": ["road", "path"],
            "old ruins": ["ruins", "ruin"],
            "village": ["village", "town"],
            "valley": ["valley"],
            "riverbank": ["river", "riverbank"],
        }

        for location, keywords in location_keywords.items():
            if any(keyword in text for keyword in keywords):
                return location

        return current_location

    def calculate_event_probabilities(
        self,
        game_state: Dict[str, Any],
        player_input: str,
        relevant_memory: List[str]
    ) -> Dict[str, Any]:
        current_location = game_state.get("location", "old forest")
        requested_location = self.detect_requested_location(
            player_input=player_input,
            current_location=current_location
        )

        memory_text = self._memory_text(relevant_memory)
        world_mood = str(game_state.get("world_mood", "mysterious")).lower()
        player_reputation = int(game_state.get("player_reputation", 50))
        world_danger = int(game_state.get("world_danger", 20))
        guard_alert = int(game_state.get("guard_alert_level", 10))
        trade_activity = int(game_state.get("trade_activity", 60))
        crime_level = int(game_state.get("crime_level", 0))

        probabilities = {
            "find_footprints": 10,
            "find_coin": 8,
            "hidden_path": 10,
            "strange_noise": 10,
            "npc_encounter": 10,
            "traveler_in_need": 6,
            "npc_returns_favor": 2,
            "enemy_ambush": 8,
            "bandit_roadblock": 4,
            "guard_patrol": 4,
            "merchant_cart": 6,
            "old_ruin_discovery": 5,
            "nothing_special": 10,
        }

        # Location influence
        if requested_location == "old forest":
            probabilities["find_footprints"] += 20
            probabilities["strange_noise"] += 15
            probabilities["hidden_path"] += 15
            probabilities["enemy_ambush"] += 10

        elif requested_location == "forest road":
            probabilities["merchant_cart"] += 20
            probabilities["npc_encounter"] += 12
            probabilities["traveler_in_need"] += 10
            probabilities["bandit_roadblock"] += 12
            probabilities["guard_patrol"] += 8

        elif requested_location == "old ruins":
            probabilities["old_ruin_discovery"] += 35
            probabilities["strange_noise"] += 15
            probabilities["enemy_ambush"] += 10
            probabilities["merchant_cart"] = 0

        elif requested_location == "village":
            probabilities["npc_encounter"] += 22
            probabilities["merchant_cart"] += 15
            probabilities["guard_patrol"] += 12
            probabilities["enemy_ambush"] = max(
                0, probabilities["enemy_ambush"] - 5
            )

        elif requested_location == "valley":
            probabilities["hidden_path"] += 15
            probabilities["strange_noise"] += 15
            probabilities["old_ruin_discovery"] += 10
            probabilities["traveler_in_need"] += 5

        elif requested_location == "riverbank":
            probabilities["find_coin"] += 8
            probabilities["hidden_path"] += 8
            probabilities["traveler_in_need"] += 8
            probabilities["merchant_cart"] = 0

        # Memory influence
        if any(word in memory_text for word in ["goblin", "enemy", "attack", "ambush"]):
            probabilities["enemy_ambush"] += 15

        if any(phrase in memory_text for phrase in ["strange lights", "old ruins", "ancient stone"]):
            probabilities["old_ruin_discovery"] += 20
            probabilities["strange_noise"] += 10

        if any(word in memory_text for word in ["merchant", "trade", "caravan"]):
            probabilities["merchant_cart"] += 10

        if any(
            phrase in memory_text
            for phrase in ["helped the traveler", "helped npc", "saved the traveler", "owed the player"]
        ):
            probabilities["npc_returns_favor"] += 30

        if any(
            phrase in memory_text
            for phrase in ["stole", "robbed", "illegal", "crime", "attacked the merchant"]
        ):
            probabilities["guard_patrol"] += 25
            probabilities["bandit_roadblock"] += 5

        # World-state influence
        probabilities["enemy_ambush"] += world_danger // 6
        probabilities["bandit_roadblock"] += world_danger // 8
        probabilities["guard_patrol"] += guard_alert // 5 + crime_level // 4
        probabilities["merchant_cart"] += trade_activity // 8

        if world_mood in {"dangerous", "hostile", "tense"}:
            probabilities["enemy_ambush"] += 12
            probabilities["bandit_roadblock"] += 8
            probabilities["strange_noise"] += 10

        if player_reputation >= 70:
            probabilities["npc_encounter"] += 10
            probabilities["npc_returns_favor"] += 8
            probabilities["enemy_ambush"] -= 5

        if player_reputation <= 25:
            probabilities["guard_patrol"] += 10
            probabilities["enemy_ambush"] += 8
            probabilities["npc_encounter"] -= 5

        probabilities = {
            event: max(int(weight), 0)
            for event, weight in probabilities.items()
        }

        return {
            "success": True,
            "message": f"Event probabilities calculated for location: {requested_location}.",
            "state_updates": {},
            "data": {
                "event_probabilities": probabilities,
                "requested_location": requested_location,
            },
        }

    def sample_event(self, event_probabilities: Dict[str, int]) -> Dict[str, Any]:
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
            "data": {"selected_event": selected_event},
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
        memory_text = self._memory_text(relevant_memory)

        impossible_locations = {
            "merchant_cart": {"old ruins", "riverbank"},
            "bandit_roadblock": {"old ruins", "riverbank"},
            "old_ruin_discovery": {"village"},
        }

        if requested_location in impossible_locations.get(selected_event, set()):
            plausible = False
            reason = f"{selected_event} is unlikely at {requested_location}."

        if selected_event == "enemy_ambush":
            enemy_alive = int(game_state.get("enemy_health", 60)) > 0
            known_threat = any(
                word in memory_text for word in ["enemy", "goblin", "bandit", "ambush"]
            )

            if not enemy_alive and not known_threat and int(game_state.get("world_danger", 20)) < 30:
                plausible = False
                reason = "No active or remembered threat supports an enemy ambush."

        if selected_event == "npc_returns_favor":
            favor_memory = any(
                phrase in memory_text
                for phrase in ["helped the traveler", "helped npc", "saved the traveler", "owed the player"]
            )
            if not favor_memory:
                plausible = False
                reason = "No remembered favor supports the NPC's return."

        if selected_event == "guard_patrol":
            guard_alert = int(game_state.get("guard_alert_level", 10))
            crime_level = int(game_state.get("crime_level", 0))
            if requested_location not in {"village", "forest road"} and guard_alert < 40 and crime_level < 30:
                plausible = False
                reason = "A guard patrol is unlikely here without elevated alert or crime."

        return {
            "success": plausible,
            "message": reason,
            "state_updates": {},
            "data": {
                "event_plausible": plausible,
                "plausibility_reason": reason,
            },
        }

    def detect_conflict(
        self,
        selected_event: str,
        event_plausible: bool,
        game_state: Dict[str, Any],
        player_input: str,
        relevant_memory: List[str]
    ) -> Dict[str, Any]:
        if not event_plausible:
            return {
                "success": True,
                "message": "No conflict detected because the event was rejected.",
                "state_updates": {},
                "data": {
                    "conflict_detected": False,
                    "conflict_type": "none",
                    "conflict_severity": 0,
                    "conflict_participants": [],
                    "conflict_reason": "Rejected event",
                },
            }

        conflict_type = "none"
        severity = 0
        reason = "The event does not create an immediate conflict."

        if selected_event in {"enemy_ambush", "bandit_roadblock"}:
            conflict_type = "combat"
            severity = 55 if selected_event == "enemy_ambush" else 45
            reason = "A hostile group directly threatens the player."

        elif selected_event == "guard_patrol":
            crime_level = int(game_state.get("crime_level", 0))
            guard_alert = int(game_state.get("guard_alert_level", 10))

            if crime_level >= 20 or guard_alert >= 40 or game_state.get("player_reputation", 50) <= 25:
                conflict_type = "law"
                severity = 35 + crime_level // 3
                reason = "Guards may confront the player because of crime, alert, or reputation."

        elif selected_event == "merchant_cart" and game_state.get("merchant_hostile", False):
            conflict_type = "social"
            severity = 30
            reason = "The merchant's existing hostility creates a social conflict."

        elif selected_event == "strange_noise" and int(game_state.get("world_danger", 20)) >= 60:
            conflict_type = "environmental"
            severity = 25
            reason = "The dangerous environment turns an uncertain sign into a possible threat."

        text = player_input.lower()
        aggressive_words = {
            "attack", "threaten", "kill", "steal", "rob",
            "fight", "draw weapon", "break"
        }
        if conflict_type == "none" and any(word in text for word in aggressive_words):
            conflict_type = "social"
            severity = 25
            reason = "The player's aggressive action creates a conflict."

        severity = self._clamp(severity)
        participants = self.EVENT_PARTICIPANTS.get(selected_event, ["player"])

        return {
            "success": True,
            "message": reason,
            "state_updates": {},
            "data": {
                "conflict_detected": conflict_type != "none",
                "conflict_type": conflict_type,
                "conflict_severity": severity,
                "conflict_participants": participants if conflict_type != "none" else [],
                "conflict_reason": reason,
            },
        }

    def escalate_conflict(
        self,
        conflict_detected: bool,
        conflict_type: str,
        conflict_severity: int,
        conflict_participants: List[str],
        conflict_reason: str,
        game_state: Dict[str, Any],
        player_input: str
    ) -> Dict[str, Any]:
        if not conflict_detected:
            return {
                "success": True,
                "message": "No conflict escalation was required.",
                "state_updates": {},
                "data": {
                    "conflict_status": "none",
                    "escalated_conflict_severity": 0,
                    "conflict_resolution_hint": "none",
                },
            }

        severity = int(conflict_severity)
        text = player_input.lower()

        if any(word in text for word in ["attack", "kill", "rob", "threaten", "fight"]):
            severity += 25

        if int(game_state.get("world_danger", 20)) >= 60:
            severity += 10

        if conflict_type == "law":
            severity += int(game_state.get("guard_alert_level", 10)) // 5

        if conflict_type == "social" and game_state.get("merchant_hostile", False):
            severity += 10

        severity = self._clamp(severity)

        if severity >= 75:
            status = "severe"
        elif severity >= 40:
            status = "active"
        else:
            status = "latent"

        resolution_hints = {
            "combat": "fight_or_flee",
            "law": "surrender_bribe_or_escape",
            "social": "apologize_negotiate_or_leave",
            "environmental": "investigate_or_avoid",
        }

        conflict_record = {
            "type": conflict_type,
            "status": status,
            "severity": severity,
            "participants": list(conflict_participants),
            "reason": conflict_reason,
        }

        return {
            "success": True,
            "message": (
                f"Conflict classified as {conflict_type}, "
                f"status={status}, severity={severity}."
            ),
            "state_updates": {},
            "data": {
                "conflict_status": status,
                "escalated_conflict_severity": severity,
                "conflict_resolution_hint": resolution_hints.get(conflict_type, "observe"),
                "conflict_record": conflict_record,
            },
        }

    def apply_event_result(
        self,
        selected_event: str,
        event_plausible: bool,
        requested_location: str,
        plausibility_reason: str,
        game_state: Dict[str, Any],
        conflict_detected: bool,
        conflict_record: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        final_event = selected_event if event_plausible else "nothing_special"

        updates: Dict[str, Any] = {"location": requested_location}
        world_danger = int(game_state.get("world_danger", 20))
        guard_alert = int(game_state.get("guard_alert_level", 10))
        trade_activity = int(game_state.get("trade_activity", 60))
        crime_level = int(game_state.get("crime_level", 0))

        messages = {
            "find_coin": "You notice a small gold coin half-buried in the dirt.",
            "find_footprints": "You find fresh footprints leading deeper into the area.",
            "hidden_path": "You discover a narrow hidden path behind thick branches.",
            "strange_noise": "You hear a strange sound somewhere nearby.",
            "npc_encounter": "You encounter a cautious traveler nearby.",
            "traveler_in_need": "A tired traveler asks you for help.",
            "npc_returns_favor": "A familiar NPC returns to repay an old favor.",
            "enemy_ambush": "An enemy emerges from cover and threatens you.",
            "bandit_roadblock": "Bandits block the road and demand payment.",
            "guard_patrol": "A guard patrol approaches and studies you carefully.",
            "merchant_cart": "A merchant cart moves slowly along the road.",
            "old_ruin_discovery": "You find old stone ruins covered by moss and dust.",
            "nothing_special": "Nothing unusual happens."
        }

        if final_event == "find_coin":
            updates["gold"] = int(game_state.get("gold", 0)) + 1
            updates["world_mood"] = "curious"

        elif final_event in {"find_footprints", "hidden_path", "old_ruin_discovery"}:
            updates["world_mood"] = "mysterious"

        elif final_event == "strange_noise":
            updates["world_mood"] = "tense"
            updates["world_danger"] = self._clamp(world_danger + 3)

        elif final_event == "npc_encounter":
            updates["world_mood"] = "watchful"

        elif final_event == "traveler_in_need":
            updates["world_mood"] = "concerned"

        elif final_event == "npc_returns_favor":
            updates["world_mood"] = "hopeful"
            updates["player_reputation"] = self._clamp(
                int(game_state.get("player_reputation", 50)) + 2
            )

        elif final_event in {"enemy_ambush", "bandit_roadblock"}:
            updates["world_mood"] = "hostile"
            updates["world_danger"] = self._clamp(world_danger + 10)

        elif final_event == "guard_patrol":
            updates["world_mood"] = "tense"
            updates["guard_alert_level"] = self._clamp(guard_alert + 5)

        elif final_event == "merchant_cart":
            updates["world_mood"] = "active"
            updates["trade_activity"] = self._clamp(trade_activity + 2)

        else:
            updates["world_mood"] = "quiet"
            updates["world_danger"] = self._clamp(world_danger - 1)

        if conflict_detected and conflict_record:
            active_conflicts = list(game_state.get("active_conflicts", []))
            active_conflicts.append(conflict_record)
            updates["active_conflicts"] = active_conflicts[-10:]

            if conflict_record.get("type") == "law":
                updates["guard_alert_level"] = self._clamp(guard_alert + 10)
                updates["crime_level"] = self._clamp(crime_level + 2)

        return {
            "success": True,
            "message": messages[final_event],
            "state_updates": updates,
            "data": {"final_event": final_event},
        }

    def generate_followup_event(
        self,
        final_event: str,
        conflict_detected: bool,
        conflict_type: str,
        conflict_status: str,
        conflict_severity: int,
        game_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        followup_event: Optional[Dict[str, Any]] = None

        if conflict_detected:
            if conflict_type == "combat":
                followup_event = {
                    "event": "enemy_response_pending",
                    "priority": "high" if conflict_severity >= 70 else "medium",
                    "trigger": "next_player_action",
                    "description": "The hostile group is ready to react to the player.",
                }

            elif conflict_type == "law":
                followup_event = {
                    "event": "guard_investigation",
                    "priority": "high" if conflict_status == "severe" else "medium",
                    "trigger": "next_village_or_road_turn",
                    "description": "Guards may question, pursue, or arrest the player.",
                }

            elif conflict_type == "social":
                followup_event = {
                    "event": "relationship_consequence",
                    "priority": "medium",
                    "trigger": "next_interaction_with_participant",
                    "description": "The participant will remember the conflict and adjust behavior.",
                }

            elif conflict_type == "environmental":
                followup_event = {
                    "event": "hidden_threat_reveal",
                    "priority": "medium",
                    "trigger": "continued_exploration",
                    "description": "Further exploration may reveal the source of the danger.",
                }

        elif final_event == "traveler_in_need":
            followup_event = {
                "event": "traveler_request_pending",
                "priority": "low",
                "trigger": "player_response",
                "description": "Helping the traveler may create a future favor.",
            }

        elif final_event == "npc_returns_favor":
            followup_event = {
                "event": "npc_assistance_available",
                "priority": "medium",
                "trigger": "current_turn",
                "description": "The returning NPC is willing to help the player.",
            }

        elif final_event == "old_ruin_discovery":
            followup_event = {
                "event": "ruin_exploration_available",
                "priority": "low",
                "trigger": "continued_exploration",
                "description": "The ruins can be investigated in a later action.",
            }

        elif final_event == "find_footprints":
            followup_event = {
                "event": "track_source_available",
                "priority": "low",
                "trigger": "continued_exploration",
                "description": "The player can choose to follow the tracks.",
            }

        updates: Dict[str, Any] = {}

        recent_events = list(game_state.get("recent_events", []))
        recent_events.append({
            "event": final_event,
            "conflict": conflict_type if conflict_detected else "none",
        })
        updates["recent_events"] = recent_events[-20:]

        if followup_event:
            pending_events = list(game_state.get("pending_events", []))
            pending_events.append(followup_event)
            updates["pending_events"] = pending_events[-10:]
            message = f"Follow-up event created: {followup_event['event']}."
        else:
            message = "No follow-up event was required."

        return {
            "success": True,
            "message": message,
            "state_updates": updates,
            "data": {"followup_event": followup_event},
        }

from typing import Dict, Any, List


class DialogueStepAgent:
    """
    Handles separate dialogue decision steps.

    This makes dialogue_action suitable for DAG execution:
    NPC attitude analysis, dialogue context analysis, strategy selection,
    and dialogue result application are separate nodes.
    """

    def analyze_npc_attitude(
        self,
        game_state: Dict[str, Any],
        target: str,
        relevant_memory: List[str],
        action_blocked: bool = False
    ) -> Dict[str, Any]:

        if action_blocked:
            return {
                "success": False,
                "message": "NPC attitude analysis skipped because the action was blocked.",
                "state_updates": {},
                "data": {
                    "npc_attitude": "blocked",
                    "attitude_modifier": -100
                }
            }

        memory_text = self._memory_text(relevant_memory)
        episodic_signal = self._analyze_episodic_memory(relevant_memory)

        npc_attitude = "neutral"
        attitude_modifier = 0

        if target == "merchant":
            relationship = game_state.get("relationship_with_merchant", 50)

            if game_state.get("merchant_hostile"):
                npc_attitude = "hostile"
                attitude_modifier -= 50
            elif relationship >= 70:
                npc_attitude = "friendly"
                attitude_modifier += 20
            elif relationship <= 25:
                npc_attitude = "distrustful"
                attitude_modifier -= 20
            else:
                npc_attitude = "neutral"

        elif target == "bartender":
            relationship = game_state.get("relationship_with_bartender", 50)

            if game_state.get("bartender_hostile"):
                npc_attitude = "hostile"
                attitude_modifier -= 50
            elif relationship >= 70:
                npc_attitude = "friendly"
                attitude_modifier += 20
            elif relationship <= 25:
                npc_attitude = "distrustful"
                attitude_modifier -= 20
            else:
                npc_attitude = game_state.get("bartender_mood", "neutral")

        elif target == "enemy":
            enemy_health = game_state.get("enemy_health", 60)

            if enemy_health <= 0:
                npc_attitude = "defeated"
                attitude_modifier -= 100
            elif enemy_health <= 15:
                npc_attitude = "fearful"
                attitude_modifier += 10
            elif enemy_health >= 45:
                npc_attitude = "hostile"
                attitude_modifier -= 30
            else:
                npc_attitude = "wary"

        else:
            npc_attitude = "unknown"
            attitude_modifier = 0

        # Structured episodic memory has priority over loose keyword matching.
        attitude_modifier += episodic_signal["attitude_modifier"]

        if episodic_signal["strong_negative"]:
            if npc_attitude not in ["defeated", "hostile"]:
                npc_attitude = (
                    "fearful"
                    if episodic_signal["fear_score"] > episodic_signal["anger_score"]
                    else "distrustful"
                )

        elif episodic_signal["strong_positive"] and npc_attitude == "neutral":
            npc_attitude = "friendly"

        # Legacy string memories remain supported, but structured episodic
        # records are not counted a second time here.
        if not episodic_signal["has_structured_memory"]:
            if "attack" in memory_text or "hostile" in memory_text or "violence" in memory_text:
                attitude_modifier -= 20

                if npc_attitude not in ["defeated", "hostile"]:
                    npc_attitude = "distrustful"

            if "friendly" in memory_text or "helped" in memory_text or "generous" in memory_text:
                attitude_modifier += 10

                if npc_attitude == "neutral":
                    npc_attitude = "friendly"

        return {
            "success": True,
            "message": (
                f"NPC attitude analysis completed. "
                f"Target: {target}. "
                f"Attitude: {npc_attitude}. "
                f"Modifier: {attitude_modifier}."
            ),
            "state_updates": {},
            "data": {
                "npc_attitude": npc_attitude,
                "attitude_modifier": attitude_modifier,
                "episodic_memory_signal": episodic_signal
            }
        }

    def analyze_dialogue_context(
        self,
        game_state: Dict[str, Any],
        target: str,
        player_input: str,
        relevant_memory: List[str],
        action_blocked: bool = False
    ) -> Dict[str, Any]:

        if action_blocked:
            return {
                "success": False,
                "message": "Dialogue context analysis skipped because the action was blocked.",
                "state_updates": {},
                "data": {
                    "dialogue_topic": "blocked",
                    "dialogue_tone": "blocked"
                }
            }

        text = player_input.lower()

        dialogue_topic = "general_conversation"
        dialogue_tone = "neutral"

        if any(word in text for word in ["rumor", "rumour", "news", "information", "strange", "weird", "odd"]):
            dialogue_topic = "information_request"

        elif any(word in text for word in ["sorry", "apologize", "forgive"]):
            dialogue_topic = "apology"

        elif any(word in text for word in ["threaten", "warn", "intimidate"]):
            dialogue_topic = "threat"

        elif any(word in text for word in ["help", "quest", "job", "work"]):
            dialogue_topic = "quest_request"

        elif any(word in text for word in ["hello", "greet", "hi", "wave"]):
            dialogue_topic = "greeting"

        if any(word in text for word in ["please", "kindly", "politely"]):
            dialogue_tone = "polite"

        elif any(word in text for word in ["flirt", "wink", "smile", "compliment"]):
            dialogue_tone = "flirtatious"

        elif any(word in text for word in ["angry", "shout", "demand", "threaten"]):
            dialogue_tone = "aggressive"

        elif target == "enemy":
            dialogue_tone = "tense"

        location = game_state.get("location", "unknown")

        return {
            "success": True,
            "message": (
                f"Dialogue context analysis completed. "
                f"Topic: {dialogue_topic}. "
                f"Tone: {dialogue_tone}. "
                f"Location: {location}."
            ),
            "state_updates": {},
            "data": {
                "dialogue_topic": dialogue_topic,
                "dialogue_tone": dialogue_tone,
                "dialogue_location": location
            }
        }

    def choose_dialogue_strategy(
        self,
        npc_attitude: str,
        dialogue_topic: str,
        dialogue_tone: str,
        target: str,
        action_blocked: bool = False
    ) -> Dict[str, Any]:

        if action_blocked:
            return {
                "success": False,
                "message": "Dialogue strategy selection skipped because the action was blocked.",
                "state_updates": {},
                "data": {
                    "dialogue_strategy": "blocked"
                }
            }

        dialogue_strategy = "neutral_response"

        if npc_attitude in ["hostile", "distrustful"]:
            if dialogue_topic == "apology":
                dialogue_strategy = "cautious_listening"
            else:
                dialogue_strategy = "cold_refusal"

        elif npc_attitude == "friendly":
            if dialogue_topic == "information_request":
                dialogue_strategy = "helpful_information"
            elif dialogue_topic == "quest_request":
                dialogue_strategy = "offer_help"
            else:
                dialogue_strategy = "warm_response"

        elif npc_attitude == "fearful":
            dialogue_strategy = "fearful_cooperation"

        elif npc_attitude == "defeated":
            dialogue_strategy = "no_response"

        elif target == "enemy":
            dialogue_strategy = "threatening_response"

        elif dialogue_tone == "aggressive":
            dialogue_strategy = "defensive_response"

        elif dialogue_tone == "flirtatious":
            dialogue_strategy = "playful_or_cautious_response"

        return {
            "success": True,
            "message": (
                f"Dialogue strategy selected: {dialogue_strategy}."
            ),
            "state_updates": {},
            "data": {
                "dialogue_strategy": dialogue_strategy
            }
        }

    @staticmethod
    def _memory_text(memories: List[Any]) -> str:
        parts: List[str] = []
        for memory in memories or []:
            if isinstance(memory, dict):
                parts.extend([
                    str(memory.get("event_type", "")),
                    str(memory.get("summary", memory.get("event", ""))),
                    str(memory.get("emotional_tag", "")),
                ])
            else:
                parts.append(str(memory))
        return " ".join(parts).lower()

    @staticmethod
    def _analyze_episodic_memory(memories: List[Any]) -> Dict[str, Any]:
        """
        Convert structured NPC memories into a deterministic attitude signal.
        Only sufficiently confident memories influence the decision strongly.
        """
        negative_types = {"insulted", "threatened", "attacked", "robbed"}
        positive_types = {"helped", "bought_goods", "bought_drink", "left_tip", "apologized"}

        modifier = 0
        fear_score = 0
        anger_score = 0
        strongest_event = None
        strongest_weight = -1.0
        has_structured = False

        for memory in memories or []:
            if not isinstance(memory, dict):
                continue

            has_structured = True
            event_type = str(memory.get("event_type", "")).strip().lower()
            importance = max(0, min(100, int(memory.get("importance", 50) or 50)))
            try:
                confidence = max(0.0, min(1.0, float(memory.get("confidence", 1.0))))
            except (TypeError, ValueError):
                confidence = 1.0

            weight = (importance / 100.0) * confidence
            impact = memory.get("emotional_impact") or {}

            if event_type in negative_types:
                modifier -= round(25 * weight)
            elif event_type in positive_types:
                modifier += round(15 * weight)

            fear_score += max(0, int(impact.get("fear", 0) or 0)) * weight
            anger_score += max(0, int(impact.get("anger", 0) or 0)) * weight

            if weight > strongest_weight:
                strongest_weight = weight
                strongest_event = {
                    "event_type": event_type or "interaction",
                    "summary": memory.get("summary", memory.get("event", "")),
                    "importance": importance,
                    "source": memory.get("source", "experienced"),
                    "confidence": confidence,
                }

        return {
            "has_structured_memory": has_structured,
            "attitude_modifier": modifier,
            "fear_score": round(fear_score, 2),
            "anger_score": round(anger_score, 2),
            "strong_negative": modifier <= -12,
            "strong_positive": modifier >= 8,
            "strongest_event": strongest_event,
        }

    def apply_dialogue_result(
        self,
        game_state: Dict[str, Any],
        target: str,
        npc_attitude: str,
        dialogue_topic: str,
        dialogue_tone: str,
        dialogue_strategy: str,
        action_blocked: bool = False,
        blocked_reason: str = ""
    ) -> Dict[str, Any]:

        if action_blocked:
            return {
                "success": False,
                "message": blocked_reason or "The dialogue action was blocked.",
                "state_updates": {}
            }

        updates = {}

        if target == "merchant":
            if dialogue_strategy == "cold_refusal":
                updates["relationship_with_merchant"] = max(
                    game_state.get("relationship_with_merchant", 50) - 2,
                    0
                )
                message = "The merchant responds coldly and refuses to continue the conversation."

            elif dialogue_strategy == "cautious_listening":
                updates["relationship_with_merchant"] = min(
                    game_state.get("relationship_with_merchant", 50) + 2,
                    100
                )
                message = "The merchant listens cautiously, but still does not fully trust the player."

            elif dialogue_strategy == "warm_response":
                updates["relationship_with_merchant"] = min(
                    game_state.get("relationship_with_merchant", 50) + 2,
                    100
                )
                message = "The merchant responds with a more open and respectful tone."

            else:
                message = "The merchant listens and gives a measured response."

        elif target == "bartender":
            if dialogue_strategy == "cold_refusal":
                updates["bartender_mood"] = "cold"
                updates["relationship_with_bartender"] = max(
                    game_state.get("relationship_with_bartender", 50) - 2,
                    0
                )
                message = "The bartender gives a cold answer and avoids further conversation."

            elif dialogue_strategy == "helpful_information":
                updates["bartender_mood"] = "talkative"
                updates["relationship_with_bartender"] = min(
                    game_state.get("relationship_with_bartender", 50) + 2,
                    100
                )
                message = "The bartender becomes more talkative and seems willing to share useful information."

            elif dialogue_strategy == "playful_or_cautious_response":
                updates["bartender_mood"] = "amused"
                updates["relationship_with_bartender"] = min(
                    game_state.get("relationship_with_bartender", 50) + 1,
                    100
                )
                message = "The bartender reacts with cautious amusement."

            else:
                message = "The bartender responds and waits to hear what the player wants next."

        elif target == "enemy":
            if dialogue_strategy == "no_response":
                message = "The enemy cannot respond."

            elif dialogue_strategy == "fearful_cooperation":
                updates["world_mood"] = "tense"
                message = "The enemy hesitates and seems willing to speak out of fear."

            elif dialogue_strategy == "threatening_response":
                updates["world_mood"] = "hostile"
                message = "The enemy answers with a threat and prepares to continue fighting."

            else:
                updates["world_mood"] = "uneasy"
                message = "The enemy listens for a moment, but remains dangerous."

        else:
            message = "There is no clear character to respond."

        return {
            "success": True,
            "message": message,
            "state_updates": updates
        }

from typing import Any, Dict, List, Optional


class DialogueAgent:
    """
    Composite agent.

    Handles conversations with NPCs such as enemies,
    merchants and bartenders.

    Dialogue depends on:
    - legacy GameState fields;
    - dynamic NPC state;
    - NPC personality and goals;
    - personal NPC memory;
    - semantic memory;
    - current player input;
    - ConsequenceAgent evaluation.

    The agent preserves the original deterministic dialogue logic,
    but enriches it with NPC-aware context.
    """

    def execute(
        self,
        game_state: Dict[str, Any],
        target: str = "unknown",
        player_input: str = "",
        relevant_memory: Optional[List[Any]] = None,
        npc_context: Optional[Dict[str, Any]] = None,
        consequence_result: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Executes an NPC dialogue action.

        Parameters
        ----------
        game_state:
            Current global game state.

        target:
            Dialogue target.

        player_input:
            Current player message.

        relevant_memory:
            Semantic memories retrieved for the current action.

        npc_context:
            Static NPC profile together with dynamic runtime state.

        consequence_result:
            Result produced by ConsequenceAgent before dialogue execution.

        Returns
        -------
        Dict[str, Any]
            Dialogue result with success, message, state updates
            and diagnostic data.
        """
        relevant_memory = relevant_memory or []
        npc_context = npc_context or {}
        consequence_result = consequence_result or {}

        normalized_target = self._normalize_target(target)

        profile = npc_context.get("profile", {})
        npc_state = npc_context.get("state", {})

        dialogue_data = self._build_dialogue_data(
            target=normalized_target,
            player_input=player_input,
            profile=profile,
            npc_state=npc_state,
            relevant_memory=relevant_memory,
            consequence_result=consequence_result
        )

        blocked_result = self._handle_blocked_dialogue(
            consequence_result=consequence_result,
            dialogue_data=dialogue_data
        )

        if blocked_result is not None:
            return blocked_result

        if normalized_target == "forest_goblin":
            return self._handle_enemy_dialogue(
                game_state=game_state,
                player_input=player_input,
                npc_state=npc_state,
                dialogue_data=dialogue_data,
                consequence_result=consequence_result
            )

        if normalized_target == "merchant":
            return self._handle_merchant_dialogue(
                game_state=game_state,
                player_input=player_input,
                npc_state=npc_state,
                dialogue_data=dialogue_data,
                consequence_result=consequence_result
            )

        if normalized_target == "bartender":
            return self._handle_bartender_dialogue(
                game_state=game_state,
                player_input=player_input,
                npc_state=npc_state,
                dialogue_data=dialogue_data,
                consequence_result=consequence_result
            )

        if normalized_target == "guard":
            return self._handle_guard_dialogue(
                game_state=game_state,
                player_input=player_input,
                npc_state=npc_state,
                dialogue_data=dialogue_data,
                consequence_result=consequence_result
            )

        if normalized_target == "traveler":
            return self._handle_traveler_dialogue(
                game_state=game_state,
                player_input=player_input,
                npc_state=npc_state,
                dialogue_data=dialogue_data,
                consequence_result=consequence_result
            )

        return self._build_result(
            success=False,
            message="There is no clear character to talk to.",
            state_updates={},
            dialogue_data=dialogue_data
        )

    # =========================================================
    # Enemy dialogue
    # =========================================================

    def _handle_enemy_dialogue(
        self,
        game_state: Dict[str, Any],
        player_input: str,
        npc_state: Dict[str, Any],
        dialogue_data: Dict[str, Any],
        consequence_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handles dialogue with the forest goblin or legacy enemy target.
        """
        enemy_health = int(
            npc_state.get(
                "health",
                game_state.get("enemy_health", 60)
            )
        )

        enemy_alive = bool(
            npc_state.get(
                "alive",
                enemy_health > 0
            )
        )

        enemy_fear = int(
            npc_state.get("fear", 10)
        )

        enemy_anger = int(
            npc_state.get("anger", 40)
        )

        enemy_stress = int(
            npc_state.get("stress", 30)
        )

        emotion = str(
            npc_state.get("emotion", "hostile")
        ).lower()

        current_goal = str(
            npc_state.get(
                "current_goal",
                "defend territory"
            )
        )

        if not enemy_alive or enemy_health <= 0:
            return self._build_result(
                success=False,
                message="The enemy is defeated and cannot respond.",
                state_updates={},
                dialogue_data=dialogue_data
            )

        if (
            enemy_health <= 15
            or enemy_fear >= 70
            or emotion == "afraid"
        ):
            message = (
                "The wounded goblin recoils and speaks in a trembling voice. "
                "It appears more interested in survival than continuing the fight."
            )

            if self._contains_any(
                player_input,
                [
                    "surrender",
                    "stop fighting",
                    "peace",
                    "spare",
                    "let you live"
                ]
            ):
                message = (
                    "The frightened goblin lowers its weapon and watches the player "
                    "carefully, willing to discuss surrender if it will be spared."
                )

            return self._build_result(
                success=True,
                message=message,
                state_updates={
                    "world_mood": "tense"
                },
                dialogue_data=dialogue_data,
                additional_data={
                    "reaction_style": "fearful",
                    "current_goal": current_goal
                }
            )

        if (
            enemy_health <= 35
            or enemy_stress >= 60
        ):
            return self._build_result(
                success=True,
                message=(
                    "The goblin growls and keeps its weapon ready, but listens "
                    "for a moment. Pain and uncertainty have made it less confident."
                ),
                state_updates={
                    "world_mood": "uneasy"
                },
                dialogue_data=dialogue_data,
                additional_data={
                    "reaction_style": "uneasy",
                    "current_goal": current_goal
                }
            )

        if (
            enemy_anger >= 70
            or emotion in {"angry", "hostile"}
        ):
            return self._build_result(
                success=True,
                message=(
                    "The goblin refuses peaceful conversation and answers with "
                    "threats, determined to defend its territory."
                ),
                state_updates={
                    "world_mood": "hostile"
                },
                dialogue_data=dialogue_data,
                additional_data={
                    "reaction_style": "hostile",
                    "current_goal": current_goal
                }
            )

        return self._build_result(
            success=True,
            message=(
                "The goblin watches the player suspiciously. It does not lower "
                "its weapon, but it appears willing to listen."
            ),
            state_updates={
                "world_mood": "uneasy"
            },
            dialogue_data=dialogue_data,
            additional_data={
                "reaction_style": "cautious",
                "current_goal": current_goal
            }
        )

    # =========================================================
    # Merchant dialogue
    # =========================================================

    def _handle_merchant_dialogue(
        self,
        game_state: Dict[str, Any],
        player_input: str,
        npc_state: Dict[str, Any],
        dialogue_data: Dict[str, Any],
        consequence_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handles dialogue with the merchant.
        """
        merchant_health = int(
            npc_state.get(
                "health",
                game_state.get("merchant_health", 100)
            )
        )

        merchant_alive = bool(
            npc_state.get(
                "alive",
                merchant_health > 0
            )
        )

        merchant_hostile = bool(
            npc_state.get(
                "hostile",
                game_state.get("merchant_hostile", False)
            )
        )

        merchant_trust = int(
            npc_state.get(
                "trust",
                game_state.get(
                    "relationship_with_merchant",
                    50
                )
            )
        )

        merchant_anger = int(
            npc_state.get("anger", 0)
        )

        merchant_fear = int(
            npc_state.get("fear", 0)
        )

        merchant_stress = int(
            npc_state.get("stress", 10)
        )

        emotion = str(
            npc_state.get("emotion", "neutral")
        ).lower()

        current_goal = str(
            npc_state.get(
                "current_goal",
                "earn profit"
            )
        )

        memory_text = dialogue_data["memory_text"]

        if not merchant_alive or merchant_health <= 0:
            return self._build_result(
                success=False,
                message="The merchant is unable to respond.",
                state_updates={},
                dialogue_data=dialogue_data
            )

        asking_for_apology = self._contains_any(
            player_input,
            [
                "sorry",
                "apologize",
                "apologise",
                "forgive me",
                "my mistake"
            ]
        )

        asking_for_trade = self._contains_any(
            player_input,
            [
                "buy",
                "sell",
                "trade",
                "price",
                "discount",
                "cheaper",
                "artifact",
                "goods",
                "item"
            ]
        )

        attack_remembered = self._contains_any(
            memory_text,
            [
                "attacked this npc",
                "robbed this npc",
                "threatened this npc",
                "attack the merchant",
                "attacks the merchant"
            ]
        )

        if merchant_hostile or attack_remembered:
            if asking_for_apology:
                return self._build_result(
                    success=True,
                    message=(
                        "The merchant listens to the apology without relaxing. "
                        "They remember the previous violence and warn that trust "
                        "will have to be earned through actions."
                    ),
                    state_updates={
                        "relationship_with_merchant": min(
                            game_state.get(
                                "relationship_with_merchant",
                                merchant_trust
                            ) + 2,
                            100
                        )
                    },
                    dialogue_data=dialogue_data,
                    additional_data={
                        "reaction_style": "cautious",
                        "current_goal": current_goal
                    }
                )

            if merchant_fear >= 70:
                message = (
                    "The merchant backs away from the player and refuses to continue "
                    "the conversation. Fear is stronger than any desire to trade."
                )

            elif merchant_anger >= 70 or emotion == "hostile":
                message = (
                    "The merchant angrily orders the player to leave and refuses "
                    "to discuss business after the previous attack or threat."
                )

            else:
                message = (
                    "The merchant refuses to talk because the player previously "
                    "attacked, robbed or threatened them."
                )

            return self._build_result(
                success=True,
                message=message,
                state_updates={
                    "merchant_hostile": True,
                    "relationship_with_merchant": max(
                        game_state.get(
                            "relationship_with_merchant",
                            merchant_trust
                        ) - 2,
                        0
                    )
                },
                dialogue_data=dialogue_data,
                additional_data={
                    "reaction_style": "hostile",
                    "current_goal": current_goal
                }
            )

        if merchant_trust >= 75:
            if asking_for_trade:
                message = (
                    "The merchant greets the player as a trusted customer and "
                    "listens carefully, appearing open to discussing prices."
                )

            else:
                message = (
                    "The merchant welcomes the player warmly and seems comfortable "
                    "sharing information beyond ordinary business."
                )

            return self._build_result(
                success=True,
                message=message,
                state_updates={
                    "relationship_with_merchant": min(
                        game_state.get(
                            "relationship_with_merchant",
                            merchant_trust
                        ) + 1,
                        100
                    )
                },
                dialogue_data=dialogue_data,
                additional_data={
                    "reaction_style": "friendly",
                    "current_goal": current_goal
                }
            )

        if merchant_trust <= 25:
            return self._build_result(
                success=True,
                message=(
                    "The merchant answers cautiously and keeps a close watch on "
                    "the player. They are willing to listen, but clearly expect trouble."
                ),
                state_updates={},
                dialogue_data=dialogue_data,
                additional_data={
                    "reaction_style": "suspicious",
                    "current_goal": current_goal
                }
            )

        if merchant_stress >= 70:
            return self._build_result(
                success=True,
                message=(
                    "The merchant appears tense and distracted. They listen briefly "
                    "but repeatedly glance toward their goods and the nearest exit."
                ),
                state_updates={},
                dialogue_data=dialogue_data,
                additional_data={
                    "reaction_style": "stressed",
                    "current_goal": current_goal
                }
            )

        return self._build_result(
            success=True,
            message=(
                "The merchant listens carefully and waits for the player "
                "to explain what they want."
            ),
            state_updates={},
            dialogue_data=dialogue_data,
            additional_data={
                "reaction_style": "neutral",
                "current_goal": current_goal
            }
        )

    # =========================================================
    # Bartender dialogue
    # =========================================================

    def _handle_bartender_dialogue(
        self,
        game_state: Dict[str, Any],
        player_input: str,
        npc_state: Dict[str, Any],
        dialogue_data: Dict[str, Any],
        consequence_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handles dialogue with the bartender.
        """
        bartender_health = int(
            npc_state.get(
                "health",
                game_state.get("bartender_health", 100)
            )
        )

        bartender_alive = bool(
            npc_state.get(
                "alive",
                bartender_health > 0
            )
        )

        bartender_hostile = bool(
            npc_state.get(
                "hostile",
                game_state.get("bartender_hostile", False)
            )
        )

        bartender_trust = int(
            npc_state.get(
                "trust",
                game_state.get(
                    "relationship_with_bartender",
                    50
                )
            )
        )

        bartender_anger = int(
            npc_state.get("anger", 0)
        )

        bartender_fear = int(
            npc_state.get("fear", 0)
        )

        bartender_stress = int(
            npc_state.get("stress", 10)
        )

        emotion = str(
            npc_state.get("emotion", "neutral")
        ).lower()

        current_goal = str(
            npc_state.get(
                "current_goal",
                "serve customers"
            )
        )

        tavern_reputation = int(
            game_state.get("tavern_reputation", 50)
        )

        memory_text = dialogue_data["memory_text"]

        if not bartender_alive or bartender_health <= 0:
            return self._build_result(
                success=False,
                message="The bartender is unable to respond.",
                state_updates={},
                dialogue_data=dialogue_data
            )

        asking_for_information = self._contains_any(
            player_input,
            [
                "rumor",
                "rumour",
                "information",
                "news",
                "strange",
                "odd",
                "nearby",
                "what happened",
                "tell me"
            ]
        )

        asking_for_apology = self._contains_any(
            player_input,
            [
                "sorry",
                "apologize",
                "apologise",
                "forgive me",
                "my mistake"
            ]
        )

        generosity_remembered = self._contains_any(
            memory_text,
            [
                "bought_drink",
                "left_tip",
                "bought drink",
                "left tip",
                "spent generously",
                "premium drink"
            ]
        )

        violence_remembered = self._contains_any(
            memory_text,
            [
                "attacked this npc",
                "threatened this npc",
                "attack the bartender",
                "attacks the bartender",
                "attack the barmaid"
            ]
        )

        if bartender_hostile or violence_remembered:
            if asking_for_apology:
                return self._build_result(
                    success=True,
                    message=(
                        "The bartender hears the apology but remains guarded. "
                        "They make it clear that peaceful behavior must continue "
                        "before trust can return."
                    ),
                    state_updates={
                        "relationship_with_bartender": min(
                            game_state.get(
                                "relationship_with_bartender",
                                bartender_trust
                            ) + 2,
                            100
                        ),
                        "bartender_mood": "suspicious"
                    },
                    dialogue_data=dialogue_data,
                    additional_data={
                        "reaction_style": "cautious",
                        "current_goal": current_goal
                    }
                )

            if bartender_fear >= 70:
                message = (
                    "The bartender keeps their distance and refuses to speak, "
                    "watching the player with obvious fear."
                )

            elif bartender_anger >= 70 or emotion == "hostile":
                message = (
                    "The bartender angrily refuses to talk and warns the player "
                    "not to cause any more trouble in the tavern."
                )

            else:
                message = (
                    "The bartender refuses to talk because the player caused "
                    "trouble in the tavern."
                )

            return self._build_result(
                success=True,
                message=message,
                state_updates={
                    "bartender_hostile": True,
                    "relationship_with_bartender": max(
                        game_state.get(
                            "relationship_with_bartender",
                            bartender_trust
                        ) - 2,
                        0
                    ),
                    "bartender_mood": "angry"
                },
                dialogue_data=dialogue_data,
                additional_data={
                    "reaction_style": "hostile",
                    "current_goal": current_goal
                }
            )

        if tavern_reputation < 25:
            return self._build_result(
                success=True,
                message=(
                    "The bartender answers coldly. The player's poor reputation "
                    "in the tavern makes the conversation tense."
                ),
                state_updates={
                    "bartender_mood": "suspicious"
                },
                dialogue_data=dialogue_data,
                additional_data={
                    "reaction_style": "suspicious",
                    "current_goal": current_goal
                }
            )

        if (
            asking_for_information
            and generosity_remembered
        ):
            return self._build_result(
                success=True,
                message=(
                    "Remembering the player's generosity, the bartender leans closer "
                    "and quietly offers a useful piece of local information."
                ),
                state_updates={
                    "bartender_mood": "friendly",
                    "relationship_with_bartender": min(
                        game_state.get(
                            "relationship_with_bartender",
                            bartender_trust
                        ) + 2,
                        100
                    )
                },
                dialogue_data=dialogue_data,
                additional_data={
                    "reaction_style": "grateful",
                    "current_goal": current_goal
                }
            )

        if bartender_trust >= 70:
            return self._build_result(
                success=True,
                message=(
                    "The bartender greets the player warmly and seems willing "
                    "to share useful information."
                ),
                state_updates={
                    "bartender_mood": "friendly"
                },
                dialogue_data=dialogue_data,
                additional_data={
                    "reaction_style": "friendly",
                    "current_goal": current_goal
                }
            )

        if bartender_trust <= 25:
            return self._build_result(
                success=True,
                message=(
                    "The bartender responds cautiously and keeps the conversation "
                    "brief, clearly uncertain whether the player can be trusted."
                ),
                state_updates={
                    "bartender_mood": "suspicious"
                },
                dialogue_data=dialogue_data,
                additional_data={
                    "reaction_style": "suspicious",
                    "current_goal": current_goal
                }
            )

        if bartender_stress >= 70:
            return self._build_result(
                success=True,
                message=(
                    "The bartender appears stressed by the situation in the tavern "
                    "and answers while carefully watching the room."
                ),
                state_updates={
                    "bartender_mood": "stressed"
                },
                dialogue_data=dialogue_data,
                additional_data={
                    "reaction_style": "stressed",
                    "current_goal": current_goal
                }
            )

        return self._build_result(
            success=True,
            message=(
                "The bartender wipes a wooden mug and waits to hear "
                "what the player wants."
            ),
            state_updates={
                "bartender_mood": "neutral"
            },
            dialogue_data=dialogue_data,
            additional_data={
                "reaction_style": "neutral",
                "current_goal": current_goal
            }
        )

    # =========================================================
    # Guard dialogue
    # =========================================================

    def _handle_guard_dialogue(
        self,
        game_state: Dict[str, Any],
        player_input: str,
        npc_state: Dict[str, Any],
        dialogue_data: Dict[str, Any],
        consequence_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handles dialogue with a guard.
        """
        health = int(
            npc_state.get("health", 100)
        )

        alive = bool(
            npc_state.get("alive", health > 0)
        )

        trust = int(
            npc_state.get("trust", 45)
        )

        hostile = bool(
            npc_state.get("hostile", False)
        )

        emotion = str(
            npc_state.get("emotion", "alert")
        ).lower()

        current_goal = str(
            npc_state.get(
                "current_goal",
                "maintain order"
            )
        )

        crime_level = int(
            game_state.get("crime_level", 0)
        )

        guard_alert = int(
            game_state.get("guard_alert_level", 10)
        )

        player_reputation = int(
            game_state.get("player_reputation", 50)
        )

        if not alive or health <= 0:
            return self._build_result(
                success=False,
                message="The guard is unable to respond.",
                state_updates={},
                dialogue_data=dialogue_data
            )

        if hostile:
            return self._build_result(
                success=True,
                message=(
                    "The guard treats the player as an immediate threat "
                    "and orders them to surrender."
                ),
                state_updates={
                    "guard_alert_level": max(
                        guard_alert,
                        70
                    )
                },
                dialogue_data=dialogue_data,
                additional_data={
                    "reaction_style": "hostile",
                    "current_goal": current_goal
                }
            )

        if (
            crime_level >= 40
            or guard_alert >= 60
            or player_reputation <= 20
            or trust <= 25
            or emotion == "suspicious"
        ):
            return self._build_result(
                success=True,
                message=(
                    "The guard questions the player formally and watches every "
                    "movement, treating them as a possible suspect."
                ),
                state_updates={},
                dialogue_data=dialogue_data,
                additional_data={
                    "reaction_style": "suspicious",
                    "current_goal": current_goal
                }
            )

        if self._contains_any(
            player_input,
            [
                "help me",
                "protect me",
                "report",
                "danger",
                "bandits",
                "crime"
            ]
        ):
            return self._build_result(
                success=True,
                message=(
                    "The guard listens carefully and asks for clear details "
                    "before deciding what assistance can be provided."
                ),
                state_updates={},
                dialogue_data=dialogue_data,
                additional_data={
                    "reaction_style": "professional",
                    "current_goal": current_goal
                }
            )

        return self._build_result(
            success=True,
            message=(
                "The guard acknowledges the player with a restrained nod "
                "and waits for them to explain their business."
            ),
            state_updates={},
            dialogue_data=dialogue_data,
            additional_data={
                "reaction_style": "formal",
                "current_goal": current_goal
            }
        )

    # =========================================================
    # Traveler dialogue
    # =========================================================

    def _handle_traveler_dialogue(
        self,
        game_state: Dict[str, Any],
        player_input: str,
        npc_state: Dict[str, Any],
        dialogue_data: Dict[str, Any],
        consequence_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handles dialogue with a traveler.
        """
        health = int(
            npc_state.get("health", 100)
        )

        alive = bool(
            npc_state.get("alive", health > 0)
        )

        trust = int(
            npc_state.get("trust", 50)
        )

        fear = int(
            npc_state.get("fear", 5)
        )

        hostile = bool(
            npc_state.get("hostile", False)
        )

        emotion = str(
            npc_state.get("emotion", "curious")
        ).lower()

        current_goal = str(
            npc_state.get(
                "current_goal",
                "travel safely"
            )
        )

        memory_text = dialogue_data["memory_text"]

        if not alive or health <= 0:
            return self._build_result(
                success=False,
                message="The traveler is unable to respond.",
                state_updates={},
                dialogue_data=dialogue_data
            )

        if hostile:
            return self._build_result(
                success=True,
                message=(
                    "The traveler refuses conversation and moves away from "
                    "the player, expecting further trouble."
                ),
                state_updates={},
                dialogue_data=dialogue_data,
                additional_data={
                    "reaction_style": "hostile",
                    "current_goal": current_goal
                }
            )

        helped_before = self._contains_any(
            memory_text,
            [
                "helped this npc",
                "saved this npc",
                "assisted this npc"
            ]
        )

        asking_for_directions = self._contains_any(
            player_input,
            [
                "road",
                "path",
                "direction",
                "where",
                "route",
                "town",
                "village"
            ]
        )

        asking_for_information = self._contains_any(
            player_input,
            [
                "rumor",
                "information",
                "danger",
                "news",
                "what happened"
            ]
        )

        if (
            helped_before
            and (
                asking_for_directions
                or asking_for_information
            )
        ):
            return self._build_result(
                success=True,
                message=(
                    "The traveler recognizes the player and responds warmly, "
                    "sharing useful details in gratitude for the earlier help."
                ),
                state_updates={},
                dialogue_data=dialogue_data,
                additional_data={
                    "reaction_style": "grateful",
                    "current_goal": current_goal
                }
            )

        if fear >= 70 or emotion == "afraid":
            return self._build_result(
                success=True,
                message=(
                    "The traveler answers nervously and keeps looking over their "
                    "shoulder, clearly worried about danger nearby."
                ),
                state_updates={},
                dialogue_data=dialogue_data,
                additional_data={
                    "reaction_style": "fearful",
                    "current_goal": current_goal
                }
            )

        if trust >= 70:
            return self._build_result(
                success=True,
                message=(
                    "The traveler speaks openly and seems pleased to exchange "
                    "stories and useful information."
                ),
                state_updates={},
                dialogue_data=dialogue_data,
                additional_data={
                    "reaction_style": "friendly",
                    "current_goal": current_goal
                }
            )

        if trust <= 25:
            return self._build_result(
                success=True,
                message=(
                    "The traveler gives a short, guarded answer and avoids "
                    "revealing anything important."
                ),
                state_updates={},
                dialogue_data=dialogue_data,
                additional_data={
                    "reaction_style": "suspicious",
                    "current_goal": current_goal
                }
            )

        return self._build_result(
            success=True,
            message=(
                "The traveler pauses and listens with curiosity, waiting "
                "to hear what the player wants to know."
            ),
            state_updates={},
            dialogue_data=dialogue_data,
            additional_data={
                "reaction_style": "curious",
                "current_goal": current_goal
            }
        )

    # =========================================================
    # Consequence handling
    # =========================================================

    def _handle_blocked_dialogue(
        self,
        consequence_result: Dict[str, Any],
        dialogue_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Converts a blocked ConsequenceAgent result into
        a standard DialogueAgent result.
        """
        if not consequence_result:
            return None

        if consequence_result.get("allow_action", True):
            return None

        message = consequence_result.get(
            "reason",
            "The NPC refuses to continue the conversation."
        )

        system_note = consequence_result.get(
            "system_note",
            ""
        )

        if system_note:
            message = f"{message} {system_note}".strip()

        return self._build_result(
            success=False,
            message=message,
            state_updates=consequence_result.get(
                "state_updates",
                {}
            ),
            dialogue_data=dialogue_data,
            additional_data={
                "reaction_style": consequence_result.get(
                    "reaction_modifier",
                    "blocked"
                ),
                "blocked_by_consequence": True
            }
        )

    # =========================================================
    # Context helpers
    # =========================================================

    def _build_dialogue_data(
        self,
        target: str,
        player_input: str,
        profile: Dict[str, Any],
        npc_state: Dict[str, Any],
        relevant_memory: List[Any],
        consequence_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Builds structured diagnostic context for the dialogue result.

        This data can later be passed directly into an LLM-based
        dialogue generator.
        """
        personal_memories = npc_state.get(
            "personal_memory",
            []
        )

        semantic_memory_text = " ".join(
            str(item)
            for item in relevant_memory
        )

        personal_memory_text = " ".join(
            str(memory.get("event", memory))
            if isinstance(memory, dict)
            else str(memory)
            for memory in personal_memories
        )

        memory_text = (
            f"{semantic_memory_text} "
            f"{personal_memory_text}"
        ).strip().lower()

        return {
            "target": target,
            "player_input": player_input,
            "profile": profile,
            "npc_state": npc_state,
            "personality": profile.get(
                "personality",
                {}
            ),
            "goals": profile.get(
                "goals",
                []
            ),
            "current_goal": npc_state.get(
                "current_goal"
            ),
            "emotion": npc_state.get(
                "emotion",
                "neutral"
            ),
            "trust": npc_state.get(
                "trust"
            ),
            "fear": npc_state.get(
                "fear"
            ),
            "anger": npc_state.get(
                "anger"
            ),
            "stress": npc_state.get(
                "stress"
            ),
            "hostile": npc_state.get(
                "hostile"
            ),
            "personal_memories": personal_memories,
            "semantic_memories": relevant_memory,
            "memory_text": memory_text,
            "consequence_result": consequence_result
        }

    def _build_result(
        self,
        success: bool,
        message: str,
        state_updates: Dict[str, Any],
        dialogue_data: Dict[str, Any],
        additional_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Produces one consistent result format for every dialogue branch.
        """
        data = {
            "dialogue_context": dialogue_data
        }

        if additional_data:
            data.update(additional_data)

        return {
            "success": success,
            "message": message,
            "state_updates": state_updates,
            "data": data
        }

    @staticmethod
    def _normalize_target(target: str) -> str:
        """
        Converts common aliases into internal NPC IDs.
        """
        if not target:
            return ""

        normalized = str(target).strip().lower()

        aliases = {
            "enemy": "forest_goblin",
            "goblin": "forest_goblin",
            "forest goblin": "forest_goblin",
            "forest_goblin": "forest_goblin",

            "merchant": "merchant",
            "vendor": "merchant",
            "shopkeeper": "merchant",
            "trader": "merchant",

            "bartender": "bartender",
            "barmaid": "bartender",
            "innkeeper": "bartender",

            "guard": "guard",
            "town guard": "guard",

            "traveler": "traveler",
            "traveller": "traveler"
        }

        return aliases.get(
            normalized,
            normalized
        )

    @staticmethod
    def _contains_any(
        text: str,
        phrases: List[str]
    ) -> bool:
        """
        Checks whether text contains at least one supplied phrase.
        """
        normalized_text = str(text).lower()

        return any(
            phrase.lower() in normalized_text
            for phrase in phrases
        )

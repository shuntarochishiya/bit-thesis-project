from typing import Dict, Any, List

from state.game_state_manager import GameStateManager
from state.npc_state_manager import NPCStateManager
from memory.memory_system import MemorySystem
from memory.semantic_memory import SemanticMemorySystem
from engine.execution_logger import ExecutionLogger
from engine.fallback_manager import FallbackManager
from engine.dag_scheduler import DAGScheduler

from agents.combat_agent import CombatAgent
from agents.persuasion_agent import PersuasionAgent
from agents.dialogue_agent import DialogueAgent
from agents.exploration_agent import ExplorationAgent
from agents.narrative_agent import NarrativeGenerationAgent
from agents.tavern_agent import TavernAgent
from agents.consequence_agent import ConsequenceAgent
from agents.precondition_agent import PreconditionAgent
from agents.combat_step_agent import CombatStepAgent
from agents.persuasion_step_agent import PersuasionStepAgent
from agents.dialogue_step_agent import DialogueStepAgent
from agents.event_agent import EventAgent


class ExecutionEngine:
    """
    Dynamic execution engine.

    It executes tasks according to the DAG created from task_templates.json.
    Semantic memory retrieval and consequence evaluation are now executed
    as normal DAG nodes.
    """

    def __init__(
        self,
        game_state_manager: GameStateManager,
        memory_system: MemorySystem,
        semantic_memory_system: SemanticMemorySystem,
        execution_logger: ExecutionLogger,
        fallback_manager: FallbackManager,
        consequence_agent: ConsequenceAgent,
        combat_agent: CombatAgent,
        combat_step_agent: CombatStepAgent,
        tavern_agent: TavernAgent,
        event_agent: EventAgent,
        persuasion_agent: PersuasionAgent,
        persuasion_step_agent: PersuasionStepAgent,
        exploration_agent: ExplorationAgent,
        dialogue_agent: DialogueAgent,
        dialogue_step_agent: DialogueStepAgent,
        precondition_agent: PreconditionAgent,
        narrative_agent: NarrativeGenerationAgent,
        npc_state_manager: NPCStateManager | None = None
    ):
        self.game_state_manager = game_state_manager
        self.npc_state_manager = npc_state_manager
        self.dag_scheduler = DAGScheduler()
        self.memory_system = memory_system
        self.semantic_memory_system = semantic_memory_system
        self.execution_logger = execution_logger
        self.fallback_manager = fallback_manager
        self.consequence_agent = consequence_agent
        self.precondition_agent = precondition_agent

        self.tavern_agent = tavern_agent
        self.event_agent = event_agent
        self.combat_agent = combat_agent
        self.combat_step_agent = combat_step_agent
        self.persuasion_agent = persuasion_agent
        self.persuasion_step_agent = persuasion_step_agent
        self.exploration_agent = exploration_agent
        self.dialogue_agent = dialogue_agent
        self.dialogue_step_agent = dialogue_step_agent
        self.narrative_agent = narrative_agent

    def apply_state_updates(
        self,
        updates: Dict[str, Any],
        source: str,
        reason: str,
        snapshot_id: int
    ):
        """
        Applies state updates and rolls back if the state becomes invalid.
        """

        if not updates:
            return

        self.game_state_manager.update_state(
            updates,
            source=source,
            reason=reason
        )

        if not self.game_state_manager.validate_state():
            self.game_state_manager.rollback_to_snapshot(snapshot_id)
            raise Exception(
                f"Invalid game state detected after update from {source}. "
                f"Rolled back to snapshot {snapshot_id}."
            )

    def build_blocked_action_result(self, execution_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Creates a standard result when ConsequenceAgent blocks an action.
        """

        consequence_result = execution_context["consequence_result"]

        return {
            "success": False,
            "message": consequence_result.get(
                "reason",
                "The action was blocked by previous consequences."
            ),
            "state_updates": {}
        }

    @staticmethod
    def _build_exploration_movement_text(
        player_input: str,
        requested_location: str | None
    ) -> str:
        """Builds a short exploration description without calling an LLM."""

        text = (player_input or "").lower()
        location = (requested_location or "the area").lower()

        if location == "old forest":
            if "deeper" in text:
                return "You go deeper into the forest."
            if any(word in text for word in ("enter", "inside", "into")):
                return "You enter the old forest."
            return "You explore the old forest."

        if location == "forest road":
            if any(word in text for word in ("continue", "follow", "forward")):
                return "You continue along the forest road."
            return "You walk along the forest road."

        if location == "old ruins":
            if any(word in text for word in ("enter", "inside")):
                return "You enter the old ruins."
            return "You approach the old ruins."

        if location == "village":
            return "You make your way toward the village."

        if location == "valley":
            return "You continue into the valley."

        if location == "riverbank":
            return "You walk toward the riverbank."

        return f"You explore {location}."

    @staticmethod
    def _build_combat_response(
        target: str,
        combat_context: Dict[str, Any],
        blocked_reason: str = ""
    ) -> str:
        """Builds the final player-facing combat text without an LLM."""

        if combat_context.get("reaction_type") == "blocked":
            return blocked_reason or "The combat action was blocked."

        target_name = {
            "enemy": "enemy",
            "merchant": "merchant",
            "bartender": "bartender"
        }.get(target, target or "target")

        hit = bool(combat_context.get("hit"))
        damage = int(combat_context.get("damage") or 0)
        defeated = bool(combat_context.get("target_defeated"))
        remaining_health = combat_context.get("target_health_after")
        reaction_type = combat_context.get("reaction_type", "none")
        retaliation_damage = int(combat_context.get("retaliation_damage") or 0)

        parts: List[str] = []

        if hit:
            parts.append(f"You hit the {target_name} for {damage} damage.")
            if defeated:
                parts.append(f"The {target_name} is defeated.")
            elif remaining_health is not None:
                parts.append(
                    f"The {target_name} has {remaining_health} health remaining."
                )
        else:
            parts.append(f"You attack the {target_name}, but miss.")

        if reaction_type == "counterattack":
            parts.append(
                f"The {target_name} retaliates and deals "
                f"{retaliation_damage} damage to you."
            )
        elif reaction_type == "call_for_help":
            parts.append("The merchant recoils and calls for help.")
            parts.append("The merchant is now hostile and refuses to trust you.")
        elif reaction_type == "flee":
            parts.append("The wounded merchant panics and flees from you.")
        elif reaction_type == "turn_tavern_hostile":
            parts.append("The bartender becomes hostile and calls the tavern to action.")
            parts.append("The room turns against you.")

        return " ".join(parts)

    def execute_plan(
        self,
        plan: List[Dict[str, Any]],
        player_input: str,
        intent: str,
        target: str
    ) -> str:
        completed_tasks = {}
        execution_result = "No specific game action was executed."
        player_response: str | None = None

        execution_levels = self.dag_scheduler.build_execution_levels(plan)
        formatted_execution_levels = self.dag_scheduler.format_execution_levels(execution_levels)
        self.execution_logger.set_dag_execution_levels(formatted_execution_levels)

        snapshot_id = self.game_state_manager.create_snapshot(
            label=f"Before action | intent={intent} | target={target} | input={player_input}"
        )

        state_before_turn = self.game_state_manager.get_state()

        execution_context: Dict[str, Any] = {
            "semantic_memory_results": [],
            "precondition_results": [],
            "combat_context": {
                "hit": False,
                "hit_roll": None,
                "damage": 0,
                "target_defeated": False,
                "projected_health": None,
                "health_key": None,
                "target_health_before": None,
                "target_health_after": None,
                "attack_applied": False,
                "reaction_type": "none",
                "retaliation_damage": 0,
                "player_health_after": None,
                "reaction_applied": False
            },
            "consequence_result": {
                "allow_action": True,
                "reason": "No consequence evaluation has been performed yet.",
                "reaction_modifier": "neutral",
                "state_updates": {},
                "system_note": ""
            },
            "persuasion_context": {
                "relationship_score": 0,
                "relationship_modifier": 0,
                "reputation_score": 0,
                "reputation_modifier": 0,
                "persuasion_chance": 0,
                "roll": None,
                "persuasion_success": False
            },
            "dialogue_context": {
                "npc_attitude": "neutral",
                "attitude_modifier": 0,
                "dialogue_topic": "general_conversation",
                "dialogue_tone": "neutral",
                "dialogue_location": None,
                "dialogue_strategy": "neutral_response"
            },
            "event_context": {
                "event_probabilities": {},
                "requested_location": None,
                "selected_event": None,
                "event_plausible": True,
                "plausibility_reason": "",
                "conflict_detected": False,
                "conflict_type": "none",
                "conflict_severity": 0,
                "conflict_participants": [],
                "conflict_reason": "",
                "conflict_status": "none",
                "escalated_conflict_severity": 0,
                "conflict_resolution_hint": "none",
                "conflict_record": None,
                "final_event": None,
                "event_message": "",
                "followup_event": None
            },
            "action_blocked": False
        }

        for level in execution_levels:
            for task in level:
                task_id = task["task_id"]
                agent_type = task["agent"]
                fallback_name = task.get("fallback", "generic_fallback")

                self.execution_logger.add_executed_task(task_id, agent_type)

                try:
                    dependencies = task.get("depends_on", [])

                    for dependency in dependencies:
                        if dependency not in completed_tasks:
                            raise Exception(
                                f"Task {task_id} depends on {dependency}, "
                                f"but it was not completed."
                            )

                    game_state = self.game_state_manager.get_state()

                    if agent_type == "validation":
                        completed_tasks[task_id] = {
                            "success": True,
                            "message": "Validation completed.",
                            "state_updates": {}
                        }

                    elif agent_type == "precondition":
                        check_name = task.get("check")

                        if check_name == "validate_location":
                            result = self.precondition_agent.validate_location(
                                intent=intent,
                                target=target,
                                game_state=game_state,
                                player_input=player_input
                            )

                        elif check_name == "check_player_resources":
                            result = self.precondition_agent.check_player_resources(
                                intent=intent,
                                target=target,
                                game_state=game_state,
                                player_input=player_input
                            )

                        elif check_name == "check_target_status":
                            result = self.precondition_agent.check_target_status(
                                intent=intent,
                                target=target,
                                game_state=game_state,
                                player_input=player_input
                            )

                        else:
                            raise Exception(f"Unknown precondition check: {check_name}")

                        execution_context["precondition_results"].append(result)

                        if not result["success"]:
                            execution_context["action_blocked"] = True
                            execution_context["consequence_result"] = {
                                "allow_action": False,
                                "reason": result["message"],
                                "reaction_modifier": "blocked_by_precondition",
                                "state_updates": {},
                                "system_note": (
                                    f"Action blocked by precondition check: "
                                    f"{result.get('precondition_type')}"
                                )
                            }

                            execution_result = result["message"]

                        else:
                            execution_result = result["message"]

                        completed_tasks[task_id] = result

                    elif agent_type == "semantic_memory":
                        current_state = self.game_state_manager.get_state()

                        semantic_query = (
                            f"Player input: {player_input}. "
                            f"Intent: {intent}. "
                            f"Target: {target}. "
                            f"Current game state: {current_state}"
                        )

                        semantic_memory_results = self.semantic_memory_system.retrieve_relevant_events(
                            query=semantic_query,
                            k=5
                        )

                        execution_context["semantic_memory_results"] = semantic_memory_results
                        self.execution_logger.set_semantic_memory_results(semantic_memory_results)

                        execution_result = "Semantic memory retrieval completed."

                        completed_tasks[task_id] = {
                            "success": True,
                            "message": execution_result,
                            "state_updates": {}
                        }

                    elif agent_type == "consequence":
                        # If a precondition already blocked the action,
                        # do not overwrite the blocking reason.
                        if execution_context["action_blocked"]:
                            consequence_result = execution_context["consequence_result"]

                            self.execution_logger.set_consequence_decision(consequence_result)

                            execution_result = consequence_result["reason"]

                            completed_tasks[task_id] = {
                                "success": False,
                                "message": execution_result,
                                "state_updates": {}
                            }

                        else:
                            current_state = self.game_state_manager.get_state()

                            consequence_result = self.consequence_agent.evaluate(
                                player_input=player_input,
                                intent=intent,
                                target=target,
                                game_state=current_state,
                                relevant_memory=execution_context["semantic_memory_results"]
                            )

                            execution_context["consequence_result"] = consequence_result
                            self.execution_logger.set_consequence_decision(consequence_result)

                            if consequence_result.get("state_updates"):
                                self.apply_state_updates(
                                    updates=consequence_result["state_updates"],
                                    source="ConsequenceAgent",
                                    reason=consequence_result.get(
                                        "reason",
                                        "Consequence-based state update"
                                    ),
                                    snapshot_id=snapshot_id
                                )

                            if not consequence_result.get("allow_action", True):
                                execution_context["action_blocked"] = True
                                execution_result = consequence_result.get(
                                    "reason",
                                    "The action was blocked by ConsequenceAgent."
                                )
                            else:
                                execution_result = consequence_result.get(
                                    "system_note",
                                    "Consequence evaluation completed."
                                )

                            completed_tasks[task_id] = {
                                "success": consequence_result.get("allow_action", True),
                                "message": execution_result,
                                "state_updates": consequence_result.get("state_updates", {})
                            }

                    elif agent_type == "combat_step":
                        step_name = task.get("step")
                        combat_context = execution_context["combat_context"]

                        if step_name == "calculate_hit":
                            result = self.combat_step_agent.calculate_hit(
                                action_blocked=execution_context["action_blocked"]
                            )

                        elif step_name == "calculate_damage":
                            result = self.combat_step_agent.calculate_damage(
                                hit=combat_context["hit"],
                                action_blocked=execution_context["action_blocked"]
                            )

                        elif step_name == "check_death":
                            result = self.combat_step_agent.check_death(
                                game_state=game_state,
                                target=target,
                                hit=combat_context["hit"],
                                damage=combat_context["damage"],
                                action_blocked=execution_context["action_blocked"]
                            )

                        elif step_name == "apply_combat_result":
                            result = self.combat_step_agent.apply_combat_result(
                                game_state=game_state,
                                target=target,
                                hit=combat_context["hit"],
                                damage=combat_context["damage"],
                                target_defeated=combat_context["target_defeated"],
                                action_blocked=execution_context["action_blocked"],
                                blocked_reason=execution_context["consequence_result"].get(
                                    "reason",
                                    "The combat action was blocked."
                                )
                            )

                            self.apply_state_updates(
                                updates=result["state_updates"],
                                source="CombatStepAgent",
                                reason=result["message"],
                                snapshot_id=snapshot_id
                            )

                        elif step_name == "calculate_target_reaction":
                            result = self.combat_step_agent.calculate_target_reaction(
                                game_state=game_state,
                                target=target,
                                hit=combat_context["hit"],
                                target_defeated=combat_context["target_defeated"],
                                action_blocked=execution_context["action_blocked"]
                            )

                        elif step_name == "apply_target_reaction":
                            result = self.combat_step_agent.apply_target_reaction(
                                game_state=game_state,
                                target=target,
                                reaction_type=combat_context["reaction_type"],
                                retaliation_damage=combat_context["retaliation_damage"],
                                action_blocked=execution_context["action_blocked"]
                            )

                            self.apply_state_updates(
                                updates=result["state_updates"],
                                source="CombatStepAgent",
                                reason=result["message"],
                                snapshot_id=snapshot_id
                            )

                        else:
                            raise Exception(f"Unknown combat step: {step_name}")

                        if "data" in result:
                            combat_context.update(result["data"])

                        if step_name == "apply_target_reaction":
                            player_response = self._build_combat_response(
                                target=target,
                                combat_context=combat_context,
                                blocked_reason=execution_context["consequence_result"].get(
                                    "reason",
                                    "The combat action was blocked."
                                )
                            )

                        execution_result = result["message"]
                        completed_tasks[task_id] = result

                    elif agent_type == "combat":
                        if execution_context["action_blocked"]:
                            result = self.build_blocked_action_result(execution_context)
                        else:
                            result = self.combat_agent.execute(game_state, target=target)

                        self.apply_state_updates(
                            updates=result["state_updates"],
                            source="CombatAgent",
                            reason=result["message"],
                            snapshot_id=snapshot_id
                        )

                        execution_result = result["message"]
                        completed_tasks[task_id] = result

                    elif agent_type == "persuasion_step":
                        step_name = task.get("step")
                        persuasion_context = execution_context["persuasion_context"]

                        if step_name == "analyze_relationship":
                            result = self.persuasion_step_agent.analyze_relationship(
                                game_state=game_state,
                                target=target,
                                relevant_memory=execution_context["semantic_memory_results"],
                                action_blocked=execution_context["action_blocked"]
                            )

                        elif step_name == "analyze_reputation":
                            result = self.persuasion_step_agent.analyze_reputation(
                                game_state=game_state,
                                relevant_memory=execution_context["semantic_memory_results"],
                                action_blocked=execution_context["action_blocked"]
                            )

                        elif step_name == "calculate_persuasion_chance":
                            result = self.persuasion_step_agent.calculate_persuasion_chance(
                                relationship_score=persuasion_context["relationship_score"],
                                reputation_score=persuasion_context["reputation_score"],
                                action_blocked=execution_context["action_blocked"]
                            )

                        elif step_name == "apply_persuasion_result":
                            result = self.persuasion_step_agent.apply_persuasion_result(
                                game_state=game_state,
                                target=target,
                                persuasion_success=persuasion_context["persuasion_success"],
                                persuasion_chance=persuasion_context["persuasion_chance"],
                                roll=persuasion_context["roll"],
                                player_input=player_input,
                                action_blocked=execution_context["action_blocked"],
                                blocked_reason=execution_context["consequence_result"].get(
                                    "reason",
                                    "The persuasion action was blocked."
                                )
                            )

                            self.apply_state_updates(
                                updates=result["state_updates"],
                                source="PersuasionStepAgent",
                                reason=result["message"],
                                snapshot_id=snapshot_id
                            )

                        else:
                            raise Exception(f"Unknown persuasion step: {step_name}")

                        if "data" in result:
                            persuasion_context.update(result["data"])

                        execution_result = result["message"]
                        completed_tasks[task_id] = result

                    elif agent_type == "persuasion":
                        if execution_context["action_blocked"]:
                            result = self.build_blocked_action_result(execution_context)
                        else:
                            result = self.persuasion_agent.execute(game_state)

                        self.apply_state_updates(
                            updates=result["state_updates"],
                            source="PersuasionAgent",
                            reason=result["message"],
                            snapshot_id=snapshot_id
                        )

                        execution_result = result["message"]
                        completed_tasks[task_id] = result

                    elif agent_type == "dialogue_step":
                        step_name = task.get("step")
                        dialogue_context = execution_context["dialogue_context"]

                        if step_name == "analyze_npc_attitude":
                            result = self.dialogue_step_agent.analyze_npc_attitude(
                                game_state=game_state,
                                target=target,
                                relevant_memory=execution_context["semantic_memory_results"],
                                action_blocked=execution_context["action_blocked"]
                            )

                        elif step_name == "analyze_dialogue_context":
                            result = self.dialogue_step_agent.analyze_dialogue_context(
                                game_state=game_state,
                                target=target,
                                player_input=player_input,
                                relevant_memory=execution_context["semantic_memory_results"],
                                action_blocked=execution_context["action_blocked"]
                            )

                        elif step_name == "choose_dialogue_strategy":
                            result = self.dialogue_step_agent.choose_dialogue_strategy(
                                npc_attitude=dialogue_context["npc_attitude"],
                                dialogue_topic=dialogue_context["dialogue_topic"],
                                dialogue_tone=dialogue_context["dialogue_tone"],
                                target=target,
                                action_blocked=execution_context["action_blocked"]
                            )

                        elif step_name == "apply_dialogue_result":
                            result = self.dialogue_step_agent.apply_dialogue_result(
                                game_state=game_state,
                                target=target,
                                npc_attitude=dialogue_context["npc_attitude"],
                                dialogue_topic=dialogue_context["dialogue_topic"],
                                dialogue_tone=dialogue_context["dialogue_tone"],
                                dialogue_strategy=dialogue_context["dialogue_strategy"],
                                action_blocked=execution_context["action_blocked"],
                                blocked_reason=execution_context["consequence_result"].get(
                                    "reason",
                                    "The dialogue action was blocked."
                                )
                            )

                            self.apply_state_updates(
                                updates=result["state_updates"],
                                source="DialogueStepAgent",
                                reason=result["message"],
                                snapshot_id=snapshot_id
                            )

                        else:
                            raise Exception(f"Unknown dialogue step: {step_name}")

                        if "data" in result:
                            dialogue_context.update(result["data"])

                        execution_result = result["message"]
                        completed_tasks[task_id] = result

                    elif agent_type == "dialogue":
                        if execution_context["action_blocked"]:
                            result = self.build_blocked_action_result(execution_context)
                        else:
                            result = self.dialogue_agent.execute(game_state, target=target)

                        self.apply_state_updates(
                            updates=result["state_updates"],
                            source="DialogueAgent",
                            reason=result["message"],
                            snapshot_id=snapshot_id
                        )

                        execution_result = result["message"]
                        completed_tasks[task_id] = result

                    elif agent_type == "event_step":
                        step_name = task.get("step")
                        event_context = execution_context["event_context"]

                        if step_name == "calculate_event_probabilities":
                            result = self.event_agent.calculate_event_probabilities(
                                game_state=game_state,
                                player_input=player_input,
                                relevant_memory=execution_context["semantic_memory_results"]
                            )

                        elif step_name == "sample_event":
                            result = self.event_agent.sample_event(
                                event_probabilities=event_context["event_probabilities"]
                            )

                        elif step_name == "check_event_plausibility":
                            result = self.event_agent.check_event_plausibility(
                                selected_event=event_context["selected_event"],
                                requested_location=event_context["requested_location"],
                                game_state=game_state,
                                relevant_memory=execution_context["semantic_memory_results"]
                            )

                        elif step_name == "detect_conflict":
                            result = self.event_agent.detect_conflict(
                                selected_event=event_context["selected_event"],
                                event_plausible=event_context["event_plausible"],
                                game_state=game_state,
                                player_input=player_input,
                                relevant_memory=execution_context["semantic_memory_results"]
                            )

                        elif step_name == "escalate_conflict":
                            result = self.event_agent.escalate_conflict(
                                conflict_detected=event_context["conflict_detected"],
                                conflict_type=event_context["conflict_type"],
                                conflict_severity=event_context["conflict_severity"],
                                conflict_participants=event_context["conflict_participants"],
                                conflict_reason=event_context["conflict_reason"],
                                game_state=game_state,
                                player_input=player_input
                            )

                        elif step_name == "apply_event_result":
                            result = self.event_agent.apply_event_result(
                                selected_event=event_context["selected_event"],
                                event_plausible=event_context["event_plausible"],
                                requested_location=event_context["requested_location"],
                                plausibility_reason=event_context["plausibility_reason"],
                                game_state=game_state,
                                conflict_detected=event_context["conflict_detected"],
                                conflict_record=event_context["conflict_record"]
                            )

                            self.apply_state_updates(
                                updates=result["state_updates"],
                                source="EventAgent",
                                reason=result["message"],
                                snapshot_id=snapshot_id
                            )

                            event_context["event_message"] = result["message"]

                            movement_message = self._build_exploration_movement_text(
                                player_input=player_input,
                                requested_location=event_context["requested_location"]
                            )

                            final_event = result.get("data", {}).get(
                                "final_event",
                                "nothing_special"
                            )

                            if final_event == "nothing_special":
                                player_response = movement_message
                            else:
                                player_response = (
                                    f"{movement_message} {result['message']}"
                                ).strip()

                        elif step_name == "generate_followup_event":
                            result = self.event_agent.generate_followup_event(
                                final_event=event_context["final_event"],
                                conflict_detected=event_context["conflict_detected"],
                                conflict_type=event_context["conflict_type"],
                                conflict_status=event_context["conflict_status"],
                                conflict_severity=event_context["escalated_conflict_severity"],
                                game_state=game_state
                            )

                            self.apply_state_updates(
                                updates=result["state_updates"],
                                source="EventAgent",
                                reason=result["message"],
                                snapshot_id=snapshot_id
                            )

                        else:
                            raise Exception(f"Unknown event step: {step_name}")

                        if "data" in result:
                            event_context.update(result["data"])

                        execution_result = result["message"]
                        completed_tasks[task_id] = result

                    elif agent_type == "exploration":
                        if execution_context["action_blocked"]:
                            result = self.build_blocked_action_result(execution_context)
                        else:
                            result = self.exploration_agent.execute(
                                game_state,
                                player_input=player_input
                            )

                        self.apply_state_updates(
                            updates=result["state_updates"],
                            source="ExplorationAgent",
                            reason=result["message"],
                            snapshot_id=snapshot_id
                        )

                        execution_result = result["message"]
                        completed_tasks[task_id] = result

                    elif agent_type == "tavern":
                        if execution_context["action_blocked"]:
                            result = self.build_blocked_action_result(execution_context)
                        else:
                            result = self.tavern_agent.execute(
                                game_state,
                                player_input=player_input
                            )

                        self.apply_state_updates(
                            updates=result["state_updates"],
                            source="TavernAgent",
                            reason=result["message"],
                            snapshot_id=snapshot_id
                        )

                        execution_result = result["message"]
                        completed_tasks[task_id] = result

                    elif agent_type == "memory":
                        state_after_action = self.game_state_manager.get_state()

                        memory_item = self.memory_system.add_event(
                            player_input=player_input,
                            intent=intent,
                            target=target,
                            system_result=player_response or execution_result,
                            state_before=state_before_turn,
                            state_after=state_after_action
                        )

                        self.semantic_memory_system.add_event(memory_item)

                        memory_event_for_log = (
                            f"Player input: {player_input}. "
                            f"Intent: {intent}. "
                            f"Target: {target}. "
                            f"System result: {player_response or execution_result}"
                        )

                        self.execution_logger.set_memory_event(memory_event_for_log)

                        completed_tasks[task_id] = {
                            "success": True,
                            "message": "Memory and semantic memory updated.",
                            "state_updates": {}
                        }

                    elif agent_type == "narrative":
                        final_state = self.game_state_manager.get_state()
                        recent_memory = self.memory_system.retrieve_recent_events()

                        combined_memory = (
                            recent_memory
                            + execution_context["semantic_memory_results"]
                            + [
                                f"Precondition results: {execution_context['precondition_results']}",
                                f"Consequence decision: {execution_context['consequence_result']}",
                                f"Event context: {execution_context['event_context']}"
                            ]
                        )

                        narrative = self.narrative_agent.generate(
                            player_input=player_input,
                            intent=intent,
                            target=target,
                            game_state=final_state,
                            execution_result=execution_result,
                            memory_events=combined_memory
                        )

                        self.execution_logger.finish_turn(
                            state_after=final_state,
                            final_response=narrative
                        )

                        completed_tasks[task_id] = narrative

                        return narrative

                    else:
                        raise Exception(f"Unknown agent type: {agent_type}")

                except Exception as error:
                    error_message = str(error)

                    self.execution_logger.add_error(task_id, error_message)
                    self.execution_logger.add_fallback(task_id, fallback_name)

                    fallback_result = self.fallback_manager.handle_fallback(
                        fallback_name=fallback_name,
                        task_id=task_id,
                        agent_type=agent_type,
                        game_state=self.game_state_manager.get_state(),
                        error_message=error_message
                    )

                    self.apply_state_updates(
                        updates=fallback_result["state_updates"],
                        source="FallbackManager",
                        reason=fallback_result["message"],
                        snapshot_id=snapshot_id
                    )

                    execution_result = fallback_result["message"]
                    completed_tasks[task_id] = fallback_result

                    if agent_type == "narrative":
                        final_state = self.game_state_manager.get_state()

                        self.execution_logger.finish_turn(
                            state_after=final_state,
                            final_response=execution_result
                        )

                        return execution_result

        final_state = self.game_state_manager.get_state()

        final_response = player_response or execution_result

        self.execution_logger.finish_turn(
            state_after=final_state,
            final_response=final_response
        )

        return final_response

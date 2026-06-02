from typing import Dict, Any, List

from state.game_state_manager import GameStateManager
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
        tavern_agent: TavernAgent,
        persuasion_agent: PersuasionAgent,
        exploration_agent: ExplorationAgent,
        dialogue_agent: DialogueAgent,
        narrative_agent: NarrativeGenerationAgent
    ):
        self.game_state_manager = game_state_manager
        self.dag_scheduler = DAGScheduler()
        self.memory_system = memory_system
        self.semantic_memory_system = semantic_memory_system
        self.execution_logger = execution_logger
        self.fallback_manager = fallback_manager
        self.consequence_agent = consequence_agent

        self.tavern_agent = tavern_agent
        self.combat_agent = combat_agent
        self.persuasion_agent = persuasion_agent
        self.exploration_agent = exploration_agent
        self.dialogue_agent = dialogue_agent
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

    def execute_plan(
        self,
        plan: List[Dict[str, Any]],
        player_input: str,
        intent: str,
        target: str
    ) -> str:
        completed_tasks = {}
        execution_result = "No specific game action was executed."

        execution_levels = self.dag_scheduler.build_execution_levels(plan)
        formatted_execution_levels = self.dag_scheduler.format_execution_levels(execution_levels)
        self.execution_logger.set_dag_execution_levels(formatted_execution_levels)

        snapshot_id = self.game_state_manager.create_snapshot(
            label=f"Before action | intent={intent} | target={target} | input={player_input}"
        )

        state_before_turn = self.game_state_manager.get_state()

        execution_context: Dict[str, Any] = {
            "semantic_memory_results": [],
            "consequence_result": {
                "allow_action": True,
                "reason": "No consequence evaluation has been performed yet.",
                "reaction_modifier": "neutral",
                "state_updates": {},
                "system_note": ""
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
                            system_result=execution_result,
                            state_before=state_before_turn,
                            state_after=state_after_action
                        )

                        self.semantic_memory_system.add_event(memory_item)

                        memory_event_for_log = (
                            f"Player input: {player_input}. "
                            f"Intent: {intent}. "
                            f"Target: {target}. "
                            f"System result: {execution_result}"
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
                                f"Consequence decision: "
                                f"{execution_context['consequence_result']}"
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

        self.execution_logger.finish_turn(
            state_after=final_state,
            final_response=execution_result
        )

        return execution_result

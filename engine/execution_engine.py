from typing import Dict, Any, List

from state.game_state_manager import GameStateManager
from memory.memory_system import MemorySystem
from engine.execution_logger import ExecutionLogger
from engine.fallback_manager import FallbackManager

from agents.combat_agent import CombatAgent
from agents.persuasion_agent import PersuasionAgent
from agents.dialogue_agent import DialogueAgent
from agents.exploration_agent import ExplorationAgent
from agents.narrative_agent import NarrativeGenerationAgent
from agents.tavern_agent import TavernAgent


class ExecutionEngine:
    """
    Simplified dynamic execution engine.
    It executes tasks according to the plan created by TaskPlanner.
    """

    def __init__(
        self,
        game_state_manager: GameStateManager,
        memory_system: MemorySystem,
        execution_logger: ExecutionLogger,
        fallback_manager: FallbackManager,
        combat_agent: CombatAgent,
        tavern_agent: TavernAgent,
        persuasion_agent: PersuasionAgent,
        exploration_agent: ExplorationAgent,
        dialogue_agent: DialogueAgent,
        narrative_agent: NarrativeGenerationAgent
    ):
        self.game_state_manager = game_state_manager
        self.memory_system = memory_system
        self.execution_logger = execution_logger
        self.fallback_manager = fallback_manager
        self.tavern_agent = tavern_agent
        self.combat_agent = combat_agent
        self.persuasion_agent = persuasion_agent
        self.exploration_agent = exploration_agent
        self.narrative_agent = narrative_agent
        self.dialogue_agent = dialogue_agent

    def execute_plan(self, plan: List[Dict[str, Any]], player_input: str, intent: str, target: str) -> str:
        completed_tasks = {}
        execution_result = "No specific game action was executed."

        state_before_turn = self.game_state_manager.get_state()

        for task in plan:
            task_id = task["task_id"]
            agent_type = task["agent"]
            fallback_name = task.get("fallback", "generic_fallback")

            self.execution_logger.add_executed_task(task_id, agent_type)

            try:
                # Check dependencies
                dependencies = task["depends_on"]
                for dependency in dependencies:
                    if dependency not in completed_tasks:
                        raise Exception(
                            f"Task {task_id} depends on {dependency}, but it was not completed."
                        )

                game_state = self.game_state_manager.get_state()

                if agent_type == "validation":
                    completed_tasks[task_id] = "Validation completed."

                elif agent_type == "combat":
                    result = self.combat_agent.execute(game_state, target=target)
                    self.game_state_manager.update_state(result["state_updates"])
                    execution_result = result["message"]
                    completed_tasks[task_id] = result

                elif agent_type == "persuasion":
                    result = self.persuasion_agent.execute(game_state)
                    self.game_state_manager.update_state(result["state_updates"])
                    execution_result = result["message"]
                    completed_tasks[task_id] = result

                elif agent_type == "dialogue":
                    result = self.dialogue_agent.execute(game_state, target=target)
                    self.game_state_manager.update_state(result["state_updates"])
                    execution_result = result["message"]
                    completed_tasks[task_id] = result

                elif agent_type == "exploration":
                    result = self.exploration_agent.execute(game_state, player_input=player_input)
                    self.game_state_manager.update_state(result["state_updates"])
                    execution_result = result["message"]
                    completed_tasks[task_id] = result

                elif agent_type == "tavern":
                    result = self.tavern_agent.execute(game_state, player_input=player_input)
                    self.game_state_manager.update_state(result["state_updates"])
                    execution_result = result["message"]
                    completed_tasks[task_id] = result

                elif agent_type == "memory":
                    state_after_action = self.game_state_manager.get_state()

                    self.memory_system.add_event(
                        player_input=player_input,
                        intent=intent,
                        target=target,
                        system_result=execution_result,
                        state_before=state_before_turn,
                        state_after=state_after_action
                    )

                    memory_event_for_log = (
                        f"Player input: {player_input}. "
                        f"Intent: {intent}. "
                        f"Target: {target}. "
                        f"System result: {execution_result}"
                    )

                    self.execution_logger.set_memory_event(memory_event_for_log)
                    completed_tasks[task_id] = "Memory updated."

                elif agent_type == "narrative":
                    final_state = self.game_state_manager.get_state()
                    recent_memory = self.memory_system.retrieve_recent_events()

                    narrative = self.narrative_agent.generate(
                        player_input=player_input,
                        intent=intent,
                        target=target,
                        game_state=final_state,
                        execution_result=execution_result,
                        memory_events=recent_memory
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

                self.game_state_manager.update_state(fallback_result["state_updates"])
                execution_result = fallback_result["message"]
                completed_tasks[task_id] = fallback_result

                # If narrative generation fails, return a basic fallback response immediately
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


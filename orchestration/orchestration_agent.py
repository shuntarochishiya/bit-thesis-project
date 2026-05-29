from config import create_llm, TASK_TEMPLATE_PATH, MEMORY_PATH

from state.game_state_manager import GameStateManager
from state.context_manager import ContextManager
from memory.memory_system import MemorySystem

from agents.intent_agent import IntentRecognitionAgent
from agents.primitive_agents import AttributeCalculationAgent, ValidationAgent
from agents.combat_agent import CombatAgent
from agents.persuasion_agent import PersuasionAgent
from agents.dialogue_agent import DialogueAgent
from agents.exploration_agent import ExplorationAgent
from agents.narrative_agent import NarrativeGenerationAgent
from agents.tavern_agent import TavernAgent

from engine.task_planner import TaskPlanner
from engine.execution_engine import ExecutionEngine
from engine.execution_logger import ExecutionLogger
from engine.fallback_manager import FallbackManager


class OrchestrationAgent:
    """
    Top-level agent.
    It coordinates the whole game flow.
    """

    def __init__(self):
        self.llm = create_llm()

        self.game_state_manager = GameStateManager()
        self.context_manager = ContextManager()
        self.memory_system = MemorySystem(memory_path=MEMORY_PATH)
        self.execution_logger = ExecutionLogger()
        self.fallback_manager = FallbackManager()

        self.intent_agent = IntentRecognitionAgent()
        self.task_planner = TaskPlanner(template_path=TASK_TEMPLATE_PATH)

        self.attribute_agent = AttributeCalculationAgent()
        self.validation_agent = ValidationAgent()

        self.combat_agent = CombatAgent(self.attribute_agent, self.validation_agent)
        self.persuasion_agent = PersuasionAgent(self.attribute_agent, self.validation_agent)
        self.exploration_agent = ExplorationAgent()
        self.dialogue_agent = DialogueAgent()
        self.tavern_agent = TavernAgent()

        self.narrative_agent = NarrativeGenerationAgent(self.llm)

        self.execution_engine = ExecutionEngine(
            game_state_manager=self.game_state_manager,
            memory_system=self.memory_system,
            execution_logger=self.execution_logger,
            fallback_manager=self.fallback_manager,
            combat_agent=self.combat_agent,
            persuasion_agent=self.persuasion_agent,
            exploration_agent=self.exploration_agent,
            narrative_agent=self.narrative_agent,
            dialogue_agent=self.dialogue_agent,
            tavern_agent=self.tavern_agent,
        )

    def process_player_input(self, player_input: str) -> str:
        state_before = self.game_state_manager.get_state()
        current_context = self.context_manager.get_context()

        self.execution_logger.start_turn(player_input, state_before)

        intent_data = self.intent_agent.recognize_intent(
            player_input=player_input,
            context=current_context
        )

        intent = intent_data["intent"]
        target = intent_data["target"]

        target = self.context_manager.resolve_target_from_context(target)

        text = player_input.lower()

        tavern_action_words = [
            "drink", "ale", "beer", "wine", "mead", "liquor",
            "glass", "cup", "bottle",
            "room", "rent", "food", "meal", "rest", "sleep",
            "order", "buy", "purchase",
            "cheap", "simple", "regular", "good", "fine", "finest",
            "expensive", "best", "premium", "royal",
            "rumor", "rumour", "information", "news",
            "odd", "strange", "weird", "nearby", "recently",
            "anything", "happened", "details", "more details",
            "explain", "tell me more", "what happened"
        ]

        if (
            intent == "general_action"
            and current_context.get("active_location") == "tavern"
            and any(word in text for word in tavern_action_words)
        ):
            intent = "tavern_action"
            target = "bartender"

        self.execution_logger.set_intent(intent, target)

        plan = self.task_planner.build_plan(intent)
        self.execution_logger.set_dag(plan)

        print("\n[DEBUG] Recognized intent:", intent)
        print("[DEBUG] Target:", target)
        print("[DEBUG] Active context:", current_context)
        print("[DEBUG] Execution DAG:")
        for task in plan:
            print(f"  - {task['task_id']} depends on {task['depends_on']}")

        response = self.execution_engine.execute_plan(
            plan=plan,
            player_input=player_input,
            intent=intent,
            target=target
        )

        state_after = self.game_state_manager.get_state()

        self.context_manager.update_after_turn(
            player_input=player_input,
            intent=intent,
            target=target,
            system_result=response,
            game_state=state_after
        )

        return response

    def show_state(self):
        self.game_state_manager.display_state()

    def show_context(self):
        self.context_manager.display_context()

    def show_memory(self):
        self.memory_system.display_memory()

    def clear_memory(self):
        self.memory_system.clear_memory()

    def show_last_log(self):
        self.execution_logger.display_last_log()

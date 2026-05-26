import random
import json
from typing import Dict, Any, List
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage


# =========================
# 1. LOCAL LLM CONFIGURATION
# =========================

llm = ChatOllama(
    model="mistral",  # если у тебя llama3.2:1b, замени здесь
    temperature=0.7
)


# =========================
# 2. GAME STATE MANAGER AGENT
# =========================

class GameStateManager:
    """
    This agent manages the current game state.
    In the supervisor's architecture, this corresponds to the Game State Manager Agent.
    """

    def __init__(self):
        self.state = {
            "location": "old forest",
            "player_health": 100,
            "enemy_health": 60,
            "gold": 50,
            "inventory": ["small knife", "map"],
            "relationship_with_merchant": 50,
            "current_enemy": "forest goblin",
            "world_mood": "mysterious"
        }

    def get_state(self) -> Dict[str, Any]:
        return self.state.copy()

    def update_state(self, updates: Dict[str, Any]):
        for key, value in updates.items():
            self.state[key] = value

    def display_state(self):
        print("\n--- Current Game State ---")
        for key, value in self.state.items():
            print(f"{key}: {value}")
        print("--------------------------\n")


# =========================
# 3. MEMORY SYSTEM
# =========================

class MemorySystem:
    """
    Simplified memory system.
    It stores important player actions and game events.
    Later this can be replaced with vector storage.
    """

    def __init__(self):
        self.events: List[str] = []

    def add_event(self, event: str):
        self.events.append(event)

    def retrieve_recent_events(self, limit: int = 5) -> List[str]:
        return self.events[-limit:]

    def search_memory(self, keyword: str) -> List[str]:
        return [event for event in self.events if keyword.lower() in event.lower()]


# =========================
# 4. INTENT RECOGNITION AGENT
# =========================

class IntentRecognitionAgent:
    """
    This agent recognizes the player's intention.
    For the MVP, we use simple rules instead of an expensive LLM call.
    """

    def recognize_intent(self, player_input: str) -> Dict[str, Any]:
        text = player_input.lower()

        if any(word in text for word in ["attack", "hit", "fight", "удар", "атак", "сраж"]):
            return {
                "intent": "combat_action",
                "target": "enemy"
            }

        if any(word in text for word in ["persuade", "convince", "discount", "merchant", "убед", "скид", "торгов"]):
            return {
                "intent": "persuasion_action",
                "target": "merchant"
            }

        if any(word in text for word in ["look", "explore", "осмотр", "смотр", "исслед"]):
            return {
                "intent": "exploration_action",
                "target": "environment"
            }

        return {
            "intent": "general_action",
            "target": "unknown"
        }


# =========================
# 5. PRIMITIVE OPERATION AGENTS
# =========================

class AttributeCalculationAgent:
    """
    Primitive agent.
    Calculates basic values such as hit chance and damage.
    """

    def calculate_hit(self) -> bool:
        hit_chance = random.randint(1, 100)
        return hit_chance <= 75

    def calculate_damage(self) -> int:
        return random.randint(8, 20)

    def calculate_persuasion_success(self, relationship_score: int) -> bool:
        base_chance = 40
        bonus = relationship_score // 2
        final_chance = min(base_chance + bonus, 90)
        roll = random.randint(1, 100)
        return roll <= final_chance


class ValidationAgent:
    """
    Primitive agent.
    Checks whether the player's action is possible.
    """

    def validate_combat(self, game_state: Dict[str, Any]) -> bool:
        return game_state["enemy_health"] > 0

    def validate_persuasion(self, game_state: Dict[str, Any]) -> bool:
        return game_state["relationship_with_merchant"] > 0


# =========================
# 6. COMPOSITE AGENTS
# =========================

class CombatAgent:
    """
    Composite agent.
    It uses primitive agents to process combat.
    """

    def __init__(self, attribute_agent: AttributeCalculationAgent, validation_agent: ValidationAgent):
        self.attribute_agent = attribute_agent
        self.validation_agent = validation_agent

    def execute(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        if not self.validation_agent.validate_combat(game_state):
            return {
                "success": False,
                "message": "There is no enemy to attack.",
                "state_updates": {}
            }

        hit = self.attribute_agent.calculate_hit()

        if not hit:
            return {
                "success": True,
                "message": "The player attacks, but misses the enemy.",
                "state_updates": {}
            }

        damage = self.attribute_agent.calculate_damage()
        new_enemy_health = max(game_state["enemy_health"] - damage, 0)

        return {
            "success": True,
            "message": f"The player hits the enemy and deals {damage} damage.",
            "state_updates": {
                "enemy_health": new_enemy_health
            }
        }


class PersuasionAgent:
    """
    Composite agent.
    It processes persuasion attempts with an NPC.
    """

    def __init__(self, attribute_agent: AttributeCalculationAgent, validation_agent: ValidationAgent):
        self.attribute_agent = attribute_agent
        self.validation_agent = validation_agent

    def execute(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        if not self.validation_agent.validate_persuasion(game_state):
            return {
                "success": False,
                "message": "The merchant refuses to listen.",
                "state_updates": {}
            }

        relationship = game_state["relationship_with_merchant"]
        success = self.attribute_agent.calculate_persuasion_success(relationship)

        if success:
            return {
                "success": True,
                "message": "The player successfully persuades the merchant to offer a discount.",
                "state_updates": {
                    "gold": game_state["gold"] + 10,
                    "relationship_with_merchant": min(relationship + 5, 100)
                }
            }

        return {
            "success": True,
            "message": "The persuasion attempt fails. The merchant remains unconvinced.",
            "state_updates": {
                "relationship_with_merchant": max(relationship - 5, 0)
            }
        }


class ExplorationAgent:
    """
    Composite agent.
    It handles exploration actions.
    """

    def execute(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        possible_events = [
            "The player finds old footprints near the trees.",
            "The player discovers a hidden path covered by leaves.",
            "The player hears strange sounds deeper in the forest.",
            "The player finds a small silver coin on the ground."
        ]

        event = random.choice(possible_events)

        updates = {}

        if "coin" in event:
            updates["gold"] = game_state["gold"] + 1

        return {
            "success": True,
            "message": event,
            "state_updates": updates
        }


# =========================
# 7. TASK PLANNER / DAG BUILDER
# =========================

class TaskPlanner:
    """
    This component creates a simplified execution DAG.

    In this version, task plans are not hardcoded.
    They are loaded from an external declarative JSON template file.
    """

    def __init__(self, template_path: str = "task_templates.json"):
        self.template_path = template_path
        self.templates = self.load_templates()

    def load_templates(self) -> Dict[str, Any]:
        try:
            with open(self.template_path, "r", encoding="utf-8") as file:
                return json.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Task template file '{self.template_path}' was not found."
            )
        except json.JSONDecodeError:
            raise ValueError(
                f"Task template file '{self.template_path}' contains invalid JSON."
            )

    def build_plan(self, intent: str) -> List[Dict[str, Any]]:
        if intent in self.templates:
            return self.templates[intent]

        return self.templates.get("general_action", [])

# =========================
# 8. NARRATIVE GENERATION AGENT
# =========================

class NarrativeGenerationAgent:
    """
    This agent uses the local LLM through Ollama.
    It generates the final text shown to the player.
    The final response is forced to be in English.
    """

    def __init__(self, llm):
        self.llm = llm

    def contains_cyrillic(self, text: str) -> bool:
        """
        Checks whether the generated text contains Russian/Cyrillic characters.
        """
        return any('\u0400' <= char <= '\u04FF' for char in text)

    def force_english(self, text: str) -> str:
        """
        If the model produces Russian text, this function asks it to rewrite the answer in English only.
        """
        response = self.llm.invoke([
            SystemMessage(content="""
You are a strict translation and rewriting assistant.

Your task:
Rewrite the given text in English only.

Rules:
1. Use English only.
2. Do not use Russian words.
3. Do not explain anything.
4. Return only the rewritten fantasy RPG narration.
"""),
            HumanMessage(content=f"""
Rewrite this text in English only:

{text}
""")
        ])

        return response.content

    def generate(
        self,
        player_input: str,
        intent: str,
        game_state: Dict[str, Any],
        execution_result: str,
        memory_events: List[str]
    ) -> str:

        system_prompt = f"""
You are an English-language fantasy RPG narrator.

IMPORTANT LANGUAGE RULE:
You must always answer in English only.
Never answer in Russian.
Never use Cyrillic characters.
Even if the player writes in another language, your final answer must be in English.

Your task is to generate a short, atmospheric and logical response to the player.

Current game state:
- Location: {game_state["location"]}
- Player health: {game_state["player_health"]}
- Enemy health: {game_state["enemy_health"]}
- Gold: {game_state["gold"]}
- Inventory: {game_state["inventory"]}
- Relationship with merchant: {game_state["relationship_with_merchant"]}
- World mood: {game_state["world_mood"]}

Recognized player intent:
{intent}

Result of internal game system:
{execution_result}

Recent memory:
{memory_events}

Output rules:
1. Write in English only.
2. Do not use Russian.
3. Do not use Cyrillic characters.
4. Do not contradict the game state.
5. Keep the answer short.
6. Make the story feel like a fantasy role-playing game.
"""

        response = self.llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""
Player input:
{player_input}

Generate the final game narration in English only.
""")
        ])

        final_text = response.content

        # Safety check: if the model still answers in Russian, rewrite it in English
        if self.contains_cyrillic(final_text):
            final_text = self.force_english(final_text)

        return final_text

class ExecutionLogger:
    """
    This class records the internal execution process of the system.
    It is useful for debugging and for demonstrating that the system uses
    a structured agent-based execution pipeline instead of a direct chatbot call.
    """

    def __init__(self):
        self.logs = []

    def start_turn(self, player_input: str, state_before: Dict[str, Any]):
        log_entry = {
            "player_input": player_input,
            "recognized_intent": None,
            "execution_dag": [],
            "executed_tasks": [],
            "state_before": state_before,
            "state_after": None,
            "state_changes": {},
            "memory_event": None,
            "final_response": None
        }

        self.logs.append(log_entry)

    def set_intent(self, intent: str):
        self.logs[-1]["recognized_intent"] = intent

    def set_dag(self, plan: List[Dict[str, Any]]):
        self.logs[-1]["execution_dag"] = [
            {
                "task_id": task["task_id"],
                "depends_on": task["depends_on"],
                "agent": task["agent"],
                "fallback": task.get("fallback")
            }
            for task in plan
        ]

    def add_executed_task(self, task_id: str, agent_type: str):
        self.logs[-1]["executed_tasks"].append({
            "task_id": task_id,
            "agent": agent_type
        })

    def set_memory_event(self, memory_event: str):
        self.logs[-1]["memory_event"] = memory_event

    def finish_turn(self, state_after: Dict[str, Any], final_response: str):
        self.logs[-1]["state_after"] = state_after
        self.logs[-1]["final_response"] = final_response
        self.logs[-1]["state_changes"] = self.calculate_state_changes(
            self.logs[-1]["state_before"],
            state_after
        )

    def calculate_state_changes(
        self,
        state_before: Dict[str, Any],
        state_after: Dict[str, Any]
    ) -> Dict[str, Any]:

        changes = {}

        for key in state_after:
            before_value = state_before.get(key)
            after_value = state_after.get(key)

            if before_value != after_value:
                changes[key] = {
                    "before": before_value,
                    "after": after_value
                }

        return changes

    def get_last_log(self) -> Dict[str, Any]:
        if not self.logs:
            return {}

        return self.logs[-1]

    def display_last_log(self):
        if not self.logs:
            print("No execution logs available.")
            return

        log = self.logs[-1]

        print("\n--- Execution Log ---")
        print(f"Player input: {log['player_input']}")
        print(f"Recognized intent: {log['recognized_intent']}")

        print("\nExecution DAG:")
        for task in log["execution_dag"]:
            print(
                f"- {task['task_id']} | agent: {task['agent']} | "
                f"depends on: {task['depends_on']} | fallback: {task['fallback']}"
            )

        print("\nExecuted tasks:")
        for task in log["executed_tasks"]:
            print(f"- {task['task_id']} by {task['agent']}")

        print("\nState changes:")
        if log["state_changes"]:
            for key, value in log["state_changes"].items():
                print(f"- {key}: {value['before']} -> {value['after']}")
        else:
            print("- No state changes")

        print(f"\nMemory event: {log['memory_event']}")
        print(f"\nFinal response: {log['final_response']}")
        print("---------------------\n")

# =========================
# 9. EXECUTION ENGINE
# =========================

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
        combat_agent: CombatAgent,
        persuasion_agent: PersuasionAgent,
        exploration_agent: ExplorationAgent,
        narrative_agent: NarrativeGenerationAgent
    ):
        self.game_state_manager = game_state_manager
        self.memory_system = memory_system
        self.execution_logger = execution_logger
        self.combat_agent = combat_agent
        self.persuasion_agent = persuasion_agent
        self.exploration_agent = exploration_agent
        self.narrative_agent = narrative_agent

    def execute_plan(self, plan: List[Dict[str, Any]], player_input: str, intent: str) -> str:
        completed_tasks = {}
        execution_result = "No specific game action was executed."

        for task in plan:
            task_id = task["task_id"]
            agent_type = task["agent"]

            self.execution_logger.add_executed_task(task_id, agent_type)

            # Check dependencies
            dependencies = task["depends_on"]
            for dependency in dependencies:
                if dependency not in completed_tasks:
                    raise Exception(f"Task {task_id} depends on {dependency}, but it was not completed.")

            game_state = self.game_state_manager.get_state()

            if agent_type == "validation":
                completed_tasks[task_id] = "Validation completed."

            elif agent_type == "combat":
                result = self.combat_agent.execute(game_state)
                self.game_state_manager.update_state(result["state_updates"])
                execution_result = result["message"]
                completed_tasks[task_id] = result

            elif agent_type == "persuasion":
                result = self.persuasion_agent.execute(game_state)
                self.game_state_manager.update_state(result["state_updates"])
                execution_result = result["message"]
                completed_tasks[task_id] = result

            elif agent_type == "exploration":
                result = self.exploration_agent.execute(game_state)
                self.game_state_manager.update_state(result["state_updates"])
                execution_result = result["message"]
                completed_tasks[task_id] = result

            elif agent_type == "memory":
                memory_event = f"Player input: {player_input}. System result: {execution_result}"
                self.memory_system.add_event(memory_event)
                self.execution_logger.set_memory_event(memory_event)
                completed_tasks[task_id] = "Memory updated."

            elif agent_type == "narrative":
                final_state = self.game_state_manager.get_state()
                recent_memory = self.memory_system.retrieve_recent_events()

                narrative = self.narrative_agent.generate(
                    player_input=player_input,
                    intent=intent,
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

        return execution_result


# =========================
# 10. ORCHESTRATION AGENT
# =========================

class OrchestrationAgent:
    """
    Top-level agent.
    It coordinates the whole game flow.
    """

    def __init__(self):
        self.game_state_manager = GameStateManager()
        self.memory_system = MemorySystem()
        self.execution_logger = ExecutionLogger()

        self.intent_agent = IntentRecognitionAgent()
        self.task_planner = TaskPlanner()

        self.attribute_agent = AttributeCalculationAgent()
        self.validation_agent = ValidationAgent()

        self.combat_agent = CombatAgent(self.attribute_agent, self.validation_agent)
        self.persuasion_agent = PersuasionAgent(self.attribute_agent, self.validation_agent)
        self.exploration_agent = ExplorationAgent()

        self.narrative_agent = NarrativeGenerationAgent(llm)

        self.execution_engine = ExecutionEngine(
            game_state_manager=self.game_state_manager,
            memory_system=self.memory_system,
            execution_logger=self.execution_logger,
            combat_agent=self.combat_agent,
            persuasion_agent=self.persuasion_agent,
            exploration_agent=self.exploration_agent,
            narrative_agent=self.narrative_agent
        )

    def process_player_input(self, player_input: str) -> str:
        state_before = self.game_state_manager.get_state()
        self.execution_logger.start_turn(player_input, state_before)

        intent_data = self.intent_agent.recognize_intent(player_input)
        intent = intent_data["intent"]

        self.execution_logger.set_intent(intent)

        plan = self.task_planner.build_plan(intent)
        self.execution_logger.set_dag(plan)

        print("\n[DEBUG] Recognized intent:", intent)
        print("[DEBUG] Execution DAG:")
        for task in plan:
            print(f"  - {task['task_id']} depends on {task['depends_on']}")

        response = self.execution_engine.execute_plan(
            plan=plan,
            player_input=player_input,
            intent=intent
        )

        return response

    def show_state(self):
        self.game_state_manager.display_state()

    def show_last_log(self):
        self.execution_logger.display_last_log()

# =========================
# 11. MAIN GAME LOOP
# =========================

def main():
    game = OrchestrationAgent()

    print("🎮 DynAgentGame — Hierarchical Agent Prototype")
    print("Local LLM: Ollama")
    print("Type 'exit' to quit.")
    print("Type 'state' to see the current game state.")
    print("Type 'log' to see the last execution log.\n")

    while True:
        player_input = input("Player: ")

        if player_input.lower() in ["exit", "quit", "выход", "q"]:
            print("👋 До свидания!")
            break

        if player_input.lower() in ["state", "статус", "состояние"]:
            game.show_state()
            continue

        if player_input.lower() in ["log", "last log", "execution log"]:
            game.show_last_log()
            continue

        response = game.process_player_input(player_input)
        print(f"\nWorld: {response}\n")


if __name__ == "__main__":
    main()

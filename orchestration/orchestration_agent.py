
from __future__ import annotations

from typing import Any
import re
import os
from pathlib import Path

from config import (
    create_llm,
    create_embeddings,
    TASK_TEMPLATE_PATH,
    MEMORY_PATH,
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME,
)

from memory.semantic_memory import SemanticMemorySystem
from memory.memory_system import MemorySystem

from state.game_state_manager import GameStateManager
from state.context_manager import ContextManager
from state.npc_state_manager import NPCStateManager

from agents.intent_agent import IntentRecognitionAgent
from agents.primitive_agents import (
    AttributeCalculationAgent,
    ValidationAgent,
)
from agents.combat_agent import CombatAgent
from agents.persuasion_agent import PersuasionAgent
from agents.dialogue_agent import DialogueAgent
from agents.dialogue_step_agent import DialogueStepAgent
from agents.exploration_agent import ExplorationAgent
from agents.narrative_agent import NarrativeGenerationAgent
from agents.tavern_agent import TavernAgent
from agents.consequence_agent import ConsequenceAgent
from agents.precondition_agent import PreconditionAgent
from agents.combat_step_agent import CombatStepAgent
from agents.persuasion_step_agent import PersuasionStepAgent
from agents.event_agent import EventAgent

from engine.task_planner import TaskPlanner
from engine.execution_engine import ExecutionEngine
from engine.execution_logger import ExecutionLogger
from engine.fallback_manager import FallbackManager


class OrchestrationAgent:
    """
    Top-level coordinator for the game backend.

    Responsibilities:
    - read the current game and dialogue context;
    - recognize the player's intent;
    - resolve contextual NPC targets;
    - build the execution DAG;
    - run the execution engine;
    - update context after the turn.

    Tavern routing rules:
    - drinks, food, rooms, entering and leaving use tavern_action;
    - questions, rumors and conversations use dialogue_action;
    - convincing or bargaining uses persuasion_action;
    - attacks use combat_action.
    """

    def __init__(self) -> None:
        self.llm = create_llm()
        self.embeddings = create_embeddings()

        # =========================================================
        # State managers
        # =========================================================

        # Opt-in SQLite persistence: set RPG_STATE_DB before starting uvicorn.
        # Both managers use the same database and retain their identities.
        configured_db = os.environ.get("RPG_STATE_DB", "").strip()
        persistence_path = Path(configured_db).expanduser().resolve() if configured_db else None
        self.game_state_manager = GameStateManager(persistence_path=persistence_path)
        self.context_manager = ContextManager()
        self.npc_state_manager = NPCStateManager(persistence_path=persistence_path)

        # =========================================================
        # Memory systems
        # =========================================================

        self.memory_system = MemorySystem(
            memory_path=MEMORY_PATH,
        )

        self.semantic_memory_system = SemanticMemorySystem(
            embeddings=self.embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
            collection_name=CHROMA_COLLECTION_NAME,
        )

        # =========================================================
        # Infrastructure
        # =========================================================

        self.execution_logger = ExecutionLogger()
        self.fallback_manager = FallbackManager()

        # =========================================================
        # Planning and intent
        # =========================================================

        self.intent_agent = IntentRecognitionAgent()

        self.task_planner = TaskPlanner(
            template_path=TASK_TEMPLATE_PATH,
        )

        # =========================================================
        # Primitive agents
        # =========================================================

        self.attribute_agent = AttributeCalculationAgent()
        self.validation_agent = ValidationAgent()

        # =========================================================
        # Domain agents
        # =========================================================

        self.combat_agent = CombatAgent(
            self.attribute_agent,
            self.validation_agent,
        )

        self.persuasion_agent = PersuasionAgent(
            self.attribute_agent,
            self.validation_agent,
        )

        self.persuasion_step_agent = PersuasionStepAgent()
        self.exploration_agent = ExplorationAgent()
        self.dialogue_agent = DialogueAgent()
        self.dialogue_step_agent = DialogueStepAgent()
        self.tavern_agent = TavernAgent()
        self.combat_step_agent = CombatStepAgent()
        self.event_agent = EventAgent()
        self.precondition_agent = PreconditionAgent()

        # =========================================================
        # Shared NPC-aware agents
        # =========================================================

        self.consequence_agent = ConsequenceAgent(
            npc_state_manager=self.npc_state_manager,
        )

        # Kept for compatibility with the current ExecutionEngine.
        # Tavern and exploration flows should no longer require it
        # once the corresponding DAG templates and engine are updated.
        self.narrative_agent = NarrativeGenerationAgent(
            self.llm,
        )

        # =========================================================
        # Execution engine
        # =========================================================

        self.execution_engine = ExecutionEngine(
            game_state_manager=self.game_state_manager,
            memory_system=self.memory_system,
            semantic_memory_system=self.semantic_memory_system,
            execution_logger=self.execution_logger,
            consequence_agent=self.consequence_agent,
            fallback_manager=self.fallback_manager,
            combat_agent=self.combat_agent,
            persuasion_agent=self.persuasion_agent,
            persuasion_step_agent=self.persuasion_step_agent,
            exploration_agent=self.exploration_agent,
            narrative_agent=self.narrative_agent,
            dialogue_agent=self.dialogue_agent,
            dialogue_step_agent=self.dialogue_step_agent,
            tavern_agent=self.tavern_agent,
            combat_step_agent=self.combat_step_agent,
            event_agent=self.event_agent,
            npc_state_manager=self.npc_state_manager,
            precondition_agent=self.precondition_agent,
        )

    # =============================================================
    # Main turn processing
    # =============================================================

    def process_player_input(
        self,
        player_input: str,
        target_hint: str | None = None,
    ) -> str:
        if not isinstance(player_input, str):
            raise TypeError("player_input must be a string")

        player_input = player_input.strip()

        if not player_input:
            return "Please enter an action."

        if target_hint is not None:
            target_hint = str(target_hint).strip().lower()
            if target_hint in {"", "unknown", "environment"}:
                target_hint = None

        state_before = self.game_state_manager.get_state()
        current_context = self.context_manager.get_context()

        self.execution_logger.start_turn(
            player_input,
            state_before,
        )

        intent_context = self._build_intent_context(
            current_context=current_context,
            game_state=state_before,
        )

        intent_data = self.intent_agent.recognize_intent(
            player_input=player_input,
            context=intent_context,
            target_hint=target_hint,
        )

        intent = intent_data.get("intent", "general_action")
        target = intent_data.get("target")

        # Explicit Unity interaction target has priority over stale
        # conversational context, while an explicitly recognized target
        # from the player's text still remains authoritative.
        target = self._apply_target_hint(
            intent=intent,
            target=target,
            target_hint=target_hint,
        )

        target = self.context_manager.resolve_target_from_context(
            target,
        )

        intent, target = self._apply_safe_context_fallback(
            player_input=player_input,
            intent=intent,
            target=target,
            current_context=current_context,
            game_state=state_before,
        )

        self.execution_logger.set_intent(
            intent,
            target,
        )

        plan = self.task_planner.build_plan(intent)

        self.execution_logger.set_dag(plan)

        self._print_debug_information(
            intent=intent,
            target=target,
            current_context=current_context,
            intent_data=intent_data,
            plan=plan,
            target_hint=target_hint,
        )

        response = self.execution_engine.execute_plan(
            plan=plan,
            player_input=player_input,
            intent=intent,
            target=target,
        )

        state_after = self.game_state_manager.get_state()

        context_target = self._context_target_after_turn(
            intent=intent,
            target=target,
        )

        self.context_manager.update_after_turn(
            player_input=player_input,
            intent=intent,
            target=context_target,
            system_result=response,
            game_state=state_after,
        )

        return response

    # =============================================================
    # Context and routing helpers
    # =============================================================

    @staticmethod
    def _apply_target_hint(
        intent: str,
        target: str | None,
        target_hint: str | None,
    ) -> str | None:
        """
        Apply an explicit interaction target supplied by a frontend such as Unity.

        The hint is used only for NPC-facing intents and only when intent
        recognition did not already resolve a concrete target. Gameplay
        preconditions remain authoritative and may still block the action.
        """
        if not target_hint:
            return target

        interaction_intents = {
            "dialogue_action",
            "persuasion_action",
            "combat_action",
            "trade_action",
            "tavern_action",
        }

        if intent not in interaction_intents:
            return target

        # Unity selected a concrete physical NPC. For NPC-facing actions,
        # that selection is authoritative and must not be replaced by stale
        # conversational context from another NPC.
        return target_hint

    @staticmethod
    def _context_target_after_turn(
        intent: str,
        target: str | None,
    ) -> str | None:
        """Keep only interaction targets in short-term context."""
        if intent in {
            "dialogue_action",
            "persuasion_action",
            "combat_action",
            "tavern_action",
        }:
            if target not in (None, "", "unknown", "environment"):
                return target
        return None

    def _build_intent_context(
        self,
        current_context: dict[str, Any],
        game_state: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Provide the intent recognizer with both short-term context
        and the current game state without mutating either object.
        """
        intent_context = dict(current_context)
        intent_context["game_state"] = game_state

        return intent_context

    def _apply_safe_context_fallback(
        self,
        player_input: str,
        intent: str,
        target: str | None,
        current_context: dict[str, Any],
        game_state: dict[str, Any] | None = None,
    ) -> tuple[str, str | None]:
        """Allow explicit tavern purchases where the bartender is present.

        The Unity prototype_hub contains both merchant and bartender, so
        the selected NPC is authoritative. Mentioning a drink in conversation
        must not automatically purchase it.
        """
        if intent not in {"general_action", "dialogue_action"}:
            return intent, target

        state = game_state or {}
        location = str(
            state.get("location")
            or current_context.get("active_location")
            or ""
        ).strip().lower()

        if location not in {"tavern", "prototype_hub"}:
            return intent, target

        # Never redirect an interaction with the merchant to the bartender.
        if target not in {None, "bartender"}:
            return intent, target
        if target is None and location != "tavern":
            return intent, target

        text = player_input.strip().lower()
        drink_nouns = (
            r"drink|ale|beer|wine|mead|water|liquor|"
            r"glass|cup|mug|bottle"
        )
        request_phrases = (
            r"\b(?:order|buy|purchase|serve|pour)\b|"
            r"\b(?:get me|give me|bring me|ask for)\b|"
            r"\b(?:can i have|could i have|may i have)\b|"
            r"\b(?:i want|i'd like|i would like|i'll have)\b"
        )
        is_drink_order = bool(
            re.search(rf"\b(?:{drink_nouns})\b", text)
            and re.search(request_phrases, text)
        )

        if is_drink_order:
            return "tavern_action", "bartender"

        # Unrecognized bartender speech remains free dialogue.
        if intent == "general_action" and target == "bartender":
            return "dialogue_action", "bartender"

        return intent, target

    @staticmethod
    def _contains_any(
        text: str,
        candidates: set[str],
    ) -> bool:
        return any(candidate in text for candidate in candidates)

    # =============================================================
    # Debug output
    # =============================================================

    def _print_debug_information(
        self,
        intent: str,
        target: str | None,
        current_context: dict[str, Any],
        intent_data: dict[str, Any],
        plan: list[dict[str, Any]],
        target_hint: str | None = None,
    ) -> None:
        print("\n[DEBUG] Recognized intent:", intent)
        print("[DEBUG] Unity target hint:", target_hint)
        print("[DEBUG] Intent raw target:", intent_data.get("target"))
        print("[DEBUG] Target:", target)
        print("[DEBUG] Intent confidence:", intent_data.get("confidence"))
        print("[DEBUG] Intent reason:", intent_data.get("reason"))
        print("[DEBUG] Active context:", current_context)
        print("[DEBUG] Execution DAG:")

        for task in plan:
            print(
                f"  - {task['task_id']} "
                f"depends on {task['depends_on']}"
            )

    # =============================================================
    # State inspection
    # =============================================================

    def show_state(self) -> None:
        self.game_state_manager.display_state()

    def show_npc_state(self, npc_id: str) -> None:
        self.npc_state_manager.display_state(npc_id)

    def show_all_npc_states(self) -> None:
        all_states = self.npc_state_manager.get_all_states()

        for npc_id in all_states:
            self.npc_state_manager.display_state(npc_id)

    def show_audit_log(self) -> None:
        self.game_state_manager.display_audit_log()

    def show_snapshots(self) -> None:
        self.game_state_manager.display_snapshots()

    def show_context(self) -> None:
        self.context_manager.display_context()

    # =============================================================
    # Memory inspection and maintenance
    # =============================================================

    def show_memory(self) -> None:
        self.memory_system.display_memory()

    def search_semantic_memory(self, query: str) -> None:
        self.semantic_memory_system.display_relevant_events(
            query=query,
        )

    def rebuild_semantic_memory(self) -> None:
        self.semantic_memory_system.rebuild_from_persistent_memory(
            self.memory_system.events,
        )

        print(
            "\nSemantic memory has been rebuilt "
            "from persistent memory.\n"
        )

    def clear_memory(self) -> None:
        self.memory_system.clear_memory()
        self.semantic_memory_system.clear_memory()

    # =============================================================
    # Logging
    # =============================================================

    def show_last_log(self) -> None:
        self.execution_logger.display_last_log()

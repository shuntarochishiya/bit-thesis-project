"""Reset helpers for the existing long-lived OrchestrationAgent.

Keep manager identities intact: ExecutionEngine and other agents hold references
into these managers. Replacing `game` or its manager attributes would leave stale
references behind. Call only while api_server.turn_lock is held.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any


def reset_dialogue(game: Any) -> dict:
    """Start fresh working conversations for ALL NPCs without erasing events.

    The dialogue_reset marker is intentionally persistent, so restarting the
    server does not resurrect the old conversation. Episodic memories, semantic
    beliefs, vector memory, and gameplay state are deliberately retained.
    """
    memory = game.memory_system
    marker = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "intent": "dialogue_reset",
        "target": None,
        "player_input": "",
        "system_result": "Working dialogue reset by developer.",
        "state_changes": {},
    }
    memory.events.append(marker)
    try:
        memory.save_memory()
    except Exception:
        memory.events.pop()
        raise

    current_location = game.game_state_manager.get_state().get("location", "old forest")
    context = game.context_manager.context
    context.update({
        "active_location": current_location,
        "active_target": None,
        "active_intent": None,
        "active_conversation": None,
        "last_player_input": None,
        "last_system_result": None,
    })
    return {
        "success": True,
        "mode": "dialogue",
        "message": "Working conversations reset; gameplay and long-term memory retained.",
        "state": game.game_state_manager.get_state(),
        "context": game.context_manager.get_context(),
    }


def reset_all(game: Any, location: str | None = None) -> dict:
    """Reset all mutable game data in place, including persistent memories.

    Do NOT delete Chroma's SQLite directory while a store is open. Drop the
    collection using the Chroma API, then recreate a fresh collection. Errors
    are not suppressed: API must not report success on an incomplete reset.
    """
    from langchain_chroma import Chroma

    semantic = game.semantic_memory_system
    semantic.vector_store.delete_collection()
    semantic.vector_store = Chroma(
        collection_name=semantic.collection_name,
        embedding_function=semantic.embeddings,
        persist_directory=semantic.persist_directory,
    )

    # Clear disk-backed events BEFORE resetting in-memory manager state.
    game.memory_system.clear_memory()

    # IMPORTANT: do not replace the manager objects; the engine references them.
    game.game_state_manager.__init__()
    game.npc_state_manager.__init__()
    game.context_manager.__init__()

    if location is not None:
        # Unity has already loaded this scene; keep authoritative state aligned.
        if game.game_state_manager.get_state().get("location") != location:
            game.game_state_manager.update_state(
                {"location": location},
                source="UnityReset",
                reason=f"New game started in loaded Unity scene: {location}",
            )
        game.context_manager.context["active_location"] = location

    return {
        "success": True,
        "mode": "all",
        "message": "New game initialized; event, vector, NPC and dialogue memories cleared.",
        "state": game.game_state_manager.get_state(),
        "context": game.context_manager.get_context(),
        "npc_states": game.npc_state_manager.get_all_states(),
    }

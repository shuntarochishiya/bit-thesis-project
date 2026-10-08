from __future__ import annotations

from threading import Lock
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from orchestration.orchestration_agent import OrchestrationAgent


app = FastAPI(
    title="DynAgentGame API",
    version="1.3.0",
    description="HTTP bridge between Unity and the DynAgentGame backend.",
)

game = OrchestrationAgent()
turn_lock = Lock()

# Locations currently supported by the prototype.
# Expand this set when new Unity scenes become real game locations.
ALLOWED_LOCATIONS = {
    "old forest",
    "village",
    "town",
    "market",
    "tavern",
    "prototype_hub"
}


class InteractionRequest(BaseModel):
    text: str = Field(..., min_length=1)
    target: str | None = None


class InteractionResponse(BaseModel):
    success: bool
    player_input: str
    requested_target: str | None
    response: str
    state: dict[str, Any]
    context: dict[str, Any]


class LocationRequest(BaseModel):
    location: str = Field(..., min_length=1)


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "service": "DynAgentGame API"}


@app.post("/interact", response_model=InteractionResponse)
def interact(data: InteractionRequest) -> InteractionResponse:
    player_input = data.text.strip()
    target_hint = data.target.strip().lower() if data.target else None

    if not player_input:
        raise HTTPException(status_code=400, detail="Player input cannot be empty.")

    try:
        with turn_lock:
            response = game.process_player_input(
                player_input,
                target_hint=target_hint,
            )
            state = dict(game.game_state_manager.get_state())
            context = dict(game.context_manager.get_context())
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Backend turn failed: {exc}",
        ) from exc

    return InteractionResponse(
        success=True,
        player_input=player_input,
        requested_target=target_hint,
        response=str(response),
        state=state,
        context=context,
    )


@app.post("/location")
def sync_location(data: LocationRequest) -> dict[str, Any]:
    """
    Synchronize a loaded Unity scene/location with authoritative Python state.

    This is an explicit scene-transition endpoint, not part of free-form
    dialogue. The location is validated against a backend whitelist and the
    mutation goes through GameStateManager.update_state so versioning and the
    audit log remain intact.
    """
    location = data.location.strip().lower()

    if location not in ALLOWED_LOCATIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported location: {location}",
        )

    try:
        with turn_lock:
            before = game.game_state_manager.get_state()
            previous_location = str(before.get("location", "")).strip().lower()

            if previous_location != location:
                game.game_state_manager.create_snapshot(
                    label=f"before Unity location sync: {previous_location} -> {location}"
                )

                game.game_state_manager.update_state(
                    {"location": location},
                    source="UnitySceneSync",
                    reason=f"Unity loaded location: {location}",
                )

            state_after = game.game_state_manager.get_state()

            # Let ContextManager observe the authoritative location change.
            # update_after_turn already contains the project's scene-transition
            # logic that clears stale targets/conversations when location changes.
            game.context_manager.update_after_turn(
                player_input=f"[Unity scene sync: {location}]",
                intent="scene_transition",
                target=None,
                system_result=f"Location synchronized to {location}.",
                game_state=state_after,
            )

            context_after = game.context_manager.get_context()

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Location synchronization failed: {exc}",
        ) from exc

    return {
        "success": True,
        "previous_location": previous_location,
        "location": location,
        "state_version": game.game_state_manager.get_version(),
        "state": dict(state_after),
        "context": dict(context_after),
    }


@app.get("/state")
def get_state() -> dict[str, Any]:
    return {"state": dict(game.game_state_manager.get_state())}


@app.get("/context")
def get_context() -> dict[str, Any]:
    return {"context": dict(game.context_manager.get_context())}


@app.get("/npc/{npc_id}")
def get_npc_state(npc_id: str) -> dict[str, Any]:
    npc_key = npc_id.strip().lower()

    if not npc_key:
        raise HTTPException(status_code=400, detail="NPC id cannot be empty.")

    all_states = game.npc_state_manager.get_all_states()
    npc_state = all_states.get(npc_key)

    if npc_state is None:
        raise HTTPException(status_code=404, detail=f"Unknown NPC: {npc_key}")

    return {"npc": npc_key, "state": npc_state}

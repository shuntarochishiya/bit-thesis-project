from pprint import pprint

from state.npc_profiles import NPCProfiles
from state.npc_state_manager import NPCStateManager


def print_separator(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def show_profile(npc_id: str) -> None:
    print_separator(f"STATIC PROFILE: {npc_id.upper()}")

    profile = NPCProfiles.get_profile(npc_id)
    pprint(profile)


def show_state(
    manager: NPCStateManager,
    npc_id: str,
    title: str
) -> None:
    print_separator(title)
    manager.display_state(npc_id)


# ============================================================
# Merchant tests
# ============================================================

def test_initial_state() -> NPCStateManager:
    """
    Creates the manager and checks the initial merchant state.
    """
    manager = NPCStateManager()

    show_profile("merchant")

    show_state(
        manager=manager,
        npc_id="merchant",
        title="TEST 1: INITIAL MERCHANT STATE"
    )

    state = manager.get_state("merchant")

    assert state["health"] == 100
    assert state["alive"] is True
    assert state["hostile"] is False
    assert 0 <= state["trust"] <= 100
    assert state["emotion"] == "neutral"

    print("\n[PASS] Initial merchant state is valid.")

    return manager


def test_helped_event(manager: NPCStateManager) -> None:
    """
    Checks whether helping the merchant improves the relationship.
    """
    state_before = manager.get_state("merchant")
    trust_before = state_before["trust"]

    manager.apply_relationship_event(
        npc_id="merchant",
        event_type="helped"
    )

    state_after = manager.get_state("merchant")

    show_state(
        manager=manager,
        npc_id="merchant",
        title="TEST 2: PLAYER HELPED THE MERCHANT"
    )

    assert state_after["trust"] > trust_before
    assert state_after["hostile"] is False
    assert len(state_after["personal_memory"]) > 0

    print("\n[PASS] Help event increased trust and created memory.")


def test_bought_goods_event(manager: NPCStateManager) -> None:
    """
    Checks whether buying goods improves the merchant's attitude.
    """
    state_before = manager.get_state("merchant")
    trust_before = state_before["trust"]

    manager.apply_relationship_event(
        npc_id="merchant",
        event_type="bought_goods"
    )

    state_after = manager.get_state("merchant")

    show_state(
        manager=manager,
        npc_id="merchant",
        title="TEST 3: PLAYER BOUGHT GOODS"
    )

    assert state_after["trust"] >= trust_before
    assert len(state_after["personal_memory"]) >= 2

    print("\n[PASS] Purchase event was applied.")


def test_insulted_event(manager: NPCStateManager) -> None:
    """
    Checks whether insulting the merchant damages the relationship.
    """
    state_before = manager.get_state("merchant")
    trust_before = state_before["trust"]
    anger_before = state_before["anger"]

    manager.apply_relationship_event(
        npc_id="merchant",
        event_type="insulted"
    )

    state_after = manager.get_state("merchant")

    show_state(
        manager=manager,
        npc_id="merchant",
        title="TEST 4: PLAYER INSULTED THE MERCHANT"
    )

    assert state_after["trust"] < trust_before
    assert state_after["anger"] > anger_before
    assert len(state_after["personal_memory"]) >= 3

    print("\n[PASS] Insult reduced trust and increased anger.")


def test_threatened_event(manager: NPCStateManager) -> None:
    """
    Checks whether threatening the merchant increases fear and hostility.
    """
    state_before = manager.get_state("merchant")
    trust_before = state_before["trust"]
    fear_before = state_before["fear"]

    manager.apply_relationship_event(
        npc_id="merchant",
        event_type="threatened"
    )

    state_after = manager.get_state("merchant")

    show_state(
        manager=manager,
        npc_id="merchant",
        title="TEST 5: PLAYER THREATENED THE MERCHANT"
    )

    assert state_after["trust"] < trust_before
    assert state_after["fear"] > fear_before
    assert state_after["hostile"] is True

    print("\n[PASS] Threat increased fear and hostility.")


def test_attacked_event(manager: NPCStateManager) -> None:
    """
    Checks whether attacking the merchant causes a hostile reaction.
    """
    state_before = manager.get_state("merchant")
    anger_before = state_before["anger"]

    manager.apply_relationship_event(
        npc_id="merchant",
        event_type="attacked"
    )

    state_after = manager.get_state("merchant")

    show_state(
        manager=manager,
        npc_id="merchant",
        title="TEST 6: PLAYER ATTACKED THE MERCHANT"
    )

    assert state_after["hostile"] is True
    assert state_after["anger"] >= anger_before
    assert state_after["trust"] <= 20

    print("\n[PASS] Attack caused a strongly hostile reaction.")


def test_memory(manager: NPCStateManager) -> None:
    """
    Checks whether the merchant remembers previous interactions.
    """
    print_separator("TEST 7: MERCHANT PERSONAL MEMORY")

    memories = manager.retrieve_memories(
        npc_id="merchant",
        limit=20
    )

    pprint(memories)

    assert len(memories) >= 5

    memory_text = " ".join(
        str(memory).lower()
        for memory in memories
    )

    assert "help" in memory_text
    assert "attack" in memory_text

    print("\n[PASS] Merchant remembers positive and negative events.")


def test_simulation_context(manager: NPCStateManager) -> None:
    """
    Checks whether profile and runtime state can be combined.
    """
    print_separator("TEST 8: MERCHANT SIMULATION CONTEXT")

    context = manager.build_simulation_context("merchant")
    pprint(context)

    assert "profile" in context
    assert "state" in context
    assert "personality" in context["profile"]
    assert "emotion" in context["state"]

    print("\n[PASS] Simulation context contains profile and state.")


def test_state_validation(manager: NPCStateManager) -> None:
    """
    Checks the final merchant state for invalid values.
    """
    print_separator("TEST 9: MERCHANT STATE VALIDATION")

    validation_result = manager.validate_state("merchant")
    pprint(validation_result)

    if isinstance(validation_result, bool):
        assert validation_result is True

    elif isinstance(validation_result, dict):
        assert validation_result.get("valid", True) is True

    state = manager.get_state("merchant")

    for field in [
        "health",
        "stress",
        "fear",
        "anger",
        "trust"
    ]:
        assert 0 <= state[field] <= 100

    print("\n[PASS] Final merchant state values are valid.")


# ============================================================
# Bartender tests
# ============================================================

def test_bartender_initial_state() -> NPCStateManager:
    """
    Creates a separate manager and checks the initial bartender state.
    """
    manager = NPCStateManager()

    show_profile("bartender")

    show_state(
        manager=manager,
        npc_id="bartender",
        title="TEST 10: INITIAL BARTENDER STATE"
    )

    state = manager.get_state("bartender")

    assert state["health"] == 100
    assert state["alive"] is True
    assert state["hostile"] is False
    assert state["trust"] == 55
    assert state["anger"] == 0
    assert state["fear"] == 0
    assert state["stress"] == 10
    assert state["emotion"] == "neutral"

    print("\n[PASS] Initial bartender state is valid.")

    return manager


def test_bought_drink_event(
    manager: NPCStateManager
) -> None:
    """
    Checks whether buying a drink improves the bartender's attitude.
    """
    state_before = manager.get_state("bartender")
    trust_before = state_before["trust"]

    manager.apply_relationship_event(
        npc_id="bartender",
        event_type="bought_drink"
    )

    state_after = manager.get_state("bartender")

    show_state(
        manager=manager,
        npc_id="bartender",
        title="TEST 11: PLAYER BOUGHT A DRINK"
    )

    assert state_after["trust"] > trust_before
    assert state_after["hostile"] is False
    assert len(state_after["personal_memory"]) >= 1

    print(
        "\n[PASS] Drink purchase increased trust "
        "and created memory."
    )


def test_left_tip_event(
    manager: NPCStateManager
) -> None:
    """
    Checks whether leaving a tip improves the bartender's attitude.
    """
    state_before = manager.get_state("bartender")

    trust_before = state_before["trust"]
    stress_before = state_before["stress"]

    manager.apply_relationship_event(
        npc_id="bartender",
        event_type="left_tip"
    )

    state_after = manager.get_state("bartender")

    show_state(
        manager=manager,
        npc_id="bartender",
        title="TEST 12: PLAYER LEFT A TIP"
    )

    assert state_after["trust"] > trust_before
    assert state_after["stress"] <= stress_before
    assert state_after["hostile"] is False
    assert len(state_after["personal_memory"]) >= 2

    print(
        "\n[PASS] Tip improved the bartender's attitude."
    )


def test_bartender_insulted_event(
    manager: NPCStateManager
) -> None:
    """
    Checks whether insulting the bartender reduces trust
    and increases anger.
    """
    state_before = manager.get_state("bartender")

    trust_before = state_before["trust"]
    anger_before = state_before["anger"]

    manager.apply_relationship_event(
        npc_id="bartender",
        event_type="insulted"
    )

    state_after = manager.get_state("bartender")

    show_state(
        manager=manager,
        npc_id="bartender",
        title="TEST 13: PLAYER INSULTED THE BARTENDER"
    )

    assert state_after["trust"] < trust_before
    assert state_after["anger"] > anger_before
    assert len(state_after["personal_memory"]) >= 3

    print(
        "\n[PASS] Insult reduced bartender trust "
        "and increased anger."
    )


def test_bartender_threatened_event(
    manager: NPCStateManager
) -> None:
    """
    Checks whether threatening the bartender causes fear
    and hostility.
    """
    state_before = manager.get_state("bartender")

    trust_before = state_before["trust"]
    fear_before = state_before["fear"]

    manager.apply_relationship_event(
        npc_id="bartender",
        event_type="threatened"
    )

    state_after = manager.get_state("bartender")

    show_state(
        manager=manager,
        npc_id="bartender",
        title="TEST 14: PLAYER THREATENED THE BARTENDER"
    )

    assert state_after["trust"] < trust_before
    assert state_after["fear"] > fear_before
    assert state_after["hostile"] is True

    print(
        "\n[PASS] Threat increased bartender fear "
        "and caused hostility."
    )


def test_bartender_attacked_event(
    manager: NPCStateManager
) -> None:
    """
    Checks whether attacking the bartender causes
    a strongly hostile reaction.
    """
    state_before = manager.get_state("bartender")

    anger_before = state_before["anger"]
    trust_before = state_before["trust"]

    manager.apply_relationship_event(
        npc_id="bartender",
        event_type="attacked"
    )

    state_after = manager.get_state("bartender")

    show_state(
        manager=manager,
        npc_id="bartender",
        title="TEST 15: PLAYER ATTACKED THE BARTENDER"
    )

    assert state_after["hostile"] is True
    assert state_after["anger"] >= anger_before
    assert state_after["trust"] < trust_before
    assert state_after["trust"] <= 20
    assert len(state_after["personal_memory"]) >= 5

    print(
        "\n[PASS] Attack caused a strongly hostile "
        "bartender reaction."
    )


def test_bartender_apology_event(
    manager: NPCStateManager
) -> None:
    """
    Checks whether an apology partially improves
    the bartender's state.
    """
    state_before = manager.get_state("bartender")

    trust_before = state_before["trust"]
    anger_before = state_before["anger"]
    fear_before = state_before["fear"]

    manager.apply_relationship_event(
        npc_id="bartender",
        event_type="apologized"
    )

    state_after = manager.get_state("bartender")

    show_state(
        manager=manager,
        npc_id="bartender",
        title="TEST 16: PLAYER APOLOGIZED TO THE BARTENDER"
    )

    assert state_after["trust"] >= trust_before
    assert state_after["anger"] <= anger_before
    assert state_after["fear"] <= fear_before
    assert len(state_after["personal_memory"]) >= 6

    print(
        "\n[PASS] Apology partially improved "
        "the bartender's emotional state."
    )


def test_bartender_memory(
    manager: NPCStateManager
) -> None:
    """
    Checks whether the bartender remembers service,
    positive and negative interactions.
    """
    print_separator("TEST 17: BARTENDER PERSONAL MEMORY")

    memories = manager.retrieve_memories(
        npc_id="bartender",
        limit=20
    )

    pprint(memories)

    assert len(memories) >= 6

    memory_text = " ".join(
        str(memory).lower()
        for memory in memories
    )

    assert "bought_drink" in memory_text
    assert "left_tip" in memory_text
    assert "insult" in memory_text
    assert "threat" in memory_text
    assert "attack" in memory_text
    assert "apolog" in memory_text

    print(
        "\n[PASS] Bartender remembers positive "
        "and negative events."
    )


def test_bartender_simulation_context(
    manager: NPCStateManager
) -> None:
    """
    Checks whether bartender profile and runtime state
    can be combined.
    """
    print_separator("TEST 18: BARTENDER SIMULATION CONTEXT")

    context = manager.build_simulation_context("bartender")
    pprint(context)

    assert "profile" in context
    assert "state" in context

    assert context["profile"]["role"] == "bartender"
    assert "personality" in context["profile"]
    assert "goals" in context["profile"]

    assert context["state"]["npc_id"] == "bartender"
    assert "emotion" in context["state"]
    assert "personal_memory" in context["state"]

    print(
        "\n[PASS] Bartender simulation context "
        "contains profile and runtime state."
    )


def test_bartender_state_validation(
    manager: NPCStateManager
) -> None:
    """
    Checks the final bartender state for invalid values.
    """
    print_separator("TEST 19: BARTENDER STATE VALIDATION")

    validation_result = manager.validate_state("bartender")
    pprint(validation_result)

    if isinstance(validation_result, bool):
        assert validation_result is True

    elif isinstance(validation_result, dict):
        assert validation_result.get("valid", True) is True

    state = manager.get_state("bartender")

    for field in [
        "health",
        "stress",
        "fear",
        "anger",
        "trust"
    ]:
        assert 0 <= state[field] <= 100

    assert isinstance(state["alive"], bool)
    assert isinstance(state["hostile"], bool)
    assert isinstance(state["personal_memory"], list)

    print(
        "\n[PASS] Final bartender state values are valid."
    )


# ============================================================
# General NPC manager tests
# ============================================================

def test_all_profiles_available() -> None:
    """
    Checks whether all configured NPC profiles
    can be loaded correctly.
    """
    print_separator("TEST 20: ALL NPC PROFILES")

    npc_ids = NPCProfiles.get_npc_ids()
    pprint(npc_ids)

    expected_npcs = {
        "merchant",
        "bartender",
        "guard",
        "traveler",
        "forest_goblin"
    }

    assert expected_npcs.issubset(set(npc_ids))

    for npc_id in expected_npcs:
        profile = NPCProfiles.get_profile(npc_id)

        assert profile["name"]
        assert profile["role"]
        assert "personality" in profile
        assert "goals" in profile
        assert "initial_state" in profile

    print("\n[PASS] All expected NPC profiles are available.")


def test_all_runtime_states() -> None:
    """
    Checks whether runtime states are initialized
    for all configured NPCs.
    """
    print_separator("TEST 21: ALL NPC RUNTIME STATES")

    manager = NPCStateManager()
    states = manager.get_all_states()

    pprint(states)

    for npc_id in NPCProfiles.get_npc_ids():
        assert npc_id in states
        assert manager.validate_state(npc_id) is True

    print(
        "\n[PASS] Runtime state exists for every NPC profile."
    )


def test_alias_normalization() -> None:
    """
    Checks whether common NPC aliases are normalized.
    """
    print_separator("TEST 22: NPC ALIAS NORMALIZATION")

    assert NPCProfiles.normalize_npc_id("merchant") == "merchant"
    assert NPCProfiles.normalize_npc_id("vendor") == "merchant"
    assert NPCProfiles.normalize_npc_id("shopkeeper") == "merchant"

    assert NPCProfiles.normalize_npc_id("bartender") == "bartender"
    assert NPCProfiles.normalize_npc_id("barmaid") == "bartender"

    assert NPCProfiles.normalize_npc_id("enemy") == "forest_goblin"
    assert NPCProfiles.normalize_npc_id("goblin") == "forest_goblin"

    print("\n[PASS] NPC aliases are normalized correctly.")


# ============================================================
# Test runner
# ============================================================

def run_all_tests() -> None:
    print_separator("NPC MODULE TESTS STARTED")

    merchant_manager = test_initial_state()

    test_helped_event(merchant_manager)
    test_bought_goods_event(merchant_manager)
    test_insulted_event(merchant_manager)
    test_threatened_event(merchant_manager)
    test_attacked_event(merchant_manager)
    test_memory(merchant_manager)
    test_simulation_context(merchant_manager)
    test_state_validation(merchant_manager)

    bartender_manager = test_bartender_initial_state()

    test_bought_drink_event(bartender_manager)
    test_left_tip_event(bartender_manager)
    test_bartender_insulted_event(bartender_manager)
    test_bartender_threatened_event(bartender_manager)
    test_bartender_attacked_event(bartender_manager)
    test_bartender_apology_event(bartender_manager)
    test_bartender_memory(bartender_manager)
    test_bartender_simulation_context(bartender_manager)
    test_bartender_state_validation(bartender_manager)

    test_all_profiles_available()
    test_all_runtime_states()
    test_alias_normalization()

    print_separator("ALL NPC MODULE TESTS PASSED")


if __name__ == "__main__":
    try:
        run_all_tests()

    except AssertionError as error:
        print_separator("TEST FAILED")
        print(f"Assertion error: {error}")
        raise

    except Exception as error:
        print_separator("UNEXPECTED ERROR")
        print(f"{type(error).__name__}: {error}")
        raise

from difflib import get_close_matches

from orchestration.orchestration_agent import OrchestrationAgent


EXACT_COMMAND_ALIASES = {
    "exit": "exit", "quit": "exit", "q": "exit",
    "state": "state", "status": "state",
    "memory": "memory", "show memory": "memory",
    "clear memory": "clear memory", "reset memory": "clear memory",
    "context": "context", "ctx": "context",
    "log": "log", "last log": "log", "execution log": "log",
    "audit": "audit", "audit log": "audit", "state audit": "audit",
    "snapshots": "snapshots", "show snapshots": "snapshots",
    "rebuild semantic": "rebuild semantic",
    "rebuild semantic memory": "rebuild semantic",
}

FUZZY_COMMANDS = {
    "state", "status", "memory", "context", "audit", "snapshots", "log",
}


def normalize_cli_command(player_input: str):
    """Return (command, argument, corrected_from)."""
    raw = " ".join(str(player_input).strip().lower().split())

    if raw in EXACT_COMMAND_ALIASES:
        return EXACT_COMMAND_ALIASES[raw], None, None

    if raw.startswith("semantic "):
        query = player_input.strip()[len("semantic "):].strip()
        if query:
            return "semantic", query, None

    # Allow a typo only in the semantic command prefix.
    parts = raw.split(maxsplit=1)
    if len(parts) == 2:
        prefix, query = parts
        if get_close_matches(prefix, ["semantic"], n=1, cutoff=0.80):
            return "semantic", query, prefix

    # Do not fuzzy-correct ordinary multi-word gameplay text.
    if " " in raw or not raw:
        return None, None, None

    match = get_close_matches(raw, FUZZY_COMMANDS, n=1, cutoff=0.72)
    if match:
        canonical = EXACT_COMMAND_ALIASES.get(match[0], match[0])
        return canonical, None, raw

    return None, None, None


def main():
    game = OrchestrationAgent()

    print("DynAgentGame — Hierarchical Agent Prototype")
    print("Local LLM: Ollama")
    print("Type 'exit' to quit.")
    print("Type 'state' to see the current game state.")
    print("Type 'audit' to see the state audit log.")
    print("Type 'snapshots' to see state snapshots.")
    print("Type 'memory' to see recent memory events.")
    print("Type 'clear memory' to erase persistent memory.")
    print("Type 'context' to see the current interaction context.")
    print("Type 'semantic <query>' to search semantic memory.")
    print("Type 'rebuild semantic' to rebuild vector memory from memory.json.")
    print("Type 'log' to see the last execution log.\n")

    while True:
        player_input = input("Player: ")
        command, argument, corrected_from = normalize_cli_command(player_input)

        if corrected_from:
            if command == "semantic":
                print(f"[CLI] Interpreting '{corrected_from}' as 'semantic'.")
            else:
                print(f"[CLI] Interpreting '{corrected_from}' as '{command}'.")

        if command == "exit":
            print("Goodbye!")
            break
        if command == "state":
            game.show_state()
            continue
        if command == "memory":
            game.show_memory()
            continue
        if command == "clear memory":
            game.clear_memory()
            continue
        if command == "context":
            game.show_context()
            continue
        if command == "log":
            game.show_last_log()
            continue
        if command == "audit":
            game.show_audit_log()
            continue
        if command == "snapshots":
            game.show_snapshots()
            continue
        if command == "semantic":
            game.search_semantic_memory(argument)
            continue
        if command == "rebuild semantic":
            game.rebuild_semantic_memory()
            continue

        response = game.process_player_input(player_input)
        print(f"\nWorld: {response}\n")


if __name__ == "__main__":
    main()

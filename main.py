from orchestration.orchestration_agent import OrchestrationAgent


def main():
    game = OrchestrationAgent()

    print("DynAgentGame — Hierarchical Agent Prototype")
    print("Local LLM: Ollama")
    print("Type 'exit' to quit.")
    print("Type 'state' to see the current game state.")
    print("Type 'memory' to see recent memory events.")
    print("Type 'clear memory' to erase persistent memory.")
    print("Type 'context' to see the current interaction context.")
    print("Type 'semantic <query>' to search semantic memory.")
    print("Type 'rebuild semantic' to rebuild vector memory from memory.json.")
    print("Type 'log' to see the last execution log.\n")

    while True:
        player_input = input("Player: ")

        if player_input.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break

        if player_input.lower() in ["state", "status"]:
            game.show_state()
            continue

        if player_input.lower() in ["memory", "show memory"]:
            game.show_memory()
            continue

        if player_input.lower() in ["clear memory", "reset memory"]:
            game.clear_memory()
            continue

        if player_input.lower() in ["context", "ctx"]:
            game.show_context()
            continue

        if player_input.lower() in ["log", "last log", "execution log"]:
            game.show_last_log()
            continue

        if player_input.lower().startswith("semantic "):
            query = player_input[len("semantic "):]
            game.search_semantic_memory(query)
            continue

        if player_input.lower() in ["rebuild semantic", "rebuild semantic memory"]:
            game.rebuild_semantic_memory()
            continue

        response = game.process_player_input(player_input)
        print(f"\nWorld: {response}\n")


if __name__ == "__main__":
    main()

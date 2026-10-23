import json
import time
import uuid
import shutil
import os
from typing import Dict, Any, List, Optional

from langchain_core.documents import Document
from langchain_chroma import Chroma


class SemanticMemorySystem:
    """
    Vector-based semantic memory system.

    It stores important game events in a local Chroma vector database.
    This version also uses metadata classification to make retrieval more precise.
    """

    def __init__(
        self,
        embeddings,
        persist_directory: str = "chroma_memory_db",
        collection_name: str = "dynagentgame_memory"
    ):
        self.embeddings = embeddings
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )

    def classify_event_type(self, event: Dict[str, Any]) -> str:
        """
        Classifies a memory event into a more precise event type.
        This makes semantic retrieval less vague.
        """

        player_input = str(event.get("player_input", "")).lower()
        system_result = str(event.get("system_result", "")).lower()
        intent = str(event.get("intent", "")).lower()
        target = str(event.get("target", "")).lower()

        combined_text = f"{player_input} {system_result}"

        drink_words = [
            "drink", "ale", "beer", "wine", "mead", "glass", "cup", "bottle"
        ]

        room_words = [
            "room", "rent", "sleep", "rest"
        ]

        food_words = [
            "food", "meal"
        ]

        rumor_words = [
            "rumor", "rumour", "information", "news",
            "odd", "strange", "weird", "nearby", "recently",
            "spotted", "heard", "seen", "details"
        ]

        flirt_words = [
            "flirt", "flirty", "wink", "compliment", "appearance", "smile"
        ]

        attack_words = [
            "attack", "hit", "strike", "stab", "kill", "hurt", "damage"
        ]

        if intent == "combat_action" or any(word in combined_text for word in attack_words):
            return "attack"

        if intent == "tavern_action" and any(word in combined_text for word in drink_words):
            return "drink_order"

        if intent == "tavern_action" and any(word in combined_text for word in room_words):
            return "room_rental"

        if intent == "tavern_action" and any(word in combined_text for word in food_words):
            return "food_order"

        if intent == "tavern_action" and any(word in combined_text for word in rumor_words):
            return "tavern_information"

        if target == "bartender" and any(word in combined_text for word in flirt_words):
            return "bartender_social"

        if intent == "dialogue_action":
            return "dialogue"

        if intent == "exploration_action":
            return "exploration"

        if intent == "persuasion_action":
            return "persuasion"

        return "general"

    def classify_query_type(self, query: str) -> Optional[str]:
        """
        Classifies the user's semantic search query.
        If possible, returns an event_type filter.
        """

        text = query.lower()

        if any(word in text for word in ["drink", "drinks", "ale", "beer", "wine", "mead", "order", "orders"]):
            return "drink_order"

        if any(word in text for word in ["room", "rent", "sleep", "rest"]):
            return "room_rental"

        if any(word in text for word in ["food", "meal"]):
            return "food_order"

        if any(word in text for word in ["rumor", "rumour", "information", "news", "strange", "weird", "odd", "nearby"]):
            return "tavern_information"

        if any(word in text for word in ["attack", "violence", "hit", "damage", "hostile", "fight"]):
            return "attack"

        if any(word in text for word in ["flirt", "wink", "compliment", "social"]):
            return "bartender_social"

        if any(word in text for word in ["talk", "dialogue", "conversation", "speak"]):
            return "dialogue"

        return None

    def format_event_for_embedding(self, event: Dict[str, Any]) -> str:
        """
        Converts a structured memory event into readable text for embedding.
        """

        player_input = event.get("player_input", "")
        intent = event.get("intent", "")
        target = event.get("target", "")
        system_result = event.get("system_result", "")
        state_changes = event.get("state_changes", {})
        important_state = event.get("important_state", {})
        event_type = self.classify_event_type(event)

        return (
            f"Event type: {event_type}\n"
            f"Player input: {player_input}\n"
            f"Intent: {intent}\n"
            f"Target: {target}\n"
            f"System result: {system_result}\n"
            f"State changes: {json.dumps(state_changes, ensure_ascii=False)}\n"
            f"Important state: {json.dumps(important_state, ensure_ascii=False)}"
        )

    def add_event(self, event: Dict[str, Any]):
        """
        Adds one structured memory event to the vector database.
        """

        event_type = self.classify_event_type(event)
        event_text = self.format_event_for_embedding(event)

        metadata = {
            "timestamp": str(event.get("timestamp", "")),
            "intent": str(event.get("intent", "")),
            "target": str(event.get("target", "")),
            "event_type": event_type,
            "player_input": str(event.get("player_input", ""))
        }

        document = Document(
            page_content=event_text,
            metadata=metadata
        )

        doc_id = str(uuid.uuid4())


        write_start = time.perf_counter()

        self.vector_store.add_documents(
            documents=[document],
            ids=[doc_id]
        )

        write_time = time.perf_counter() - write_start

        print(
            f"[SEMANTIC MEMORY WRITE] "
            f"Embedding + Chroma insertion: {write_time:.4f}s"
        )


    def retrieve_relevant_events(
        self,
        query: str,
        k: int = 3,
        npc_id: Optional[str] = None,
    ) -> List[str]:
        """Search world events, optionally scoped to interactions with one NPC.

        npc_id=None retains the original global search for world reasoning.
        An NPC-scoped search never falls back to global events: that would
        expose another NPC's interactions as if they were personal history.
        The target field records the interaction target, not eyewitness status.
        """
        total_start = time.perf_counter()

        try:
            classify_start = time.perf_counter()
            query_type = self.classify_query_type(query)
            classify_time = time.perf_counter() - classify_start

            normalized_npc = str(npc_id).strip().lower() if npc_id else None
            if normalized_npc:
                normalized_npc = normalized_npc.replace("-", "_").replace(" ", "_")
                # Match the canonical target IDs used by the execution engine.
                normalized_npc = {
                    "vendor": "merchant", "shopkeeper": "merchant",
                    "trader": "merchant", "barmaid": "bartender",
                    "barkeep": "bartender", "innkeeper": "bartender",
                    "goblin": "enemy", "forest_goblin": "enemy",
                }.get(normalized_npc, normalized_npc)

            conditions = []
            if query_type:
                conditions.append({"event_type": query_type})
            if normalized_npc:
                conditions.append({"target": normalized_npc})

            if len(conditions) == 2:
                search_filter = {"$and": conditions}
            elif conditions:
                search_filter = conditions[0]
            else:
                search_filter = None

            search_start = time.perf_counter()
            search_kwargs = {"query": query, "k": k}
            if search_filter is not None:
                search_kwargs["filter"] = search_filter
            results = self.vector_store.similarity_search(**search_kwargs)
            search_time = time.perf_counter() - search_start

            format_start = time.perf_counter()
            formatted_results = []
            for doc in results:
                metadata = doc.metadata or {}
                # The stored document contains the confirmed system result and
                # state changes, unlike the old metadata-only representation.
                formatted_results.append(
                    f"[World event; interaction target: {metadata.get('target', 'unknown')}; "
                    f"not proof that other NPCs witnessed it]\n"
                    f"{doc.page_content}"
                )
            format_time = time.perf_counter() - format_start
            total_time = time.perf_counter() - total_start

            print("\n[SEMANTIC MEMORY PROFILE]")
            print(f"Query classification: {classify_time:.4f}s")
            print(f"Vector search:        {search_time:.4f}s")
            print(f"Result formatting:    {format_time:.4f}s")
            print(f"Total retrieval:      {total_time:.4f}s")
            print(f"Results found:        {len(results)}")
            print(f"Event type filter:    {query_type}")
            print(f"NPC target filter:    {normalized_npc}")
            print("-" * 45)
            return formatted_results

        except Exception as error:
            print(f"[SEMANTIC MEMORY ERROR] {error}")
            # Failed retrieval must not be mistaken for a factual memory.
            return []


    def rebuild_from_persistent_memory(self, events: List[Dict[str, Any]]):
        """
        Rebuilds vector memory from memory.json events.
        """

        for event in events:
            if isinstance(event, dict) and "player_input" in event:
                self.add_event(event)

    def display_relevant_events(self, query: str, k: int = 3):
        """
        Displays relevant semantic memory events in terminal.
        """

        query_type = self.classify_query_type(query)
        results = self.retrieve_relevant_events(query=query, k=k)

        print("\n--- Semantic Memory Search ---")
        print(f"Query: {query}")
        print(f"Detected query type: {query_type}")

        if not results:
            print("No relevant semantic memories found.")
        else:
            for index, result in enumerate(results, start=1):
                print(f"\nResult {index}:")
                print(result)

    def clear_memory(self):
        """
        Clears the local Chroma vector memory.
        This removes old semantic memory records.
        """

        try:
            self.vector_store.delete_collection()
        except Exception:
            pass

        try:
            if os.path.exists(self.persist_directory):
                shutil.rmtree(self.persist_directory, ignore_errors=True)
        except Exception:
            pass

        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )

        print("\nSemantic memory has been cleared.\n")
        print("------------------------------\n")

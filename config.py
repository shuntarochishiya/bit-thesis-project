from langchain_ollama import ChatOllama, OllamaEmbeddings


LLM_MODEL = "mistral"
LLM_TEMPERATURE = 0.3

EMBEDDING_MODEL = "nomic-embed-text"

TASK_TEMPLATE_PATH = "task_templates.json"
MEMORY_PATH = "memory.json"

CHROMA_PERSIST_DIR = "chroma_memory_db"
CHROMA_COLLECTION_NAME = "dynagentgame_memory"


def create_llm():
    """
    Creates the local LLM connection through Ollama.
    """

    return ChatOllama(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE
    )


def create_embeddings():
    """
    Creates local embedding model through Ollama.
    This model is used for semantic memory search.
    """

    return OllamaEmbeddings(
        model=EMBEDDING_MODEL
    )

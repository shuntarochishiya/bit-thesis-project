from langchain_ollama import ChatOllama


LLM_MODEL = "mistral"
LLM_TEMPERATURE = 0.3

TASK_TEMPLATE_PATH = "task_templates.json"
MEMORY_PATH = "memory.json"


def create_llm():
    """
    Creates the local LLM connection through Ollama.
    """

    return ChatOllama(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE
    )

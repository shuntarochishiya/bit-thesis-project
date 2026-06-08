from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
import requests

app = FastAPI()

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL_NAME = "mistral"

# Память NPC по имени
npc_memory: Dict[str, List[str]] = {}


class PlayerAction(BaseModel):
    player_input: str
    location: str = "tavern"
    npc: str | None = "bartender"


@app.post("/player_action")
def player_action(data: PlayerAction):
    npc_name = data.npc or "Unknown NPC"

    # Инициализируем память для этого NPC
    if npc_name not in npc_memory:
        npc_memory[npc_name] = []

    # Добавляем сообщение игрока в память
    npc_memory[npc_name].append(f"Player: {data.player_input}")

    # Обрезаем память до последних 10 сообщений
    if len(npc_memory[npc_name]) > 10:
        npc_memory[npc_name] = npc_memory[npc_name][-10:]

    # Собираем контекст для промпта
    conversation_history = "\n".join(npc_memory[npc_name])
    prompt = f"""
You are an NPC in a fantasy RPG game.

Name: {npc_name}
Role: {'tavern bartender' if npc_name.lower() == 'bartender' else 'unknown'}
Personality: friendly, slightly suspicious
Current location: {data.location}

Conversation so far:
{conversation_history}

Player says: "{data.player_input}"
Reply as {npc_name}. Keep the answer short, atmospheric, and natural. Do not explain that you are an AI.
"""

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }

    try:
        ollama_response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=60
        )
        ollama_response.raise_for_status()
        result = ollama_response.json()
        response_text = result.get("response", "The NPC says nothing.")
    except Exception as e:
        response_text = f"LLM error: {str(e)}"

    # Добавляем ответ NPC в память
    npc_memory[npc_name].append(f"{npc_name}: {response_text}")

    return {
        "response": response_text,
        "intent": "llm_dialogue",
        "location": data.location,
        "npc": npc_name,
        "conversation_history": npc_memory[npc_name]  # Можно выводить для отладки
    }


@app.post("/reset_memory")
def reset_memory():
    npc_memory.clear()
    return {"status": "memory cleared"}


@app.get("/history/{npc_name}")
def get_history(npc_name: str):
    npc_name = npc_name.lower()
    return {
        "npc": npc_name,
        "conversation_history": npc_memory.get(npc_name, [])
    }

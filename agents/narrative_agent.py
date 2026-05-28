from typing import Dict, Any, List
from langchain_core.messages import HumanMessage, SystemMessage


class NarrativeGenerationAgent:
    """
    This agent uses the local LLM through Ollama.
    It generates the final text shown to the player.
    The final response is forced to be in English.
    """

    def __init__(self, llm):
        self.llm = llm

    def contains_cyrillic(self, text: str) -> bool:
        """
        Checks whether the generated text contains Russian/Cyrillic characters.
        """
        return any('\u0400' <= char <= '\u04FF' for char in text)

    def force_english(self, text: str) -> str:
        """
        If the model produces Russian text, this function asks it to rewrite the answer in English only.
        """
        response = self.llm.invoke([
            SystemMessage(content="""
You are a strict translation and rewriting assistant.

Your task:
Rewrite the given text in English only.

Rules:
1. Use English only.
2. Do not use Russian words.
3. Do not explain anything.
4. Return only the rewritten fantasy RPG narration.
"""),
            HumanMessage(content=f"""
Rewrite this text in English only:

{text}
""")
        ])

        return response.content

    def generate(
        self,
        player_input: str,
        intent: str,
        target: str,
        game_state: Dict[str, Any],
        execution_result: str,
        memory_events: List[str]
    ) -> str:

        system_prompt = f"""
You are an English-language fantasy RPG narrator.

IMPORTANT LANGUAGE RULE:
You must always answer in English only.
Never answer in Russian.
Never use Cyrillic characters.
Even if the player writes in another language, your final answer must be in English.

Your task is to generate a short, atmospheric and logical response to the player.

Current game state:
- Location: {game_state["location"]}
- Player health: {game_state["player_health"]}
- Enemy health: {game_state["enemy_health"]}
- Gold: {game_state["gold"]}
- Inventory: {game_state["inventory"]}
- Relationship with merchant: {game_state["relationship_with_merchant"]}
- World mood: {game_state["world_mood"]}
- Bartender health: {game_state["bartender_health"]}
- Relationship with bartender: {game_state["relationship_with_bartender"]}
- Bartender hostile: {game_state["bartender_hostile"]}
- Bartender mood: {game_state["bartender_mood"]}
- Tavern reputation: {game_state["tavern_reputation"]}

Recognized player intent:
{intent}

Result of internal game system:
{execution_result}

Target of player action:
{target}

Recent memory:
{memory_events}

Output rules:
1. Write in English only.
2. Do not use Russian.
3. Do not use Cyrillic characters.
4. Do not contradict the game state.
5. Keep the answer short.
6. Make the story feel like a fantasy role-playing game.
7. Keep the answer short.
8. Make the story feel like a fantasy role-playing game.
9. If the player talks to an enemy, reflect the enemy's current health and hostility.
10. If the enemy is badly wounded, it may hesitate, bargain, or ask for mercy.
11. If the enemy is healthy, it should usually remain threatening.
12. If the player interacts with the bartender, reflect the bartender's mood, tavern reputation, and previous player behavior.
13. If the bartender is hostile, do not describe them as friendly or helpful.
14. If the player is in the tavern, make the atmosphere feel like an inn or medieval tavern.
"""

        response = self.llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""
Player input:
{player_input}

Generate the final game narration in English only.
""")
        ])

        final_text = response.content

        # Safety check: if the model still answers in Russian, rewrite it in English
        if self.contains_cyrillic(final_text):
            final_text = self.force_english(final_text)

        return final_text


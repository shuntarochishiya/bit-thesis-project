from copy import deepcopy
from typing import Any, Dict, List


class NPCProfiles:
    """Static NPC configuration used to initialize runtime NPC state."""

    _ALIASES: Dict[str, str] = {
        "vendor": "merchant",
        "shopkeeper": "merchant",
        "trader": "merchant",
        "barmaid": "bartender",
        "barkeep": "bartender",
        "innkeeper": "bartender",
        "enemy": "forest_goblin",
        "goblin": "forest_goblin",
        "forest goblin": "forest_goblin",
    }

    _PROFILES: Dict[str, Dict[str, Any]] = {
        "merchant": {
            "name": "Merchant",
            "role": "merchant",
            "personality": {
                "openness": 45,
                "conscientiousness": 75,
                "extraversion": 55,
                "agreeableness": 50,
                "neuroticism": 40,
            },
            "goals": ["earn profit", "protect goods", "avoid unnecessary danger"],
            "initial_state": {
                "health": 100,
                "alive": True,
                "hostile": False,
                "emotion": "neutral",
                "trust": 50,
                "anger": 0,
                "fear": 0,
                "stress": 10,
            },
        },
        "bartender": {
            "name": "Bartender",
            "role": "bartender",
            "personality": {
                "openness": 60,
                "conscientiousness": 70,
                "extraversion": 75,
                "agreeableness": 65,
                "neuroticism": 35,
            },
            "goals": ["serve customers", "keep the tavern peaceful", "collect useful rumors"],
            "initial_state": {
                "health": 100,
                "alive": True,
                "hostile": False,
                "emotion": "neutral",
                "trust": 55,
                "anger": 0,
                "fear": 0,
                "stress": 10,
            },
        },
        "guard": {
            "name": "Guard",
            "role": "guard",
            "personality": {
                "openness": 30,
                "conscientiousness": 85,
                "extraversion": 45,
                "agreeableness": 40,
                "neuroticism": 30,
            },
            "goals": ["maintain order", "protect civilians", "stop crime"],
            "initial_state": {
                "health": 100,
                "alive": True,
                "hostile": False,
                "emotion": "alert",
                "trust": 45,
                "anger": 0,
                "fear": 0,
                "stress": 15,
            },
        },
        "traveler": {
            "name": "Traveler",
            "role": "traveler",
            "personality": {
                "openness": 80,
                "conscientiousness": 45,
                "extraversion": 60,
                "agreeableness": 70,
                "neuroticism": 45,
            },
            "goals": ["travel safely", "exchange information", "find shelter"],
            "initial_state": {
                "health": 100,
                "alive": True,
                "hostile": False,
                "emotion": "curious",
                "trust": 50,
                "anger": 0,
                "fear": 5,
                "stress": 10,
            },
        },
        "forest_goblin": {
            "name": "Forest Goblin",
            "role": "enemy",
            "personality": {
                "openness": 25,
                "conscientiousness": 20,
                "extraversion": 40,
                "agreeableness": 10,
                "neuroticism": 70,
            },
            "goals": ["survive", "defend territory", "steal valuables"],
            "initial_state": {
                "health": 60,
                "alive": True,
                "hostile": True,
                "emotion": "hostile",
                "trust": 0,
                "anger": 40,
                "fear": 10,
                "stress": 30,
            },
        },
    }

    @classmethod
    def normalize_npc_id(cls, npc_id: str) -> str:
        if not npc_id:
            return ""
        normalized = str(npc_id).strip().lower().replace("-", "_")
        normalized = " ".join(normalized.split())
        normalized = cls._ALIASES.get(normalized, normalized)
        return normalized.replace(" ", "_")

    @classmethod
    def get_profile(cls, npc_id: str) -> Dict[str, Any]:
        normalized = cls.normalize_npc_id(npc_id)
        if normalized not in cls._PROFILES:
            raise KeyError(f"Unknown NPC profile: {npc_id}")
        return deepcopy(cls._PROFILES[normalized])

    @classmethod
    def get_all_profiles(cls) -> Dict[str, Dict[str, Any]]:
        return deepcopy(cls._PROFILES)

    @classmethod
    def get_npc_ids(cls) -> List[str]:
        return list(cls._PROFILES.keys())

    @classmethod
    def list_npc_ids(cls) -> List[str]:
        """Backward-compatible alias used by NPCStateManager."""
        return cls.get_npc_ids()

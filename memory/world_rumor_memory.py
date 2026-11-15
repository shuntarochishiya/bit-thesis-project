"""Shared, persistent *narrative* rumor memory.

Public rumors circulate among local NPCs independently of private dialogues.
An NPC's version is separate from the shared subject. Neither rumors nor
retellings mutate authoritative game mechanics.
"""
from __future__ import annotations

import json
import re
import random
from datetime import datetime, timezone
from pathlib import Path
from agents.world_facts import RUMOR_TEMPLATES, is_news_question

_CUES = re.compile(r"\b(?:rumou?rs?|whispers?|some say|they say|it.s said|i.ve heard|i heard|there(?: is|\'s) talk|people say|legends?|tales?)\b", re.I)
_NEWS = re.compile(r"\b(?:rumou?rs?|gossip|news|whispers?|stories|strange|weird|heard anything)\b", re.I)
_STOP = set("have heard anything about there some that those this they them from with where what when your here were been could would into more tell news story rumors rumor recently woods wilds near these around very said says seen player something strange people traveler adventure adventures welcome friend indeed unknown subject offer knows known history".split())


def keywords(text):
    return set(w for w in re.findall(r"[a-z]{4,}", str(text or "").lower()) if w not in _STOP)


# Pool IDs are identified by subject, NOT by exact template text.
_POOL_SUBJECTS = {
    "strange_noise": re.compile(r"\b(?:strange|odd|mysterious) (?:noises?|sounds?)\b|\b(?:noises?|sounds?)\b", re.I),
    "bandit_roadblock": re.compile(r"\b(?:bandits?|highwaymen|roadblocks?|robbers?)\b", re.I),
    "old_ruin_discovery": re.compile(r"\b(?:ruins?|ancient temples?|forgotten temples?)\b", re.I),
    "merchant_cart": re.compile(r"\b(?:caravans?|merchant carts?|trading wagons?)\b", re.I),
    "traveler_in_need": re.compile(r"\b(?:travell?ers?|wanderers?)\b(?:\W+\w+){0,3}\W+(?:need(?:s|ing)? help|in trouble)\b|\b(?:travell?ers?|wanderers?) in need\b", re.I),
}
_MORE = re.compile(r"\b(?:another|different|something else|other rumors?|more rumors?)\b", re.I)


def narrative_anchor(story, pool_event_id=None):
    """Compact topic cues rather than another NPC's complete spoken reply.

    This is a deliberately lightweight, deterministic summary. It is NOT an
    alternate dialogue generator; it supplies shared subjects to the LLM.
    """
    if pool_event_id in RUMOR_TEMPLATES:
        return "Pool subject: " + pool_event_id.replace("_", " ") + ". " + RUMOR_TEMPLATES[pool_event_id]
    text = str(story or "")
    # Keep named entities so a place/character has the same identity across NPCs.
    # Do not copy whole quoted utterances or long sentence fragments.
    names = re.findall(r"\b(?:[A-Z][a-z]+(?:['’]s)?\s+){1,3}[A-Z][a-z]+\b", text)
    names = list(dict.fromkeys(n for n in names if not n.startswith(("Some ", "They ", "What ", "Might ", "I've "))))[:4]
    # Content-bearing words indicate topic; avoid emitting verbatim prose.
    words = re.findall(r"[a-z]{4,}", text.lower())
    cues = list(dict.fromkeys(w for w in words if w not in _STOP and w not in {
        "there", "these", "their", "might", "would", "perhaps", "truth", "could",
        "claim", "heard", "talk", "folks", "often", "among", "along", "about",
    }))[:14]
    return "Subject cues: " + ", ".join(cues) + (". Names: " + "; ".join(names) if names else "")


def excessive_shared_overlap(reply, prior_reply):
    """Reject extended verbatim copying, not references to shared lore/names."""
    words = lambda t: re.findall(r"[a-z0-9']+", str(t or "").lower())
    a, b = words(reply), words(prior_reply)
    if min(len(a), len(b)) < 18:
        return False
    # An eight-word exact sequence is strong evidence of copied narration.
    prior = {tuple(b[i:i + 8]) for i in range(len(b) - 7)}
    duplicated = sum(tuple(a[i:i + 8]) in prior for i in range(len(a) - 7))
    return duplicated >= 3 and duplicated / max(1, len(a) - 7) >= .18


def pool_subject(text):
    matches = [event_id for event_id, pattern in _POOL_SUBJECTS.items() if pattern.search(text or "")]
    return matches[0] if len(matches) == 1 else None


def selected_pool_alignment(reply, linked_rumors):
    """Validate that a selected pool *subject* survived the retelling.

    This operates on canonical event IDs for any NPC, not on sample dialog or
    a particular generated rumor. It tests the chosen subject, not wording.
    Creative details outside the pool are allowed when they expand the topic.
    """
    subjects = [r.get("pool_event_id") for r in (linked_rumors or [])
                if isinstance(r, dict) and r.get("pool_event_id") in _POOL_SUBJECTS]
    if not subjects:
        return True
    # A pool-specific topical question or one selected pool story gets a single
    # anchor. Multiple supplied anchors may be combined in a free narrative.
    return any(_POOL_SUBJECTS[subject].search(str(reply or "")) for subject in subjects)




class WorldRumorMemory:
    def __init__(self, memory_path="memory.json"):
        self.path = Path(memory_path).with_name("world_rumors.json")
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.rumors = data if isinstance(data, list) else []
        except (OSError, ValueError):
            self.rumors = []

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.rumors, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)

    def _similarity(self, text, record):
        a, b = keywords(text), set(record.get("keywords", []))
        # Two topic-bearing words are needed to avoid conflating unrelated tales.
        if not a or not b:
            return 0
        shared = len(a & b)
        return shared / max(1, min(len(a), len(b))) if shared >= 2 else 0

    def find(self, query, limit=2):
        ranked = [(self._similarity(query, r), r) for r in self.rumors]
        ranked = [(score, r) for score, r in ranked if score >= .35]
        ranked.sort(key=lambda x: (x[0], x[1].get("created_at", "")), reverse=True)
        return [r for _, r in ranked[:limit]]

    def _seed_pool_rumor(self, event_id, npc, location=""):
        """Register a public candidate from EventAgent's five template IDs.

        A candidate is a RUMOR, never a completed mechanical game event.
        The pool record is shared between NPCs; their spoken versions differ.
        """
        for rumor in self.rumors:
            if rumor.get("pool_event_id") == event_id and rumor.get("location", "") == location:
                return rumor
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        next_id = 1 + max((int(re.search(r"(\d+)$", r.get("rumor_id", "")).group(1))
                           for r in self.rumors if re.search(r"\d+$", r.get("rumor_id", ""))), default=0)
        template = RUMOR_TEMPLATES[event_id]
        rumor = {
            "rumor_id": f"rumor_{next_id:04d}", "story": template,
            "keywords": sorted(keywords(template)), "origin_npc": npc,
            "source_type": "static_pool", "pool_event_id": event_id,
            "status": "unconfirmed", "location": location,
            "known_by": [], "versions": {}, "created_at": now,
        }
        self.rumors.append(rumor)
        self._save()
        print(f"[WORLD RUMOR POOL] selected={event_id} rumor_id={rumor['rumor_id']} status=unconfirmed")
        return rumor

    def clear(self):
        """New-game reset: erase this world's rumors, not only NPC dialogue."""
        self.rumors = []
        self._save()

    def _location_records(self, location):
        return [r for r in self.rumors
                if not r.get("location") or r.get("location") == location]

    def _select_general_news(self, npc, location):
        """Pick a fresh public subject; never default to newest persisted rumor.

        Pool candidates are tracked by canonical ID. Existing generated stories
        compete after unused pool subjects; after exhaustion, choose a subject
        least recently *told* by this NPC rather than repeatedly returning the
        same global record. Merely registering a seed does not count as telling it.
        """
        records = self._location_records(location)
        pool_by_id = {r.get("pool_event_id"): r for r in records
                      if r.get("pool_event_id") in RUMOR_TEMPLATES}
        never_told_pool = [eid for eid in RUMOR_TEMPLATES
                           if not pool_by_id.get(eid, {}).get("versions")]
        if never_told_pool:
            chosen = random.choice(never_told_pool)
            return self._seed_pool_rumor(chosen, npc, location)
        # Existing shared tales are eligible, but don't monopolize the pool.
        candidates = list(records)
        if not candidates:
            return self._seed_pool_rumor(random.choice(list(RUMOR_TEMPLATES)), npc, location)
        least = min(int(r.get("tell_count", 0)) for r in candidates)
        choices = [r for r in candidates if int(r.get("tell_count", 0)) == least]
        last = getattr(self, "_last_general_subject", {}).get((npc, location))
        alternatives = [r for r in choices if r["rumor_id"] != last]
        selected = random.choice(alternatives or choices)
        return selected

    def prepare_for_npc(self, npc, player_input, location=""):
        """Resolve a specific shared subject, or select one diverse news topic.

        A greeting does not trigger a rumor; a follow-up does not auto-select a
        different story. Only explicit broad news requests select a fresh topic.
        """
        npc = str(npc or "").strip().lower()
        if not npc:
            return []
        query = str(player_input or "")
        matches = self.find(query)
        pool_id = pool_subject(query)
        if pool_id and (_NEWS.search(query) or is_news_question(query) or
                        re.search(r"\b(?:heard|know|tell|about)\b", query, re.I)):
            matches = [self._seed_pool_rumor(pool_id, npc, location)]
        elif (_NEWS.search(query) or is_news_question(query)) and not matches:
            matches = [self._select_general_news(npc, location)]
        # Specific requests or references take precedence over selection; other
        # intents do not randomly launch a story.
        result = []
        for rumor in matches[:1]:
            versions = rumor.get("versions", {})
            own = versions.get(npc)
            result.append({
                "rumor_id": rumor["rumor_id"],
                "story_anchor": narrative_anchor(rumor["story"], rumor.get("pool_event_id")),
                "origin_npc": rumor.get("origin_npc"),
                "status": rumor.get("status", "unconfirmed"),
                "source_type": rumor.get("source_type", "llm_generated"),
                "pool_event_id": rumor.get("pool_event_id"),
                "pool_template": RUMOR_TEMPLATES.get(rumor.get("pool_event_id"), ""),
                "knowledge": "own_version" if own else "shared_rumor",
                "npc_version": own,
                # Only used by validators to detect copying. Never supplied as
                # a full monologue in the Qwen prompt.
                "origin_version": versions.get(rumor.get("origin_npc"), ""),
            })
        if result and (_NEWS.search(query) or is_news_question(query)) and not pool_id:
            self._last_general_subject = getattr(self, "_last_general_subject", {})
            self._last_general_subject[(npc, location)] = result[0]["rumor_id"]
        return result

    def record_dialogue(self, npc, player_input, npc_reply, delivered=None):
        """Keep a shared event ID while recording NPC-specific versions of it."""
        npc = str(npc or "").lower().strip()
        reply = str(npc_reply or "").strip()
        if not npc or not reply or (not _CUES.search(reply) and not delivered):
            return []
        links = delivered or []
        referenced = next((r for r in self.rumors
                           if any(x.get("rumor_id") == r.get("rumor_id") for x in links)
                           and (self._similarity(reply, r) >= .35 or
                                self._similarity(player_input, r) >= .35)
                           and (not r.get("pool_event_id") or
                                selected_pool_alignment(reply, [r]))), None)
        related = referenced or next((r for r in self.rumors if self._similarity(reply, r) >= .5 and (not r.get("pool_event_id") or selected_pool_alignment(reply, [r]))), None)
        # A provided pool anchor identifies the story even when Qwen retells it
        # without the exact original wording. Never infer a pool ID by copying
        # arbitrary narrative text; only a unique subject can match it.
        if related is None:
            candidate_id = pool_subject(reply)
            related = next((r for r in self.rumors if r.get("pool_event_id") == candidate_id), None) if candidate_id else None
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if related is None:
            next_id = 1 + max((int(re.search(r"(\d+)$", r.get("rumor_id", "")).group(1))
                               for r in self.rumors if re.search(r"\d+$", r.get("rumor_id", ""))), default=0)
            pool_id = pool_subject(reply)
            source = "static_pool" if pool_id else "llm_generated"
            related = {
                "rumor_id": f"rumor_{next_id:04d}", "story": reply[:650],
                "keywords": sorted(keywords(reply)), "origin_npc": npc,
                "source_type": source, "pool_event_id": pool_id,
                "status": "unconfirmed", "known_by": [npc, "player"],
                "versions": {}, "created_at": now,
            }
            self.rumors.append(related)
        related.setdefault("versions", {})[npc] = reply[:650]
        related["tell_count"] = int(related.get("tell_count", 0)) + 1
        if npc not in related.setdefault("known_by", []):
            related["known_by"].append(npc)
        # Keep the original topic as an anchor; never replace it with retellings.
        self._save()
        return [related["rumor_id"]]

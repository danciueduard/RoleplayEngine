# relationship_analyzer.py
import json
import httpx

NUMERIC_FIELDS = {
    "affection",
    "attraction",
    "trust",
    "respect",
    "resentment",
    "friendship",
}
CATEGORICAL_FIELDS = {
    "romantic_interest": {
        "none",
        "curious",
        "interested",
        "developing_feelings",
        "in_love",
    },
    "sexual_interest": {
        "none",
        "curious",
        "interested",
        "strong_attraction",
        "casual_encounters",
    },
}
ALLOWED_FIELDS = NUMERIC_FIELDS | set(CATEGORICAL_FIELDS)


async def analyze_relationship_changes(
    user_message: str,
    assistant_reply: str,
    characters: list,
    relationships: list,
    existing_proposals: list,
    ollama_url: str,
    model: str,
):
    """Return validated relationship suggestions without changing the database."""
    characters = [dict(item) for item in characters]
    relationships = [dict(item) for item in relationships]
    existing_proposals = [dict(item) for item in existing_proposals]

    by_id = {item["id"]: item for item in characters if "id" in item}
    by_name = {item["name"].casefold(): item for item in characters if "name" in item}
    current = {
        item["character_id"]: item
        for item in relationships
        if item.get("character_id") in by_id
    }
    if not current:
        return []

    pending = {
        (item.get("character_id"), item.get("field_name"))
        for item in existing_proposals
        if item.get("status", "pending") == "pending"
    }

    context = []
    for character_id, relationship in current.items():
        character = by_id[character_id]
        context.append(
            {
                "character_id": character_id,
                "name": character["name"],
                "personality": character.get("personality", ""),
                "background": character.get("background", ""),
                "relationship": {
                    field: relationship.get(field) for field in sorted(ALLOWED_FIELDS)
                },
            }
        )

    system_prompt = f"""You analyze changes in how independent roleplay characters feel toward the user.
The user plays themselves. The characters have autonomous personalities and emotions.
Only evaluate the LATEST exchange, using the character's actual words, actions, and established context.

CURRENT CHARACTERS AND RELATIONSHIPS:
{json.dumps(context, ensure_ascii=False, indent=2)}

RULES:
1. Most ordinary exchanges cause no measurable relationship change. Return an empty list when appropriate.
2. A polite remark, ordinary greeting, or simple question does not automatically increase affection or trust.
3. Propose only changes supported by specific behavior or events in this exchange. Do not invent events or motives.
4. Interpret reactions according to each character's personality and existing relationship.
5. Affection, attraction, trust, respect, resentment, friendship, romantic_interest, and sexual_interest are independent.
6. Sexual attraction or interest in casual encounters does not imply romance. Romance does not imply sexual interest.
7. Keep numeric changes small (usually 1 to 5 points). Exceptional, strongly supported events may justify up to 10 points.
8. Numeric values must remain between 0 and 100. Never change a numeric value by more than 10 in one exchange.
9. Categorical changes require clear, strong evidence and normally progress gradually. Never infer love or casual sexual intent from generic friendliness.
10. A character can forgive slowly; an apology need not erase resentment or restore trust immediately.
11. Only update characters actually involved or clearly affected by the exchange.
12. Return at most one proposal per character and field.
13. Return JSON only, using the structure below.

ALLOWED NUMERIC FIELDS: {json.dumps(sorted(NUMERIC_FIELDS))}
ALLOWED CATEGORICAL VALUES: {json.dumps({field: sorted(values) for field, values in CATEGORICAL_FIELDS.items()}, sort_keys=True)}

REQUIRED JSON STRUCTURE:
{{"proposals": [{{"character_name": "Isabella", "field_name": "trust", "proposed_value": 12, "reason": "Specific observed reason"}}]}}
If no meaningful change is justified, return {{"proposals": []}}.
"""
    payload = {
        "model": model,
        "think": False,
        "stream": False,
        "format": "json",
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"LATEST USER MESSAGE:\n{user_message}\n\n"
                    f"NARRATOR RESPONSE:\n{assistant_reply}\n\n"
                    "Identify justified relationship changes."
                ),
            },
        ],
        "options": {"num_predict": 700, "temperature": 0.1},
    }
    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.post(ollama_url, json=payload)
    response.raise_for_status()
    analysis = json.loads(response.json()["message"]["content"])
    if not isinstance(analysis, dict) or not isinstance(
        analysis.get("proposals", []), list
    ):
        raise ValueError("Relationship analyzer returned invalid proposals JSON")

    validated = []
    seen = set()
    for item in analysis.get("proposals", []):
        if not isinstance(item, dict):
            continue
        name = item.get("character_name")
        field = item.get("field_name")
        if not isinstance(name, str) or not isinstance(field, str):
            continue
        character = by_name.get(name.casefold())
        if character is None or field not in ALLOWED_FIELDS:
            continue
        character_id = character["id"]
        key = (character_id, field)
        if character_id not in current or key in pending or key in seen:
            continue
        old_value = current[character_id][field]
        proposed = item.get("proposed_value")
        if field in NUMERIC_FIELDS:
            if isinstance(proposed, bool) or not isinstance(proposed, int):
                continue
            if not 0 <= proposed <= 100 or abs(proposed - old_value) > 10:
                continue
        else:
            if (
                not isinstance(proposed, str)
                or proposed not in CATEGORICAL_FIELDS[field]
            ):
                continue
        if proposed == old_value:
            continue
        reason = item.get("reason", "")
        if not isinstance(reason, str) or not reason.strip():
            continue
        validated.append(
            {
                "character_id": character_id,
                "character_name": character["name"],
                "field_name": field,
                "old_value": str(old_value),
                "proposed_value": str(proposed),
                "reason": reason.strip()[:500],
            }
        )
        seen.add(key)
    return validated

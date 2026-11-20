# knowledge_analyzer.py

import json
import httpx

ALLOWED_KNOWLEDGE_STATUSES = {
    "knows",
    "believes",
    "suspects",
}


async def analyze_character_knowledge(
    user_message: str,
    assistant_reply: str,
    characters: list,
    world_events: list,
    existing_knowledge: list,
    existing_proposals: list,
    ollama_url: str,
    model: str,
):
    """
    Analyze a roleplay exchange and identify changes
    to individual characters' knowledge.

    This function does not modify the database.
    It returns validated knowledge proposals for review.
    """

    character_data = [dict(character) for character in characters]

    world_event_data = [dict(event) for event in world_events]

    knowledge_data = [dict(record) for record in existing_knowledge]

    proposal_data = [dict(proposal) for proposal in existing_proposals]

    character_context = json.dumps(
        character_data,
        ensure_ascii=False,
        indent=2,
    )

    event_context = json.dumps(
        world_event_data,
        ensure_ascii=False,
        indent=2,
    )

    knowledge_context = json.dumps(
        knowledge_data,
        ensure_ascii=False,
        indent=2,
    )

    proposal_context = json.dumps(
        proposal_data,
        ensure_ascii=False,
        indent=2,
    )

    system_prompt = f"""
You are a careful character-knowledge analyzer for
a persistent roleplay simulation engine.

Your task is to identify what individual characters
learned during the latest roleplay exchange.

CHARACTERS:
{character_context}

ACCEPTED WORLD EVENTS:
{event_context}

EXISTING CHARACTER KNOWLEDGE:
{knowledge_context}

EXISTING PENDING KNOWLEDGE PROPOSALS:
{proposal_context}

CORE PRINCIPLE:

Each character has an individual understanding of
the world.

A character can only know information they personally
experienced, were told, discovered, or otherwise
obtained through an established source.

Never give a character knowledge simply because the
information appears in the narrator's response.

KNOWLEDGE STATUS DEFINITIONS:

1. "knows":
   The character directly witnessed the information,
   received reliable confirmation, or otherwise has
   established knowledge of it.

2. "believes":
   The character accepts the information as plausible
   based on something they heard or were told, but
   has not independently verified it.

3. "suspects":
   The character considers something possible based
   on observations, clues, or incomplete evidence.

RULES:

1. Only propose knowledge changes supported by the
   latest roleplay exchange.

2. Identify which character actually obtained each
   piece of information.

3. Do not give characters information they could not
   reasonably know.

4. Characters who were absent from an event do not
   automatically learn about it.

5. A character's private thoughts and suspicions
   remain private unless communicated or discovered.

6. Distinguish between witnessing an event and
   understanding what caused it.

7. A character may know that a sound occurred while
   only suspecting who caused it.

8. A character may believe a rumor without knowing
   whether the rumor is true.

9. Do not invent conversations, witnesses, messages,
   discoveries, or information sources.

10. Do not create proposals for characters whose
    knowledge has not changed.

11. Do not duplicate existing accepted knowledge.

12. Do not duplicate existing pending proposals.

13. If a character already has a knowledge record
    about an event, do not create another proposal
    for the same character and event.

14. Use character names and IDs exactly as supplied.

15. Only reference event IDs present in ACCEPTED
    WORLD EVENTS.

16. If an event ID cannot be established, do not
    create a proposal for it.

17. Keep knowledge details concise and factual.

18. Describe how the character obtained the information
    in source_description.

19. Return valid JSON only.

REQUIRED JSON STRUCTURE:

{{
    "proposals": [
        {{
            "character_id": 1,
            "event_id": 4,
            "knowledge_status": "knows",
            "knowledge_detail": "Isabella personally heard the metallic sound.",
            "source_description": "Personally heard during the scene."
        }}
    ]
}}

If no new knowledge was obtained, return:

{{
    "proposals": []
}}
"""

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": f"""
LATEST USER MESSAGE:
{user_message}

NARRATOR'S RESPONSE:
{assistant_reply}

Identify new character knowledge proposals.
""",
        },
    ]

    payload = {
        "model": model,
        "think": False,
        "stream": False,
        "format": "json",
        "messages": messages,
        "options": {
            "num_predict": 900,
            "temperature": 0.1,
        },
    }

    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.post(
            ollama_url,
            json=payload,
        )

    response.raise_for_status()

    result = response.json()
    content = result["message"]["content"]

    analysis = json.loads(content)

    if not isinstance(analysis, dict):
        raise ValueError("Knowledge analyzer returned an invalid JSON object.")

    raw_proposals = analysis.get("proposals", [])

    if not isinstance(raw_proposals, list):
        raise ValueError("Analyzer proposals must be a list.")

    existing_records = {
        (
            record.get("character_id"),
            record.get("event_id"),
        )
        for record in knowledge_data
    }

    existing_records.update(
        (
            proposal.get("character_id"),
            proposal.get("event_id"),
        )
        for proposal in proposal_data
        if proposal.get("status", "pending") == "pending"
    )

    valid_character_ids = {character.get("id") for character in character_data}

    valid_event_ids = {event.get("id") for event in world_event_data}

    validated_proposals = []

    for proposal in raw_proposals:

        if not isinstance(proposal, dict):
            continue

        character_id = proposal.get("character_id")
        event_id = proposal.get("event_id")
        knowledge_status = proposal.get("knowledge_status")

        knowledge_detail = proposal.get(
            "knowledge_detail",
            "",
        )

        source_description = proposal.get(
            "source_description",
            "",
        )

        if character_id not in valid_character_ids:
            continue

        if event_id not in valid_event_ids:
            continue

        if (character_id, event_id) in existing_records:
            continue

        if knowledge_status not in ALLOWED_KNOWLEDGE_STATUSES:
            continue

        if not isinstance(knowledge_detail, str):
            continue

        if not isinstance(source_description, str):
            source_description = ""

        knowledge_detail = knowledge_detail.strip()

        if not knowledge_detail:
            continue

        validated_proposals.append(
            {
                "character_id": character_id,
                "event_id": event_id,
                "knowledge_status": knowledge_status,
                "knowledge_detail": knowledge_detail,
                "source_description": source_description.strip(),
            }
        )

        existing_records.add((character_id, event_id))

    return validated_proposals

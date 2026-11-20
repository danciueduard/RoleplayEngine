# state_analyzer.py
import json
import httpx

ALLOWED_FIELDS = {
    "location",
    "physical_condition",
    "emotional_state",
    "appearance",
    "clothing",
    "possessions",
    "current_goals",
}


async def analyze_state_changes(
    user_message: str,
    assistant_reply: str,
    character_states: list,
    ollama_url: str,
    model: str,
):
    """
    Analyze a roleplay exchange and return proposed character
    state changes.

    This function does not modify the database.
    """

    # Convert database records into ordinary dictionaries
    states = [dict(state) for state in character_states]

    # Build a list of characters the analyzer is allowed to modify
    character_names = [state["name"] for state in states]

    if not character_names:
        return []

    # Prepare current state information
    current_states = json.dumps(
        states,
        ensure_ascii=False,
        indent=2,
    )

    # Build analyzer instructions
    system_prompt = f"""
You are a careful state-change analyzer for a persistent
roleplay simulation engine.

Your job is to identify meaningful changes to established
character states based on the latest roleplay exchange.

CURRENT CHARACTER STATES:
{current_states}

CHARACTERS YOU ARE ALLOWED TO UPDATE:
{json.dumps(character_names, ensure_ascii=False)}

ALLOWED STATE FIELDS:
{json.dumps(sorted(ALLOWED_FIELDS))}

RULES:

1. Only propose changes supported by the user's message
   or the narrator's response.

2. Use the narrator's response as evidence of what actually
   happened in the scene.

3. Do not invent events, actions, possessions, injuries,
   emotions, or changes.

4. Only propose a change when the new information differs
   meaningfully from the currently stored value.

5. Use the exact character names provided above.

6. Only use the allowed state fields.

7. Do not propose changes for characters who were not
   involved in the events.

8. Be conservative. If an event is ambiguous, leave it out.

9. For old_value, reproduce the currently stored value.

10. For proposed_value, provide the complete new value
    that should replace the old value.

11. Keep reasons short and explain what happened.

12. Return valid JSON only. Do not include Markdown,
    explanations, or text outside the JSON object.

REQUIRED JSON STRUCTURE:

{{
    "proposals": [
        {{
            "character_name": "Isabella",
            "field_name": "location",
            "old_value": "Venice courtyard",
            "proposed_value": "Nearby café",
            "reason": "The narrator describes Isabella entering the café."
        }}
    ]
}}

If no meaningful changes occurred, return:

{{
    "proposals": []
}}
"""

    # Build the analysis request
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

Identify any supported character state changes.
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
            "num_predict": 700,
            "temperature": 0.1,
        },
    }

    # Ask Ollama to analyze the exchange
    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.post(
            ollama_url,
            json=payload,
        )

    response.raise_for_status()

    result = response.json()
    content = result["message"]["content"]

    # Parse the returned JSON
    analysis = json.loads(content)

    if not isinstance(analysis, dict):
        raise ValueError("Analyzer returned an invalid JSON object.")

    raw_proposals = analysis.get("proposals", [])

    if not isinstance(raw_proposals, list):
        raise ValueError("Analyzer proposals must be a list.")

    # Validate the proposed changes
    valid_names = {name.casefold(): name for name in character_names}

    current_state_lookup = {state["name"].casefold(): state for state in states}

    validated_proposals = []

    for proposal in raw_proposals:

        if not isinstance(proposal, dict):
            continue

        character_name = proposal.get("character_name")
        field_name = proposal.get("field_name")
        proposed_value = proposal.get("proposed_value")

        if not isinstance(character_name, str):
            continue

        if not isinstance(field_name, str):
            continue

        if not isinstance(proposed_value, str):
            continue

        character_key = character_name.casefold()

        if character_key not in valid_names:
            continue

        if field_name not in ALLOWED_FIELDS:
            continue

        proposed_value = proposed_value.strip()

        if not proposed_value:
            continue

        character_name = valid_names[character_key]
        current_state = current_state_lookup[character_key]

        old_value = current_state.get(field_name) or ""

        # Ignore changes that do not actually change the value
        if old_value.strip().casefold() == proposed_value.casefold():
            continue

        reason = proposal.get("reason", "")

        if not isinstance(reason, str):
            reason = ""

        validated_proposals.append(
            {
                "character_name": character_name,
                "field_name": field_name,
                "old_value": old_value,
                "proposed_value": proposed_value,
                "reason": reason.strip(),
            }
        )

    return validated_proposals

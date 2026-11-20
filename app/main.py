# main.py

import json
import sqlite3
import httpx
from fastapi import FastAPI, HTTPException
from pathlib import Path
from contextlib import contextmanager
from pydantic import BaseModel, Field
from app.state_analyzer import analyze_state_changes
from app.event_analyzer import analyze_world_events
from app.knowledge_analyzer import analyze_character_knowledge
from app.relationship_analyzer import analyze_relationship_changes
from app.memory_analyzer import (
    ensure_memory_table,
    load_memories,
    analyze_conversation_memories,
)

app = FastAPI(title="Roleplay Engine")
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen-aggressive"
DATABASE = Path(__file__).resolve().parent.parent / "database" / "roleplay.db"
# Database connection


@contextmanager
def get_db():

    connection = sqlite3.connect(DATABASE)

    connection.row_factory = sqlite3.Row

    connection.execute("PRAGMA foreign_keys = ON")

    try:

        yield connection

        connection.commit()

    except Exception:

        connection.rollback()

        raise

    finally:

        connection.close()


# Request models


class ChatRequest(BaseModel):

    campaign_id: int

    message: str


class CampaignCreate(BaseModel):

    name: str

    description: str = ""


class WorldEventReview(BaseModel):

    certainty: str = Field(default="confirmed", pattern="^(confirmed|uncertain)$")


class WorldEventCreate(BaseModel):

    event_type: str = "general"

    description: str = Field(min_length=1)

    location: str = ""

    participants: list[str] = Field(default_factory=list)

    occurred_at: str = ""


class CharacterKnowledgeCreate(BaseModel):

    character_id: int

    knowledge_status: str = Field(pattern="^(knows|believes|suspects)$")

    knowledge_detail: str = ""

    source_description: str = ""


class CharacterKnowledgeProposalReview(BaseModel):

    pass


class PlayerDiscoveryCreate(BaseModel):

    discovery_note: str = ""


class CharacterCreate(BaseModel):

    name: str

    age: int | None = Field(default=None, ge=1, le=120)

    description: str = ""

    personality: str = ""

    background: str = ""


class CharacterStateUpdate(BaseModel):

    location: str = ""

    physical_condition: str = ""

    emotional_state: str = ""

    appearance: str = ""

    clothing: str = ""

    possessions: str = ""

    current_goals: str = ""


class CharacterRelationshipUpdate(BaseModel):

    affection: int | None = Field(default=None, ge=0, le=100)

    attraction: int | None = Field(default=None, ge=0, le=100)

    trust: int | None = Field(default=None, ge=0, le=100)

    respect: int | None = Field(default=None, ge=0, le=100)

    resentment: int | None = Field(default=None, ge=0, le=100)

    friendship: int | None = Field(default=None, ge=0, le=100)

    romantic_interest: str | None = Field(
        default=None,
        pattern="^(none|curious|interested|developing_feelings|in_love)$",
    )

    sexual_interest: str | None = Field(
        default=None,
        pattern="^(none|curious|interested|strong_attraction|casual_encounters)$",
    )

    reason: str = ""


# Apply relationship suggestions transactionally, with an audit trail.
def apply_relationship_proposal(
    connection, campaign_id, character_id, field_name, old_value, proposed_value, reason
):
    numeric = {
        "affection",
        "attraction",
        "trust",
        "respect",
        "resentment",
        "friendship",
    }
    categorical = {
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
    if field_name not in numeric and field_name not in categorical:
        return False
    if field_name in numeric:
        try:
            old_number, new_number = int(old_value), int(proposed_value)
        except (TypeError, ValueError):
            return False
        if not (0 <= new_number <= 100 and 0 < abs(new_number - old_number) <= 10):
            return False
        proposed_value = new_number
    elif proposed_value not in categorical[field_name]:
        return False

    current = connection.execute(
        f"SELECT {field_name} FROM character_relationships WHERE campaign_id = ? AND character_id = ?",
        (campaign_id, character_id),
    ).fetchone()
    if current is None or str(current[field_name]) != str(old_value):
        return False

    connection.execute(
        f"UPDATE character_relationships SET {field_name} = ?, updated_at = CURRENT_TIMESTAMP "
        "WHERE campaign_id = ? AND character_id = ?",
        (proposed_value, campaign_id, character_id),
    )
    connection.execute(
        """INSERT INTO relationship_history
           (campaign_id, character_id, field_name, old_value, new_value, reason, source_description)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            campaign_id,
            character_id,
            field_name,
            str(old_value),
            str(proposed_value),
            reason,
            "Automatic relationship analyzer",
        ),
    )
    return True


# Home


@app.get("/")
def home():

    return {"status": "Roleplay Engine is running"}


# AI chat with campaign history


@app.post("/chat")
async def chat(request: ChatRequest):

    with get_db() as connection:

        campaign = connection.execute(
            """

            SELECT id, name, description

            FROM campaigns

            WHERE id = ?

            """,
            (request.campaign_id,),
        ).fetchone()

        if campaign is None:

            raise HTTPException(
                status_code=404,
                detail="Campaign not found",
            )

        characters = connection.execute(
            """

            SELECT id, name, age, description,

                   personality, background, current_state

            FROM characters

            WHERE campaign_id = ?

            ORDER BY id ASC

            """,
            (request.campaign_id,),
        ).fetchall()

        character_states = connection.execute(
            """

            SELECT

                characters.id AS character_id,

                characters.name,

                character_states.location,

                character_states.physical_condition,

                character_states.emotional_state,

                character_states.appearance,

                character_states.clothing,

                character_states.possessions,

                character_states.current_goals

            FROM characters

            LEFT JOIN character_states

                ON characters.id = character_states.character_id

            WHERE characters.campaign_id = ?

            ORDER BY characters.id ASC

            """,
            (request.campaign_id,),
        ).fetchall()

        conversation_memories = load_memories(connection, request.campaign_id)

        history = connection.execute(
            """

            SELECT role, content

            FROM messages

            WHERE campaign_id = ?

            ORDER BY id DESC

            LIMIT 20

            """,
            (request.campaign_id,),
        ).fetchall()

        world_events = connection.execute(
            """

            SELECT

                id,

                event_type,

                description,

                location,

                participants,

                occurred_at,

                certainty

            FROM world_events

            WHERE campaign_id = ?

              AND status = 'accepted'

            ORDER BY id ASC

            """,
            (request.campaign_id,),
        ).fetchall()

        character_knowledge = connection.execute(
            """

            SELECT

                character_knowledge.character_id,

                characters.name AS character_name,

                world_events.id AS event_id,

                world_events.description AS world_event,

                character_knowledge.knowledge_status,

                character_knowledge.knowledge_detail,

                character_knowledge.source_description

            FROM character_knowledge

            JOIN characters

                ON character_knowledge.character_id = characters.id

            JOIN world_events

                ON character_knowledge.event_id = world_events.id

            WHERE characters.campaign_id = ?

              AND world_events.status = 'accepted'

            ORDER BY character_knowledge.id ASC

            """,
            (request.campaign_id,),
        ).fetchall()

        player_discoveries = connection.execute(
            """

            SELECT

                world_events.id AS event_id,

                world_events.description AS world_event,

                player_discoveries.discovery_note

            FROM player_discoveries

            JOIN world_events

                ON player_discoveries.event_id = world_events.id

            WHERE player_discoveries.campaign_id = ?

              AND world_events.status = 'accepted'

            ORDER BY player_discoveries.id ASC

            """,
            (request.campaign_id,),
        ).fetchall()

        existing_events = connection.execute(
            """

            SELECT

                id,

                event_type,

                description,

                location,

                participants,

                occurred_at,

                status,

                certainty

            FROM world_events

            WHERE campaign_id = ?

            ORDER BY id ASC

            """,
            (request.campaign_id,),
        ).fetchall()

        pending_knowledge_proposals = connection.execute(
            """

            SELECT

                character_id,

                event_id,

                knowledge_status,

                knowledge_detail,

                source_description,

                status

            FROM character_knowledge_proposals

            WHERE campaign_id = ?

              AND status = 'pending'

            ORDER BY id ASC

            """,
            (request.campaign_id,),
        ).fetchall()

        # Apply any older pending relationship proposals before building narrator context.
        # Stale proposals are rejected so they cannot overwrite newer relationship values.
        older_proposals = connection.execute(
            """SELECT * FROM relationship_change_proposals
               WHERE campaign_id = ? AND status = 'pending' ORDER BY id ASC""",
            (request.campaign_id,),
        ).fetchall()
        for proposal in older_proposals:
            applied = apply_relationship_proposal(
                connection,
                request.campaign_id,
                proposal["character_id"],
                proposal["field_name"],
                proposal["old_value"],
                proposal["proposed_value"],
                proposal["reason"],
            )
            connection.execute(
                "UPDATE relationship_change_proposals SET status = ? WHERE id = ?",
                ("accepted" if applied else "rejected", proposal["id"]),
            )
        if older_proposals:
            print(
                f"Processed {len(older_proposals)} older relationship proposal(s).",
                flush=True,
            )

        # Relationship values are private engine context for the narrator.

        relationships = connection.execute(
            """

            SELECT cr.character_id, ch.name AS character_name,

                   cr.affection, cr.attraction, cr.trust, cr.respect,

                   cr.resentment, cr.friendship,

                   cr.romantic_interest, cr.sexual_interest

            FROM character_relationships AS cr

            JOIN characters AS ch ON ch.id = cr.character_id

            WHERE cr.campaign_id = ? AND ch.campaign_id = ?

            ORDER BY ch.id ASC

            """,
            (request.campaign_id, request.campaign_id),
        ).fetchall()

        pending_relationship_proposals = connection.execute(
            """

            SELECT character_id, field_name, status

            FROM relationship_change_proposals

            WHERE campaign_id = ? AND status = 'pending'

            """,
            (request.campaign_id,),
        ).fetchall()

    character_details = []

    for character in characters:

        character_details.append(
            {
                "id": character["id"],
                "name": character["name"],
                "age": character["age"],
                "description": character["description"],
                "personality": character["personality"],
                "background": character["background"],
                "previous_notes": character["current_state"],
            }
        )

    character_states_data = [dict(state) for state in character_states]

    world_events_data = []

    for event in world_events:

        item = dict(event)

        try:

            item["participants"] = json.loads(item["participants"] or "[]")

        except (json.JSONDecodeError, TypeError):

            item["participants"] = []

        world_events_data.append(item)

    character_knowledge_data = [dict(record) for record in character_knowledge]

    player_discoveries_data = [dict(record) for record in player_discoveries]

    relationships_data = [dict(item) for item in relationships]

    system_prompt = f"""

You are the narrator and storyteller for a persistent

roleplay simulation campaign.

CAMPAIGN:

{campaign['name']}

CAMPAIGN DESCRIPTION:

{campaign['description']}

CHARACTER PROFILES:

{json.dumps(character_details, ensure_ascii=False, indent=2)}

CURRENT CHARACTER STATES:

{json.dumps(character_states_data, ensure_ascii=False, indent=2)}

OBJECTIVE WORLD TRUTH:

{json.dumps(world_events_data, ensure_ascii=False, indent=2)}

INDIVIDUAL CHARACTER KNOWLEDGE:

{json.dumps(character_knowledge_data, ensure_ascii=False, indent=2)}

CHARACTER-SPECIFIC CONVERSATION MEMORIES:

{json.dumps(conversation_memories, ensure_ascii=False, indent=2)}

Each memory belongs ONLY to its named character. Preserve uncertainty and do not transfer private memories to other characters.

PLAYER DISCOVERIES:

{json.dumps(player_discoveries_data, ensure_ascii=False, indent=2)}



CURRENT RELATIONSHIPS WITH THE USER (PRIVATE ENGINE STATE):

{json.dumps(relationships_data, ensure_ascii=False, indent=2)}



These relationship values describe each character's current feelings.

Use them to guide behavior, but do not reveal hidden feelings or numeric

scores through dialogue unless the character has a credible reason.

The user plays themselves. All named characters are independent NPCs.

The user controls world events and major narrative outcomes.

Never speak, decide, or act on the user's behalf.

IMPORTANT KNOWLEDGE RULES:

1. OBJECTIVE WORLD TRUTH:

These records describe events that actually happened

within the campaign.

Use them to maintain continuity and understand the

underlying situation.

Events marked uncertain represent unresolved information.

Preserve their uncertainty.

2. CHARACTER KNOWLEDGE:

Each character has their own individual knowledge.

A character may know, believe, or suspect something.

Respect the recorded knowledge_status exactly.

A character who suspects something must not behave

as though they have confirmed proof.

Characters must not automatically know events simply

because those events appear in objective world truth.

Characters must remember conversations in which they

participated and information they received from others.

Distinguish between:

- Knowing that someone made a statement.

- Believing that the statement is true.

- Suspecting something based on available clues.

3. PLAYER DISCOVERIES:

These records describe information the player has

personally discovered.

The player's knowledge and each character's knowledge

are separate.

Do not automatically transfer player discoveries

to characters.

4. SECRETS:

Keep undiscovered information secret.

Do not reveal hidden events through dialogue or

narration unless the current scene provides a

believable way for the player to discover them.

5. INFORMATION SOURCES:

Characters may learn new information through dialogue,

observation, evidence, or events during the story.

Pay particular attention to conversations.

When one character tells another something, the

receiving character should remember being told,

even when they remain uncertain about its truth.

Never invent conversations, observations, evidence,

or information sources.

6. CONTINUITY:

Respect established campaign events, character

personalities, backgrounds, relationships, memories

and current states.

Do not silently rewrite established facts.

INTERACTIVE ROLEPLAY GUIDELINES:

1. This is an interactive roleplay, not a novel.

2. The user participates as themselves.

Never write dialogue, decisions, thoughts, feelings

or actions on behalf of the user.

3. You control the other characters and the surrounding

world.

4. Prioritize natural, realistic dialogue.

Characters should speak directly to the user and

to each other.

5. Keep narration brief.

Use descriptions only when needed to establish actions,

expressions, reactions or changes in the surroundings.

6. Avoid long literary passages, lengthy descriptions

and summaries of conversations.

7. Let conversations develop naturally through

questions, answers, disagreements, misunderstandings,

emotional reactions and interruptions.

8. Give every character a distinct personality,

voice, opinions, motivations and conversational style.

9. Characters may initiate conversations, ask questions,

disagree, express feelings and take reasonable

independent actions.

10. Give the user room to respond.

Do not advance through several major conversational

or narrative events in a single reply.

11. Never assume the user agrees, complies, follows

someone or takes an action unless they choose to do so.

12. Keep responses concise, natural and conversational.

Prioritize meaningful interaction over descriptive prose.

13. ADAPTIVE RESPONSE LENGTH AND STYLE:

Match the length and detail of each response to the

user's contribution.

- Casual remarks and simple observations should

  usually receive short, natural replies.

- Simple questions should receive direct answers.

- Complex questions deserve thoughtful and detailed

  answers when appropriate.

- Emotional statements should receive emotionally

  appropriate responses.

- Jokes, teasing and light conversation should feel

  spontaneous and natural.

Do not turn every user message into a major scene.

Do not force dramatic reactions into ordinary

conversations.

Characters should be allowed to respond with a single

sentence when that is all the situation requires.

Avoid unnecessary narration, repetition and

overexplaining.

Let the conversation flow naturally, like a real

conversation between people.

"""

    history = list(reversed(history))

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        }
    ]

    messages.extend(
        [
            {
                "role": item["role"],
                "content": item["content"],
            }
            for item in history
        ]
    )

    messages.append(
        {
            "role": "user",
            "content": request.message,
        }
    )

    payload = {
        "model": MODEL,
        "think": False,
        "stream": False,
        "messages": messages,
        "options": {
            "num_ctx": 8192,
            "num_predict": 500,
            "temperature": 0.8,
        },
    }

    try:

        async with httpx.AsyncClient(timeout=300.0) as client:

            response = await client.post(
                OLLAMA_URL,
                json=payload,
            )

        response.raise_for_status()

        result = response.json()

        reply = result["message"]["content"]

        # Ollama diagnostics: identify why a reply might stop mid-sentence.
        print("OLLAMA STOP REASON:", result.get("done_reason"), flush=True)
        print("OLLAMA OUTPUT TOKENS:", result.get("eval_count"), flush=True)
        print("OLLAMA PROMPT TOKENS:", result.get("prompt_eval_count"), flush=True)

    except httpx.HTTPError as error:

        raise HTTPException(
            status_code=502,
            detail=f"Ollama connection error: {error}",
        )

    except (KeyError, ValueError):

        raise HTTPException(
            status_code=502,
            detail="Unexpected response from Ollama",
        )

    # Save the conversation before running the analyzers.

    with get_db() as connection:

        connection.executemany(
            """

            INSERT INTO messages (campaign_id, role, content)

            VALUES (?, ?, ?)

            """,
            [
                (request.campaign_id, "user", request.message),
                (request.campaign_id, "assistant", reply),
            ],
        )

    # Automatically extract and persist character-specific conversation memories.
    try:
        new_memories = await analyze_conversation_memories(
            user_message=request.message,
            assistant_reply=reply,
            characters=[dict(character) for character in characters],
            existing_memories=conversation_memories,
            ollama_url=OLLAMA_URL,
            model=MODEL,
        )
        if new_memories:
            with get_db() as connection:
                ensure_memory_table(connection)
                connection.executemany(
                    """INSERT INTO character_memories
                       (campaign_id, character_id, memory, knowledge_status, source_description)
                       VALUES (?, ?, ?, ?, ?)""",
                    [
                        (
                            request.campaign_id,
                            m["character_id"],
                            m["memory"],
                            m["knowledge_status"],
                            m["source_description"],
                        )
                        for m in new_memories
                    ],
                )
        print(
            f"Conversation memory saved {len(new_memories)} memory/memories.",
            flush=True,
        )
    except Exception as error:
        print(f"Conversation memory analysis failed: {error}", flush=True)

    # Analyze character state changes.

    try:

        proposals = await analyze_state_changes(
            user_message=request.message,
            assistant_reply=reply,
            character_states=character_states,
            ollama_url=OLLAMA_URL,
            model=MODEL,
        )

        character_ids = {
            state["name"]: state["character_id"] for state in character_states
        }

        applied_count = 0

        if proposals:
            with get_db() as connection:
                for proposal in proposals:
                    character_id = character_ids[proposal["character_name"]]
                    field_name = proposal["field_name"]
                    old_value = proposal["old_value"]
                    new_value = proposal["proposed_value"]

                    # Ensure a character state row exists.
                    connection.execute(
                        "INSERT OR IGNORE INTO character_states (character_id) VALUES (?)",
                        (character_id,),
                    )

                    # Apply only if the stored value has not changed since analysis.
                    cursor = connection.execute(
                        f"""UPDATE character_states
                            SET {field_name} = ?, updated_at = CURRENT_TIMESTAMP
                            WHERE character_id = ? AND COALESCE({field_name}, '') = ?""",
                        (new_value, character_id, old_value),
                    )
                    applied = cursor.rowcount == 1

                    # Keep an audit record using the existing proposals table.
                    connection.execute(
                        """INSERT INTO state_change_proposals
                           (campaign_id, character_id, field_name, old_value,
                            proposed_value, reason, status)
                           VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (
                            request.campaign_id,
                            character_id,
                            field_name,
                            old_value,
                            new_value,
                            proposal["reason"],
                            "accepted" if applied else "rejected",
                        ),
                    )
                    if applied:
                        applied_count += 1

        print(
            f"State analyzer suggested {len(proposals)} change(s); "
            f"automatically applied {applied_count}.",
            flush=True,
        )

    except Exception as error:

        print(f"State analysis failed: {error}")

    # Analyze new objective world events.

    try:

        event_proposals = await analyze_world_events(
            user_message=request.message,
            assistant_reply=reply,
            existing_events=existing_events,
            ollama_url=OLLAMA_URL,
            model=MODEL,
        )

        if event_proposals:

            with get_db() as connection:

                connection.executemany(
                    """

                    INSERT INTO world_events (

                        campaign_id,

                        event_type,

                        description,

                        location,

                        participants,

                        occurred_at,

                        status,

                        certainty

                    )

                    VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)

                    """,
                    [
                        (
                            request.campaign_id,
                            event["event_type"],
                            event["description"],
                            event["location"],
                            json.dumps(
                                event["participants"],
                                ensure_ascii=False,
                            ),
                            event["occurred_at"],
                            event["certainty"],
                        )
                        for event in event_proposals
                    ],
                )

        print(f"Event analyzer created {len(event_proposals)} proposal(s).")

    except Exception as error:

        print(f"Event analysis failed: {error}")

    # Analyze individual character knowledge.

    try:

        knowledge_proposals = await analyze_character_knowledge(
            user_message=request.message,
            assistant_reply=reply,
            characters=[dict(character) for character in characters],
            world_events=world_events_data,
            existing_knowledge=character_knowledge_data,
            existing_proposals=[
                dict(proposal) for proposal in pending_knowledge_proposals
            ],
            ollama_url=OLLAMA_URL,
            model=MODEL,
        )

        if knowledge_proposals:

            with get_db() as connection:

                connection.executemany(
                    """

                    INSERT INTO character_knowledge_proposals (

                        campaign_id,

                        character_id,

                        event_id,

                        knowledge_status,

                        knowledge_detail,

                        source_description,

                        status

                    )

                    VALUES (?, ?, ?, ?, ?, ?, 'pending')

                    """,
                    [
                        (
                            request.campaign_id,
                            proposal["character_id"],
                            proposal["event_id"],
                            proposal["knowledge_status"],
                            proposal["knowledge_detail"],
                            proposal["source_description"],
                        )
                        for proposal in knowledge_proposals
                    ],
                )

        print("Knowledge analyzer created " f"{len(knowledge_proposals)} proposal(s).")

    except Exception as error:

        print(f"Knowledge analysis failed: {error}")

    # Analyze changes in each NPC's relationship toward the user.

    try:

        relationship_proposals = await analyze_relationship_changes(
            user_message=request.message,
            assistant_reply=reply,
            characters=[dict(character) for character in characters],
            relationships=relationships_data,
            existing_proposals=[dict(item) for item in pending_relationship_proposals],
            ollama_url=OLLAMA_URL,
            model=MODEL,
        )

        applied_count = 0
        if relationship_proposals:
            with get_db() as connection:
                for item in relationship_proposals:
                    applied = apply_relationship_proposal(
                        connection,
                        request.campaign_id,
                        item["character_id"],
                        item["field_name"],
                        item["old_value"],
                        item["proposed_value"],
                        item["reason"],
                    )
                    # Keep the proposal record for auditing, already resolved.
                    connection.execute(
                        """INSERT INTO relationship_change_proposals
                           (campaign_id, character_id, field_name, old_value,
                            proposed_value, reason, status)
                           VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (
                            request.campaign_id,
                            item["character_id"],
                            item["field_name"],
                            item["old_value"],
                            item["proposed_value"],
                            item["reason"],
                            "accepted" if applied else "rejected",
                        ),
                    )
                    if applied:
                        applied_count += 1

        print(
            f"Relationship analyzer suggested {len(relationship_proposals)} change(s); "
            f"automatically applied {applied_count}.",
            flush=True,
        )
    except Exception as error:

        print(f"Relationship analysis failed: {error}")

    return {"reply": reply}


# Create campaign


@app.post("/campaigns")
def create_campaign(campaign: CampaignCreate):

    if not campaign.name.strip():

        raise HTTPException(status_code=400, detail="Campaign name cannot be empty")

    with get_db() as connection:

        cursor = connection.execute(
            """

            INSERT INTO campaigns (name, description)

            VALUES (?, ?)

            """,
            (campaign.name.strip(), campaign.description),
        )

        campaign_id = cursor.lastrowid

    return {
        "id": campaign_id,
        "name": campaign.name.strip(),
        "description": campaign.description,
    }


# Retrieve campaigns


@app.get("/campaigns")
def get_campaigns():

    with get_db() as connection:

        campaigns = connection.execute("""

            SELECT id, name, description, created_at

            FROM campaigns

            ORDER BY id DESC

            """).fetchall()

    return [dict(campaign) for campaign in campaigns]


# Create character


@app.post("/campaigns/{campaign_id}/characters")
def create_character(campaign_id: int, character: CharacterCreate):

    if not character.name.strip():

        raise HTTPException(status_code=400, detail="Character name cannot be empty")

    with get_db() as connection:

        campaign = connection.execute(
            "SELECT id FROM campaigns WHERE id = ?", (campaign_id,)
        ).fetchone()

        if campaign is None:

            raise HTTPException(status_code=404, detail="Campaign not found")

        try:

            cursor = connection.execute(
                """

                INSERT INTO characters (

                    campaign_id, name, age,

                    description, personality, background

                )

                VALUES (?, ?, ?, ?, ?, ?)

                """,
                (
                    campaign_id,
                    character.name.strip(),
                    character.age,
                    character.description,
                    character.personality,
                    character.background,
                ),
            )

        except sqlite3.IntegrityError:

            raise HTTPException(
                status_code=409, detail="A character with this name already exists"
            )

        character_id = cursor.lastrowid

    return {
        "id": character_id,
        "campaign_id": campaign_id,
        "name": character.name.strip(),
        "age": character.age,
        "description": character.description,
        "personality": character.personality,
        "background": character.background,
    }


# Retrieve characters


@app.get("/campaigns/{campaign_id}/characters")
def get_characters(campaign_id: int):

    with get_db() as connection:

        campaign = connection.execute(
            "SELECT id FROM campaigns WHERE id = ?", (campaign_id,)
        ).fetchone()

        if campaign is None:

            raise HTTPException(status_code=404, detail="Campaign not found")

        characters = connection.execute(
            """

            SELECT *

            FROM characters

            WHERE campaign_id = ?

            ORDER BY id ASC

            """,
            (campaign_id,),
        ).fetchall()

    return [dict(character) for character in characters]


# Retrieve conversation history


@app.get("/campaigns/{campaign_id}/messages")
def get_messages(campaign_id: int):

    with get_db() as connection:

        campaign = connection.execute(
            "SELECT id FROM campaigns WHERE id = ?", (campaign_id,)
        ).fetchone()

        if campaign is None:

            raise HTTPException(status_code=404, detail="Campaign not found")

        messages = connection.execute(
            """

            SELECT id, role, content, created_at

            FROM messages

            WHERE campaign_id = ?

            ORDER BY id ASC

            """,
            (campaign_id,),
        ).fetchall()

    return [dict(message) for message in messages]


@app.get("/campaigns/{campaign_id}/characters/{character_id}/state")
def get_character_state(campaign_id: int, character_id: int):

    with get_db() as connection:

        character = connection.execute(
            """

            SELECT id

            FROM characters

            WHERE id = ? AND campaign_id = ?

        """,
            (character_id, campaign_id),
        ).fetchone()

        if character is None:

            raise HTTPException(status_code=404, detail="Character not found")

        connection.execute(
            """

            INSERT OR IGNORE INTO character_states (character_id)

            VALUES (?)

        """,
            (character_id,),
        )

        state = connection.execute(
            """

            SELECT *

            FROM character_states

            WHERE character_id = ?

        """,
            (character_id,),
        ).fetchone()

        return dict(state)


@app.put("/campaigns/{campaign_id}/characters/{character_id}/state")
def update_character_state(
    campaign_id: int, character_id: int, state: CharacterStateUpdate
):

    with get_db() as connection:

        character = connection.execute(
            """

            SELECT id

            FROM characters

            WHERE id = ? AND campaign_id = ?

        """,
            (character_id, campaign_id),
        ).fetchone()

        if character is None:

            raise HTTPException(status_code=404, detail="Character not found")

        connection.execute(
            """

            INSERT INTO character_states (

                character_id,

                location,

                physical_condition,

                emotional_state,

                appearance,

                clothing,

                possessions,

                current_goals,

                updated_at

            )

            VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)



            ON CONFLICT(character_id) DO UPDATE SET

                location = excluded.location,

                physical_condition = excluded.physical_condition,

                emotional_state = excluded.emotional_state,

                appearance = excluded.appearance,

                clothing = excluded.clothing,

                possessions = excluded.possessions,

                current_goals = excluded.current_goals,

                updated_at = CURRENT_TIMESTAMP

        """,
            (
                character_id,
                state.location,
                state.physical_condition,
                state.emotional_state,
                state.appearance,
                state.clothing,
                state.possessions,
                state.current_goals,
            ),
        )

        connection.commit()

        saved_state = connection.execute(
            """

            SELECT *

            FROM character_states

            WHERE character_id = ?

        """,
            (character_id,),
        ).fetchone()

        return dict(saved_state)


# --------------------------------------------------

# STATE CHANGE PROPOSAL ENDPOINTS

# --------------------------------------------------


@app.get("/campaigns/{campaign_id}/proposals")
async def get_state_proposals(campaign_id: int):

    with get_db() as connection:

        proposals = connection.execute(
            """

            SELECT

                state_change_proposals.id,

                state_change_proposals.character_id,

                characters.name AS character_name,

                state_change_proposals.field_name,

                state_change_proposals.old_value,

                state_change_proposals.proposed_value,

                state_change_proposals.reason,

                state_change_proposals.status,

                state_change_proposals.created_at

            FROM state_change_proposals

            JOIN characters

                ON state_change_proposals.character_id = characters.id

            WHERE state_change_proposals.campaign_id = ?

              AND state_change_proposals.status = 'pending'

            ORDER BY state_change_proposals.id ASC

            """,
            (campaign_id,),
        ).fetchall()

    return {"proposals": [dict(proposal) for proposal in proposals]}


@app.post("/campaigns/{campaign_id}/proposals/{proposal_id}/accept")
async def accept_state_proposal(
    campaign_id: int,
    proposal_id: int,
):

    allowed_fields = {
        "location",
        "physical_condition",
        "emotional_state",
        "appearance",
        "clothing",
        "possessions",
        "current_goals",
    }

    with get_db() as connection:

        proposal = connection.execute(
            """

            SELECT *

            FROM state_change_proposals

            WHERE id = ?

              AND campaign_id = ?

            """,
            (proposal_id, campaign_id),
        ).fetchone()

        if proposal is None:

            raise HTTPException(
                status_code=404,
                detail="Proposal not found",
            )

        if proposal["status"] != "pending":

            raise HTTPException(
                status_code=409,
                detail="This proposal has already been processed",
            )

        field_name = proposal["field_name"]

        if field_name not in allowed_fields:

            raise HTTPException(
                status_code=400,
                detail="Invalid state field",
            )

        current_state = connection.execute(
            f"""

            SELECT {field_name}

            FROM character_states

            WHERE character_id = ?

            """,
            (proposal["character_id"],),
        ).fetchone()

        if current_state is None:

            raise HTTPException(
                status_code=404,
                detail="Character state not found",
            )

        current_value = current_state[field_name] or ""

        old_value = proposal["old_value"] or ""

        if current_value != old_value:

            raise HTTPException(
                status_code=409,
                detail=(
                    "The character's current state has changed "
                    "since this proposal was created. "
                    "Review the latest state before proceeding."
                ),
            )

        connection.execute(
            f"""

            UPDATE character_states

            SET {field_name} = ?,

                updated_at = CURRENT_TIMESTAMP

            WHERE character_id = ?

            """,
            (
                proposal["proposed_value"],
                proposal["character_id"],
            ),
        )

        connection.execute(
            """

            UPDATE state_change_proposals

            SET status = 'accepted'

            WHERE id = ?

            """,
            (proposal_id,),
        )

    return {
        "message": "State proposal accepted successfully",
        "proposal_id": proposal_id,
        "field_name": field_name,
        "new_value": proposal["proposed_value"],
    }


@app.post("/campaigns/{campaign_id}/proposals/{proposal_id}/reject")
async def reject_state_proposal(
    campaign_id: int,
    proposal_id: int,
):

    with get_db() as connection:

        proposal = connection.execute(
            """

            SELECT id, status

            FROM state_change_proposals

            WHERE id = ?

              AND campaign_id = ?

            """,
            (proposal_id, campaign_id),
        ).fetchone()

        if proposal is None:

            raise HTTPException(
                status_code=404,
                detail="Proposal not found",
            )

        if proposal["status"] != "pending":

            raise HTTPException(
                status_code=409,
                detail="This proposal has already been processed",
            )

        connection.execute(
            """

            UPDATE state_change_proposals

            SET status = 'rejected'

            WHERE id = ?

            """,
            (proposal_id,),
        )

    return {
        "message": "State proposal rejected successfully",
        "proposal_id": proposal_id,
    }


# --------------------------------------------------

# WORLD EVENT ENDPOINTS

# --------------------------------------------------


@app.get("/campaigns/{campaign_id}/knowledge-proposals")
async def get_knowledge_proposals(campaign_id: int):

    with get_db() as connection:

        campaign = connection.execute(
            """

            SELECT id

            FROM campaigns

            WHERE id = ?

            """,
            (campaign_id,),
        ).fetchone()

        if campaign is None:

            raise HTTPException(
                status_code=404,
                detail="Campaign not found",
            )

        proposals = connection.execute(
            """

            SELECT

                character_knowledge_proposals.id,

                character_knowledge_proposals.character_id,

                characters.name AS character_name,

                character_knowledge_proposals.event_id,

                world_events.description AS world_event,

                character_knowledge_proposals.knowledge_status,

                character_knowledge_proposals.knowledge_detail,

                character_knowledge_proposals.source_description,

                character_knowledge_proposals.status,

                character_knowledge_proposals.created_at

            FROM character_knowledge_proposals

            JOIN characters

                ON character_knowledge_proposals.character_id = characters.id

            JOIN world_events

                ON character_knowledge_proposals.event_id = world_events.id

            WHERE character_knowledge_proposals.campaign_id = ?

              AND character_knowledge_proposals.status = 'pending'

            ORDER BY character_knowledge_proposals.id ASC

            """,
            (campaign_id,),
        ).fetchall()

    return [dict(proposal) for proposal in proposals]


@app.post("/campaigns/{campaign_id}/knowledge-proposals/{proposal_id}/accept")
async def accept_knowledge_proposal(
    campaign_id: int,
    proposal_id: int,
):

    with get_db() as connection:

        proposal = connection.execute(
            """

            SELECT

                id,

                character_id,

                event_id,

                knowledge_status,

                knowledge_detail,

                source_description

            FROM character_knowledge_proposals

            WHERE id = ?

              AND campaign_id = ?

              AND status = 'pending'

            """,
            (proposal_id, campaign_id),
        ).fetchone()

        if proposal is None:

            raise HTTPException(
                status_code=404,
                detail="Pending knowledge proposal not found",
            )

        event = connection.execute(
            """

            SELECT id

            FROM world_events

            WHERE id = ?

              AND campaign_id = ?

              AND status = 'accepted'

            """,
            (proposal["event_id"], campaign_id),
        ).fetchone()

        if event is None:

            raise HTTPException(
                status_code=400,
                detail="The associated world event must be accepted first",
            )

        try:

            connection.execute(
                """

                INSERT INTO character_knowledge (

                    character_id,

                    event_id,

                    knowledge_status,

                    knowledge_detail,

                    source_description

                )

                VALUES (?, ?, ?, ?, ?)

                """,
                (
                    proposal["character_id"],
                    proposal["event_id"],
                    proposal["knowledge_status"],
                    proposal["knowledge_detail"],
                    proposal["source_description"],
                ),
            )

        except sqlite3.IntegrityError:

            raise HTTPException(
                status_code=409,
                detail="This character already has a knowledge record for this event",
            )

        connection.execute(
            """

            UPDATE character_knowledge_proposals

            SET status = 'accepted'

            WHERE id = ?

            """,
            (proposal_id,),
        )

    return {
        "message": "Character knowledge proposal accepted",
        "proposal_id": proposal_id,
    }


@app.post("/campaigns/{campaign_id}/knowledge-proposals/{proposal_id}/reject")
async def reject_knowledge_proposal(
    campaign_id: int,
    proposal_id: int,
):

    with get_db() as connection:

        proposal = connection.execute(
            """

            SELECT id

            FROM character_knowledge_proposals

            WHERE id = ?

              AND campaign_id = ?

              AND status = 'pending'

            """,
            (proposal_id, campaign_id),
        ).fetchone()

        if proposal is None:

            raise HTTPException(
                status_code=404,
                detail="Pending knowledge proposal not found",
            )

        connection.execute(
            """

            UPDATE character_knowledge_proposals

            SET status = 'rejected'

            WHERE id = ?

            """,
            (proposal_id,),
        )

    return {
        "message": "Character knowledge proposal rejected",
        "proposal_id": proposal_id,
    }


@app.post("/campaigns/{campaign_id}/events")
async def create_world_event(
    campaign_id: int,
    request: WorldEventCreate,
):

    with get_db() as connection:

        campaign = connection.execute(
            """

            SELECT id

            FROM campaigns

            WHERE id = ?

            """,
            (campaign_id,),
        ).fetchone()

        if campaign is None:

            raise HTTPException(
                status_code=404,
                detail="Campaign not found",
            )

        # Confirm that all participants belong to this campaign

        participants = list(dict.fromkeys(request.participants))

        if participants:

            placeholders = ",".join("?" for _ in participants)

            existing_characters = connection.execute(
                f"""

                SELECT name

                FROM characters

                WHERE campaign_id = ?

                  AND name IN ({placeholders})

                """,
                (campaign_id, *participants),
            ).fetchall()

            existing_names = {row["name"] for row in existing_characters}

            unknown_names = [
                name for name in participants if name not in existing_names
            ]

            if unknown_names:

                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown characters: {unknown_names}",
                )

        cursor = connection.execute(
            """

            INSERT INTO world_events (

                campaign_id,

                event_type,

                description,

                location,

                participants,

                occurred_at

            )

            VALUES (?, ?, ?, ?, ?, ?)

            """,
            (
                campaign_id,
                request.event_type,
                request.description.strip(),
                request.location,
                json.dumps(participants, ensure_ascii=False),
                request.occurred_at,
            ),
        )

        event_id = cursor.lastrowid

        event = connection.execute(
            """

            SELECT *

            FROM world_events

            WHERE id = ?

            """,
            (event_id,),
        ).fetchone()

    result = dict(event)

    result["participants"] = json.loads(result["participants"])

    return result


@app.get("/campaigns/{campaign_id}/events")
async def get_world_events(campaign_id: int):

    with get_db() as connection:

        campaign = connection.execute(
            """

            SELECT id

            FROM campaigns

            WHERE id = ?

            """,
            (campaign_id,),
        ).fetchone()

        if campaign is None:

            raise HTTPException(
                status_code=404,
                detail="Campaign not found",
            )

        events = connection.execute(
            """

            SELECT *

            FROM world_events

            WHERE campaign_id = ?

            ORDER BY id ASC

            """,
            (campaign_id,),
        ).fetchall()

    results = []

    for event in events:

        item = dict(event)

        try:

            item["participants"] = json.loads(item["participants"] or "[]")

        except (json.JSONDecodeError, TypeError):

            item["participants"] = []

        results.append(item)

    return {"events": results}


# --------------------------------------------------

# CHARACTER KNOWLEDGE ENDPOINTS

# --------------------------------------------------


@app.post("/campaigns/{campaign_id}/events/{event_id}/knowledge")
async def create_character_knowledge(
    campaign_id: int,
    event_id: int,
    request: CharacterKnowledgeCreate,
):

    with get_db() as connection:

        # Confirm that the event belongs to this campaign

        event = connection.execute(
            """

            SELECT id

            FROM world_events

            WHERE id = ?

              AND campaign_id = ?

            """,
            (event_id, campaign_id),
        ).fetchone()

        if event is None:

            raise HTTPException(
                status_code=404,
                detail="World event not found",
            )

        # Confirm that the character belongs to this campaign

        character = connection.execute(
            """

            SELECT id, name

            FROM characters

            WHERE id = ?

              AND campaign_id = ?

            """,
            (request.character_id, campaign_id),
        ).fetchone()

        if character is None:

            raise HTTPException(
                status_code=404,
                detail="Character not found in this campaign",
            )

        # Insert or update this character's knowledge of the event

        connection.execute(
            """

            INSERT INTO character_knowledge (

                character_id,

                event_id,

                knowledge_status,

                knowledge_detail,

                source_description

            )

            VALUES (?, ?, ?, ?, ?)



            ON CONFLICT(character_id, event_id)

            DO UPDATE SET

                knowledge_status = excluded.knowledge_status,

                knowledge_detail = excluded.knowledge_detail,

                source_description = excluded.source_description

            """,
            (
                request.character_id,
                event_id,
                request.knowledge_status,
                request.knowledge_detail,
                request.source_description,
            ),
        )

        knowledge = connection.execute(
            """

            SELECT *

            FROM character_knowledge

            WHERE character_id = ?

              AND event_id = ?

            """,
            (request.character_id, event_id),
        ).fetchone()

    return dict(knowledge)


@app.get("/campaigns/{campaign_id}/characters/{character_id}/knowledge")
async def get_character_knowledge(
    campaign_id: int,
    character_id: int,
):

    with get_db() as connection:

        character = connection.execute(
            """

            SELECT id

            FROM characters

            WHERE id = ?

              AND campaign_id = ?

            """,
            (character_id, campaign_id),
        ).fetchone()

        if character is None:

            raise HTTPException(
                status_code=404,
                detail="Character not found in this campaign",
            )

        knowledge_records = connection.execute(
            """

            SELECT

                character_knowledge.id,

                character_knowledge.event_id,

                world_events.event_type,

                world_events.description AS world_event,

                character_knowledge.knowledge_status,

                character_knowledge.knowledge_detail,

                character_knowledge.source_description,

                character_knowledge.created_at

            FROM character_knowledge

            JOIN world_events

                ON character_knowledge.event_id = world_events.id

            WHERE character_knowledge.character_id = ?

            ORDER BY character_knowledge.id ASC

            """,
            (character_id,),
        ).fetchall()

    return {
        "character_id": character_id,
        "knowledge": [dict(record) for record in knowledge_records],
    }


# --------------------------------------------------

# PLAYER DISCOVERY ENDPOINTS

# --------------------------------------------------


@app.post("/campaigns/{campaign_id}/events/{event_id}/discover")
async def create_player_discovery(
    campaign_id: int,
    event_id: int,
    request: PlayerDiscoveryCreate,
):

    with get_db() as connection:

        # Confirm that the event belongs to this campaign

        event = connection.execute(
            """

            SELECT id, description

            FROM world_events

            WHERE id = ?

              AND campaign_id = ?

            """,
            (event_id, campaign_id),
        ).fetchone()

        if event is None:

            raise HTTPException(
                status_code=404,
                detail="World event not found",
            )

        # Insert or update the player's discovery

        connection.execute(
            """

            INSERT INTO player_discoveries (

                campaign_id,

                event_id,

                discovery_note

            )

            VALUES (?, ?, ?)



            ON CONFLICT(campaign_id, event_id)

            DO UPDATE SET

                discovery_note = excluded.discovery_note

            """,
            (
                campaign_id,
                event_id,
                request.discovery_note,
            ),
        )

        discovery = connection.execute(
            """

            SELECT *

            FROM player_discoveries

            WHERE campaign_id = ?

              AND event_id = ?

            """,
            (campaign_id, event_id),
        ).fetchone()

    return dict(discovery)


@app.get("/campaigns/{campaign_id}/discoveries")
async def get_player_discoveries(campaign_id: int):

    with get_db() as connection:

        campaign = connection.execute(
            """

            SELECT id

            FROM campaigns

            WHERE id = ?

            """,
            (campaign_id,),
        ).fetchone()

        if campaign is None:

            raise HTTPException(
                status_code=404,
                detail="Campaign not found",
            )

        discoveries = connection.execute(
            """

            SELECT

                player_discoveries.id,

                player_discoveries.event_id,

                world_events.event_type,

                world_events.description AS world_event,

                player_discoveries.discovery_note,

                player_discoveries.created_at

            FROM player_discoveries

            JOIN world_events

                ON player_discoveries.event_id = world_events.id

            WHERE player_discoveries.campaign_id = ?

            ORDER BY player_discoveries.id ASC

            """,
            (campaign_id,),
        ).fetchall()

    return {
        "campaign_id": campaign_id,
        "discoveries": [dict(discovery) for discovery in discoveries],
    }


# --------------------------------------------------

# WORLD EVENT REVIEW ENDPOINTS

# --------------------------------------------------


@app.post("/campaigns/{campaign_id}/events/{event_id}/accept")
async def accept_world_event(
    campaign_id: int,
    event_id: int,
    request: WorldEventReview,
):

    with get_db() as connection:

        event = connection.execute(
            """

            SELECT id, status

            FROM world_events

            WHERE id = ?

              AND campaign_id = ?

            """,
            (event_id, campaign_id),
        ).fetchone()

        if event is None:

            raise HTTPException(
                status_code=404,
                detail="World event not found",
            )

        if event["status"] != "pending":

            raise HTTPException(
                status_code=409,
                detail="This event has already been processed",
            )

        connection.execute(
            """

            UPDATE world_events

            SET status = 'accepted',

                certainty = ?

            WHERE id = ?

            """,
            (request.certainty, event_id),
        )

    return {
        "message": "World event accepted successfully",
        "event_id": event_id,
        "status": "accepted",
        "certainty": request.certainty,
    }


@app.post("/campaigns/{campaign_id}/events/{event_id}/reject")
async def reject_world_event(
    campaign_id: int,
    event_id: int,
):

    with get_db() as connection:

        event = connection.execute(
            """

            SELECT id, status

            FROM world_events

            WHERE id = ?

              AND campaign_id = ?

            """,
            (event_id, campaign_id),
        ).fetchone()

        if event is None:

            raise HTTPException(
                status_code=404,
                detail="World event not found",
            )

        if event["status"] != "pending":

            raise HTTPException(
                status_code=409,
                detail="This event has already been processed",
            )

        connection.execute(
            """

            UPDATE world_events

            SET status = 'rejected'

            WHERE id = ?

            """,
            (event_id,),
        )

    return {
        "message": "World event rejected successfully",
        "event_id": event_id,
        "status": "rejected",
    }


# --------------------------------------------------

# CHARACTER RELATIONSHIP ENDPOINTS

# --------------------------------------------------


def ensure_character_relationship(connection, campaign_id, character_id):

    connection.execute(
        """

        INSERT OR IGNORE INTO character_relationships (

            campaign_id,

            character_id

        )

        VALUES (?, ?)

        """,
        (campaign_id, character_id),
    )


@app.get("/campaigns/{campaign_id}/relationships")
async def get_campaign_relationships(campaign_id: int):

    with get_db() as connection:

        campaign = connection.execute(
            "SELECT id FROM campaigns WHERE id = ?",
            (campaign_id,),
        ).fetchone()

        if campaign is None:

            raise HTTPException(
                status_code=404,
                detail="Campaign not found",
            )

        # Ensure every character has a relationship record.

        connection.execute(
            """

            INSERT OR IGNORE INTO character_relationships (

                campaign_id,

                character_id

            )

            SELECT campaign_id, id

            FROM characters

            WHERE campaign_id = ?

            """,
            (campaign_id,),
        )

        relationships = connection.execute(
            """

            SELECT

                character_relationships.*,

                characters.name AS character_name

            FROM character_relationships

            JOIN characters

                ON character_relationships.character_id = characters.id

            WHERE character_relationships.campaign_id = ?

            ORDER BY characters.id ASC

            """,
            (campaign_id,),
        ).fetchall()

    return {
        "campaign_id": campaign_id,
        "relationships": [dict(record) for record in relationships],
    }


@app.get("/campaigns/{campaign_id}/characters/{character_id}/relationship")
async def get_character_relationship(
    campaign_id: int,
    character_id: int,
):

    with get_db() as connection:

        character = connection.execute(
            """

            SELECT id, name

            FROM characters

            WHERE id = ?

              AND campaign_id = ?

            """,
            (character_id, campaign_id),
        ).fetchone()

        if character is None:

            raise HTTPException(
                status_code=404,
                detail="Character not found in this campaign",
            )

        ensure_character_relationship(
            connection,
            campaign_id,
            character_id,
        )

        relationship = connection.execute(
            """

            SELECT *

            FROM character_relationships

            WHERE character_id = ?

            """,
            (character_id,),
        ).fetchone()

    return {
        "character_name": character["name"],
        "relationship": dict(relationship),
    }


@app.put("/campaigns/{campaign_id}/characters/{character_id}/relationship")
async def update_character_relationship(
    campaign_id: int,
    character_id: int,
    request: CharacterRelationshipUpdate,
):

    with get_db() as connection:

        character = connection.execute(
            """

            SELECT id, name

            FROM characters

            WHERE id = ?

              AND campaign_id = ?

            """,
            (character_id, campaign_id),
        ).fetchone()

        if character is None:

            raise HTTPException(
                status_code=404,
                detail="Character not found in this campaign",
            )

        ensure_character_relationship(
            connection,
            campaign_id,
            character_id,
        )

        relationship = connection.execute(
            """

            SELECT *

            FROM character_relationships

            WHERE character_id = ?

            """,
            (character_id,),
        ).fetchone()

        submitted_data = request.model_dump(
            exclude_unset=True,
            exclude_none=True,
        )

        reason = submitted_data.pop("reason", "")

        if not submitted_data:

            raise HTTPException(
                status_code=400,
                detail="No relationship fields were provided",
            )

        valid_fields = {
            "affection",
            "attraction",
            "trust",
            "respect",
            "resentment",
            "friendship",
            "romantic_interest",
            "sexual_interest",
        }

        changes = []

        for field, new_value in submitted_data.items():

            if field not in valid_fields:

                continue

            old_value = relationship[field]

            if old_value == new_value:

                continue

            connection.execute(
                """

                UPDATE character_relationships

                SET

                    """
                + field
                + """ = ?,

                    updated_at = CURRENT_TIMESTAMP

                WHERE character_id = ?

                """,
                (new_value, character_id),
            )

            connection.execute(
                """

                INSERT INTO relationship_history (

                    campaign_id,

                    character_id,

                    field_name,

                    old_value,

                    new_value,

                    reason

                )

                VALUES (?, ?, ?, ?, ?, ?)

                """,
                (
                    campaign_id,
                    character_id,
                    field,
                    str(old_value),
                    str(new_value),
                    reason,
                ),
            )

            changes.append(
                {
                    "field": field,
                    "old_value": old_value,
                    "new_value": new_value,
                }
            )

        updated_relationship = connection.execute(
            """

            SELECT *

            FROM character_relationships

            WHERE character_id = ?

            """,
            (character_id,),
        ).fetchone()

    return {
        "message": "Relationship updated successfully",
        "character_name": character["name"],
        "changes": changes,
        "relationship": dict(updated_relationship),
    }


@app.get("/campaigns/{campaign_id}/characters/{character_id}/relationship/history")
async def get_character_relationship_history(
    campaign_id: int,
    character_id: int,
):

    with get_db() as connection:

        character = connection.execute(
            """

            SELECT id, name

            FROM characters

            WHERE id = ?

              AND campaign_id = ?

            """,
            (character_id, campaign_id),
        ).fetchone()

        if character is None:

            raise HTTPException(
                status_code=404,
                detail="Character not found in this campaign",
            )

        history = connection.execute(
            """

            SELECT

                id,

                field_name,

                old_value,

                new_value,

                reason,

                source_description,

                created_at

            FROM relationship_history

            WHERE campaign_id = ?

              AND character_id = ?

            ORDER BY id DESC

            """,
            (campaign_id, character_id),
        ).fetchall()

    return {
        "character_name": character["name"],
        "history": [dict(record) for record in history],
    }

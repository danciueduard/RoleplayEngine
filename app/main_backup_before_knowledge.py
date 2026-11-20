import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pathlib import Path
from contextlib import contextmanager
import sqlite3
import httpx
from app.state_analyzer import analyze_state_changes
from app.event_analyzer import analyze_world_events

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
            SELECT name, age, description,
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

        # Only accepted events become objective world truth.
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

        # Only knowledge linked to accepted events is supplied.
        character_knowledge = connection.execute(
            """
            SELECT
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

        # Only discoveries linked to accepted events are supplied.
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

        # Include existing pending events when checking for duplicates.
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

    character_details = []

    for character in characters:
        character_details.append(
            {
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

PLAYER DISCOVERIES:
{json.dumps(player_discoveries_data, ensure_ascii=False, indent=2)}

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

3. PLAYER DISCOVERIES:
These records describe information the player has
personally discovered.
The player's knowledge and each character's knowledge
are separate.
Do not automatically transfer player discoveries
to characters.

4. SECRETS:
Keep undiscovered information secret.
Do not reveal hidden events through narration unless
the current scene provides a believable way for the
player to discover them.

5. INFORMATION SOURCES:
Characters may learn new information through dialogue,
observation, evidence, or events during the story.
Only treat information as established knowledge when
the story provides a credible reason for the
character to acquire it.

6. CONTINUITY:
Respect established campaign events, character
personalities, backgrounds, and current states.
Do not silently rewrite established facts.

ROLEPLAY GUIDELINES:
- Write immersive, natural and coherent narrative.
- Treat the user's messages as contributions to
  the ongoing story.
- Maintain consistent character personalities.
- Allow characters to have independent motives,
  opinions, memories and knowledge.
- Preserve mysteries and dramatic tension.
- Avoid inventing knowledge that characters have
  no reason to possess.
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

        if proposals:
            with get_db() as connection:
                connection.executemany(
                    """
                    INSERT INTO state_change_proposals (
                        campaign_id,
                        character_id,
                        field_name,
                        old_value,
                        proposed_value,
                        reason,
                        status
                    )
                    VALUES (?, ?, ?, ?, ?, ?, 'pending')
                    """,
                    [
                        (
                            request.campaign_id,
                            character_ids[proposal["character_name"]],
                            proposal["field_name"],
                            proposal["old_value"],
                            proposal["proposed_value"],
                            proposal["reason"],
                        )
                        for proposal in proposals
                    ],
                )

        print(f"State analyzer created {len(proposals)} proposal(s).")

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

        print(f"Event analyzer created " f"{len(event_proposals)} proposal(s).")

    except Exception as error:
        print(f"Event analysis failed: {error}")

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

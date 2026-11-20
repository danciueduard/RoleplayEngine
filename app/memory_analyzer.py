"""Automatic, character-specific conversational memory for the roleplay engine."""
import json
import re
import httpx


def ensure_memory_table(connection):
    connection.execute("""
        CREATE TABLE IF NOT EXISTS character_memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            campaign_id INTEGER NOT NULL,
            character_id INTEGER NOT NULL,
            memory TEXT NOT NULL,
            knowledge_status TEXT NOT NULL DEFAULT 'knows',
            source_description TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (campaign_id) REFERENCES campaigns(id) ON DELETE CASCADE,
            FOREIGN KEY (character_id) REFERENCES characters(id) ON DELETE CASCADE
        )
    """)
    connection.execute("""
        CREATE INDEX IF NOT EXISTS idx_character_memories_lookup
        ON character_memories(campaign_id, character_id, id)
    """)


def load_memories(connection, campaign_id, limit_per_character=35):
    ensure_memory_table(connection)
    rows = connection.execute("""
        SELECT cm.id, c.name AS character_name, cm.memory,
               cm.knowledge_status, cm.source_description
        FROM character_memories cm
        JOIN characters c ON c.id = cm.character_id
        WHERE cm.campaign_id = ? AND cm.id IN (
            SELECT id FROM character_memories m
            WHERE m.campaign_id = cm.campaign_id
              AND m.character_id = cm.character_id
            ORDER BY id DESC LIMIT ?
        )
        ORDER BY cm.id ASC
    """, (campaign_id, limit_per_character)).fetchall()
    return [dict(row) for row in rows]


async def analyze_conversation_memories(user_message, assistant_reply, characters,
                                        existing_memories, ollama_url, model):
    if not characters:
        return []
    names = {c['name']: c['id'] for c in characters}
    instructions = """You are a strict continuity-memory extractor for a roleplay simulation.
Extract ONLY durable, character-specific information genuinely learned in the CURRENT exchange.
Return JSON only: {"memories": [{"character_name": "...", "memory": "...", "knowledge_status": "knows|believes|suspects", "source_description": "..."}]}.
Rules:
- A character must be present, addressed, or credibly informed to acquire a memory.
- A narrator's description of private thoughts is NOT knowledge for other characters.
- Record promises as 'X promised Y', never as proof that Y will happen.
- A suspicion is 'suspects'; an unverified assertion is 'believes' or 'knows they were told'.
- No future events, invented witnesses, omniscience, or speculation presented as fact.
- Only meaningful information worth recalling later: promises, personal disclosures, discoveries, plans, conflicts, secrets, important observations.
- Skip routine pleasantries, repeated facts, momentary emotions, and trivial details.
- Avoid duplicates of existing memories; return [] when nothing new.
- Each memory must be a short standalone sentence with clear people and context.
- Max 4 memories for the entire exchange.
"""
    payload = {
        'model': model, 'think': False, 'stream': False,
        'format': 'json',
        'messages': [
            {'role': 'system', 'content': instructions},
            {'role': 'user', 'content': json.dumps({
                'characters': list(names),
                'existing_memories': existing_memories[-100:],
                'player_message': user_message,
                'narrator_reply': assistant_reply,
            }, ensure_ascii=False)},
        ],
        'options': {'temperature': 0.1, 'num_ctx': 4096, 'num_predict': 400},
    }
    async with httpx.AsyncClient(timeout=180.0) as client:
        response = await client.post(ollama_url, json=payload)
        response.raise_for_status()
        raw = response.json()['message']['content']
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return []
    if not isinstance(data, dict) or not isinstance(data.get('memories'), list):
        return []
    result = []
    seen = {(m['character_name'].casefold(), m['memory'].strip().casefold())
            for m in existing_memories}
    for item in data['memories'][:4]:
        if not isinstance(item, dict):
            continue
        name = item.get('character_name')
        memory = item.get('memory')
        status = item.get('knowledge_status')
        source = item.get('source_description', '')
        if (not isinstance(name, str) or name not in names or
                not isinstance(memory, str) or not 12 <= len(memory.strip()) <= 400 or
                status not in ('knows', 'believes', 'suspects') or
                not isinstance(source, str)):
            continue
        key = (name.casefold(), memory.strip().casefold())
        if key in seen:
            continue
        seen.add(key)
        result.append({'character_id': names[name], 'memory': memory.strip(),
                       'knowledge_status': status,
                       'source_description': source.strip()[:300]})
    return result

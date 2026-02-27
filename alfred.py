import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from pathlib import Path
from dotenv import load_dotenv
from mem0 import Memory
from mem0.llms.anthropic import AnthropicLLM
import anthropic
import os
import datetime
import json

# ─── Mem0 Patches ─────────────────────────────────────────────
# Patch 1: Mem0's AnthropicLLM always passes both temperature and top_p,
# which causes a 400 error from the Anthropic API. Strip top_p before
# the API call.
_original_get_common_params = AnthropicLLM._get_common_params

def _patched_get_common_params(self, **kwargs):
    params = _original_get_common_params(self, **kwargs)
    params.pop("top_p", None)
    return params

AnthropicLLM._get_common_params = _patched_get_common_params

# Patch 2: Local Qdrant doesn't support payload indexes, so filtered
# scroll() calls (e.g. filtering by user_id) silently return no results.
# Patch list() and search() to fetch all points then filter in Python.
from mem0.vector_stores.qdrant import Qdrant as QdrantStore
from qdrant_client.models import FieldCondition, MatchValue

_original_qdrant_list = QdrantStore.list
_original_qdrant_search = QdrantStore.search

def _patched_qdrant_list(self, filters: dict = None, limit: int = 100):
    if not self.is_local or not filters:
        return _original_qdrant_list(self, filters=filters, limit=limit)
    # Fetch all points unfiltered, then filter payloads in Python
    result = self.client.scroll(
        collection_name=self.collection_name,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    points, _ = result
    return [p for p in points if all(
        p.payload.get(k) == v for k, v in filters.items()
    )], None

def _patched_qdrant_search(self, query: str, vectors: list, limit: int = 5, filters: dict = None):
    if not self.is_local or not filters:
        return _original_qdrant_search(self, query=query, vectors=vectors, limit=limit, filters=filters)
    # Fetch all points by score (no Qdrant filter), then Python-filter by user_id etc.
    # Use a large fetch limit so we don't miss relevant results after filtering.
    hits = self.client.query_points(
        collection_name=self.collection_name,
        query=vectors,
        limit=1000,
    )
    return [h for h in hits.points if all(
        h.payload.get(k) == v for k, v in filters.items()
    )][:limit]

QdrantStore.list = _patched_qdrant_list
QdrantStore.search = _patched_qdrant_search

# ─── Environment ───────────────────────────────────────────────
load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "dhwafD61uVd8h85wAZSE")
FFMPEG_PATH = os.getenv("FFMPEG_PATH", r"C:\ffmpeg\ffmpeg-8.0.1-essentials_build\bin\ffmpeg.exe")

if not ANTHROPIC_API_KEY:
    raise EnvironmentError("ANTHROPIC_API_KEY not found in .env")

# ─── Local Paths ──────────────────────────────────────────────
alfred_dir = Path.home() / ".alfred"
qdrant_path = alfred_dir / "qdrant"
db_path = alfred_dir / "alfred_memory.db"
alfred_dir.mkdir(parents=True, exist_ok=True)
qdrant_path.mkdir(parents=True, exist_ok=True)


# ─── Mem0 Config ──────────────────────────────────────────────
def _local_mem0_config(anthropic_api_key: str) -> dict:
    """
    Build a Mem0 config that uses:
    - Anthropic (Claude) as the LLM for memory extraction
    - HuggingFace sentence-transformers for local embeddings (no OpenAI key needed)
    - Qdrant (on-disk) as the vector store
    - SQLite for history
    """
    return {
        "llm": {
            "provider": "anthropic",
            "config": {
                "model": "claude-haiku-4-5-20251001",
                "api_key": anthropic_api_key,
                "max_tokens": 2000,
                "temperature": 0,
            },
        },
        "embedder": {
            "provider": "huggingface",
            "config": {
                "model": "multi-qa-MiniLM-L6-cos-v1",
                "embedding_dims": 384,
            },
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": "alfred_memories",
                "on_disk": True,
                "path": str(qdrant_path),
                "embedding_model_dims": 384,
            },
        },
        "history_db_path": str(db_path),
        "version": "v1.1",
    }


# ─── Initialize Services ─────────────────────────────────────
print("⏳ Initializing Alfred's memory systems...")
memory = Memory.from_config(_local_mem0_config(ANTHROPIC_API_KEY))
client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
print("✅ Memory systems online.\n")

# ─── Tavily Web Search ───────────────────────────────────────
TAVILY_TOOL = {
    "name": "web_search",
    "description": (
        "Search the internet for current information. Use this when the user asks "
        "about recent events, current prices, news, weather, sports scores, or "
        "anything that requires up-to-date information beyond your training data."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to look up.",
            },
            "max_results": {
                "type": "integer",
                "description": "Number of results to return (1-10). Default is 5.",
                "default": 5,
            },
        },
        "required": ["query"],
    },
}


def web_search(query: str, max_results: int = 5) -> str:
    """Run a Tavily search and return formatted results."""
    if not TAVILY_API_KEY:
        return "Web search is unavailable — TAVILY_API_KEY is not set."
    try:
        import urllib.request
        payload = json.dumps({
            "api_key": TAVILY_API_KEY,
            "query": query,
            "max_results": max_results,
            "include_answer": True,
        }).encode()
        req = urllib.request.Request(
            "https://api.tavily.com/search",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())

        lines = []
        if data.get("answer"):
            lines.append(f"Summary: {data['answer']}\n")
        for i, r in enumerate(data.get("results", []), 1):
            lines.append(f"{i}. {r['title']}")
            lines.append(f"   {r['url']}")
            if r.get("content"):
                lines.append(f"   {r['content'][:300].strip()}...")
            lines.append("")
        return "\n".join(lines) if lines else "No results found."
    except Exception as e:
        return f"Search error: {e}"


# ─── Voice: Text-to-Speech (ElevenLabs) ─────────────────────
def speak(text: str) -> None:
    """Convert text to speech via ElevenLabs and play it."""
    if not ELEVENLABS_API_KEY:
        return
    try:
        import urllib.request
        import urllib.error
        import tempfile
        import re

        # Strip markdown formatting so it sounds natural spoken aloud
        clean = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", text)  # bold/italic
        clean = re.sub(r"`[^`]+`", "", clean)                    # inline code
        clean = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", clean)  # links
        clean = re.sub(r"#{1,6}\s*", "", clean)                  # headings
        clean = re.sub(r"\n{2,}", " ", clean).strip()

        # Truncate to ElevenLabs limit (5000 chars on free/starter tier)
        clean = clean[:4500]

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
        payload = json.dumps({
            "text": clean,
            "model_id": "eleven_turbo_v2_5",
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
        }, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "xi-api-key": ELEVENLABS_API_KEY,
                "Content-Type": "application/json; charset=utf-8",
                "Accept": "audio/mpeg",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                audio = resp.read()
        except urllib.error.HTTPError as http_err:
            print(f"⚠️  ElevenLabs error {http_err.code}: {http_err.read().decode()}")
            return

        # Write to temp file and play (blocks until done)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            f.write(audio)
            tmp_path = f.name

        from playsound import playsound
        playsound(tmp_path)
        os.unlink(tmp_path)
    except Exception as e:
        print(f"⚠️  Voice output error: {e}")


# ─── Voice: Speech-to-Text (Google via speech_recognition) ───
def listen() -> str | None:
    """Record from microphone and return transcribed text, or None on failure."""
    try:
        import speech_recognition as sr
        r = sr.Recognizer()
        with sr.Microphone() as source:
            print("🎤 Listening... (speak now)")
            r.adjust_for_ambient_noise(source, duration=0.5)
            audio = r.listen(source, timeout=10, phrase_time_limit=30)
        print("🔄 Transcribing...")
        text = r.recognize_google(audio)
        print(f"You (voice): {text}")
        return text
    except ImportError:
        print("⚠️  speech_recognition not installed. Run: pip install SpeechRecognition pyaudio")
        return None
    except Exception as e:
        print(f"⚠️  Voice input error: {e}")
        return None


# ─── Family Registry ─────────────────────────────────────────
FAMILY_MEMBERS = {
    "dad": {
        "name": "Dad",
        "user_id": "dad",
        "role": "Head of household",
    },
    "jenny": {
        "name": "Jenny",
        "user_id": "jenny",
        "role": "Wife",
    },
}

DEFAULT_USER = "dad"


# ─── System Prompt ────────────────────────────────────────────
SYSTEM_PROMPT = """You are Alfred, a distinguished and deeply devoted family AI advisor.
You serve as the trusted orchestrator for the family's AI advisor system — a butler
of the modern age, equal parts wise counsel and capable organizer.

Your character:
- Warm, composed, and unflappable — like a great butler, you handle everything with grace
- Deeply knowledgeable about the family's needs, routines, and goals
- You speak with quiet confidence and a touch of dry wit when appropriate
- You treat every request — from the trivial to the consequential — with full attention

You have a persistent memory system. You remember things the family tells you across
conversations. When you recall relevant memories, weave them naturally into your responses.

You have access to a web_search tool for current information — news, prices, weather,
sports, research, anything beyond your training data. Use it proactively when a question
would benefit from up-to-date information. Cite sources naturally in your response.

Today's date is: {date}

MEMORIES:
{memories}
"""


# ─── Memory Helpers ───────────────────────────────────────────
def recall_memories(query: str, user_id: str = DEFAULT_USER) -> str:
    """Search memory for relevant context."""
    results = memory.search(query=query, user_id=user_id)
    if not results:
        return "No relevant memories found."
    # results is a dict with a "results" key (v1.1 API)
    items = results.get("results", []) if isinstance(results, dict) else results
    if not items:
        return "No relevant memories found."
    memories = []
    for r in items:
        if isinstance(r, dict) and "memory" in r:
            memories.append(f"- {r['memory']}")
        elif isinstance(r, dict) and "text" in r:
            memories.append(f"- {r['text']}")
        elif hasattr(r, "payload"):
            memories.append(f"- {r.payload.get('data', str(r))}")
        else:
            memories.append(f"- {str(r)}")
    return "\n".join(memories)


def store_memory(conversation: list, user_id: str = DEFAULT_USER):
    """Store conversation in memory for future recall."""
    try:
        result = memory.add(
            messages=conversation,
            user_id=user_id,
        )
        stored = result.get("results", []) if isinstance(result, dict) else []
        if stored:
            print(f"💾 Stored {len(stored)} new memory item(s).")
    except Exception as e:
        import traceback
        print(f"⚠️  Memory storage error: {e}\n{traceback.format_exc()}")


# ─── Conversation Engine ─────────────────────────────────────
def build_system_prompt(user_input: str, user_id: str = DEFAULT_USER) -> str:
    """Build system prompt with injected memories and date."""
    memories = recall_memories(user_input, user_id)
    today = datetime.datetime.now().strftime("%A, %B %d, %Y %I:%M %p")
    return SYSTEM_PROMPT.format(date=today, memories=memories)


def chat(user_input: str, conversation_history: list, user_id: str = DEFAULT_USER) -> str:
    """Send a message to Alfred and get a response, handling tool use in a loop."""
    system = build_system_prompt(user_input, user_id)

    conversation_history.append({
        "role": "user",
        "content": user_input,
    })

    tools = [TAVILY_TOOL] if TAVILY_API_KEY else []

    # Agentic loop — keeps going until Alfred returns a final text response
    while True:
        kwargs = dict(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            system=system,
            messages=conversation_history,
        )
        if tools:
            kwargs["tools"] = tools

        response = client.messages.create(**kwargs)

        if response.stop_reason == "tool_use":
            # Append Alfred's tool-use turn to history
            conversation_history.append({
                "role": "assistant",
                "content": response.content,
            })

            # Execute each tool call and collect results
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    if block.name == "web_search":
                        print(f"🔍 Searching: {block.input.get('query', '')}")
                        result = web_search(
                            query=block.input["query"],
                            max_results=block.input.get("max_results", 5),
                        )
                    else:
                        result = f"Unknown tool: {block.name}"

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

            # Feed results back to Alfred and loop
            conversation_history.append({
                "role": "user",
                "content": tool_results,
            })

        else:
            # Final text response
            assistant_message = next(
                (b.text for b in response.content if hasattr(b, "text")), ""
            )
            conversation_history.append({
                "role": "assistant",
                "content": assistant_message,
            })

            # Store the user input + final response in memory
            store_memory(conversation_history[-2:], user_id)

            return assistant_message


# ─── Main Loop ────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("🎩 Alfred — Family AI Advisor")
    print("=" * 55)
    print("Type your message, or press Enter to speak")
    print("'voice on' / 'voice off' to toggle spoken responses")
    print("'quit' to exit  |  'memories' to view stored memories")
    print("=" * 55)
    print()

    conversation_history = []
    user_id = DEFAULT_USER
    voice_mode = False  # toggled with 'voice on' / 'voice off'

    # Opening greeting
    greeting = chat(
        "Please greet me warmly as Alfred, my family AI butler.",
        conversation_history,
        user_id,
    )
    print(f"Alfred: {greeting}\n")
    if voice_mode:
        speak(greeting)

    while True:
        try:
            typed = input("You (or Enter to speak): ").strip()

            if not typed:
                # Empty input — trigger microphone
                user_input = listen()
                if user_input is None:
                    continue
            else:
                user_input = typed

            cmd = user_input.lower().strip()

            if cmd == "quit":
                farewell = "Very good, sir. Until next time. 🎩"
                print(f"\nAlfred: {farewell}")
                if voice_mode:
                    speak(farewell)
                break

            if cmd == "voice on":
                if not ELEVENLABS_API_KEY:
                    print("⚠️  ELEVENLABS_API_KEY is not set — voice output unavailable.")
                voice_mode = True
                print("🔊 Voice responses ON")
                continue

            if cmd == "voice off":
                voice_mode = False
                print("🔇 Voice responses OFF")
                continue

            if cmd == "memories":
                all_memories = memory.get_all(user_id=user_id)
                print("\n📋 All stored memories:")
                print("-" * 40)
                items = all_memories.get("results", []) if isinstance(all_memories, dict) else all_memories
                if items:
                    for m in items:
                        if isinstance(m, dict) and "memory" in m:
                            print(f"  - {m['memory']}")
                        else:
                            print(f"  - {m}")
                else:
                    print("  No memories stored yet.")
                print("-" * 40 + "\n")
                continue

            response = chat(user_input, conversation_history, user_id)
            print(f"\nAlfred: {response}\n")
            if voice_mode:
                speak(response)

        except KeyboardInterrupt:
            print("\n\nAlfred: Very good, sir. Until next time. 🎩")
            break


if __name__ == "__main__":
    main()

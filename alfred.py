from pathlib import Path
from dotenv import load_dotenv
from mem0 import Memory
import anthropic
import os
import datetime

# ─── Environment ───────────────────────────────────────────────
load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

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
    memories = []
    for r in results:
        if isinstance(r, dict) and "memory" in r:
            memories.append(f"- {r['memory']}")
        elif isinstance(r, dict) and "text" in r:
            memories.append(f"- {r['text']}")
        else:
            memories.append(f"- {str(r)}")
    return "\n".join(memories)


def store_memory(conversation: list, user_id: str = DEFAULT_USER):
    """Store conversation in memory for future recall."""
    try:
        memory.add(
            messages=conversation,
            user_id=user_id,
        )
    except Exception as e:
        print(f"⚠️  Memory storage note: {e}")


# ─── Conversation Engine ─────────────────────────────────────
def build_system_prompt(user_input: str, user_id: str = DEFAULT_USER) -> str:
    """Build system prompt with injected memories and date."""
    memories = recall_memories(user_input, user_id)
    today = datetime.datetime.now().strftime("%A, %B %d, %Y %I:%M %p")
    return SYSTEM_PROMPT.format(date=today, memories=memories)


def chat(user_input: str, conversation_history: list, user_id: str = DEFAULT_USER) -> str:
    """Send a message to Alfred and get a response."""
    system = build_system_prompt(user_input, user_id)

    conversation_history.append({
        "role": "user",
        "content": user_input,
    })

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        system=system,
        messages=conversation_history,
    )

    assistant_message = response.content[0].text

    conversation_history.append({
        "role": "assistant",
        "content": assistant_message,
    })

    # Store the exchange in memory
    store_memory(conversation_history[-2:], user_id)

    return assistant_message


# ─── Main Loop ────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("🎩 Alfred — Family AI Advisor")
    print("=" * 55)
    print("Type 'quit' to exit | 'memories' to view all memories")
    print("=" * 55)
    print()

    conversation_history = []
    user_id = DEFAULT_USER

    # Opening greeting
    greeting = chat(
        "Please greet me warmly as Alfred, my family AI butler.",
        conversation_history,
        user_id,
    )
    print(f"Alfred: {greeting}\n")

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() == "quit":
                print("\nAlfred: Very good, sir. Until next time. 🎩")
                break

            if user_input.lower() == "memories":
                all_memories = memory.get_all(user_id=user_id)
                print("\n📋 All stored memories:")
                print("-" * 40)
                if all_memories:
                    for m in all_memories:
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

        except KeyboardInterrupt:
            print("\n\nAlfred: Very good, sir. Until next time. 🎩")
            break


if __name__ == "__main__":
    main()

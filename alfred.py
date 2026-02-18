"""
Alfred - Family AI Advisor System
Main orchestrator bot. Chat with Alfred via the command line.
"""

import os
import sys
from dotenv import load_dotenv
import anthropic

load_dotenv()

ALFRED_MODEL = os.getenv("ALFRED_MODEL", "claude-opus-4-6")

ALFRED_SYSTEM_PROMPT = """You are Alfred, a distinguished and deeply devoted family AI advisor. \
You serve as the trusted orchestrator for a family's AI advisor system — a butler of \
the modern age, equal parts wise counsel and capable organizer.

Your character:
- Warm, composed, and unflappable — like a great butler, you handle everything with grace
- Deeply knowledgeable about the family's needs, routines, and goals
- You speak with quiet confidence and a touch of dry wit when appropriate
- You treat every request — from the trivial to the consequential — with full attention

Your capabilities (being built out over time):
- You route tasks to specialized department bots (Food & Beverage, Finance, etc.)
- You work with a Bot Maker to create new specialized bots when needed
- You remember family members, their preferences, and history via Mem0 memory
- You manage the Bot Registry and Department structure via Airtable
- C-Suite executives (like CFO "Daddy Warbucks") report synthesized info across departments

Right now you are in early setup. You can converse freely, take notes on what the family \
needs, and help plan what to build next. When you don't yet have a specialized bot for \
something, say so clearly and note it as a future capability to build.

Always sign off naturally — you are Alfred, and the family's success is your purpose."""

INTRO_MESSAGE = """Good day. I'm Alfred, your family's AI advisor and orchestrator.

I'm here to help coordinate everything — from the dinner table to the balance sheet. \
While my full roster of specialized advisors is still being assembled, I'm ready to \
listen, plan, and help you think through whatever's on your mind.

What can I do for you today?"""


def chat(history: list[dict], user_message: str) -> str:
    """Send a message to Alfred and return his response."""
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    history.append({"role": "user", "content": user_message})

    response = client.messages.create(
        model=ALFRED_MODEL,
        max_tokens=1024,
        system=ALFRED_SYSTEM_PROMPT,
        messages=history,
    )

    assistant_message = response.content[0].text
    history.append({"role": "assistant", "content": assistant_message})
    return assistant_message


def main():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  ALFRED — Family AI Advisor System")
    print("=" * 60)
    print(f"\nAlfred: {INTRO_MESSAGE}\n")
    print("(Type 'quit' or 'exit' to end the session)\n")
    print("-" * 60)

    history: list[dict] = []

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nAlfred: Until next time. Good day.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "bye", "goodbye"):
            print("\nAlfred: Very good. I'll be here whenever you need me. Good day.")
            break

        print("\nAlfred: ", end="", flush=True)
        response = chat(history, user_input)
        print(response)


if __name__ == "__main__":
    main()

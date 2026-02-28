from fastapi import FastAPI
from pydantic import BaseModel
import httpx
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Alfred Orchestrator")

# Agent registry — add new agents here as they're built
AGENTS = {
    "food_beverage": {
        "url": os.getenv("FOOD_BEVERAGE_URL", "http://food_beverage:8001"),
        "description": "Manages grocery lists, meal planning, and food-related tasks",
        "keywords": [
            "grocery", "groceries", "food", "eat", "milk", "eggs", "meal",
            "recipe", "buy", "shopping", "fridge", "pantry", "dinner",
            "lunch", "breakfast", "ingredient", "produce", "dairy", "meat",
        ],
    }
}


class ChatRequest(BaseModel):
    message: str
    user_id: str = "mike"
    conversation_history: list = []
    context: dict = {}


def route_to_agent(message: str) -> str | None:
    """Return agent name if any keyword matches, else None."""
    msg_lower = message.lower()
    for agent_name, info in AGENTS.items():
        if any(kw in msg_lower for kw in info.get("keywords", [])):
            return agent_name
    return None


@app.get("/agents/status")
async def agents_status():
    """Ping every registered agent's /health endpoint and report status."""
    status = {}
    async with httpx.AsyncClient(timeout=5.0) as client:
        for agent_name, agent_info in AGENTS.items():
            try:
                r = await client.get(f"{agent_info['url']}/health")
                if r.status_code == 200:
                    status[agent_name] = r.json()
                else:
                    status[agent_name] = {"status": "error", "code": r.status_code}
            except Exception as e:
                status[agent_name] = {"status": "unreachable", "error": str(e)}
    return {"orchestrator": "ok", "agents": status}


@app.post("/chat")
async def chat(req: ChatRequest):
    """Route message to the appropriate agent based on keywords."""
    target = route_to_agent(req.message)

    if target:
        agent_url = AGENTS[target]["url"]
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(
                f"{agent_url}/handle",
                json={
                    "message": req.message,
                    "user_id": req.user_id,
                    "conversation_history": req.conversation_history,
                    "context": req.context,
                },
            )
            return r.json()

    return {
        "response": (
            "I'm not sure which agent handles that. "
            f"Available agents: {', '.join(AGENTS.keys())}"
        ),
        "status": "unrouted",
    }


@app.get("/health")
async def health():
    return {"service": "orchestrator", "status": "ok"}

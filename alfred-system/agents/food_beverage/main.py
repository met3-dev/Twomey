from fastapi import FastAPI
from pydantic import BaseModel
import anthropic
import json
import os
from dotenv import load_dotenv
from tools.grocery import get_grocery_list, add_grocery_item, update_item_status, delete_grocery_item

load_dotenv()

app = FastAPI(title="Food & Beverage Agent")
claude = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

SYSTEM = """You are the Food & Beverage Manager for the Alfred family system.
You manage the family grocery list, meal planning, and food-related tasks.
You have access to tools to view, add, update, and remove items from the grocery list.
Be concise and practical. Always confirm actions taken."""

# Tool definitions for Claude
TOOLS = [
    {
        "name": "get_grocery_list",
        "description": "Get the current grocery list",
        "input_schema": {"type": "object", "properties": {}, "required": []}
    },
    {
        "name": "add_grocery_item",
        "description": "Add an item to the grocery list",
        "input_schema": {
            "type": "object",
            "properties": {
                "item": {"type": "string", "description": "Name of the item"},
                "quantity": {"type": "string", "description": "Quantity needed"},
                "category": {"type": "string", "description": "Category like Produce, Dairy, Meat, etc."}
            },
            "required": ["item"]
        }
    },
    {
        "name": "update_item_status",
        "description": "Update the status of a grocery item (e.g. mark as bought)",
        "input_schema": {
            "type": "object",
            "properties": {
                "record_id": {"type": "string", "description": "The Airtable record ID"},
                "status": {"type": "string", "description": "New status: To Buy, Bought, Skip"}
            },
            "required": ["record_id", "status"]
        }
    },
    {
        "name": "delete_grocery_item",
        "description": "Remove an item from the grocery list",
        "input_schema": {
            "type": "object",
            "properties": {
                "record_id": {"type": "string", "description": "The Airtable record ID"}
            },
            "required": ["record_id"]
        }
    }
]


def execute_tool(tool_name: str, tool_input: dict) -> str:
    if tool_name == "get_grocery_list":
        result = get_grocery_list()
        return json.dumps(result)
    elif tool_name == "add_grocery_item":
        result = add_grocery_item(**tool_input)
        return json.dumps({"success": True, "record": result})
    elif tool_name == "update_item_status":
        result = update_item_status(**tool_input)
        return json.dumps({"success": True})
    elif tool_name == "delete_grocery_item":
        delete_grocery_item(**tool_input)
        return json.dumps({"success": True})
    return json.dumps({"error": "Unknown tool"})


class AgentRequest(BaseModel):
    message: str
    user_id: str = "mike"
    conversation_history: list = []
    context: dict = {}


@app.post("/handle")
async def handle(req: AgentRequest):
    messages = req.conversation_history.copy()
    messages.append({"role": "user", "content": req.message})

    # Agentic loop — let Claude use tools until it's done
    while True:
        response = claude.messages.create(
            model="claude-opus-4-6",
            max_tokens=1024,
            system=SYSTEM,
            tools=TOOLS,
            messages=messages
        )

        # If Claude wants to use a tool
        if response.stop_reason == "tool_use":
            # Add Claude's response to message history
            messages.append({"role": "assistant", "content": response.content})

            # Execute each tool call and collect results
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = execute_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })
            messages.append({"role": "user", "content": tool_results})

        else:
            # Claude is done, extract final text response
            final_text = next(
                (block.text for block in response.content if hasattr(block, "text")),
                "Done."
            )
            return {"response": final_text, "status": "success"}


@app.get("/health")
async def health():
    return {"agent": "food_beverage", "status": "ok"}

import os
from pyairtable import Api


def _get_table():
    api = Api(os.getenv("AIRTABLE_API_KEY"))
    base_id = os.getenv("AIRTABLE_BASE_ID")
    return api.table(base_id, "Grocery List")


def get_grocery_list() -> list:
    """Return all records from the Grocery List table."""
    table = _get_table()
    records = table.all()
    return [{"id": r["id"], **r["fields"]} for r in records]


def add_grocery_item(item: str, quantity: str = None, category: str = None) -> dict:
    """Add a new item to the grocery list."""
    table = _get_table()
    fields = {"Item": item, "Status": "To Buy"}
    if quantity:
        fields["Quantity"] = quantity
    if category:
        fields["Category"] = category
    record = table.create(fields)
    return {"id": record["id"], **record["fields"]}


def update_item_status(record_id: str, status: str) -> dict:
    """Update the status of an existing grocery item."""
    table = _get_table()
    record = table.update(record_id, {"Status": status})
    return {"id": record["id"], **record["fields"]}


def delete_grocery_item(record_id: str) -> None:
    """Delete a grocery item by its Airtable record ID."""
    table = _get_table()
    table.delete(record_id)

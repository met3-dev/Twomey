"""
Shopping List Analyzer

Reads a shopping list CSV (item, quantity, unit_price) and produces:
- Total cost
- Most expensive item
- Cheapest item
- Average item cost
- Total number of items (units)
- Per-item subtotals, sorted by subtotal descending
"""

import sys
from pathlib import Path


CATEGORIES = {
    "produce": {"apples", "bananas", "spinach", "carrots", "onions", "garlic"},
    "dairy": {"milk", "eggs", "yogurt", "cheese", "butter"},
    "meat": {"chicken breast"},
    "pantry": {"pasta", "tomato sauce", "rice", "olive oil", "salt", "coffee"},
    "bakery": {"bread"},
    "beverages": {"orange juice"},
}


def categorize(item_name: str) -> str:
    name = item_name.strip().lower()
    for category, items in CATEGORIES.items():
        if name in items:
            return category
    return "other"


def parse_list(path: str) -> list[dict]:
    items = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 3:
                print(f"  [warn] line {lineno}: expected 3 fields, got {len(parts)} — skipping")
                continue
            name, qty_str, price_str = parts
            try:
                qty = int(qty_str)
                unit_price = float(price_str)
            except ValueError:
                print(f"  [warn] line {lineno}: invalid quantity or price — skipping")
                continue
            items.append({
                "name": name,
                "quantity": qty,
                "unit_price": unit_price,
                "subtotal": round(qty * unit_price, 2),
                "category": categorize(name),
            })
    return items


def analyze(items: list[dict]) -> None:
    if not items:
        print("No valid items found.")
        return

    total_cost = round(sum(i["subtotal"] for i in items), 2)
    total_units = sum(i["quantity"] for i in items)
    avg_unit_price = round(sum(i["unit_price"] for i in items) / len(items), 2)
    most_expensive = max(items, key=lambda i: i["subtotal"])
    cheapest = min(items, key=lambda i: i["subtotal"])

    # Category breakdown
    category_totals: dict[str, float] = {}
    for item in items:
        cat = item["category"]
        category_totals[cat] = round(category_totals.get(cat, 0) + item["subtotal"], 2)

    # --- Output ---
    print("=" * 55)
    print("  SHOPPING LIST ANALYSIS")
    print("=" * 55)

    print("\nItem Breakdown (sorted by subtotal):")
    print(f"  {'Item':<20} {'Qty':>5} {'Unit $':>8} {'Subtotal':>10}  Category")
    print(f"  {'-'*20} {'---':>5} {'------':>8} {'--------':>10}  --------")
    for item in sorted(items, key=lambda i: i["subtotal"], reverse=True):
        print(
            f"  {item['name']:<20} {item['quantity']:>5} "
            f"${item['unit_price']:>7.2f} ${item['subtotal']:>9.2f}  {item['category']}"
        )

    print("\nCategory Totals:")
    for cat, total in sorted(category_totals.items(), key=lambda x: x[1], reverse=True):
        pct = round(100 * total / total_cost, 1)
        print(f"  {cat:<12} ${total:>7.2f}  ({pct}%)")

    print("\nSummary:")
    print(f"  Total items (lines):   {len(items)}")
    print(f"  Total units:           {total_units}")
    print(f"  Avg unit price:       ${avg_unit_price:.2f}")
    print(f"  Most expensive item:   {most_expensive['name']} (${most_expensive['subtotal']:.2f})")
    print(f"  Cheapest item:         {cheapest['name']} (${cheapest['subtotal']:.2f})")
    print(f"  TOTAL COST:           ${total_cost:.2f}")
    print("=" * 55)


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "shopping_list.txt"
    if not Path(path).exists():
        print(f"Error: file '{path}' not found.")
        sys.exit(1)
    print(f"Analyzing: {path}\n")
    items = parse_list(path)
    analyze(items)


if __name__ == "__main__":
    main()

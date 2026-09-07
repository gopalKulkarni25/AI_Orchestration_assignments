"""Transaction loading and validation."""
from typing import Any


def load_transactions(filepath: str) -> list[dict[str, Any]]:
    """Load transactions from a JSON file and return them as a list of dicts."""
    import json
    with open(filepath, encoding="utf-8") as fh:
        return json.load(fh)


def validate_transaction(tx: dict[str, Any]) -> bool:
    """Return True if a transaction dict has amount, currency and timestamp keys."""
    return all(k in tx for k in ("amount", "currency", "timestamp"))

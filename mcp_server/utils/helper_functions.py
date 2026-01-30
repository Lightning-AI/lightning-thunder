from datetime import datetime, timezone
from typing import Any
from dataclasses import asdict


def calculate_days_diff(date_str: str) -> int:
    """Function to calculate the days difference between two dates
    Args:
        date_str: The date string

    Returns:
        The days difference
    """
    date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    now = datetime.now(timezone.utc)
    return (now - date).days


def dataclass_to_dict(obj: Any) -> Any:
    """Convert dataclass to dict recursively.

    Args:
        obj: The object to convert

    Returns:
        The object as a dict
    """
    if hasattr(obj, "__dataclass_fields__"):
        return {k: dataclass_to_dict(v) for k, v in asdict(obj).items()}
    return obj

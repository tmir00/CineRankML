"""JSON encode and decode helpers for Kafka event payloads. Serialization and deserialization conversions."""

from __future__ import annotations

import json

from typing import Any
from pydantic import BaseModel


def encode_event(event: BaseModel) -> bytes:
    """
    Serialize a Pydantic event to UTF-8 JSON bytes for Kafka.

    ============================ Arguments ============================
    event: A validated Kafka event model.

    ============================ Returns ============================
    JSON bytes ready for produce().
    """
    return event.model_dump_json().encode("utf-8")


def decode_json(raw: bytes | None) -> dict[str, Any]:
    """
    Decode Kafka message bytes into a Python dict.

    Do this by:
    1. Checking that the payload is not empty.
    2. Parsing UTF-8 JSON into a dict.

    ============================ Arguments ============================
    raw: The message value bytes from Kafka.

    ============================ Returns ============================
    The decoded JSON object as a dict.

    ============================ Raises ============================
    ValueError: When the payload is empty or not valid JSON.
    """
    # Check if the payload is empty or not valid JSON.
    if raw is None or len(raw) == 0:
        raise ValueError("empty_payload")

    # Try to decode the payload as JSON.
    try:
        payload = json.loads(raw.decode("utf-8"))
    
    # If the payload is not valid JSON, raise a ValueError.
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("json_decode_error") from exc

    # Check if the payload is a dictionary.
    if not isinstance(payload, dict):
        raise ValueError("json_decode_error")

    return payload

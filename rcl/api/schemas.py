"""Request/response models (field names fixed by docs/architecture/api.md)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel


class EntryInfo(BaseModel):
    id: str
    kind: Literal["environment", "controller"]
    metadata: dict[str, Any]
    params_schema: dict[str, Any]

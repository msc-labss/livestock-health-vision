"""Versioned record base for every stage boundary.

Each stage boundary is a record schema that carries its own name and version, so
a breaking change to a contract is visible in the data rather than silent. The
base class also enforces required fields at construction time, because several
requirements turn on refusing an incomplete record by name rather than emitting
a partially populated one.
"""

from __future__ import annotations

import dataclasses
import types
from dataclasses import dataclass, field, fields
from datetime import date, datetime
from enum import Enum
from typing import Any, ClassVar, Union, get_args, get_origin, get_type_hints

from .errors import MissingFieldError

__all__ = ["MISSING", "Record", "req", "canonical_json"]


class _Missing:
    """Sentinel distinguishing "not supplied" from "supplied as null"."""

    _instance: _Missing | None = None

    def __new__(cls) -> _Missing:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "MISSING"

    def __bool__(self) -> bool:
        return False


MISSING: Any = _Missing()


def req(**kwargs: Any) -> Any:
    """Declare a required field.

    The field defaults to the MISSING sentinel so that omitting it, or passing
    ``None`` for it, both fail at construction naming the field.
    """
    metadata = dict(kwargs.pop("metadata", {}))
    metadata["required"] = True
    return field(default=MISSING, metadata=metadata, **kwargs)


def opt(default: Any = None, **kwargs: Any) -> Any:
    """Declare an optional field with an explicit default."""
    metadata = dict(kwargs.pop("metadata", {}))
    metadata["required"] = False
    if isinstance(default, (list, dict, set)):
        return field(default_factory=lambda d=default: type(d)(d), metadata=metadata, **kwargs)
    return field(default=default, metadata=metadata, **kwargs)


@dataclass(frozen=True)
class Record:
    """Base for a record written at a stage boundary."""

    SCHEMA_NAME: ClassVar[str] = "record"
    SCHEMA_VERSION: ClassVar[str] = "0"

    def __post_init__(self) -> None:
        for f in fields(self):
            if not f.metadata.get("required"):
                continue
            value = getattr(self, f.name)
            if value is MISSING or value is None:
                raise MissingFieldError(self.schema_id, f.name)

    # -- identity -----------------------------------------------------------

    @property
    def schema_version(self) -> str:
        return self.SCHEMA_VERSION

    @property
    def schema_name(self) -> str:
        return self.SCHEMA_NAME

    @property
    def schema_id(self) -> str:
        return f"{self.SCHEMA_NAME}@{self.SCHEMA_VERSION}"

    # -- serialisation ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Encode to plain JSON-compatible types, version included.

        The schema name and version travel inside the record so a consumer can
        read them without out-of-band knowledge.
        """
        out: dict[str, Any] = {
            "schema_name": self.SCHEMA_NAME,
            "schema_version": self.SCHEMA_VERSION,
        }
        for f in fields(self):
            out[f.name] = encode_value(getattr(self, f.name))
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Record:
        """Decode a record, refusing one that omits a required field by name."""
        hints = get_type_hints(cls)
        kwargs: dict[str, Any] = {}
        for f in fields(cls):
            required = bool(f.metadata.get("required"))
            if f.name not in data or data[f.name] is None:
                if required:
                    raise MissingFieldError(f"{cls.SCHEMA_NAME}@{cls.SCHEMA_VERSION}", f.name)
                continue
            kwargs[f.name] = decode_value(hints.get(f.name, Any), data[f.name])
        return cls(**kwargs)

    def replace(self, **changes: Any) -> Record:
        return dataclasses.replace(self, **changes)


# -- codec ------------------------------------------------------------------


def encode_value(value: Any) -> Any:
    if value is MISSING:
        return None
    if isinstance(value, Record):
        return value.to_dict()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, (list, tuple)):
        return [encode_value(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted(encode_value(v) for v in value)
    if isinstance(value, dict):
        return {str(k): encode_value(v) for k, v in value.items()}
    if dataclasses.is_dataclass(value):
        return {f.name: encode_value(getattr(value, f.name)) for f in fields(value)}
    return value


def _union_args(tp: Any) -> tuple[Any, ...] | None:
    origin = get_origin(tp)
    if origin is Union or isinstance(tp, types.UnionType):
        return get_args(tp)
    return None


def decode_value(tp: Any, value: Any) -> Any:
    if value is None:
        return None

    union = _union_args(tp)
    if union is not None:
        non_none = [a for a in union if a is not type(None)]
        for candidate in non_none:
            try:
                return decode_value(candidate, value)
            except (TypeError, ValueError):
                continue
        return value

    origin = get_origin(tp)
    if origin in (list, tuple, set, frozenset):
        args = get_args(tp)
        inner = args[0] if args else Any
        if origin is tuple and len(args) == 2 and args[1] is Ellipsis:
            return tuple(decode_value(inner, v) for v in value)
        if origin is tuple:
            return tuple(decode_value(a, v) for a, v in zip(args, value, strict=False))
        decoded = [decode_value(inner, v) for v in value]
        return origin(decoded) if origin is not list else decoded
    if origin is dict:
        args = get_args(tp)
        vt = args[1] if len(args) == 2 else Any
        return {k: decode_value(vt, v) for k, v in value.items()}

    if isinstance(tp, type):
        if issubclass(tp, Record):
            return tp.from_dict(value)
        if issubclass(tp, Enum):
            return tp(value)
        if tp is datetime:
            return datetime.fromisoformat(value) if isinstance(value, str) else value
        if tp is date:
            return date.fromisoformat(value) if isinstance(value, str) else value
        if dataclasses.is_dataclass(tp) and isinstance(value, dict):
            hints = get_type_hints(tp)
            return tp(
                **{
                    f.name: decode_value(hints.get(f.name, Any), value[f.name])
                    for f in fields(tp)
                    if f.name in value
                }
            )
    return value


def canonical_json(value: Any) -> str:
    """Serialise to JSON with mapping keys sorted, so ordering is not material."""
    import json

    return json.dumps(encode_value(value), sort_keys=True, separators=(",", ":"), default=str)

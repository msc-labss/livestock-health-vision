"""Exception hierarchy shared by every stage.

Errors are named after the contract they violate, because several requirements
turn on the difference between "refused with a named cause" and "silently
degraded".
"""

from __future__ import annotations


class LhvError(Exception):
    """Base class for every error raised by this package."""


class SchemaError(LhvError):
    """A record does not satisfy the schema it declares."""


class MissingFieldError(SchemaError):
    """A required field was absent. Carries the field name so callers can report it."""

    def __init__(self, schema_name: str, field_name: str) -> None:
        self.schema_name = schema_name
        self.field_name = field_name
        super().__init__(f"{schema_name}: required field {field_name!r} is missing")


class RegistrationError(LhvError):
    """A source or dataset was registered without the keys later stages depend on."""

    def __init__(self, message: str, *, missing_field: str | None = None) -> None:
        self.missing_field = missing_field
        super().__init__(message)


class DerivedMediaIsolationError(LhvError):
    """A configured output root is tracked by version control."""

    def __init__(self, message: str, *, path: str) -> None:
        self.path = path
        super().__init__(message)


class UndeclaredFeatureError(LhvError):
    """Extraction produced a value whose name is not in the declared feature set."""

    def __init__(self, feature_name: str, feature_set: str) -> None:
        self.feature_name = feature_name
        self.feature_set = feature_set
        super().__init__(f"feature {feature_name!r} is not declared in feature set {feature_set!r}")


class LeakageError(LhvError):
    """A requested split could not be satisfied without overlap on its disjointness key."""

    def __init__(self, key_kind: str, overlapping: tuple[str, ...]) -> None:
        self.key_kind = key_kind
        self.overlapping = tuple(overlapping)
        listed = ", ".join(self.overlapping)
        super().__init__(
            f"{key_kind}-disjoint split requested but these keys appear in more than one "
            f"partition: {listed}"
        )


class ProfileError(LhvError):
    """A species profile is malformed or unavailable."""


class ConfigError(LhvError):
    """A resolved configuration is malformed or internally inconsistent."""

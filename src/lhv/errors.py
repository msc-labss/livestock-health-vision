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


class FeatureViewMismatchError(LhvError):
    """A declared feature's view does not match the skeleton in use.

    A feature computed in image coordinates measures a different physical
    quantity under a different camera geometry while its name, unit and priors
    stay the same. Refusing is the only outcome that does not silently
    substitute one quantity for another.
    """

    def __init__(
        self, *, feature_name: str, feature_view: str, skeleton_id: str, skeleton_view: str
    ) -> None:
        self.feature_name = feature_name
        self.feature_view = feature_view
        self.skeleton_id = skeleton_id
        self.skeleton_view = skeleton_view
        super().__init__(
            f"feature {feature_name!r} is defined for a {feature_view!r} view but skeleton "
            f"{skeleton_id!r} is {skeleton_view!r}. Computing it here would measure a "
            f"different quantity than its name, unit and priors describe."
        )


class UnimplementedFeatureError(LhvError):
    """A profile declares a feature available that extraction cannot compute.

    Distinct from a feature declared unavailable, which is an accepted state
    with a recorded reason. This one is a profile and code that disagree, and
    emitting nothing for it would look exactly like the accepted state.
    """

    def __init__(self, feature_name: str, feature_set: str) -> None:
        self.feature_name = feature_name
        self.feature_set = feature_set
        super().__init__(
            f"feature {feature_name!r} in feature set {feature_set!r} is declared available "
            f"but has no implementation; declare it unavailable with a reason, or implement it"
        )


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


class ViewMismatchError(LhvError):
    """A source's camera view does not match the view its skeleton is defined for.

    A skeleton's view decides what its keypoints mean spatially, so pose emitted
    against a mismatched view measures the wrong quantity rather than measuring
    the right one badly. An undeclared view is refused on the same grounds: it
    cannot be checked, and assuming one is how the wrong quantity gets computed
    quietly.
    """

    def __init__(
        self,
        *,
        source_id: str,
        source_view: str,
        skeleton_id: str,
        skeleton_view: str,
    ) -> None:
        self.source_id = source_id
        self.source_view = source_view
        self.skeleton_id = skeleton_id
        self.skeleton_view = skeleton_view
        if not source_view:
            message = (
                f"source {source_id!r} declares no view, and skeleton "
                f"{skeleton_id!r} is defined for a {skeleton_view!r} view. Declare the "
                f"source's view on its registration; pose refuses to assume one."
            )
        else:
            message = (
                f"source {source_id!r} was recorded {source_view!r} but skeleton "
                f"{skeleton_id!r} is defined for a {skeleton_view!r} view. Keypoints from "
                f"this pairing would measure a different quantity than their names claim."
            )
        super().__init__(message)


class ProfileError(LhvError):
    """A species profile is malformed or unavailable."""


class ConfigError(LhvError):
    """A resolved configuration is malformed or internally inconsistent."""

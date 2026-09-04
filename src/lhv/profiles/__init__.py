"""The species seam.

Everything species-specific lives here: skeleton definition, weight references
and their licences, feature-set definition, normal-range priors and the scoring
scale. No other module in the package is permitted to branch on species; the
check in ``tools/check_species_seam.py`` enforces that.
"""

from .profile import (
    PROFILE_DIR,
    FeatureDefinition,
    FeatureSet,
    KeypointDefinition,
    NormalRange,
    ScoringScale,
    SkeletonDefinition,
    SpeciesProfile,
    WeightReference,
    available_profiles,
    default_profile_name,
    load_profile,
    profile_from_dict,
)

__all__ = [
    "PROFILE_DIR",
    "FeatureDefinition",
    "FeatureSet",
    "KeypointDefinition",
    "NormalRange",
    "ScoringScale",
    "SkeletonDefinition",
    "SpeciesProfile",
    "WeightReference",
    "available_profiles",
    "default_profile_name",
    "load_profile",
    "profile_from_dict",
]

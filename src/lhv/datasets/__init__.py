"""Dataset registration records and their recorded licence and access terms."""

from .coco import CocoKeypointLabels, LabelledPoseBackend, load_coco_keypoints
from .registration import (
    REGISTRATION_DIR,
    AccessTerms,
    CountSpec,
    CountVerification,
    DatasetRegistration,
    VerificationReport,
    available_registrations,
    load_registration,
    registration_from_dict,
)

__all__ = [
    "REGISTRATION_DIR",
    "CocoKeypointLabels",
    "LabelledPoseBackend",
    "AccessTerms",
    "CountSpec",
    "CountVerification",
    "DatasetRegistration",
    "VerificationReport",
    "available_registrations",
    "load_coco_keypoints",
    "load_registration",
    "registration_from_dict",
]

"""Dataset registration: the keys a source must supply, and count verification."""

from __future__ import annotations

import pytest

from lhv.datasets import (
    AccessTerms,
    CountSpec,
    DatasetRegistration,
    available_registrations,
    load_registration,
    registration_from_dict,
)
from lhv.errors import MissingFieldError, RegistrationError


def _terms() -> AccessTerms:
    return AccessTerms(licence="CC-BY-4.0", access_route="open-download")


def _registration(**overrides) -> DatasetRegistration:
    base = dict(
        name="example",
        version="1",
        site_key="site-a",
        camera_ids=("cam-1",),
        animal_set_keys=("herd-a",),
        access=_terms(),
    )
    base.update(overrides)
    return DatasetRegistration(**base)


# -- 2.3 registration records -----------------------------------------------


def test_both_sources_are_registered() -> None:
    assert set(available_registrations()) >= {"cattleeyeview", "multicamcows2024"}


@pytest.mark.parametrize("name", ["cattleeyeview", "multicamcows2024"])
def test_registration_supplies_the_split_keys(name: str) -> None:
    registration = load_registration(name)
    assert registration.site_key
    assert registration.camera_ids
    assert registration.animal_set_keys


@pytest.mark.parametrize("name", ["cattleeyeview", "multicamcows2024"])
def test_registration_records_licence_and_access_terms(name: str) -> None:
    access = load_registration(name).access
    assert access.licence
    assert access.commercial_use
    assert access.citation
    assert access.recorded_on
    if access.requires_human_request:
        assert access.request_url, "a gated source must record where the request is made"
    else:
        assert access.download_url, "an open source must record where it is downloaded from"


@pytest.mark.parametrize("name", ["cattleeyeview", "multicamcows2024"])
def test_registration_declares_limitations(name: str) -> None:
    assert load_registration(name).limitations


@pytest.mark.parametrize("field_name", ["site_key", "camera_ids", "animal_set_keys"])
def test_registration_is_refused_when_a_required_key_is_absent(field_name: str) -> None:
    kwargs = dict(
        name="example",
        version="1",
        site_key="site-a",
        camera_ids=("cam-1",),
        animal_set_keys=("herd-a",),
        access=_terms(),
    )
    kwargs.pop(field_name)
    with pytest.raises(MissingFieldError) as excinfo:
        DatasetRegistration(**kwargs)
    assert excinfo.value.field_name == field_name


@pytest.mark.parametrize(
    ("field_name", "empty"),
    [("site_key", ""), ("camera_ids", ()), ("animal_set_keys", ())],
)
def test_registration_is_refused_when_a_required_key_is_empty(field_name, empty) -> None:
    with pytest.raises(RegistrationError) as excinfo:
        _registration(**{field_name: empty})
    assert excinfo.value.missing_field == field_name


def test_registration_without_access_terms_is_refused() -> None:
    with pytest.raises(RegistrationError, match="access"):
        registration_from_dict(
            {
                "name": "example",
                "version": "1",
                "site_key": "site-a",
                "camera_ids": ["cam-1"],
                "animal_set_keys": ["herd-a"],
            }
        )


def test_unknown_registration_names_what_is_available() -> None:
    with pytest.raises(RegistrationError, match="cattleeyeview"):
        load_registration("no-such-dataset")


# -- count verification: declared figures are not evidence ------------------


def test_verification_reports_absence_rather_than_passing(tmp_path) -> None:
    report = _registration().verify(tmp_path / "missing")
    assert not report.present
    assert not report.ok
    assert "not present" in report.describe()


def test_verification_counts_what_is_on_disk(tmp_path) -> None:
    (tmp_path / "seq-1").mkdir()
    (tmp_path / "seq-2").mkdir()
    for i in range(5):
        (tmp_path / "seq-1" / f"{i}.jpg").write_bytes(b"x")

    registration = _registration(
        count_specs={
            "sequences": CountSpec(kind="dirs", glob="*", declared=2),
            "frames": CountSpec(kind="files", glob="**/*.jpg", declared=5),
        }
    )
    report = registration.verify(tmp_path)
    assert report.ok
    assert {c.name: c.observed for c in report.counts} == {"sequences": 2, "frames": 5}


def test_verification_fails_when_the_literature_disagrees_with_the_download(tmp_path) -> None:
    """The declared figure is a claim; the count on disk is the evidence."""
    (tmp_path / "a.jpg").write_bytes(b"x")
    registration = _registration(
        count_specs={"frames": CountSpec(kind="files", glob="**/*.jpg", declared=30703)}
    )
    report = registration.verify(tmp_path)
    assert report.present
    assert not report.ok
    mismatch = report.counts[0]
    assert mismatch.declared == 30703
    assert mismatch.observed == 1
    assert "MISMATCH" in report.describe()

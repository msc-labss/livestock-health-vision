# P0 outcome, against the gate stated in the proposal

**Gate (from the proposal and the README phase table):** *the pipeline runs end
to end, with no manual step between stages.*

**Status: the spine is built and runs end to end; the gate is not yet met,
because neither public dataset has been obtained.** The pipeline has been run
from source registration to exported events over a labelled source, but not over
CattleEyeView or MultiCamCows2024. Everything below separates what has evidence
behind it from what does not.

## What runs

One invocation carries a registered dataset through every stage:

```
register -> ingest -> detect -> track -> pose -> identity
         -> passes -> features -> time series -> baseline
         -> risk -> threshold -> event -> evidence clip -> export
```

`lhv run --dataset <name> --data-root <path> --output <path>` runs all of it.
`--stages baseline,events` re-runs the cheap tail without re-running detection;
this is verified to produce byte-identical events to a full rerun.

## What has been validated, and against what

| Claim | Evidence | Status |
|---|---|---|
| Every stage boundary is a versioned record that refuses an incomplete record by name | Unit tests per schema | Verified |
| Ingest is deterministic and resumable, and never invents a timestamp | Unit tests over synthetic video | Verified |
| A frame with no animal yields an empty result, not an absent one | Verified against **real pretrained weights** (`yolo11m`) and the weights-free detector | Verified |
| Pose output is emitted against the profile skeleton, with unobserved keypoints marked not-visible and carrying no coordinate | Verified against **real pretrained weights** (`yolo11m-pose`) | Verified |
| Swapping weights makes outputs distinguishable by model identity alone | Unit test | Verified |
| Tracking records exit, occlusion loss and detection failure as distinct reasons | Unit tests | Verified |
| Identity prefers the external anchor, marks the visual fallback, and leaves ambiguity unresolved | Unit tests | Verified |
| Two time-overlapping tracklets resolving to one animal are both withheld and recorded as a conflict | Unit test | Verified |
| A partial pass is labelled, never extrapolated | Unit tests | Verified |
| An undeclared feature fails extraction naming the offender | Unit test | Verified |
| An invalid pass is retained for audit and presents no measurements downstream | Unit tests | Verified |
| The own-history baseline excludes the observation being scored | Unit test | Verified |
| A herd-wide shift raises own-history deviation while leaving herd-relative deviation flat | Unit test with a synthetic herd shift | Verified |
| Cold start yields an explicit insufficient-history state and no numeric score | Unit test | Verified |
| A risk score is recomputable from the stored series using only its recorded windows | Unit test | Verified |
| Alerts retain a bounded, human-masked clip; routine observations retain none | Unit and end-to-end tests | Verified |
| A clip that cannot be masked is not retained, and the event says so | Unit test | Verified |
| Redelivery is detectable from the idempotency key; undelivered events are retained and counted | Unit tests | Verified |
| Leakage aborts the run naming the overlap, before any metric is computed | Unit tests | Verified |
| Metric families stay separate and no aggregate is emitted across them | Unit and end-to-end tests | Verified |
| A report is reproducible from its recorded inputs | Unit and end-to-end tests | Verified |
| Every report states the stubbed inference and the site count | Unit and end-to-end tests | Verified |
| Every stage is independently re-runnable, matching a full rerun | End-to-end test | Verified |
| Version control carries no video, derived media or model weights, in the tree or in a clean clone | `tools/check_no_media_tracked.sh`, `tools/check_clean_clone.sh` | Verified |
| Only the species profile names a species | `tools/check_species_seam.py`, in CI | Verified |

## What rests on injected data

**All of it, above the phenotype layer.** The baseline, risk score, threshold
policy, alerting and evidence retention have been exercised only against
synthetic deviations injected into the phenotype time series with a declared
magnitude and temporal shape. No public livestock dataset carries clinical
outcomes, so there was nothing to fit or test them against.

Concretely: an injected ramp on one animal's `step_asymmetry_front` raises that
animal's risk score, crosses the threshold policy, produces an alert-level event
with a masked evidence clip, and is exported — with the injection marker intact
at every step and on the exported JSON. That demonstrates the machinery
functions. It demonstrates nothing about lameness.

Every event produced in P0 carries `stub_derived: true` and `non_clinical: true`,
and every report states this in its declared limitations.

## What has not been validated

- **Neither public dataset was obtained**, so nothing has run against real
  livestock imagery. See "Blocked" below.
- **Clinical correlation is entirely unevaluated.** The phenotype family
  measures reproducibility and stability across repeated passes — whether the
  same animal measured twice gives the same number. That is a precondition for a
  feature meaning something, not evidence that it does.
- **Lead time is unmeasured.** There is no reference event in P0 to measure it
  against. It becomes measurable in P1 against parallel human locomotion scoring.
- **Alarm burden is uncalibrated.** The harness reports it per thousand
  animal-days alongside the policy that produced it, but the threshold policy was
  chosen arbitrarily and has never met a real prevalence. That is P2's work.
- **Site-disjoint validation did not run.** Both registered sources are single
  site. The split machinery and the provenance keys are built for it; the data
  is not there. Every report states this.
- **The pose weights are a placeholder.** No cattle-trained checkpoint for the
  24-keypoint top-down convention is publicly downloadable, and training one is
  an explicit P0 non-goal. The profile currently points the pose stage at
  COCO-human weights with a declared, explicitly approximate keypoint map, and
  emits every profile keypoint the model cannot produce as not-visible. Pose
  accuracy measured in P0 measures that placeholder, not the achievable ceiling.
  Substituting a cattle-trained checkpoint is an edit to
  `src/lhv/profiles/definitions/cattle.yaml` and nothing else.
- **Interface stability is asserted, not tested.** P4's species swap and P3's
  edge split are the tests. Until then, every stage boundary is a versioned
  record schema, which makes a breaking change visible rather than silent.

## Blocked

| Task | Blocker |
|---|---|
| Obtain CattleEyeView (2.1), tracklet-id uniqueness over a full pass through it (4.5), perception metrics against its labels (9.3), the end-to-end run over it (10.1) | The dataset is distributed only through a Google Form request to the authors, requiring a named requester and an affiliation, and awaiting their approval. It cannot be fetched programmatically. Request URL is recorded in `src/lhv/datasets/registrations/cattleeyeview.yaml`. |
| Obtain MultiCamCows2024 (2.2), fallback reporting against it (9.5), the end-to-end run over it (10.2) | Open download, but 36.5 GB as a single archive, needing roughly 73 GB free to unpack. The build machine has ~20 GB free on one filesystem. Download URL is recorded in `src/lhv/datasets/registrations/multicamcows2024.yaml`. |

Both registrations are written, carry their licence and access terms, and
declare content counts from the literature that `lhv datasets verify` will check
against the download rather than take on trust. Nothing else in the pipeline
needs to change to run over either source once it is present.

## Honest summary

The scarce component the proposal set out to build — the spine — exists, is
tested, and runs. What is missing is the data to run it over. The gate as
written is one dataset acquisition away, and that acquisition is a human step
(a form, or 40 GB of free disk), not an engineering one.

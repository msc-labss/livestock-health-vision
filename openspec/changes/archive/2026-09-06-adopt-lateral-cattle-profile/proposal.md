## Why

The cattle profile was shaped by the only public source that carries cattle
keypoints, and that source is top-down. P0 measured the consequence — paws are
labelled visible in 4.5–9.1% of instances, because the body occludes its own
limbs from above — and the verified literature in
[docs/P1-LITERATURE-FINDINGS.md](../../../docs/P1-LITERATURE-FINDINGS.md) then
retired the rest of the inheritance: the geometry, the feature set chosen for
what a top-down skeleton could compute, and the skeleton itself.
[docs/P1-MODEL-DECISION.md](../../../docs/P1-MODEL-DECISION.md) records the
decisions. This change executes the part of them that does not need footage the
project does not yet have.

There is a second reason to do it now rather than at pilot time. P0's protection
against wrong pose output was accidental: the COCO-human pose weights are mapped
onto the cattle skeleton by analogy, and on top-down footage the detector ahead
of them found nothing, so nothing downstream computed. Under the lateral
geometry the detector works, the analogy map would emit confident paw keypoints,
and every limb feature would compute from them with no report saying so. The
safeguard has to become deliberate before the geometry improves, not after.

## What Changes

- **BREAKING** — `feature_set` moves from version 0 to version 1. Nine features
  chosen for published discrimination replace twelve chosen for computability.
  Records emitted under version 0 are not comparable with version 1 and are
  distinguishable by the version field that the existing spec already requires.
- **BREAKING** — the skeleton changes from `cattleeyeview-topdown-24` to a
  provisional nine-point lateral skeleton: four hooves, a three-point dorsal
  line, and a head point. Nothing in the new feature set depends on a fetlock,
  carpal, ear tip or tail end. Recorded as provisional pending one outstanding
  literature question.
- Three features are implemented against the existing keypoint-track machinery:
  `stride_length`, `speed`, `head_bob`.
- Six features are **declared but marked unavailable**, each with its reason.
  Five — `stance_duration`, `swing_duration`, `stride_duration`,
  `support_phase_asymmetry` and `tracking_distance` — need hoof ground-contact
  detection, which does not exist and cannot be tuned without lateral footage.
  The sixth, `back_posture`, needs a mid-dorsal keypoint that the chosen pose
  backend does not emit: sagittal arch needs three points along the back and
  AP-10K supplies two. Declaring them unavailable rather than omitting them
  keeps the decided feature set visible and keeps the reason for each absence
  recorded. A single labelled mid-dorsal point is the cheapest route to
  `back_posture`, and is the first thing to test on the pilot's labelled
  frames.
- Three features retire because they measure the wrong axis under a lateral
  mount: `lateral_sway`, `head_lateral_offset`, `spine_lateral_curvature`. The
  quantities they would accidentally measure return, correctly named, as
  `head_bob` and `back_posture`.
- `stride_frequency_front`/`_back` retire in favour of `stride_duration`, and
  with them `requires_sampling_hz: 5.0`, which was twice the upper bound of a
  guessed stride band the literature does not support.
- `step_asymmetry_front`/`_back` retire in favour of `support_phase_asymmetry`,
  and `speed_variability` retires for want of any supporting evidence.
- `tracking_jitter` leaves the clinical feature set and becomes observation
  quality metadata on the pass. It measures the tracker, not the animal.
- Priors gain a provenance that is not a guess. Four features carry a measured
  healthy anchor and a direction of change under lameness, sourced to Flower et
  al. 2005 and recorded as group means rather than as population bands.
- The pose backend becomes AP-10K, whose distal limb reaches the paw, and the
  COCO-human analogy map retires.
- `skeleton.view` becomes load-bearing. It is currently parsed, printed by
  `lhv profiles show`, and consulted by nothing.
- Weights declared as placeholder are named in every report that carries their
  output.

## Capabilities

### New Capabilities

None. Every requirement below extends an existing capability.

### Modified Capabilities

- `animal-perception`: pose output must be refused, rather than emitted, when
  the skeleton's declared view disagrees with the source's declared view; and
  weights declared placeholder must be identifiable as such on their output.
- `gait-phenotype`: a feature may be declared by the profile yet unavailable
  under the current skeleton or backend, and must be reported as unavailable
  with its reason rather than omitted, zeroed, or silently computed from
  keypoints that cannot support it.
- `evaluation-harness`: the existing requirement that declared limitations
  appear in the report extends to placeholder weights, so a family whose output
  came from weights not trained for the task says so.

## Impact

- `src/lhv/profiles/definitions/cattle.yaml` — skeleton, feature set, priors,
  `weights.pose`.
- `src/lhv/profiles/profile.py` — the `view` field gains a consumer.
- `src/lhv/phenotype/features.py` — `_compute` rewritten for the four
  implementable features; the retired three removed.
- `src/lhv/perception/pose.py` — AP-10K backend; the analogy keypoint map
  retires.
- `src/lhv/config.py` — `min_usable_features`, and the `PhenotypeConfig` comment
  that reasons about a top-down view.
- `src/lhv/evaluation/report.py` and `src/lhv/perception/report.py` — placeholder
  weights surfaced.
- Tests, and the species-seam check in `tools/check.sh`.
- Existing runs under feature-set version 0 remain readable and remain
  distinguishable by their version field. They are not migrated.

- The top-down profile is **retained, frozen, as `cattle-topdown`**, and the
  default profile becomes a declaration rather than an inference from there
  being only one file. Without this the lateral profile's view check would
  correctly refuse CattleEyeView and leave the project with no end-to-end run
  over real data, turning P0's findings from reproducible measurements into
  assertions in a document. Feature extraction accordingly runs one
  implementation per declared feature, taking its keypoints from that feature's
  own `depends_on` and the body axis from a declared skeleton role, so one
  extractor serves both skeletons without naming a keypoint of its own.

### Out of scope

- Hoof ground-contact detection and the five features that depend on it. Gated
  on the pilot recording that R1 requires, not on developer time.
- The representation question (pose versus hoof-event tracking versus direct
  video), which no published study settles and which needs an experiment.
- Frame rate, study design and exposure requirements, whose supporting evidence
  is unverified or absent.

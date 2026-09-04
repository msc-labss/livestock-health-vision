## Why

Precision-livestock computer vision has a well-documented shape: detectors, pose estimators, temporal models and trackers all reach strong within-dataset numbers, while deployments fail on occlusion, identity loss, farm-to-farm domain shift, noisy clinical labels, rare positives and uncalibrated alarms. The scarce component is not a model — it is the **spine** that carries video through to a calibrated, auditable, per-animal health signal. Published work stops at a metric; commercial systems keep the spine closed.

P0 builds that spine on public datasets, where it can be built immediately with a GPU and no farm dependency. Arranging a cooperating dairy is the critical path for later phases and takes longer to negotiate than P0 takes to write, so the two run in parallel rather than in series.

## What Changes

- Add a video ingest layer that turns recorded video into frames carrying provenance (source, camera, timestamp, dataset split keys) so that no downstream stage has to guess where a frame came from.
- Add a perception layer: animal detection, multi-object tracking into tracklets, and keypoint pose estimation. Weights are pretrained and swappable; this change owns the interfaces and the evaluation, not new architectures.
- Add identity anchoring that resolves a tracklet to a stable animal identity, with an **external anchor** (parlour/AMS/RFID identifier) as the primary path and visual re-identification as fallback. Anchoring is explicit and auditable, never implicit.
- Add gait phenotype extraction that converts a single lane pass into named, biomechanically meaningful locomotion features with per-feature quality flags.
- Add a per-animal phenotype time series and a baseline model producing a deviation-based risk score against both the animal's own history and the herd.
- Add a health event schema — the canonical contract carrying animal id, timestamp, observation window, phenotype values, confidence, source sensors, context and provenance — plus alert generation with a retained evidence clip and an export adapter.
- Add an evaluation harness as a first-class module: split construction (frame, animal-disjoint, day-disjoint, farm-disjoint), separated perception / phenotype / operational metric families, and a reproducible report.
- **Health inference is stubbed, deliberately.** Public livestock datasets carry identity and behaviour labels, not clinical outcomes. P0 validates perception and phenotype layers against real public labels, and exercises the baseline, risk and alerting layers against **injected synthetic anomalies**. The stub is a declared limitation surfaced in every report, not a hidden shortcut.

Not in this change: clinical diagnosis of disease aetiology; thermal, depth or audio modalities; mastitis, respiratory disease, body condition or heat stress; farm recording and human locomotion scoring (P1); alarm-burden calibration against real prevalence (P2); on-device edge deployment (P3); the sheep species swap (P4). These are named so later phases inherit a stated boundary rather than an assumed one.

## Capabilities

### New Capabilities

- `video-ingest`: Video sources decoded to frames with provenance and split-key metadata attached; deterministic, resumable iteration over recorded material.
- `animal-perception`: Animal detection, tracking into tracklets, and keypoint pose estimation behind stable interfaces with swappable pretrained weights.
- `identity-anchoring`: Resolution of a tracklet to a stable animal identity via an external anchor, with visual re-identification fallback and explicit confidence and provenance on every assignment.
- `gait-phenotype`: Conversion of one lane pass into named locomotion features with per-feature quality flags and an overall pass-validity decision.
- `health-baseline`: Per-animal phenotype time series, own-history and herd baselines, and a deviation-based risk score with uncertainty.
- `health-events`: Canonical health event schema, alert generation with retained evidence clips, threshold policy, and export to an external consumer.
- `evaluation-harness`: Split construction, separated perception / phenotype / operational metric families, and reproducible evaluation reports carrying declared limitations.

### Modified Capabilities

None. This is the first change in a greenfield repository; `openspec/specs/` is empty.

## Impact

- **Code**: Greenfield. No existing source to modify. This change establishes the package layout, the stage interfaces and the data contracts that every later phase extends.
- **Dependencies**: A deep-learning runtime, a pretrained detector, a pretrained tracker, a pretrained keypoint model, a time-series store, and a video decoder. Specific choices are settled in design, not here.
- **Data**: Public livestock computer-vision datasets only. No farm footage. The repository never commits video, derived media or model weights.
- **Downstream phases**: P1 attaches real ground truth to the same event schema; P2 recalibrates thresholds against real prevalence; P3 moves the same stages onto edge hardware; P4 swaps the species-specific column. Each depends on the interfaces this change fixes, so interface stability is a first-class concern rather than an implementation detail.
- **Known limitation carried forward**: A single-farm POC eventually yields n=1 for domain diversity. The split construction and event schema are built multi-farm from the start so that farm-disjoint validation later becomes a data problem, not a rewrite.

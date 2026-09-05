# P0 outcome, against the gate stated in the proposal

**Gate (proposal and README phase table):** *the pipeline runs end to end, with
no manual step between stages.*

**Status: met on CattleEyeView. 66 of 67 tasks complete.** One invocation
carries the dataset from registration to exported events — all 14 sequences,
30,706 frames, 145 events, in two and a half minutes. Both public datasets were
obtained and verified against their own published figures. The single remaining
task, 10.2, cannot be satisfied by the source it names, for reasons given below.

## What runs

```
register -> ingest -> detect -> track -> pose -> identity
         -> passes -> features -> time series -> baseline
         -> risk -> threshold -> event -> evidence clip -> export
```

`lhv run --dataset <name> --data-root <path> --output <path>` runs all of it.
`--stages baseline,events` re-runs the cheap tail without re-running detection,
and is verified to produce byte-identical events to a full rerun.

## Verified against the real datasets

**CattleEyeView.** 14 sequences ✓. `images.tar.gz` holds exactly the 30,703
frames the paper reports ✓, counted inside the archive without unpacking its
12 GB. The videos decode to 30,706: sequences 03, 05 and 06 are each missing
their final frame from the extracted set, and the other eleven agree frame for
frame. Both figures are right and measure different things.

**MultiCamCows2024.** 101,329 images ✓ (the paper), 137 videos ✓ (the readme),
7 days ✓, 90 animals ✓, 3 cameras ✓, all 101,329 files accounted for by the
declared layout.

**The species profile's skeleton was checked against the release**, not against
the paper's prose. The 24-keypoint index order matches exactly. The link
topology did not and now follows the release: every limb attaches to the
withers, and there is no head-to-neck link. Flip pairs and the published OKS
sigmas are recorded.

**Recording dates were recovered from the pixels.** CattleEyeView puts no date
in any path, but the camera burns one into every frame. Reading them gives 13
distinct days and independently confirms the paper's stated window: sequence 01
is 2021-11-09 and sequence 14 is 2022-03-04. Day-disjoint splits over this
source are real rather than a sequence-shaped proxy.

## What the data said that the papers did not

**A top-down camera cannot see a cow's legs.** Across all 24,054 labelled
CattleEyeView instances, the neck, ears, head, withers and elbows are visible
74–93% of the time; knees 4–10%; paws 4.5–9.1%. The body occludes its own limbs
from directly above. Stride length, stride frequency and step asymmetry each
depend on a paw and are unmeasurable on this source, while the body-axis
features are well observed. **This is a P1 design input: a purely top-down lane
camera will not yield limb kinematics.**

**Frame rate varies by sequence and the paper does not say so** — 8 fps for
01–06, 3 fps for 07–11, 5 fps for 12–14. Five of fourteen sequences sample below
twice the stride band the profile's own priors describe. A feature now declares
the sampling rate it needs and is marked unusable below it, naming both figures.

**The ramp carries several animals at once.** Anchor windows genuinely overlap
in 251 of 1176 pairs, which is the geometry the README says the target
deployment avoids, and a risk the design named.

**CattleEyeView labels instances, not individuals.** 753 instances, each
appearing in exactly one sequence. No animal accumulates history, so every
assessment correctly reports insufficient history and no longitudinal phenotype
is possible from this source.

**The visual fallback's confidence says nothing.** Enrolling on five days and
identifying 432 held-out tracklets from two unseen days against 89 animals gives
17.8% top-1, against 1.1% chance. Its absolute similarity carries almost no
information about correctness — AUROC 0.527, with correct and incorrect matches
averaging 0.9965 and 0.9964 — so the spec'd confidence floor filters nothing and
would admit 82% wrong identities at nominal high confidence. The margin over the
runner-up does discriminate (AUROC 0.578) and yields a real precision-recall
trade, so that is what an assignment now records and what a floor can sit on.

## What measures this system, and what does not

Detection and pose on CattleEyeView were driven by the release's own labels,
because no pretrained detector can see the animal: COCO-pretrained YOLO11 finds
zero cattle at any confidence down to 0.05 on this footage, while finding the
stockperson in shot. Their metrics are near-perfect by construction and measure
nothing about a model. Every report says so automatically whenever a family came
from a backend whose identity names it a label.

**Tracking is the one perception family that was not handed its answer.** Over
76 held-out animals of real multi-animal ramp footage, the motion-only tracker
achieves tracklet purity 0.9999, 96% of labelled tracks mostly tracked, and one
identity switch. 761 tracklets across 14 sources, no duplicate identifiers.

## What rests on injected data

Everything above the phenotype layer. The baseline, risk score, threshold
policy, alerting and evidence retention have only ever been exercised against
synthetic deviations injected into the phenotype time series with a declared
magnitude and shape. No public livestock dataset carries clinical outcomes.

Every event carries `stub_derived: true` and `non_clinical: true`, and every
report states it.

## What has not been validated

- **Clinical correlation is entirely unevaluated.** The phenotype family
  measures reproducibility, not validity.
- **Lead time is unmeasured** — P0 has no reference event. P1 supplies one.
- **Alarm burden is uncalibrated** — the threshold policy has never met a real
  prevalence. That is P2.
- **Site-disjoint validation did not run.** Both sources are single-site.
- **Interface stability is asserted, not tested.** P4's species swap and P3's
  edge split are the tests.

## The remaining task

**10.2 — "Run the full pipeline over MultiCamCows2024 and verify the identity
fallback path and multi-day series populate."** The pipeline runs over all 1,584
sources and **the identity fallback path populates**: 432 assignments, reported
standalone. The **multi-day series cannot populate**, for two independent
structural reasons, either sufficient alone:

1. The release carries no keypoint annotations, so no pose is available, so
   every one of the 19,008 feature values a full run produced is unusable and no
   pass is valid.
2. The tracklet stills carry no capture time in any path, filename or header,
   and a per-animal series is keyed by observation time. None of the 1,584
   passes has one.

This is a property of the source, not a defect in the spine: the same pipeline
populates a series from CattleEyeView, which has keypoints and recoverable
times. Populating a multi-day series from real data needs a source with both,
which is what P1's farm recording provides.

## Honest summary

The spine exists, is tested, and runs end to end on real public data. What P0
cannot do is say whether any of its measurements mean anything clinically —
no public dataset can answer that, and the reports say so rather than implying
otherwise. The most valuable output of running it on real data is the list of
things the papers do not mention: overhead cameras cannot see legs, frame rates
vary enough to make a periodic feature meaningless, and a plausible-looking
re-identification confidence can be statistically worthless.

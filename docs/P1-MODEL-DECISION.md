# P1 model decision

What the species profile becomes, decided from the verified literature.

[docs/P1-LITERATURE-SCOPE.md](P1-LITERATURE-SCOPE.md) asked the questions.
[docs/P1-LITERATURE-FINDINGS.md](P1-LITERATURE-FINDINGS.md) records which
answers survived checking. This takes the decisions those answers gate, and
takes only those — a decision resting on an unverified claim is not taken here,
and the reason is stated where it would have gone.

Nothing below is executed yet. These are decisions about what the profile
*declares*; the edit to `src/lhv/profiles/definitions/cattle.yaml` and the
feature computation that follows it is development, listed at the end.

---

## What this decides

| | decision | rests on |
|---|---|---|
| **D1** | `feature_set` version 1 — its members, units and priors | L3, confirmed |
| **D2** | the skeleton, and the pose bootstrap that replaces the P0 placeholder | L4, confirmed apart from one open count |

## What it does not decide, and why

- **The representation fork (L2).** No published study compares pose against
  hoof-event tracking or direct video on the same data, the same split and the
  same clinical reference. That is an experiment, not a search, and it stays
  open. D2 below assumes pose; the paragraph *If L2 goes the other way* says
  what survives if it does not.
- **Frame rate (L5).** The stride band is settled — dairy cattle walk at
  roughly 0.68–0.79 Hz per limb, not the 0.40–2.50 Hz the profile currently
  guesses — but the capture rate that a stance-timing measurement needs rests on
  an unverified setup claim. R2 does not move until Kang et al. 2020 is read.
- **Study design (L8).** Its supporting citation could not be found. R7 and R8's
  cadence stay as they are, marked as placeholders.
- **Exposure (L9).** Closed as *no usable evidence found*. That is an answer and
  it stands: motion blur becomes a pilot measurement, not a literature number.

---

## D1 — Feature set version 1

**Decision.** Replace the twelve top-down features with nine chosen for
published discrimination rather than for computability from the skeleton that
happened to be available.

| feature | unit | evidence | status |
|---|---|---|---|
| `support_phase_asymmetry` | seconds | Kang 2020: max−min supporting phase across hooves, Spearman ρ = 0.864 against locomotion score, 100 cows, commercial lane | confirmed |
| `stance_duration` | seconds | Flower 2005: 0.69 s healthy vs 0.91 s sole ulcer. Also one of Russello's six | confirmed |
| `swing_duration` | seconds | one of Russello's six traits | confirmed |
| `stride_duration` | seconds | Flower 2005: 1.26 ± 0.03 vs 1.48 ± 0.05 s | confirmed |
| `stride_length` | body_lengths | Flower 2005: 139.5 → 130.0 cm. Also one of Russello's six | confirmed |
| `tracking_distance` | body_lengths | one of Russello's six — the hind-to-fore hoof overlap clinicians call tracking-up | confirmed |
| `head_bob` | body_lengths | one of Russello's six; vertical oscillation, sagittal plane | confirmed |
| `back_posture` | body_lengths | one of Russello's six; sagittal arch. Van Hertem's standalone AUC ≈ 0.70 is unverified, so this is carried as a member, not as a strong one | confirmed as a member |

> **Executed, with one correction.** Implementation found that AP-10K supplies
> no mid-dorsal keypoint — its 17 points reach the paw but carry no
> mid-thoracic — and sagittal arch needs three points along the back. So
> `back_posture` joined the unavailable six rather than the implemented three,
> and version 1 computes `speed`, `stride_length` and `head_bob`. A single
> labelled mid-dorsal point is the cheapest route to recovering it, and is the
> first thing to test on the pilot's 200 labelled frames.
>
> A second decision was taken during implementation and is recorded here rather
> than only in the change. The top-down profile is **retained as a frozen
> sibling**, `cattle-topdown`, because CattleEyeView is overhead footage and the
> lateral profile correctly refuses it — which would otherwise have left this
> project with no end-to-end run over real data and turned P0's findings into
> claims. Keeping it costs one file and exercises the species seam for the first
> time. The extractor now runs one implementation per declared feature, taking
> its keypoints from that feature's own `depends_on` and the body axis from the
> skeleton's declared roles, so a single extractor serves both skeletons without
> naming a keypoint of its own.
| `speed` | body_lengths_per_second | Flower 2005: 1.11 ± 0.03 vs 0.90 ± 0.05 m/s | confirmed |

Russello et al. 2024 reached 80.1% on 98 cows using six of these together, and
79.9% using its three most important — so the set is evidenced as a set, not
only feature by feature.

### Retired

- `lateral_sway`, `head_lateral_offset`, `spine_lateral_curvature` — image-plane
  lateral quantities. Under the lateral geometry R1 now specifies, the
  image-perpendicular axis is largely vertical, so these would silently measure
  vertical bob and sagittal arch under names that say otherwise. Those
  quantities are wanted; they enter as `head_bob` and `back_posture`, named for
  what they measure.
- `stride_frequency_front` / `_back` — superseded by `stride_duration`. The same
  information, expressed as the quantity the literature actually measures and
  the one the priors exist for. This also retires `requires_sampling_hz: 5.0`,
  which was twice the upper bound of a guessed band.
- `step_asymmetry_front` / `_back` — superseded by `support_phase_asymmetry`,
  which is the better-evidenced form of the same idea.
- `speed_variability` — no supporting evidence found. Not retained on the
  strength of plausibility.

### Moved, not retired

`tracking_jitter` leaves the clinical feature set and becomes observation-quality
metadata on the pass. It measures the tracker, not the animal, and the review
found no clinical study of it. It is worth keeping for exactly the reason it is
worth removing from the phenotype.

### Two properties of this set worth stating

**Four of the nine are calibration-free.** `stance_duration`, `swing_duration`,
`stride_duration` and `support_phase_asymmetry` are times. They need the frame
rate and nothing else — no camera calibration, no scale reference, no body-length
normalisation. The best-evidenced feature in the set, Kang's support-phase
asymmetry, is one of them. This is a marked improvement on version 0, where
every feature depended on a withers-to-tailbase pixel distance.

**The lateral geometry stabilises the rest.** The remaining five normalise by
body length in pixels. Under an oblique mount that reference varies across a
pass through perspective foreshortening, and a single median per pass biases
whatever it divides. Under a lateral mount with travel roughly perpendicular to
the optical axis, scale is near-constant across the frame. L1's choice fixes a
problem that the earlier "oblique" wording would have introduced.

### Priors

Flower et al. supply real measured numbers, and this is the first non-guessed
provenance the profile will carry. They must be recorded for what they are:
**group means with standard errors, not population ranges.** A mean ± 0.03 s is
far narrower than the spread of a herd, and entering it as a `low`/`high` band
would produce a confident wrong plausibility check.

So version 1 records, for the temporal features, an **anchor** (the healthy
group mean), a **direction** (which way lameness moves it) and the source —
`flower-2005-group-mean` — rather than a band. Bands stay
`unvalidated-p0-prior` until measured on P1 footage.

| feature | healthy anchor | direction with lameness |
|---|---|---|
| `stride_duration` | 1.26 s | longer (1.48 s at sole ulcer) |
| `stance_duration` | 0.69 s | longer (0.91 s) |
| `speed` | 1.11 m/s | slower (0.90 m/s) |
| `stride_length` | 139.5 cm | shorter (130.0 cm) |

The absolute units here are a further reason to prefer the temporal features:
`speed` and `stride_length` anchors are in metres and this project has no
calibration, so they can inform a direction but not a threshold.

---

## D2 — Skeleton and pose bootstrap

**Decision.** Retire `cattleeyeview-topdown-24` and its `view: top-down`. Adopt
a lateral cattle skeleton covering exactly what D1's features depend on:

```
  four hooves        -> every temporal feature, stride length, tracking distance
  a dorsal line      -> back_posture   (withers, mid-thoracic, sacrum)
  a head point       -> head_bob
```

That is the whole requirement. Nothing in D1 depends on a fetlock, a carpal, an
ear tip or a tail end, and no point is labelled because a previous convention
had it.

**Point count is left open.** The review states T-LEAP/CoWalk uses 17 landmarks;
Russello et al. 2024 states that T-LEAP extracted motion from **nine**. Both
cannot be right, and the difference is roughly half the labelling bill. Reading
Russello et al. 2022 settles it — see *Outstanding* below. If nine points
support an 80.1% result, nine is the target.

**Bootstrap before labelling.** SuperAnimal-Quadruped (downloadable weights,
39-point unified vocabulary) and AP-10K (CC-BY-4.0, 17 keypoints, four
checkpoints published) are both available. Run them zero-shot on pilot footage
first and label only if distal accuracy is inadequate.

**Declared, and the runtime is unavailable — not merely deferred.** AP-10K's
published checkpoints are mmpose HRNet files, and the only implemented pose
backend reads Ultralytics weights. The profile declares the runtime it needs and
the pipeline refuses to build one it does not implement, by name.

This was first recorded as deferred work waiting on footage. **That reason was
wrong.** mmcv does not support Python 3.12, let alone the 3.13 this project runs
on, and carries an open defect building its extension against torch 2.9 with
CUDA 13 while this environment is torch 2.14 on CUDA 13.0. The runtime is not
installable here at all, so no amount of footage would have unblocked it.

The distinction matters because the two reasons imply different work. Deferred
work waits. An unavailable runtime means **the bootstrap decision has to be
re-opened**, and D2's choice of AP-10K needs separating into two parts that were
conflated:

- **AP-10K as a keypoint convention** — still right. Seventeen points, CC-BY-4.0,
  reaching the paw, which is all any version-1 feature needs.
- **mmpose as the way to obtain a model trained on it** — wrong for this project.

**A verified alternative exists.** ViTPose++ carries an AP-10K expert head
(dataset index 3 of its six), is supported natively by HuggingFace
`transformers` through `VitPoseForPoseEstimation`, and its checkpoints are on the
Hugging Face hub. `transformers` installs on Python 3.13 and needs no mmcv. That
reaches the same convention through a runtime this project can actually run.

Not adopted here, because swapping the declared backend is a decision rather
than an implementation detail and the licence position of the ViTPose
checkpoints has not been checked with the care AP-10K's was. What is recorded is
that the obstacle is the runtime and not the footage, and that a route around it
has been verified to exist.

A useful convergence: **AP-10K's distal limb ends at the paw, with no fetlock or
carpal** — confirmed against its repository. That was recorded as a limitation.
Against D1's feature set it is not one. Every temporal feature needs hoof
position and ground contact; a paw point supplies both. The intermediate joints
were only ever needed by a skeleton chosen before the features were.

**Retire the COCO-human keypoint map.** The P0 placeholder maps `left_wrist →
left_front_paw` and similar analogies onto the cattle skeleton. On top-down
footage that failed visibly, because the detector ahead of it found nothing. On
a lateral view the detector works, so the placeholder would emit confident paw
keypoints, every temporal feature would compute from them, and no report would
mark them as placeholder output. This is the P0 safeguard that was accidental,
and it disappears exactly when the geometry improves.

### If L2 goes the other way

If the representation comparison favours hoof-event tracking over pose, seven of
the nine features survive unchanged — every temporal feature, `stride_length`,
`tracking_distance` and `speed` need hoof positions over time and nothing else.
`head_bob` and `back_posture` are the two that require the head and dorsal
points, and therefore the two that a keypoint-free representation would cost.
That is the actual price of the fork, and it is smaller than it looked before
D1 narrowed the feature set.

---

## What executes this

Development, once explore is over and in this order:

1. `src/lhv/profiles/definitions/cattle.yaml` — skeleton (identifier, `view`,
   points, flip pairs, sigmas), `feature_set` version 1, priors with their new
   provenance, `weights.pose`.
2. `src/lhv/phenotype/features.py` — `_compute` rewritten. The temporal features
   need hoof ground-contact detection, which version 0 never had; this is the
   substantial piece of work, not the YAML.
3. `src/lhv/config.py` — `min_usable_features`, and the comment at the
   `PhenotypeConfig` boundary that reasons about a top-down view.
4. Make `skeleton.view` load-bearing. It is currently parsed, printed by
   `lhv profiles show`, and consulted by nothing — which is what allowed a
   top-down profile to be pointed at any footage at all.
5. Tests, and the species-seam check.

## Outstanding before the remaining decisions

1. **Kang et al. 2020**, institutional access — the recording setup settles L5's
   frame rate and R4's lane length together.
2. **Russello et al. 2022** — settles nine versus seventeen keypoints, and with
   it the labelling estimate in D2.
3. **Siachos et al. 2026** — find it, or re-ask L8 without the treatment-policy
   claim.
4. **The T-LEAP trajectory release's licence** — an enquiry to the authors. It
   is the only identified route to testing feature-to-score correlation before a
   farm exists.

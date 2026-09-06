## Context

The species profile is the seam this project is built around: adding a species
means adding a sibling file, not editing a stage. P0 filled that file from
CattleEyeView, because it was the only public cattle source carrying keypoints.
The consequence was that the skeleton, the feature set and every prior were
shaped by one dataset's camera position rather than by what lameness is.

`docs/P1-LITERATURE-FINDINGS.md` records which literature answers survived
checking, and `docs/P1-MODEL-DECISION.md` records the decisions they support.
This change executes those decisions as far as evidence and available footage
allow, and no further.

Three properties of the current code shape the design:

- `features.py` computes in image coordinates, decomposing motion into a
  principal direction and its image-plane perpendicular. Under a top-down mount
  that perpendicular is horizontal in the world; under a lateral mount it is
  largely vertical. The same code therefore measures a different physical
  quantity under a different mount, with no error and no change in the feature's
  name, unit or priors.
- `skeleton.view` is parsed into the profile, printed by `lhv profiles show`,
  and consulted by nothing. Nothing would have caught the substitution above.
- The pose backend is COCO-human weights mapped onto the cattle skeleton by
  analogy — `left_wrist -> left_front_paw`. On top-down footage the detector
  ahead of it found no cattle, so nothing downstream computed and the placeholder
  failed loudly. Under lateral footage the detector works.

## Goals / Non-Goals

**Goals:**

- Replace the top-down skeleton, feature set and priors with the decided lateral
  ones, and implement the features that need no capability the project lacks.
- Make the substitution described above structurally impossible rather than
  corrected once: views are declared, checked, and refused when they disagree.
- Make placeholder weights impossible to mistake for model output.
- Give the profile its first priors that are not guesses, recorded as what they
  actually are.

**Non-Goals:**

- Hoof ground-contact detection, and the features that depend on it. There is no
  lateral cattle footage in this project to tune a detector against, and tuning
  it against top-down footage where paws are visible in 4.5–9.1% of instances
  would be worse than not writing it.
- Settling whether pose is the right intermediate representation at all. No
  published study compares representations on the same data, split and clinical
  reference; that needs an experiment, not a decision.
- Migrating existing feature records from version 0. They stay readable and stay
  distinguishable by their version field.
- Training or fine-tuning any model.

## Decisions

### Feature set version 1, and what "implemented" covers

The nine decided features are declared in full. Only those computable from
keypoints an available backend actually emits are implemented; the rest are
declared unavailable with the reason.

| feature | status in this change | blocked on |
|---|---|---|
| `stride_length` | implemented | — |
| `speed` | implemented | — |
| `head_bob` | implemented | — |
| `back_posture` | declared unavailable | a mid-dorsal keypoint (see below) |
| `stance_duration` | declared unavailable | hoof ground-contact detection |
| `swing_duration` | declared unavailable | hoof ground-contact detection |
| `stride_duration` | declared unavailable | hoof ground-contact detection |
| `support_phase_asymmetry` | declared unavailable | hoof ground-contact detection |
| `tracking_distance` | declared unavailable | hoof ground-contact detection |

**Alternative considered:** omitting the six rather than declaring them. Rejected
because a reader of the profile would then have no way to distinguish a feature
that was decided against from one that is merely not yet possible, and the
decision record would live only in a document.

### The mid-dorsal point, and why `back_posture` slips

`back_posture` measures sagittal arch as the deviation of the mid-back from the
line joining the cranial and caudal ends of the dorsal line. That needs three
points along the back. AP-10K supplies neck and root of tail — and shoulders,
from which a withers position is derivable — but has **no mid-thoracic or
mid-dorsal keypoint**. Two of three.

**Decision:** declare `back_posture` unavailable rather than approximate it from
shoulder and hip. An approximation would produce a number under a name the
literature attaches to a specific measurement, which is the failure this change
exists to prevent.

**Consequence worth stating:** a single labelled mid-dorsal keypoint is the
cheapest route to one of Russello's six traits. That is the first thing to test
on the pilot's 200 labelled frames, and it is a much smaller ask than a full
pose training set.

### Skeleton: provisional nine-point lateral

Four hooves, a three-point dorsal line (withers, mid-thoracic, sacrum), a head
point, and nothing else. Nothing in the decided feature set depends on a
fetlock, carpal, ear tip or tail end.

Recorded as **provisional**. The review states T-LEAP/CoWalk uses 17 landmarks;
Russello et al. 2024 states T-LEAP extracted motion from nine. Reading Russello
et al. 2022 settles it, and the skeleton is revised then if nine is wrong.

**Alternative considered:** waiting for that reading before writing any
skeleton. Rejected because the top-down skeleton blocks everything else in the
change, and because the point set is derived from the feature set rather than
from either paper — the papers only settle whether nine is enough, not which
nine this project needs.

### Pose backend: AP-10K, not SuperAnimal

**Decision:** AP-10K, because it is the option this project verified directly —
CC-BY-4.0 dataset, 17 keypoints enumerated, four checkpoints published — while
SuperAnimal-Quadruped's licence and point vocabulary could not be read from its
publisher, which returned an authentication redirect.

SuperAnimal remains the better candidate on paper: 39 points, purpose-built for
cross-species zero-shot transfer, and plausibly carrying the mid-dorsal point
AP-10K lacks. It becomes a benchmark comparison once its licence is confirmed,
which is an outstanding action rather than a rejection.

**Open risk:** AP-10K's CC-BY-4.0 is stated for the *dataset*. The checkpoints
are hosted by OpenMMLab and may carry a different licence. This must be
established before the weights are fetched, not after.

### Representing unavailability

A new explicit state on the feature record, distinct from the existing reduced
quality flag.

**Alternative considered:** reusing the reduced-quality flag with a special
reason. Rejected because reduced quality counts toward the pass validity
decision and unavailability must not — a feature the *configuration* cannot
supply is not evidence that a *pass* was bad, and conflating them would reject
every pass for a property of the camera. The existing config comment on
`min_usable_features` already reasons this way; this makes it a declared state
rather than a counting convention.

### Where a source declares its view

On the dataset registration, and on `register_source`, alongside the site and
camera identity it already carries. The view is a property of the recording, so
it belongs with the recording's provenance rather than in configuration.

Existing registrations gain a declared view: CattleEyeView is top-down and
documented as such. MultiCamCows2024's view must be established from its release
before it can run under any profile, which the spec's refuse-on-undeclared
scenario forces rather than leaves to chance.

### Placeholder declaration

The profile's weight entries gain an explicit boolean rather than the prose note
they carry today. Prose in a `notes:` field is not reachable by a report. Model
identity records already travel with every output, so the flag rides along the
path that already exists.

### Priors as anchors, not bands

Four features gain a measured healthy anchor and a direction, sourced to Flower
et al. 2005, recorded under a provenance distinct from `unvalidated-p0-prior`.

**Decision:** anchors and directions, not low/high bands. Flower's figures are
group means with standard errors — a mean ± 0.03 s describes where a group sat,
not how far a population spreads, and entering it as a band would produce a
plausibility check that rejects most normal animals.

Note that two of the four anchors (`speed`, `stride_length`) are in metres and
this project has no camera calibration, so they inform a direction but cannot
become a threshold.

## Risks / Trade-offs

- **The phenotype layer computes three features after this change, down from
  twelve.** → This is the honest state, not a regression: of the twelve, the
  limb features were unmeasurable on top-down footage anyway and three of the
  body features were measuring the wrong axis. Three well-founded features are
  worth more than twelve that were not.

- **AP-10K checkpoint licence may not be CC-BY-4.0.** → Establish it before
  fetching. If it is unusable, SuperAnimal or a licence-clean alternative
  replaces it; this is a change to `weights.pose` alone, which is what the seam
  is for.

- **AP-10K is trained on many species from natural imagery, not on cattle in a
  lane.** → Zero-shot accuracy on this footage is unknown and may be poor,
  particularly on hooves. The change wires it and reports its identity; it does
  not claim its accuracy. The pilot's labelled frames measure it.

- **The provisional skeleton may be wrong in its point count.** → It is recorded
  as provisional and derived from the feature set rather than from a paper, so a
  revision changes the count, not the design.

- **`back_posture` slipping costs one of the six traits Russello validated
  together.** → Their reported 79.9% on the three most important traits, against
  80.1% on all six, suggests the set degrades gracefully. But this change makes
  no performance claim of any kind, and should not be read as making one.

- **Refusing on an undeclared source view will break existing runs.** → That is
  the intent. Both current registrations gain a declared view as part of this
  change, so the breakage is confined to sources registered outside them.

### Retaining the top-down profile

**Decision, taken during implementation.** Keep the version-0 profile as a
frozen sibling, `cattle-topdown`, rather than deleting it.

The view check refuses to pair a lateral skeleton with CattleEyeView, which is
correct and is the point of the guard. But CattleEyeView is the only public
source this project can run end to end, so refusing it also retires the five
tests that are P0's evidence — including, with some irony, the one proving that
a top-down camera cannot see a cow's legs, which is the finding the lateral
profile exists because of.

**Alternatives considered.** Rewriting those tests to assert the refusal was
smaller, and would have cost the end-to-end-on-real-data claim in `README.md`
and `docs/P0-OUTCOME.md` until pilot footage exists. Dropping the view check was
rejected outright: it is the mechanism that stops the quiet failure this change
exists to prevent.

**Cost, which was under-priced when the option was chosen.** One extractor now
has to serve two skeletons whose keypoints do not share names. Resolved by
making extraction name no keypoint at all: the body axis arrives through a
declared skeleton role, every other point through the feature's own
`depends_on`, and which implementations run is decided by which features the
profile declares available. A feature declared available with no implementation
is refused by name rather than silently skipped, because silence there would be
indistinguishable from a feature legitimately declared unavailable.

That is a better arrangement than the one it replaces, and it is what the
species-seam claim always implied — a stage should not know a keypoint's name.

## Migration Plan

1. Land the profile, spec and code changes together — the feature set and
   `features.py` cannot move independently, because extraction refuses any
   computed feature not in the declared set.
2. Existing runs under feature-set version 0 are not migrated. They remain
   readable and remain distinguishable by the version field the existing spec
   already requires on every record.
3. Rollback is reverting the change: nothing here writes a persistent store
   whose format changes, and version 0 records are untouched.

## Open Questions

- ~~Is AP-10K's published checkpoint licence the same as its dataset's
  CC-BY-4.0?~~ **Resolved.** The AP-10K repository's `LICENSE` is the CC-BY-4.0
  text; mmpose, which trains and hosts the checkpoints, is Apache-2.0. No
  checkpoint-specific statement exists, so the conservative reading is that
  CC-BY-4.0 attribution flows through from the training data over Apache-2.0
  framework code. Both permit commercial use with attribution, which is less
  restrictive than the incumbent AGPL-3.0 weights. Usable.
- Does SuperAnimal-Quadruped's 39-point vocabulary include a mid-dorsal point,
  and what licence do its weights carry?
- Nine keypoints or seventeen — settled by Russello et al. 2022.
- What zero-shot hoof accuracy does AP-10K reach on lateral cattle footage?
  Unanswerable until the pilot recording exists, and the reason `back_posture`
  is not the only thing waiting on it.

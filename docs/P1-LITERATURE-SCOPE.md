# P1 literature scope

What the literature must settle before a model is chosen.

P0 built the spine and ran it on public data. What it measured became
[docs/P1-RECORDING-SPECIFICATION.md](P1-RECORDING-SPECIFICATION.md), which states
what a cooperating dairy must install. Neither document says what the system
should *measure*, or *how*. That gap is not an oversight: P0 held itself to "no
requirement without a measurement", and the four items under its own *What P0
could not determine* are precisely the ones no further running of the spine can
answer. They are answerable from published work. This document is the list of
what must be asked, and what each answer decides.

The governing rule, matching its sibling: **a question with no decision attached
is reading, not review.** Every entry names the decision it unblocks and the
thing the project currently assumes in its absence, with the file that records
the assumption. An entry that cannot name what it would change does not belong
here.

---

## What P0 settled, and this review does not reopen

- **A top-down mount cannot see limbs.** Measured over 24,054 labelled
  instances: paws visible in 4.5–9.1%, knees in 4.1–10.0%. The *direction* of R1
  is not in question. Only its specifics are, which is L1.
- **A periodic quantity sampled below twice its band is not measured.** That is
  an argument, not an empirical claim, and needs no literature support. What
  does need support is the band itself, which is L5.
- **An external identity anchor must be the primary path.** Visual
  re-identification gave 17.8% top-1 against a 1.1% chance level, and its
  confidence was uninformative — AUROC 0.527. A stronger embedding would raise
  the accuracy; the architectural conclusion holds either way, so R5 stands
  whatever the re-identification literature reports.
- **A lane admitting one animal at a time removes an identity problem rather
  than mitigating it.** 70.7% of CattleEyeView's annotated frames hold more than
  one animal, and 507 of 761 tracklets were unresolvable in consequence. R3
  needs no further support.

---

## L1 — What viewing geometry do working systems use?

**Question.** Which camera positions do published cattle lameness systems adopt
— lateral, oblique, rear, overhead, or a combination — and what does each make
measurable, at what cost?

**What it decides.** R1's specifics (angle and height), R4's framing, and
whether more than one camera is warranted. The recording specification states
that the angle "has to be found empirically at the site"; every angle the
literature has already rejected is one the pilot does not have to spend an
iteration on.

**Currently assumed.** Oblique, "looking along and across the lane", from R1 —
a direction inferred from a top-down failure rather than from a positive
finding. Rear view has not been considered at all, although weight-bearing
asymmetry is assessed from behind by human scorers.

**What counts as an answer.** A table of published systems against view, target
feature and reported performance, including at least one entry that measured
limb kinematics on a commercial lane rather than on a walkway built for the
study. That distinction is the one this project cares about.

---

## L2 — Are keypoints the right intermediate representation?

**Question.** Do published lameness systems estimate pose, or do they work from
silhouettes, back-curvature contours, optical flow, leg-region tracking, or
end-to-end video models — and how do the pose-based ones compare with the rest?

**What it decides.** The largest fork in the project.
`detect -> track -> pose -> keypoint tracks -> features` is assumed throughout
the spine. If a keypoint-free representation performs comparably, the cattle
pose-labelling workstream — for which no public source exists at the required
view — disappears, and L4 disappears with it.

**Currently assumed.** Keypoints, throughout `src/lhv/phenotype/features.py`,
which computes every feature from named keypoint tracks. The assumption came
from CattleEyeView being the only public cattle source carrying keypoints, which
is a fact about data availability, not about method.

**What counts as an answer.** For each representation family: what it measures,
what supervision it needs, and what it reported — with the labelling cost stated
explicitly, because that is what the project is actually choosing between.

---

## L3 — Which gait and posture quantities carry lameness signal?

**Question.** Which measurable quantities have been shown to separate lame from
sound animals, at what effect size, and which are merely plausible?

**What it decides.** `feature_set` version 1 — its members, their units, and the
direction of each `higher_is_worse` flag — and through it, what P1's gate is
actually testing.

**Currently assumed.** Twelve features in
`src/lhv/profiles/definitions/cattle.yaml`, chosen for computability from a
top-down skeleton rather than for clinical relevance, with priors every one of
which is marked `source: unvalidated-p0-prior`. Three of them —
`lateral_sway`, `head_lateral_offset`, `spine_lateral_curvature` — are
image-plane quantities whose physical meaning changes with the mount, and the
profile's own notes list back arch, head height and sagittal limb flight as
deliberately absent.

**What counts as an answer.** A ranked list of candidate features carrying, for
each, the view it requires, the reported discrimination, and whether that was
measured against a scored reference or asserted. Features the literature
reports and this project cannot see are as important to record as the ones it
can.

---

## L4 — Conditional on L2: which keypoint convention, and what is trained on it?

**Question.** If keypoints are the chosen representation, is there a standard
cattle keypoint convention for a lateral or oblique view, and are there
downloadable weights for it? Failing that, what do the general quadruped
conventions cover, and at what cost in fidelity?

**What it decides.** The skeleton fork — identifier, `view`, keypoint set, flip
pairs, OKS sigmas — and whether P1 contains a pose-labelling and training
workstream at all.

**Currently assumed.** `cattleeyeview-topdown-24` with `view: top-down`, and a
pose backend (`yolo11m-pose`) that is COCO-human weights mapped onto the cattle
skeleton by analogy — `left_wrist -> left_front_paw` — declared in the profile
as a P0 placeholder. On top-down footage that placeholder failed visibly,
because the detector ahead of it found nothing. On an oblique view the detector
works, the placeholder will emit confident paw keypoints, and nothing in the
reports currently marks them as placeholder output.

**What counts as an answer.** The available conventions with keypoint counts and
their coverage of the distal limb, the public weights for each with licences,
and an estimate of the labelling volume required if none fits.

---

## L5 — What is the stride frequency band for walking dairy cattle?

**Question.** What stride and step frequencies have been measured for dairy
cattle walking a lane, sound and lame, and what temporal resolution do published
gait-event methods require?

**What it decides.** R2's frame-rate floor, and the `requires_sampling_hz`
declaration each periodic feature carries.

**Currently assumed.** 0.40–2.50 Hz, recorded in `cattle.yaml` as
`source: unvalidated-p0-prior`, and a 15 fps floor with 25 fps preferred in
`Requirements` in `src/lhv/recording.py`. The whole Nyquist argument in R2 rests
on the upper end of a guessed band.

**What counts as an answer.** A measured band with its source and, separately,
the timing resolution that stance-time or foot-strike asymmetry requires. Those
are different constraints — a frequency needs samples per cycle, an event needs
temporal precision — and the stricter of the two sets the requirement.

---

## L6 — What ceiling does the reference standard impose?

**Question.** What inter- and intra-observer agreement has been reported for the
Sprecher 1–5 locomotion scale, and for the alternatives to it?

**What it decides.** P1's pass mark. The gate is *gait features correlate with
human locomotion score*, and the agreement between two human scorers bounds the
correlation any system can demonstrate against one of them. Without that bound
the gate has no threshold and cannot be failed.

**Currently assumed.** Sprecher 1–5 is recorded in `cattle.yaml` as the
convention, with no agreement figure attached, and R8 asks for "at least weekly"
scoring with no derivation of the cadence.

**What counts as an answer.** Reported agreement statistics for the scale, a
target correlation defensibly derived from them, and the reported agreement of
at least one alternative scale — so that choosing Sprecher is a decision rather
than an inheritance.

---

## L7 — What is already recorded and labelled?

**Question.** Is there a public cattle dataset at a lateral or oblique view,
carrying keypoints, lameness scores, or both? Under what licence, by what access
route?

**What it decides.** Whether P1 has any validation available before the farm's
own footage exists. It also feeds L2 and L4 directly, and decides whether
P0-OUTCOME's statement that no public source carries clinical outcomes still
holds.

**Currently assumed.** That no such source exists. That is a true statement
about the two datasets P0 examined, and it was never a survey.

**What counts as an answer.** The datasets, with view, size, labels, licence and
access route, in the same form as the existing entries in
`src/lhv/datasets/registrations/`. A negative result is a real result here and
must be recorded as one rather than left implicit.

---

## L8 — How are these studies actually designed?

**Question.** How many animals, over how long, at what scoring cadence, at what
observed prevalence — and how did each study handle animals that were treated
during it?

**What it decides.** The sample-size and protocol requirements the recording
specification does not contain: herd size, expected case count, R7's duration,
R8's cadence. And the treatment protocol, which is presently the largest
unwritten item in P1.

**Currently assumed.** Three weeks (R7), derived from the baseline
configuration's own 21-day lookback rather than from any study; weekly scoring
(R8), stated without derivation; and no position at all on prevalence, herd
size, or what happens clinically when the scorer finds a lame animal.

**What counts as an answer.** A table of comparable studies against n, duration,
cadence, prevalence and reference standard, with an explicit statement of how
each handled treatment — because treating a case on detection truncates the
natural history that a lead-time endpoint depends on.

---

## L9 — What illumination and exposure do these systems require?

**Question.** What lighting do published barn and lane systems use, at what
exposure times, and is infrared reported as acceptable for gait?

**What it decides.** A requirement the recording specification does not
currently have. Motion blur is the mechanism by which poor lighting destroys
limb measurement — a long exposure smears the limb during swing phase, when it
moves fastest — and unlike illuminance it is specifiable and testable directly.

**Currently assumed.** Nothing. The recording specification files lighting under
*what P0 could not determine* and states no exposure requirement anywhere.

**What counts as an answer.** A maximum exposure time defensible from published
practice, and a statement on whether infrared or supplementary lighting is used,
tolerated, or avoided.

---

## What a literature review cannot settle

Stated so nobody mistakes a finding for a result.

- **Whether the chosen geometry fits this farm's lane.** Site-specific. Only the
  pilot recording and the recording specification's acceptance procedure settle
  it.
- **Whether the chosen features correlate here.** That is P1's gate. The
  literature supplies candidates and expected effect sizes; it cannot supply a
  result on this herd, this camera and this scorer.
- **Whether a published method reproduces.** Reported performance is on the
  authors' data, frequently a purpose-built walkway. Nothing found in this
  review may be carried forward as an expectation for a commercial lane.
- **The treatment protocol.** L8 establishes what others did. What this study
  does is an agreement with the farm, and plausibly an ethics submission.

---

## Where to look

The relevant work is split across two literatures that cite each other sparsely,
and both must be searched. Animal and production side: Journal of Dairy Science,
Computers and Electronics in Agriculture, Biosystems Engineering, Preventive
Veterinary Medicine, Animals. Vision side: the agriculture and animal workshops
at the major vision conferences (CV4Animals, Agriculture-Vision), and Sensors.
A method described as a lameness detection system in one literature and as a
quadruped pose benchmark in the other may be the same method under two names.

---

## What closes this review

Every entry above carries either an answer with its source, or an explicit *no
usable evidence found*. Then the decision each entry gates is taken and written
into the artefact that holds it:

| entry | artefact the answer updates |
|---|---|
| L1 | R1 and R4 in the recording specification |
| L2 | the model decision record; the pipeline stage contract, if it changes |
| L3 | `cattle.yaml` — `feature_set` version 1, and its priors |
| L4 | `cattle.yaml` — `skeleton` and `weights.pose` |
| L5 | `cattle.yaml` priors and `requires_sampling_hz`; `Requirements` in `src/lhv/recording.py`; R2 |
| L6 | the P1 gate's threshold, in the README phase table |
| L7 | `src/lhv/datasets/registrations/` |
| L8 | R7, R8, and a new requirement covering herd size and treatment |
| L9 | a new requirement in the recording specification |

The output of this review is a set of decisions, each traceable to a source. A
bibliography is not an output.

Nothing below the model decision — no profile fork, no weights, no code — begins
before this document is closed.

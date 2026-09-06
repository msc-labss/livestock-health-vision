# P1 literature prompt

The question set from [docs/P1-LITERATURE-SCOPE.md](P1-LITERATURE-SCOPE.md), in
the exact form it is given to a research tool.

It is kept in the repository for one reason: the scope document states what must
be asked, and this states how it was asked. When the model decision record cites
a thin answer, a reader can then tell whether the evidence is thin or the
question was badly put. Those need different remedies, and without the prompt
they are indistinguishable.

Amend this file when a question is reworded, and note it, rather than rewording
it silently in a research session.

**Running it.** One pass first. If the answers come back shallow, split into
three runs sharing the context and rules blocks — Q1–Q3 (what to see and
measure), Q4/Q5/Q9 (engineering parameters), Q6–Q8 (study design and
resources). Q2 and Q3 are worth a dedicated run regardless, being the two that
fork the project.

**Before an answer becomes a decision, verify its citations.** The prompt guards
against fabrication and the guard is not sufficient. Every artefact in this
project carries the provenance of its claims; an invented DOI in the model
decision record would be worse than an open question left open.

---

```
You are conducting a literature review that must produce decisions, not a
bibliography. Every question below is blocking a concrete engineering choice.

## Project context

I am building a computer-vision system to detect lameness in dairy cattle at a
constrained parlour/AMS exit lane. Single RGB camera; no depth, thermal or
audio. The pipeline is: detect -> track -> identity anchor (external, from the
AMS/parlour record) -> gait phenotype extraction -> per-animal daily time
series -> baseline from the animal's own history -> risk score -> alert with an
evidence clip.

The architecture is built and runs end to end on public datasets. What is NOT
decided is the perception method: viewing geometry, intermediate
representation, and feature set. That is what this review must settle.

## Already established — do not re-derive or argue these

1. A top-down/overhead mount cannot see limbs. Measured over 24,054 labelled
   instances of a public top-down cattle dataset: paw keypoints labelled
   visible in 4.5-9.1% of instances, knees in 4.1-10.0%, while body-axis
   points (withers, neck, tail base) are visible 72-93%. The camera will
   therefore not be overhead. Only the replacement geometry is open.
2. Identity comes from an external anchor (AMS/parlour/RFID), not from vision.
   Visual re-identification on a public 90-animal, 7-day cattle dataset gave
   17.8% top-1 against a 1.1% chance level, with uninformative confidence
   (AUROC 0.527 for correct-vs-incorrect). Do not propose visual re-ID as the
   identity mechanism.
3. The lane physically admits one animal at a time, so inter-animal occlusion
   is not a constraint on the answer.
4. A periodic feature sampled below twice its own frequency band is not
   measured. This is an argument, not a claim needing support. What needs
   support is the band itself (see Q5).

## Questions

For each, I give what the answer decides and what would count as an answer.

Q1 — VIEWING GEOMETRY
Which camera positions do published cattle lameness systems use — lateral,
oblique, rear, overhead, or combinations — and what does each make measurable
at what cost?
Decides: the mounting specification given to the farm, and whether more than
one camera is warranted. Rear view is of specific interest, since human
scorers assess weight-bearing asymmetry from behind, and I have not
considered it.
Counts as an answer: a table of systems against view, target feature and
reported performance, INCLUDING at least one entry that measured limb
kinematics on a working commercial lane rather than a walkway built for the
study. Label every study as commercial-lane or purpose-built.

Q2 — INTERMEDIATE REPRESENTATION
Do published lameness systems estimate skeletal pose, or do they work from
silhouettes, back-curvature contours, optical flow, leg-region tracking, or
end-to-end video models? How do pose-based approaches compare with the rest?
Decides: the largest fork in the project. My pipeline currently assumes
keypoints throughout. If a keypoint-free representation performs comparably, a
cattle pose-labelling and training workstream disappears entirely.
Counts as an answer: for each representation family — what it measures, what
supervision it requires, what it reported, and the LABELLING COST it implies.
That cost is what I am actually choosing between.

Q3 — WHICH QUANTITIES CARRY LAMENESS SIGNAL
Which measurable gait and posture quantities have been shown to separate lame
from sound cattle, at what effect size? Which are merely plausible?
Decides: my feature set. It was chosen for computability from a top-down
skeleton, not for clinical relevance, and currently contains: speed, speed
variability, stride frequency (front/back), stride length (front/back), step
asymmetry (front/back), lateral sway, head lateral offset, spine lateral
curvature, tracking jitter.
Counts as an answer: a ranked list of candidate features with, for each, the
view it requires, the reported discrimination (AUC, sensitivity/specificity,
or correlation, with n), and whether it was measured against a scored
reference or asserted. Features I cannot see from a single lane camera are as
important to list as ones I can.

Q4 — KEYPOINT CONVENTION AND AVAILABLE WEIGHTS (only if Q2 favours keypoints)
Is there a standard cattle keypoint convention for a lateral or oblique view,
with downloadable weights? Failing that, what do general quadruped
conventions (e.g. AP-10K, APT-36K, AnimalPose and successors) cover, and how
well do they cover the distal limb — the part lameness lives in?
Decides: my skeleton definition, and whether P1 contains a pose-labelling and
training workstream at all.
Counts as an answer: conventions with keypoint counts and distal-limb
coverage; public checkpoints with licences; and an estimate of labelling
volume required if nothing fits.

Q5 — STRIDE FREQUENCY AND GAIT TIMING
What stride and step frequencies have been measured for dairy cattle walking
a lane, sound and lame? What temporal resolution do published gait-event
detection methods require?
Decides: my minimum frame rate. I currently assume a 0.40-2.50 Hz stride band
(an unvalidated guess) and a 15 fps floor with 25 fps preferred, derived
entirely from the upper end of that guess.
Counts as an answer: a measured band with its source AND, separately, the
timing precision that stance-time or foot-strike asymmetry measurement
requires. These are different constraints — one needs samples per cycle, the
other needs event precision — and the stricter sets the requirement.

Q6 — THE CEILING IMPOSED BY THE REFERENCE STANDARD
What inter- and intra-observer agreement is reported for the Sprecher et al.
(1997) 1-5 locomotion scale, and for its alternatives?
Decides: my project's pass mark. My success criterion is "gait features
correlate with human locomotion score", and agreement between two human
scorers bounds the correlation any system can demonstrate against one of
them. Without that bound the criterion has no threshold and cannot be failed.
Counts as an answer: agreement statistics (kappa, weighted kappa, ICC) for
Sprecher and for at least one alternative scale, plus a defensible target
correlation derived from them.

Q7 — EXISTING DATASETS
Is there a public cattle dataset at a lateral or oblique view carrying
keypoints, lameness scores, or both? Licence and access route for each?
Decides: whether I have any validation available before the farm's own
footage exists.
Counts as an answer: datasets with view, size, label type, licence and access
route. A well-supported negative — "no such public dataset exists" — is a
genuine and valuable result here. State it explicitly if that is the finding.

Q8 — STUDY DESIGN
In studies validating automated lameness detection: how many animals, over
how long, at what scoring cadence, at what observed prevalence? Critically,
how did each handle animals that were TREATED during the study?
Decides: sample size, study duration, scoring cadence, and the treatment
protocol I must agree with the farm. Treating a case on detection truncates
the natural history that a lead-time endpoint depends on, and I have no
position on this at all.
Counts as an answer: a table against n, duration, cadence, prevalence and
reference standard, with each study's treatment handling stated explicitly.

Q9 — ILLUMINATION AND EXPOSURE
What lighting do published barn and lane systems use, at what exposure times?
Is infrared reported as acceptable for gait measurement?
Decides: a camera requirement I currently do not have. Motion blur is the
mechanism by which poor lighting destroys limb measurement — a long exposure
smears the limb during swing phase, when it moves fastest — and unlike
illuminance it is directly specifiable and testable.
Counts as an answer: a maximum exposure time defensible from published
practice, and whether IR or supplementary lighting is used, tolerated, or
avoided.

## Output format

For EACH question Q1-Q9, in order:

**Answer** — 3-6 sentences, direct, no preamble.
**Evidence** — table: source (authors, year, venue, DOI or stable URL) |
  what it did | n | setting (commercial / purpose-built / lab) | what it
  reported.
**Confidence** — strong / moderate / weak / no usable evidence found.
**Decision implied** — the concrete change to the assumption I stated above.
**Still open** — what this could not settle.

Then a closing section: **Contradictions and disagreements** — where sources
disagree, stated as disagreement rather than averaged away.

## Rules

- Real, checkable citations only. Give DOIs or stable URLs. Do not invent a
  source, an author list, or a number. If you are unsure a citation is real,
  say so.
- Report effect sizes and n. "Feature X correlates with lameness" without a
  number is not an answer.
- "No usable evidence found" is an acceptable answer and I want it when it is
  true. Do not fill a gap with plausible reasoning presented as a finding.
- Always distinguish evidence from purpose-built walkways or treadmills from
  evidence gathered on working commercial lanes. My deployment is the latter.
- Note when a claim rests on a preprint, a thesis, or a vendor whitepaper
  rather than peer-reviewed work.
- Search BOTH literatures, which cite each other sparsely: animal/production
  science (Journal of Dairy Science, Computers and Electronics in
  Agriculture, Biosystems Engineering, Preventive Veterinary Medicine,
  Animals) and computer vision (CV4Animals and Agriculture-Vision workshops
  at CVPR/ICCV/ECCV, Sensors). The same method may appear as a "lameness
  detection system" in one and a "quadruped pose benchmark" in the other.
- Useful term variants: lameness / locomotion score / gait scoring / hoof
  health; automatic / automated / computer vision / machine vision / sensor-
  based; back arch / back posture / spine curvature; head bob / head bobbing;
  tracking-up / overlap; dairy cow / dairy cattle / Holstein.
```

---

## What the answers feed

The mapping is in the scope document's closing table: each question updates a
named artefact, and the review is closed only when every one of them carries
either an answer with its source or an explicit *no usable evidence found*.
The answers themselves belong in `docs/P1-MODEL-DECISION.md`, which does not
exist yet and should not be started before the review runs.

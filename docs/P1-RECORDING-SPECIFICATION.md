# P1 recording specification

What a cooperating dairy must install and record for P1 to be possible.

P1's gate is *gait features correlate with human locomotion score*. Reaching it
needs footage the spine can actually measure, and P0 established — by running on
two public datasets — that several properties which sound like details decide
whether any measurement is possible at all. Three of them contradict what P0's
own design assumed.

Every requirement below carries the P0 measurement it comes from and a test that
can be run against a pilot recording before the farm commits to anything. A
requirement with no test is a hope, and this document has none of those.

---

## R1 — The camera must not look straight down

**Requirement.** Mount the camera obliquely, looking along and across the lane,
so that the legs are visible against the background rather than beneath the
animal. A purely overhead mount is not acceptable.

**Why.** P0 assumed top-down was the right geometry: it removes occlusion
between animals and makes heading unambiguous. It does — by occluding the legs
with the body. Measured over all 24,054 labelled CattleEyeView instances:

| keypoint group | labelled visible |
|---|---|
| neck, ear bases, head | 89–93% |
| withers, front elbows | 83–88% |
| rear elbows, tail base | 72–74% |
| **knees** | **4.1–10.0%** |
| **paws** | **4.5–9.1%** |

Every limb feature — stride length, stride frequency, step asymmetry — depends on
a paw. On a top-down source all of them were unusable in over 90% of passes,
while body-axis features (speed, lateral sway, spine curvature) were good in
roughly 70%. Lameness is a limb phenomenon, so a camera that cannot see limbs
cannot serve P1's gate.

**Test.** On a pilot recording, label 200 frames with the profile skeleton and
require **paw keypoints visible in at least 60% of instances**. Below that,
change the mounting before recording anything else. A useful proxy needing no
labelling: a COCO-pretrained detector finds *zero* cattle in the top-down
footage at any confidence down to 0.05, because overhead cattle are outside its
training distribution. On a correctly angled view it should detect them readily,
so `lhv recording-check` doubles as a mounting check.

---

## R2 — At least 15 frames per second, 25 preferred

**Requirement.** Record at **≥ 15 fps**, constant, with the rate recorded in the
container. 25 fps is preferred. Below 15 fps is not acceptable for gait.

**Why.** Stride frequency is periodic, and a periodic quantity sampled below
twice its own band is not measured badly — it is not measured at all, and the
number that comes back describes the sampling. Against the cattle profile's own
prior band of 0.4–2.5 Hz:

| rate | samples per stride at 2.5 Hz | verdict |
|---|---|---|
| 3 fps | 1.2 | below Nyquist — unusable |
| 5 fps | 2.0 | at the limit — unusable in practice |
| 8 fps | 3.2 | marginal |
| 10 fps | 4.0 | marginal |
| **15 fps** | **6.0** | adequate |
| 25 fps | 10.0 | comfortable |

CattleEyeView varies between 3, 5 and 8 fps across its 14 sequences and the
paper does not mention it. Five of the fourteen are below Nyquist for the fast
end of the band. The pipeline refuses to report a stride frequency from them,
which is correct and also means a third of that dataset can never contribute a
limb measurement.

**Test.** Read the container's frame rate for every file; require ≥ 15 and
constant. `lhv recording-check` reports it.

---

## R3 — One animal in the measurement zone at a time

**Requirement.** The lane must physically admit one animal at a time through the
measured section — a race, a sorting gate, or the AMS exit — not an open ramp or
holding area.

**Why.** This is the load-bearing assumption behind the whole identity design,
and P0 showed what happens without it. In CattleEyeView's loading ramp, **70.7%
of annotated frames contain more than one animal**, up to nine at once:

| animals in frame | share of annotated frames |
|---|---|
| 1 | 29.3% |
| 2 | 42.3% |
| 3 | 20.1% |
| 4 or more | 8.2% |

An external identifier — a parlour reading, an AMS record, an RFID gate — tells
you *who* passed and *when*. When several animals share a window, that is not
enough to say which tracklet is which, and the identity layer correctly refuses
rather than guessing. On the ramp, 507 of 761 tracklets were left unresolved as
ambiguous. Adding the reading's *location* recovered a large part of it, but a
lane that admits one animal at a time removes the problem instead of mitigating
it, and is what the README's design has assumed from the start.

**Test.** On a pilot recording, require **≥ 95% of frames containing an animal to
contain exactly one** in the measured section.

---

## R4 — The animal must fill enough of the frame, for long enough

**Requirement.** At the measurement point the animal's body should span roughly
a third of the frame's long side, and the field of view should cover **at least
three body lengths along the direction of travel**, so a full pass is captured
with the animal wholly in view at entry and exit.

**Why.** In CattleEyeView the median labelled animal is 717 × 304 px in a
1920 × 1080 frame — the long side is 37% of frame width, so about 2.7 body
lengths fit across the view. That was just enough to segment a pass between
boundaries at 0.3 and 0.7 of frame width; tighter framing would have left no
room for the entry and exit crossings a complete pass requires.

Transit across the field of view took a median of **4.3 s** (p10 2.6, p90 9.7),
containing 1.7–10.8 stride cycles at the prior band. At 15 fps a median transit
yields about 65 frames, which is ample; at 3 fps it yielded 13.

**Test.** Detect animals on a pilot recording and require the median box's long
side to fall between **25% and 50%** of the frame's long side, and the median
transit to last **≥ 3 s**.

---

## R5 — A per-individual identity feed, persistent across days

**Requirement.** The farm must supply an identifier stream keyed to the
**individual animal** — AMS, parlour, or RFID — with a timestamp and, ideally,
the reader's position. The same animal must carry the same identifier on every
day of the study.

**Why.** This is the single thing that stops P0 producing a real risk score, and
neither public dataset has it. CattleEyeView labels 753 *instances* — one
animal's appearance in one sequence — so an animal never appears twice and can
never accumulate history. Every assessment in the full run correctly reported
*insufficient history*, and no longitudinal phenotype was possible.

Visual re-identification is not a substitute. Measured on MultiCamCows2024,
which does track 90 individuals over 7 days, enrolling on five days and
identifying 432 held-out tracklets from two unseen days gave **17.8% top-1
accuracy** against a 1.1% chance level. Worse, the embedding's confidence was
uninformative: correct and incorrect matches averaged 0.9965 and 0.9964
similarity, an AUROC of 0.527, so no confidence floor could filter its errors.
A stronger embedding would improve the accuracy; the architectural point stands
either way, which is why the design makes the external anchor the primary path.

**Test.** Join a day of the identifier feed against a day of footage and require
**≥ 95% of passes** to match exactly one identifier within a 5-second window.
Confirm that identifiers seen on day 1 reappear on day 20.

---

## R6 — Capture time must survive to the file

**Requirement.** Every recorded file must carry a capture timestamp readable
without pixel inspection — in the container metadata, or failing that in the
filename — synchronised to the identity feed's clock, with the timezone offset
recorded.

**Why.** A per-animal time series is keyed by observation time, so a pass with
no time cannot enter one. Both public datasets lost this, in different ways.
MultiCamCows2024's tracklet stills carry no time anywhere: none of the 1,584
passes a full run produced had an observation time, which alone prevents any
series. CattleEyeView carries the time only as pixels burned into the frame
corner; P0 recovered all 14 by reading them, which is not a procedure anyone
should have to repeat.

**Test.** Read the capture time from every pilot file programmatically. Require
100%, and require the offset between the video clock and the identity feed's
clock to be **under 1 second**.

---

## R7 — Record for long enough before expecting a score

**Requirement.** Plan for **at least three weeks** of continuous silent
recording before any risk score is meaningful, and longer to observe cases
developing.

**Why.** The baseline is deviation from an animal's own history. With the
cattle profile's declared minimum of 5 prior observations and animals passing
twice daily, the first score arrives on about day 3; the declared 21-day
lookback is only full after three weeks. A herd baseline additionally needs at
least 3 animals within a 3-day window. Any lameness case that begins before the
baseline is established is invisible to it, because there is nothing to deviate
from.

**Test.** Arithmetic, not measurement: passes per animal per day × days ≥ the
configured minimum, with margin for missed passes.

---

## R8 — Parallel human locomotion scoring

**Requirement.** A trained scorer must score locomotion on the same animals,
on a stated scale, on stated days, recorded per individual with the same
identifiers as R5.

**Why.** P1's gate is correlation with human score, and P0 has no clinical
label of any kind — every event it produces is marked `stub_derived` and
`non_clinical` for exactly this reason. Lead time and alarm burden also become
measurable only once a reference event exists; P0 reports lead time as
unmeasured rather than as zero.

The cattle profile records the Sprecher 1–5 scale as the convention. Scoring
cadence should be at least weekly, and every scoring event needs a date, because
lead time is measured against it.

**Test.** Not a recording test. Confirm before the study starts that the scorer
is trained, the scale is fixed, and the schedule is agreed.

---

## R9 — People will be in shot

**Requirement.** Assume the footage contains identifiable people, and enable
masking before any clip is retained.

**Why.** Not hypothetical: the first CattleEyeView frame P0 examined contains a
stockperson, and the COCO detector found them when it could not find a single
cow. Barn and lane cameras capture workers and visitors, who are identifiable
natural persons under GDPR even though the animals are not.

The spine already refuses to retain a clip it cannot mask, and records the
retention failure on the event. What the farm must supply is the agreement:
camera placement, signage, retention period, and who may view a retained clip.

**Test.** Run the pilot recording through clip retention with masking enabled
and confirm no unmasked frame is written. The existing test suite covers the
code path; the farm side is a paperwork check.

---

## What P0 could not determine

Stated so nobody mistakes silence for a settled answer.

- **The correct mounting angle and height.** P0 can say that straight down fails
  and why, and can give a visibility threshold to test against, but the angle
  that best exposes limbs without introducing occlusion between animals has to
  be found empirically at the site.
- **Lighting.** Both public sources are daylight or barn-lit and neither
  documents illumination. Whether the lane needs supplementary lighting, and
  whether infrared is acceptable, is unmeasured.
- **What resolution is genuinely needed.** The 25–50% framing requirement is
  derived from what worked at 1920 × 1080; it is not evidence that a lower
  resolution fails.
- **Which features actually correlate with lameness.** That is P1's entire
  point. P0 measured reproducibility, never validity, and no number it produced
  stands in for a clinical result.

---

## Acceptance procedure

Before the study begins, record **one hour at the intended mounting**, and:

1. Run `lhv recording-check` over it for R2, R4 and R6, and for the R1 proxy.
2. Label 200 frames with the profile skeleton and check R1's paw visibility.
3. Join one day of the identifier feed against the footage for R5 and R6.
4. Confirm R3 by counting animals per frame in the measured section.
5. Settle R7, R8 and R9 on paper with the farm.

If R1, R2, R3, R5 or R6 fails, fix it before recording in earnest. Each of them
makes some measurement impossible rather than merely noisy, and no amount of
later processing recovers a stride that was never sampled or an identity that
was never recorded.

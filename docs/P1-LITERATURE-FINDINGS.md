# P1 literature findings

The first-pass review's answers, each carried at its verified status.

[docs/P1-LITERATURE-SCOPE.md](P1-LITERATURE-SCOPE.md) states what had to be
asked. [docs/P1-LITERATURE-PROMPT.md](P1-LITERATURE-PROMPT.md) states how it was
asked. This holds what came back, and how much of it survived checking.

The prompt's rule was *real, checkable citations only*. Checking is the other
half of that rule, and a claim that has not been checked is recorded here as
unchecked rather than quietly promoted. No requirement changes on the strength
of anything below marked otherwise than **confirmed**.

---

## What verification meant, and where it could not reach

A claim is **confirmed** only when both halves hold: the paper exists with the
stated authors, venue and DOI, *and* the specific number quoted was found in an
accessible source. Both matter. A real paper misquoted damages a decision
exactly as much as an invented one, and is harder to notice.

Several publishers refuse automated fetching — the Journal of Dairy Science
(both full text and PDF), Wiley, ScienceDirect and Nature all returned 403 or an
authentication redirect. Where the primary source was unreachable, the claim was
checked against indexes, repository READMEs, preprints and citing works, and
where that failed it is marked unverified rather than assumed. **Institutional
access would close most of the remaining gaps**, and three of them are worth the
trip.

---

## Status of every load-bearing claim

| claim | source | status |
|---|---|---|
| 100 cows, r = 0.864 support-phase vs score, 96% cows / 93% hooves | Kang, Zhang & Liu 2020, *J Dairy Sci* 103(11):10628–10638, `10.3168/jds.2020-18288` | **confirmed** |
| stride duration 1.26 ± 0.03 s healthy vs 1.48 ± 0.05 s sole ulcer; groups n = 17 / 14 / 7 | Flower, Sanderson & Weary 2005, *J Dairy Sci* | **confirmed** |
| triple support more than doubled in sole-ulcer cows | Flower et al. 2005 | **confirmed** (the 18% → 42% figures themselves not seen) |
| interobserver κw 0.24–0.68, means 0.48 and 0.52 before/after training | Thomsen et al. 2008, *J Dairy Sci* 91:119–126 | **confirmed** |
| 10 observers, 207 cows, 9 farms; Gwet AC1 35.6–74.5%; prevalence estimates 36.2–57.0% | Wilson, Thorup & Bell 2026, *Vet Record*, `10.1002/vetr.70801` | **confirmed** |
| overall Fleiss κ = 0.28 | Wilson et al. 2026 | **unverified** — direction confirmed ("overall agreement was poor"); the value was not reachable |
| T-LEAP code Apache-2.0; cattle training data not open | github.com/hrussel/t-leap | **confirmed**, verbatim in the README |
| AP-10K CC-BY-4.0, 17 keypoints, distal limb ends at the paw | github.com/AlexTheBad/AP-10K | **confirmed** |
| classification 76.6% / 79.9% / 80.1% for one, three and six traits | Russello et al. 2024, arXiv:2401.05202 | **confirmed** |
| 99.6% correct keypoints under variable outdoor lighting | Russello et al. 2024 | **confirmed** |
| 272 keypoint trajectories with per-trajectory lameness scores, no raw video | github.com/hrussel/lstm-lameness-detection | **confirmed** |
| 633 milking sessions, 224 ± 10 cows identified per session, single-lane alley after milking | Van Hertem et al. 2018, `10.1016/j.biosystemseng.2017.08.011` | **confirmed** |
| AUC 0.719 hip curvature, 0.702 back posture | Van Hertem et al. 2018 | **unverified** |
| ~49% of identified cows yielded an automatic score | Van Hertem et al. 2018 | **unverified**, and the accessible figure is 197 ± 16 videos recorded of 224 identified (88.1%), which measures something else |
| 11 farms, 42 mobility-scoring sessions, four veterinarians, 0–3 scale | Siachos et al. 2025, *J Dairy Sci* | **confirmed** |
| RCT, 419 cows, severe lameness 7.9% → 2.0%, chronic 9.8% → 3.9% | Siachos et al. 2026, *J Dairy Sci*, PMID 42398719 | **paper confirmed, figures unverified** — see below |
| T-LEAP/CoWalk uses 17 landmarks including fetlock and carpal per limb | Russello et al. 2022, arXiv:2104.08029 | **confirmed** — see below |
| Kang recorded at 50 fps over a 4 × 1.2 m passage, camera 6 m to the side | review's claim | **unverified** — see below |

---

## The three that did not survive

### Siachos et al. 2026 randomised controlled trial — found, figures still unverified

**Resolved 2026-09-07.** The paper exists: *A randomized controlled trial
evaluating the use of an intelligent, fully automated 2D imaging system to
detect lame cows and control lameness*, Siachos, Wilson, Anagnostopoulos, Neary,
Smith & Oikonomou, University of Liverpool, *Journal of Dairy Science* 2026,
online 3 July 2026, PMID 42398719. The earlier *not found* was a failure of
search, not a fabricated citation, and the distinction matters: the first
reading would have had the review inventing a source.

Two things remain open. The **specific figures** — 419 cows, severe lameness
7.9% → 2.0%, chronic 9.8% → 3.9% — could not be read, because PubMed and Europe
PMC both refuse automated fetching. And the system under trial is an **overhead
camera roughly 4 m above the parlour return alley**, not a lateral one, so it
speaks to the treatment protocol rather than to the geometry.

Until the abstract is read, the treatment-policy conclusion in L8 rests on a
paper that certainly exists but whose numbers this project has not seen. That is
a materially better position than *not found* and still not good enough to put a
figure in a requirement.

### T-LEAP's keypoint count — resolved: seventeen, and the nine are a subset

**Resolved 2026-09-07** from the paper itself (arXiv:2104.08029), which states:
*"Seventeen anatomical landmarks on the cow's body were annotated."* They are
hoof, fetlock and carpal on each forelimb; hoof, fetlock and tarsal on each hind
limb; and nose, forehead, withers, caudal thoracic vertebrae and sacrum. The
review was right and the reconciliation guessed above was the correct one: the
2024 lameness work used a nine-point subset of the seventeen.

This settles the L4 labelling estimate, and it does something better than that.
The five body landmarks — **nose, forehead, withers, caudal thoracic vertebrae,
sacrum** — are exactly the five this project chose independently for its own
skeleton, before the count was known. The provisional `lateral-gait-9` is
therefore not a guess that happens to work: it is CoWalk-17 with the eight
intermediate limb joints removed, and those eight are removed because no feature
in version 1 depends on one.

Two further figures came with it, neither of which the review reported:

- **A mounting geometry that works.** The camera sat **2 m above ground and
  4.5 m from the walkway fence**, side view. `docs/P1-RECORDING-SPECIFICATION.md`
  lists the correct mounting angle and height under *what P0 could not
  determine*; this is the first published pair of numbers to put against it.
- **30 fps, not 50.** T-LEAP's footage was recorded at 30 fps and reached 87.6%
  PCKh on unseen cows. That is a working system below the review's preferred
  rate, and it weakens the case for 50 fps for pose — though not for the
  foot-strike timing that L5's recommendation actually rests on.

### Kang's recording setup — unverified

The 50 fps capture rate, the 4 × 1.2 m passage and the 6 m camera distance
appear in no accessible source. The abstract does carry a nearby and confusable
number: **83.3 frames per second is the detector's processing throughput**, not
a capture rate.

Two decisions rest on the unverified figures. L5's "50 fps preferred"
partly derives from the capture rate. And the passage length bears directly on
R4's requirement of three body lengths of field of view — if the strongest
commercial-lane result was obtained over roughly 1.7 body lengths, R4 is
over-specified, since it was derived from CattleEyeView's framing rather than
from what a measurement needs. Neither moves until the paper is read.

---

## What verification added

**The Russello trajectory release carries no licence.** It contains what was
claimed — 272 T-LEAP keypoint trajectories, each with a lameness score in
`data/videos_lameness_scores.csv`, no raw video — but states no licence for
either code or data. Under this project's provenance rules that blocks
registration. The remedy is an enquiry to the authors, not an assumption.

This dataset is worth the enquiry. It is the only identified route to testing
**feature-to-score correlation before a farm exists** — P1's own gate, in its
cross-sectional form, on 98 cows. It cannot exercise detection, tracking or
pose, since the trajectories are already extracted, and at roughly 2.8
trajectories per cow it is not a longitudinal series, so it does nothing for the
baseline layer. Within those limits it addresses the largest untested claim in
the project: P0-OUTCOME records that everything above the phenotype layer has
only ever seen injected synthetic deviations.

**Russello et al. 2024 supplies a validated feature set outright.** Its six
traits are back posture, head bobbing, tracking distance, stride length, stance
duration and swing duration, reaching 80.1% on 98 cows. That is very nearly the
L3 recommendation, from a single confirmed source with a number attached, and is
a better starting point for `feature_set` version 1 than a set assembled from
six separate papers.

---

### A dataset that reads like the answer to L7 and is not

**CowScreeningDB** (Ismail, Diaz, Carmona-Duarte, Vilar & Ferrer, 2024,
arXiv:2405.15550, doi:10.1016/j.compag.2023.108500) is published as *a public
benchmark dataset for lameness detection in dairy cows*, CC BY-NC-ND 4.0, 43
cows from a farm in Gran Canaria. The title is almost exactly L7's question.

It is **Apple Watch accelerometer and gyroscope data**, not video, with binary
healthy/lame labels and no keypoints. It therefore does not disturb L7's
negative, and it is recorded here so the next reader does not spend the same
search discovering that for themselves. A negative that has been checked twice
is worth more than one checked once, and worth much more than one checked once
and forgotten.

## Where the decisions stand

```
  SAFE TO DECIDE
  L1  lateral geometry      Kang's result confirmed; its setup details are not
  L3  feature set           Flower and Russello confirmed; adopt Russello's six
  L4  skeleton              AP-10K licence and coverage confirmed; T-LEAP data
                            confirmed unavailable; keypoint count unresolved
  L6  the gate              Thomsen and Wilson confirmed -- the strongest result
                            in the review, and the one that changes the README
  L7  qualified negative    confirmed; the one usable resource has no licence

  HELD
  L2  representation        the review did not close it, and said so
  L5  frame rate            the stride band is solid; the fps figure is not
  L8  treatment policy      sole citation not found
  L9  exposure              closed as "no usable evidence found" -- which is an
                            answer, and stands
```

L6 is the result to act on first. Both of its supporting papers verified
cleanly, and together they say that a single human locomotion score is too noisy
a reference for the gate the README currently states.

---

## What this document is not

- **Not the decision record.** `docs/P1-MODEL-DECISION.md` does not exist yet
  and should be written from the confirmed rows above, not from this document
  wholesale.
- **Not a second pass.** L2 remains open by the review's own account: no
  study compares representations on the same data, the same split and the same
  clinical reference. That comparison is an experiment, not a search.
- **Not a substitute for reading the papers.** Verification here establishes
  that a claim is real and quoted correctly. Whether it transfers to a
  commercial AMS exit lane is a separate judgement, and the review's own warning
  applies: most reported performance comes from purpose-built walkways.

## Outstanding

1. **Kang et al. 2020**, institutional access — the recording setup settles L5's
   frame rate and R4's lane length together. The only one of these four still
   wholly unresolved. Corroborated without it: side view, near a milking-parlour
   entrance, 100 multiparous Holsteins on one farm, kappa 0.93 against human
   score. Not corroborated: the 50 fps capture rate and the 4 x 1.2 m passage,
   both of which carry decisions.
2. ~~Obtain Russello et al. 2022; settle the nine-versus-seventeen keypoint
   question.~~ **Done, 2026-09-07.** Seventeen, from the paper itself. The nine
   are a subset; this profile's own nine are a principled subset of the same
   seventeen; and two figures came free — a 2 m / 4.5 m side mounting that
   works, and 30 fps rather than 50.
3. ~~Find Siachos et al. 2026.~~ **Found, 2026-09-07**, PMID 42398719. Its
   figures remain unread behind two publishers that refuse automated fetching,
   so the treatment-policy claim is still not quotable.
4. ~~Write to the T-LEAP authors about the trajectory release's licence.~~
   **Sent 2026-09-07**, see
   [docs/P1-TLEAP-LICENCE-ENQUIRY.md](P1-TLEAP-LICENCE-ENQUIRY.md). Awaiting a
   reply.

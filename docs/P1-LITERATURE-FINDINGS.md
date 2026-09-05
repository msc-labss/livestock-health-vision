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
| RCT, 419 cows, severe lameness 7.9% → 2.0%, chronic 9.8% → 3.9% | Siachos et al. 2026, `10.3168/jds.2026-28255` | **not found** — see below |
| T-LEAP/CoWalk uses 17 landmarks including fetlock and carpal per limb | review's claim | **contradicted** — see below |
| Kang recorded at 50 fps over a 4 × 1.2 m passage, camera 6 m to the side | review's claim | **unverified** — see below |

---

## The three that did not survive

### Siachos et al. 2026 randomised controlled trial — not found

Three searches under different phrasings returned the 2025 evaluation paper —
which is real, and confirmed above — but nothing matching a 2026 randomised
controlled trial of 419 cows. It may exist behind indexing that automated search
cannot reach.

It matters because it is the **sole support for the treatment-policy conclusion
in L8**: that detections should trigger examination and treatment, that
treatment must not be withheld to preserve a lead-time endpoint, and that
treatment becomes a censoring event. That conclusion is methodologically
attractive and currently unsupported. It stays out of the requirements until the
paper is in hand or another source carries it.

### T-LEAP's keypoint count — contradicted, in a useful direction

The review states 17 landmarks: hoof, fetlock and carpal on each limb plus nose,
forehead, withers, caudal thoracic vertebra and sacrum. Russello et al. 2024
states that T-LEAP extracted motion data from **nine keypoints**.

The likely reconciliation is that the 2022 pose paper defines 17 and the 2024
lameness work used a nine-point subset. That distinction *is* the L4 decision.
If nine points support an 80.1% result, the labelling obligation is roughly half
what the review costed, and the finding independently supports L2's
recommendation to shrink the pose rather than adopt a full skeleton. Resolving
it needs the 2022 paper.

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

1. Obtain Kang et al. 2020 through institutional access; read the recording
   setup. Settles the L5 frame rate and the R4 lane-length question together.
2. Obtain Russello et al. 2022; settle the nine-versus-seventeen keypoint
   question. Settles the L4 labelling estimate.
3. Find Siachos et al. 2026, or drop the treatment-policy conclusion to
   unsupported and re-ask L8.
4. Write to the T-LEAP authors about the trajectory release's licence.

# livestock-health-vision

Computer vision for detecting and monitoring animal health issues on farms.
Proof of concept — MSC Labs.

## Thesis

Model architecture is no longer the bottleneck in precision-livestock vision.
Detectors, pose estimators, temporal models and trackers all produce strong
within-dataset results. Deployments fail on occlusion, identity loss, farm-to-farm
domain shift, noisy clinical labels, rare positives and uncalibrated alarms.

So this project does not build another classifier. It builds the **spine**:

```
video -> detect -> track -> identity anchor
      -> phenotype extraction
      -> per-animal daily time series
      -> baseline (own history + herd)
      -> risk score + calibration
      -> alert with evidence clip
      -> event schema -> farm database
```

Perception weights are downloadable. The spine is what papers skip and vendors
keep closed.

## Scope

**First target: dairy cattle lameness, at a constrained parlour/AMS exit lane.**

Chosen for ground-truth economics, not model maturity. Lameness is the only
livestock health target whose clinical labels can be produced by a trained scorer
with a clipboard at the lane exit — same day, no lab. Mastitis needs SCC/CMT/culture,
respiratory needs physiological reference and veterinary diagnosis, body condition
needs consensus expert scoring.

The lane geometry is a deliberate design lever: one animal at a time removes
occlusion, known heading makes gait features well defined, fixed lighting shrinks
domain shift, and parlour/AMS identity is supplied for free — which deletes the
unsolved problem of re-identifying an individual across weeks. Animals pass twice
daily, so the longitudinal baseline builds itself.

**Sensor stack: one RGB camera.** No depth, no thermal, no microphone. Each added
modality costs calibration and buys nothing for this target.

### Non-goals

- Diagnosing disease aetiology from pixels. This is a screening and measurement
  system that hands a phenotype to a clinical decision layer.
- Mastitis, respiratory disease, body condition, heat stress.
- Multi-species coverage at the outset. Sheep is added later, as a test that the
  species seam holds.

## Species seam

| Species-specific (swappable) | Species-agnostic (built once) |
|---|---|
| detector weights | tracking / identity anchoring |
| keypoint skeleton definition | time-series store |
| gait feature definitions | baseline model |
| normal-range priors | calibration and thresholds |
| scoring scale | alerting and evidence clips |
| | event schema |
| | evaluation harness |

The right column is most of the code and does not care whether the animal is a cow
or a ewe. Sheep is implemented later to prove that — but the seam is already
carrying weight without it. Two cattle profiles now sit side by side, differing
in camera geometry rather than species: a lateral one for the farm recording and
a frozen top-down one that keeps the public dataset runnable. They disagree on
skeleton, feature set, keypoint names and view, and one extractor serves both
without naming a keypoint of its own.

## Phases and gates

| Phase | Work | Gate |
|---|---|---|
| P0 | Spine on public datasets, animal-disjoint splits | Pipeline runs end to end, no manual steps |
| P1 | One cooperating farm, silent recording, parallel locomotion scoring by two blinded scorers | The system agrees with a held-out scorer consensus as well as a held-out scorer does |
| P2 | Calibration | Lead time and alarms per 1,000 animal-days a farmer would tolerate |
| P3 | Edge deployment on Jetson-class device | Real-time on device, local filtering |
| P4 | Sheep seam | Only species-specific column changes |

P0 needs a GPU and nothing else. **P1 depends on a cooperating dairy — that is the
critical path and takes longer to arrange than P0 takes to build.**

## Evaluation

The evaluation harness is a first-class module, not a notebook. Perception metrics,
clinical metrics and operational metrics are reported separately.

Validation hierarchy, in order:

1. Frame / image hold-out
2. Animal-disjoint
3. Day / production-cycle hold-out
4. Farm-disjoint external
5. Prospective silent deployment
6. Prospective alert trial

Primary endpoints are **lead time** and **alarm burden**, not accuracy. Test sets
stay prevalence-realistic so positive predictive value and daily false-alarm counts
mean something.

Known limitation: a single-farm POC gives n=1 for domain diversity. The schema and
evaluation harness are built multi-farm from the start so that stage 4 is a data
problem, not a rewrite.

## Data handling

No farm footage, derived media or model weights are committed to this repository.
Barn cameras can capture workers and visitors, who are identifiable natural persons
under GDPR even though the animals are not. Camera placement, human masking, access
control and retention are design inputs, not post-deployment additions.

## Running it

```bash
uv sync --extra dev              # resolve the environment
python tools/fetch_weights.py    # pretrained weights, with their licences printed

lhv profiles list                # installed species profiles
lhv profiles show                # the default profile: skeleton, features, weights, licences
lhv profiles show cattle-topdown # the frozen top-down profile P0 was measured against
lhv datasets list                # registered sources and how each is obtained
lhv datasets verify <name> --data-root <path>   # counts on disk vs counts in the paper

# The profile must match the geometry the source was recorded under — pose
# refuses the pairing otherwise, by name. CattleEyeView is overhead footage, so
# it runs under the frozen top-down profile, not the lateral default.
lhv run --dataset cattleeyeview --data-root <path> --output runs/first \
        --profile cattle-topdown
lhv evaluate --dataset cattleeyeview --output runs/first --profile cattle-topdown

lhv recording-check <video>       # judge a pilot recording against the P1 spec
```

`lhv run` carries a registered dataset from source registration to exported
events in one invocation. Stages are separately invocable — `--stages
baseline,events` re-runs the cheap tail against the records perception already
wrote, and produces the same events a full rerun would.

`./tools/check.sh` runs the lint, the species-seam check, the profile-declaration
check and the tests. The last of those enforces that every weight a profile
declares is either built by a stage or says why it is not — a profile is where
decisions get written down, and a declaration nobody honours reads exactly like
one that works.

Weights, video, derived media and dataset payloads are never committed;
`tools/check_no_media_tracked.sh` and `tools/check_clean_clone.sh` enforce that
in CI.

## Status

**P0 is done.** It is built and runs end to end over real public data — see
[docs/P0-OUTCOME.md](docs/P0-OUTCOME.md) for what has been validated against
real labels, what rests on injected data, and what remains unmeasurable.

**P1's groundwork is done; P1 itself waits on a farm.** What P0 measured became
requirements for the recording P1 depends on
([docs/P1-RECORDING-SPECIFICATION.md](docs/P1-RECORDING-SPECIFICATION.md)), and
three of those nine contradict what P0's own design assumed — most importantly
that a top-down camera, chosen to remove occlusion between animals, cannot see
the limbs whose motion lameness consists of.

That left four questions no amount of running the spine could answer, so they
were put to the literature instead. The questions
([scope](docs/P1-LITERATURE-SCOPE.md)), the exact form they were asked in
([prompt](docs/P1-LITERATURE-PROMPT.md)) and which answers survived checking
([findings](docs/P1-LITERATURE-FINDINGS.md)) are all recorded, because an answer
is only as traceable as the question that produced it — and three of the review's
own claims did not survive verification.

The answers that did survive were taken as decisions
([docs/P1-MODEL-DECISION.md](docs/P1-MODEL-DECISION.md)): a lateral geometry, a
feature set chosen for published discrimination rather than for what a top-down
skeleton could compute, and a skeleton that follows the feature set instead of
preceding it. Three of its nine features compute today; six are declared
unavailable and say why.

The P1 gate moved twice. *Correlates with human locomotion score* had no
threshold and could not be failed, because a single scorer is too noisy a
reference to carry one. Its replacement, agreement with a scorer consensus, then
turned out to be unpassable — scorers sit inside the consensus and a system does
not, which is worth about 0.19 of kappa. Both are fixed, and sizing the result
gives the farm a question it can answer:
[docs/P1-GATE-POWER.md](docs/P1-GATE-POWER.md).

**What P1 is waiting on**, none of it work this repository can do alone:

1. **A cooperating dairy** — the critical path, as it has been from the start.
   The ask is now specific: 80–160 cows past the lane, two trained scorers plus
   an adjudicator, compared leave-one-out on a named agreement statistic.
   [docs/P1-FARM-BRIEF.md](docs/P1-FARM-BRIEF.md) is that ask in plain language,
   written to be handed to a farm as it is — what we need, what it costs them,
   what we will not do, and what we cannot promise.
2. **A reply from the T-LEAP authors**
   ([enquiry](docs/P1-TLEAP-LICENCE-ENQUIRY.md)) — their trajectory release is
   the only identified route to testing feature-to-score correlation before farm
   footage exists, and it carries no licence.
3. **One paper behind a paywall** — Kang et al. 2020, which settles the frame
   rate floor and the lane length R4 requires. Both currently sit behind a
   single attributed constant apiece, so moving them is one edit each.

**Every event this system currently produces is marked `stub_derived` and
`non_clinical`.** Health inference is exercised against injected synthetic
deviations, not fitted to clinical outcomes. Nothing it emits is clinical
evidence.

## Workflow

Specifications and changes are managed with OpenSpec under `openspec/`.

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
or a ewe. Sheep is implemented later to prove that.

## Phases and gates

| Phase | Work | Gate |
|---|---|---|
| P0 | Spine on public datasets, animal-disjoint splits | Pipeline runs end to end, no manual steps |
| P1 | One cooperating farm, silent recording, parallel human locomotion scoring | Gait features correlate with human score |
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

lhv profiles show                # the species seam: skeleton, features, weights, licences
lhv datasets list                # registered sources and how each is obtained
lhv datasets verify <name> --data-root <path>   # counts on disk vs counts in the paper

lhv run --dataset <name> --data-root <path> --output runs/first
lhv evaluate --dataset <name> --output runs/first

lhv recording-check <video>       # judge a pilot recording against the P1 spec
```

`lhv run` carries a registered dataset from source registration to exported
events in one invocation. Stages are separately invocable — `--stages
baseline,events` re-runs the cheap tail against the records perception already
wrote, and produces the same events a full rerun would.

`./tools/check.sh` runs the lint, the species-seam check and the tests.
Weights, video, derived media and dataset payloads are never committed;
`tools/check_no_media_tracked.sh` and `tools/check_clean_clone.sh` enforce that
in CI.

## Status

P0 is built and runs end to end over real public data — see
[docs/P0-OUTCOME.md](docs/P0-OUTCOME.md) for what has been validated against
real labels, what rests on injected data, and what remains unmeasurable.

What P0 measured turned into requirements for the farm recording P1 depends on:
[docs/P1-RECORDING-SPECIFICATION.md](docs/P1-RECORDING-SPECIFICATION.md). Three
of its nine requirements contradict what P0's own design assumed — most
importantly that a top-down camera, chosen to remove occlusion between animals,
cannot see the limbs whose motion lameness consists of.

**Every event this system currently produces is marked `stub_derived` and
`non_clinical`.** Health inference is exercised against injected synthetic
deviations, not fitted to clinical outcomes. Nothing it emits is clinical
evidence.

## Workflow

Specifications and changes are managed with OpenSpec under `openspec/`.

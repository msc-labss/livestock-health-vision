## Context

See `proposal.md` — Why. This is the first code in a greenfield repository, so the design fixes the interfaces every later phase inherits rather than optimising anything.

Three constraints shape everything below.

**No farm data.** P0 runs on public livestock datasets. Those carry identity and behaviour labels, not clinical outcomes, so the health-inference layer has no ground truth to fit against and is exercised against injected deviations instead.

**No external identifier stream.** The parlour or AMS identifier that will anchor identity in production does not exist in public data. P0 substitutes the dataset's own ground-truth animal id, routed through the same anchor interface a real identifier will use later.

**Later phases must not force a rewrite.** P1 attaches real ground truth, P2 recalibrates thresholds, P3 moves stages onto edge hardware, P4 swaps species. Each is a substitution behind an interface this change defines, so interface stability outranks convenience.

## Goals / Non-Goals

**Goals:**

- Stage boundaries that are durable and independently re-runnable, so recomputing a baseline never requires recomputing detections.
- One record schema per stage boundary, versioned, carrying provenance from ingest through to exported event.
- A species profile that isolates everything species-specific into one replaceable object.
- Determinism sufficient for the reproducible-report requirement: pinned model identities, seeded stochastic steps, a configuration digest on every artefact.

**Non-Goals:**

- Throughput or latency optimisation. P0 is batch over recorded video; real-time is P3's gate and is served by keeping stages separable, not by tuning them now.
- Distributed execution. Single-node is assumed; if the data outgrows it, that is a later change.
- A user interface. P0's outputs are event records and evaluation reports.
- Training or fine-tuning any model. All weights are pretrained and treated as swappable inputs.

## Decisions

### Materialised stage boundaries over an in-process pipeline

Each stage reads durable records and writes durable records: frames → detections → tracklets → poses → passes → features → time series → scores → events. Stages are separately invocable.

*Why:* perception is the expensive stage and the one that changes least; baseline and threshold logic are cheap and change constantly. Coupling them into one process would make every threshold experiment pay for re-detection. Materialisation also gives the evaluation harness a natural place to attach, and gives P3 a clean cut line when perception moves to the edge and inference stays central.

*Alternative considered:* a single streaming process with in-memory handoff. Fewer moving parts and closer to the eventual production shape, but it makes reprocessing and per-stage evaluation painful, and P0's whole purpose is per-stage auditability.

### Columnar files for bulk records, an embedded analytical database for series and events

Bulk per-frame and per-tracklet records are written as partitioned columnar files, partitioned by site and day. The per-animal time series, the event store and the evaluation queries run against an embedded analytical database that reads those files directly.

*Why:* per-frame records are large, append-only and never updated — a columnar file is the right shape. Time series and events are small, queried by animal and window, and benefit from real query support. An embedded engine that reads the columnar files in place avoids maintaining two copies.

*Alternatives considered:* everything in a relational database — poor fit for millions of per-frame rows in a POC, and imposes a load step before anything can be inspected. Everything in flat files — makes the baseline and evaluation queries hand-rolled and slow to iterate.

### Species profile as the single seam

One profile object supplies: skeleton definition and version, detector and pose weight references, feature-set definition and version, normal-range priors, and the scoring scale. Nothing else in the codebase is allowed to branch on species.

*Why:* the proposal's species-seam claim is only proven if adding sheep touches exactly this object. Making it a single named artefact turns that claim into something P4 can falsify.

*Alternative considered:* per-species subclasses across stages. Spreads species knowledge through the pipeline and makes the P4 gate unmeasurable.

### Dataset ground-truth identity as a simulated external anchor

P0 wires the dataset's ground-truth animal id into the external-anchor interface, marked with its anchor source. Visual re-identification is implemented as the fallback path and evaluated against identity-labelled public data, but is not the primary path even in P0.

*Why:* this exercises the production control flow — anchor first, fallback second, unresolved as a real outcome — without a farm. If P0 instead used visual re-identification as the primary path, P1 would be re-plumbing rather than substituting a source.

*Trade-off:* an anchor with perfect accuracy makes P0's identity layer look better than production will. The evaluation harness therefore also reports the fallback path's standalone performance, so the anchor is not hiding the visual path's weakness.

### Motion-based tracking, appearance embeddings only in the fallback

Tracking within a pass uses motion association without appearance embeddings. Appearance-based re-identification exists only in the identity fallback.

*Why:* the target geometry is a lane carrying one animal at a time in a known direction. Appearance association solves a problem that geometry has already removed, and costs compute that P3 will not have. Keeping embeddings confined to the fallback also keeps the expensive path optional.

*Alternative considered:* appearance-based tracking throughout, for robustness in unconstrained barn footage. Rejected because it optimises for a geometry this change deliberately does not target — see proposal non-goals.

### Skeleton and feature set follow available public labels

The skeleton definition adopts the top-down keypoint convention used by the public dataset the perception layer is evaluated against. Feature-set version zero contains only features computable from that skeleton on that data.

*Why:* defining a biomechanically ideal feature set that public data cannot support would produce features with no evaluable quality. Version zero is honest about what the data allows; the feature set is versioned precisely so P1 can extend it against real lane footage without invalidating earlier records.

### Injection at the time-series layer, not the pixel layer

Synthetic deviations are injected into the phenotype time series with a declared magnitude and temporal shape, and every derived artefact carries the injection marker.

*Why:* the stub exists to exercise baseline, scoring, thresholding and alerting — all of which consume the time series. Injecting at the pixel layer would additionally test perception's response to synthetic imagery, which is a different question and one the literature shows is easy to answer misleadingly.

*Trade-off:* injection cannot validate that a real lame cow produces a detectable deviation. Nothing available in P0 can. The evaluation harness states this rather than obscuring it.

### Configuration digest as the reproducibility key

Every artefact records a digest over the resolved configuration, the model identities and versions, and the dataset version. Reports are keyed on it.

*Why:* the reproducible-report requirement needs a single value that changes when anything material changes. Recording individual fields alone makes "is this the same run?" a manual comparison.

### Export as a pluggable adapter, file-backed in P0

The event export interface is defined now; P0 ships a file-backed adapter with the idempotency key written into each record.

*Why:* there is no farm database to integrate with yet, and inventing one would be speculative. Defining the interface and delivery semantics now is what P1 needs; the transport is a later substitution.

## Risks / Trade-offs

- **Public data may not match the lane geometry the design targets.** → Feature-set version zero is scoped to what the available top-down data supports, and pass segmentation boundaries are configurable rather than assumed. If a public source has no usable pass structure, it still serves the detection, tracking, pose and identity layers, and pass segmentation is evaluated on whatever subset does.

- **The gait-phenotype layer cannot be validated against lameness labels in P0.** → P0 evaluates feature reproducibility, stability across repeated passes by the same animal, and quality-flag behaviour, and states plainly that clinical correlation is unevaluated. Presenting a stability result as a clinical result is the specific failure mode being guarded against.

- **A perfect simulated anchor flatters the identity layer.** → The harness reports the visual fallback's standalone performance separately, so the fallback's real weakness is visible even while the anchor path carries production flow.

- **Public datasets carry a single site each, so site-disjoint validation may be unexercised.** → The split machinery and provenance keys are built for it regardless, and the report is required to state the site count and that site-disjoint validation did not run. This keeps the gap loud rather than latent.

- **Stage materialisation multiplies storage.** → Partitioning by site and day makes selective deletion cheap, and intermediate per-frame records are declared disposable and regenerable from the configuration digest.

- **Pretrained weights may be licensed incompatibly with later commercial use.** → Weight references live in the species profile with their licences recorded, so a substitution is a profile edit rather than a code change. Resolving licensing is P1 work, not P0's.

- **Interface stability is asserted, not yet tested.** → P4's species swap and P3's edge split are the tests. Until then, every stage boundary is a versioned record schema, which at least makes a breaking change visible instead of silent.

## Migration Plan

Greenfield; nothing to migrate. Deployment in P0 means running the pipeline locally over recorded public data. Rollback is discarding generated artefacts, which are all regenerable from the configuration digest and the source datasets.

## Open Questions

- The exact composition of feature-set version one, beyond version zero's data-constrained subset. Deferrable: the feature set is versioned, and adding features does not change any stage boundary.
- Whether the embedded analytical database remains adequate once P1 adds continuous farm recording. Deferrable: the storage interface is behind the time-series and event stores, and the columnar files are engine-independent.
- ~~Which public dataset becomes the primary perception benchmark.~~ **Resolved:** CattleEyeView is the primary perception source, as the only candidate carrying keypoints and therefore the only one able to exercise gait phenotype. MultiCamCows2024 is the secondary source, exercising the identity fallback. It was also expected to supply a genuine multi-day series; running it showed it cannot, and the expectation was wrong rather than unmet. The release carries no keypoint annotations, so no pose is available and every feature value a full run produced was unusable; and its tracklet stills carry no capture time in any path, filename or header, while a per-animal series is keyed by observation time. Either absence alone is sufficient. A multi-day series from real data therefore waits for P1's farm recording, which is specified in docs/P1-RECORDING-SPECIFICATION.md. Ingest remains source-registered, so further sources stay configuration rather than code.

## 1. Foundation

- [ ] 1.1 Create the package layout with one module per stage boundary (ingest, perception, identity, phenotype, baseline, events, evaluation) and verify an import of each module succeeds from a clean environment
- [ ] 1.2 Add the dependency set covering the deep-learning runtime, video decoding, columnar storage and the embedded analytical database, and verify a clean install resolves and imports without version conflicts
- [ ] 1.3 Implement the resolved-configuration model and its digest, and verify that two configurations differing in any material field produce different digests while a reordered but equivalent configuration produces the same digest
- [ ] 1.4 Implement the species profile object carrying skeleton definition, weight references with licence fields, feature-set definition, priors and scoring scale, and verify a profile loads and reports its version
- [ ] 1.5 Add a lint and test harness with a check that fails if any module outside the species profile branches on species, and verify the check runs in CI

## 2. Dataset access

- [ ] 2.1 Obtain CattleEyeView, record its licence and access terms in the repository, and verify the recorded content counts match what was actually downloaded rather than what the literature reported
- [ ] 2.2 Obtain MultiCamCows2024 on the same terms and verify its identity labels and camera partitioning are present and usable
- [ ] 2.3 Write the dataset registration records supplying site key, camera identifiers and animal-set keys for both sources, and verify registration is refused when a required key is absent

## 3. Video ingest

- [ ] 3.1 Implement source registration and the provenance record, and verify a source missing a site key or camera identifier is refused at registration
- [ ] 3.2 Implement frame emission with full provenance attached, and verify every emitted frame carries source, camera, timestamp, frame index, animal-set key, day key and site key
- [ ] 3.3 Implement deterministic iteration and verify two runs over the same source with the same configuration emit equal frame counts, order and provenance
- [ ] 3.4 Implement resumable iteration from a recorded position and verify a run interrupted after N frames resumes at N+1 without re-emission
- [ ] 3.5 Implement timestamp integrity handling and verify that a missing timestamp is marked unreliable rather than defaulted, and that non-monotonic timestamps are reported with their frame indices
- [ ] 3.6 Enforce derived-media isolation and verify ingest fails before writing when configured with a version-controlled output root

## 4. Perception

- [ ] 4.1 Define the detection, tracklet and pose record schemas at their stage boundaries, and verify each schema carries its own version and rejects a record missing a required field
- [ ] 4.2 Implement the detection stage against pretrained weights and verify a frame containing no animal yields an empty result rather than being omitted from the stream
- [ ] 4.3 Implement model identity recording and verify that reprocessing the same source with different weights yields outputs distinguishable by the model identity field alone
- [ ] 4.4 Implement motion-based tracking into tracklets with termination reasons, and verify exit, occlusion loss and detection failure are recorded as distinct reasons
- [ ] 4.5 Verify tracklet identifier uniqueness within a source across a full CattleEyeView pass through
- [ ] 4.6 Implement pose estimation against the species profile skeleton and verify outputs record the skeleton identifier and version
- [ ] 4.7 Implement keypoint visibility handling and verify an unobservable keypoint is emitted not-visible rather than interpolated
- [ ] 4.8 Implement low-confidence marking for detection and pose, and verify low-confidence results remain in the stream and are counted in the run summary

## 5. Identity anchoring

- [ ] 5.1 Define the external-anchor interface and wire CattleEyeView and MultiCamCows2024 ground-truth animal identity into it as a simulated anchor source, and verify assignments record the anchor source
- [ ] 5.2 Implement ambiguous-match handling and verify a tracklet matching more than one anchor record is left unresolved with its competing candidates recorded
- [ ] 5.3 Implement the visual re-identification fallback and verify its assignments are marked as fallback with a confidence value
- [ ] 5.4 Implement the unresolved outcome and verify a tracklet below the confidence floor with no anchor is excluded from time series while remaining retrievable for audit
- [ ] 5.5 Implement assignment provenance and verify method, confidence, assignment time and evidence reference are retrievable without reprocessing video
- [ ] 5.6 Implement identity conflict detection and verify that two time-overlapping tracklets resolving to one identity are both withheld from the time series and recorded as a conflict

## 6. Gait phenotype

- [ ] 6.1 Implement configurable pass segmentation with entry and exit boundaries, and verify a complete pass spans exactly the two boundary crossings
- [ ] 6.2 Implement partial-pass labelling and verify a pass missing a boundary is labelled partial with no extrapolated boundary
- [ ] 6.3 Define feature-set version zero limited to what the CattleEyeView skeleton supports, document each feature's name and unit, and verify the emitted record names its feature-set version
- [ ] 6.4 Implement rejection of undeclared features and verify extraction fails naming the offending feature rather than silently emitting it
- [ ] 6.5 Implement per-feature quality flags derived from keypoint confidence and coverage, and verify a feature depending on an intermittently visible keypoint carries a reduced flag naming that keypoint
- [ ] 6.6 Implement the pass validity decision with reasons, and verify an invalid pass is excluded from the time series, retained for audit, and emits no feature values to downstream consumers

## 7. Baseline and time series

- [ ] 7.1 Implement the per-animal time series store over the columnar layout partitioned by site and day, and verify a valid resolved pass is appended at its pass timestamp
- [ ] 7.2 Verify a valid pass with unresolved identity modifies no time series and lands in the unattributed store
- [ ] 7.3 Implement the own-history baseline over a declared lookback window and verify the scored observation is excluded from its own baseline
- [ ] 7.4 Implement the herd baseline and verify that a synthetic herd-wide shift elevates own-history deviation while leaving herd-relative deviation flat, with both reported
- [ ] 7.5 Implement the cold-start state and verify an animal below the minimum observation count yields an explicit insufficient-history result and no numeric risk score
- [ ] 7.6 Implement the risk score with uncertainty and recorded inputs, and verify the score can be recomputed from the stored series using only its recorded baseline and window references
- [ ] 7.7 Implement time-series anomaly injection with declared magnitude and temporal shape, and verify an injected deviation of known magnitude moves the risk score
- [ ] 7.8 Implement injection marking and verify every artefact derived from injected data carries the marker through to the exported event

## 8. Events, alerting and export

- [ ] 8.1 Define the versioned health event schema with its required fields, and verify construction fails naming the missing field when the observation window is absent
- [ ] 8.2 Verify the schema version is readable from an exported event without out-of-band knowledge
- [ ] 8.3 Implement the named versioned threshold policy and verify each alert records the policy identity and version that raised it
- [ ] 8.4 Verify that reprocessing comparable scores under a changed policy yields alerts distinguishable by recorded policy identity
- [ ] 8.5 Implement bounded evidence clip retention on alert-level events and verify a routine observation retains no clip
- [ ] 8.6 Implement human masking applied before clip write, and verify no unmasked copy reaches storage
- [ ] 8.7 Implement masking-failure handling and verify a clip that cannot be masked is not retained and the event records the retention failure
- [ ] 8.8 Implement the file-backed export adapter with idempotency keys and verify a redelivered event is detectable as a duplicate from its key
- [ ] 8.9 Implement export retry and verify that an unavailable consumer causes retention and retry, with undelivered counts reported for the run
- [ ] 8.10 Implement the stub-derived marker and verify every P0 event carries the non-clinical marking

## 9. Evaluation harness

- [ ] 9.1 Implement split construction from provenance keys and verify animal-disjoint and site-disjoint splits share no key across partitions
- [ ] 9.2 Implement leakage rejection and verify a requested animal-disjoint split with an overlapping animal aborts the run naming the overlap and produces no metrics
- [ ] 9.3 Implement the perception metric family and verify detection, tracking and pose metrics are reported against CattleEyeView labels under an animal-disjoint split
- [ ] 9.4 Implement the phenotype metric family covering feature reproducibility and stability across repeated passes by the same animal, and verify it reports without any clinical label present
- [ ] 9.5 Implement standalone reporting of the visual identity fallback against MultiCamCows2024, and verify it is reported separately from the anchored path
- [ ] 9.6 Implement the operational metric family and verify alarm burden per thousand animal-days is reported alongside the threshold policy identity, and lead time names its reference event
- [ ] 9.7 Enforce metric-family separation and verify no aggregate score is emitted across families
- [ ] 9.8 Implement the reproducible report recording dataset version, model identities, split definition and configuration digest, and verify regeneration from those recorded inputs reproduces the report
- [ ] 9.9 Implement declared-limitation reporting and verify every report states that health inference is stubbed and reports the site count with a note when site-disjoint validation did not run

## 10. P0 gate

- [ ] 10.1 Run the full pipeline end to end over CattleEyeView from registration to exported events with no manual step between stages, and verify the run completes from a single invocation
- [ ] 10.2 Run the full pipeline over MultiCamCows2024 and verify the identity fallback path and multi-day series populate
- [ ] 10.3 Verify each stage is independently re-runnable by recomputing the baseline and event stages without recomputing detection, and confirm outputs match a full rerun
- [ ] 10.4 Produce the P0 evaluation report and verify it carries separated metric families, the stubbed-inference limitation and the site count
- [ ] 10.5 Confirm no video, derived media or model weights are tracked by version control, and verify a clean clone contains none
- [ ] 10.6 Record the P0 outcome against the gate stated in the proposal, naming what was validated against real labels and what rests on injected data

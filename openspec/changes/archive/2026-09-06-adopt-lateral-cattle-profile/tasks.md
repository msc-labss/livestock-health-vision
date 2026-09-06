## 1. Settle the licence before anything is fetched

- [x] 1.1 Establish the licence of AP-10K's published checkpoints, which are hosted by OpenMMLab and may not carry the dataset's CC-BY-4.0. Record it with the weights entry the way every other licence in the profile is recorded.
- [x] 1.2 If the checkpoint licence is unusable, stop and reopen the backend decision rather than proceeding — this is a change to `weights.pose` alone, which is what the seam is for.

## 2. Make the view load-bearing

- [x] 2.1 Add a declared view to the dataset registration schema, and to `register_source` in `src/lhv/ingest/source.py`, alongside the site and camera identity it already carries.
- [x] 2.2 Declare `top-down` on the CattleEyeView registration.
- [x] 2.3 Establish MultiCamCows2024's view from its release and declare it. If it cannot be established from the release, leave it undeclared and let the refusal in 2.4 stand.
- [x] 2.4 Refuse pose estimation when the skeleton's declared view disagrees with the source's, naming both views; refuse equally when the source declares none.
- [x] 2.5 Record the view on emitted pose records.
- [x] 2.6 Tests: mismatched view aborts; undeclared view aborts; matching view proceeds and records the view.

## 3. Placeholder weights become identifiable

- [x] 3.1 Add an explicit placeholder boolean to the profile's weight entries, replacing the prose in the current `notes:` field, which no report can reach.
- [x] 3.2 Carry the flag on model identity records, which already travel with every output.
- [x] 3.3 Name placeholder weights in the perception and evaluation reports, stating that the family's metrics measure the placeholder rather than an achievable result.
- [x] 3.4 Tests: placeholder output carries the mark; replacing the weights with non-placeholder ones removes it and makes the two runs distinguishable by that field alone.

## 4. The skeleton

- [x] 4.1 Replace `cattleeyeview-topdown-24` with the provisional nine-point lateral skeleton: four hooves, withers, mid-thoracic, sacrum, head. Set `view: lateral`.
- [x] 4.2 Record it as provisional, naming the open question (nine keypoints or seventeen) that would revise it.
- [x] 4.3 Record flip pairs for the four hooves. Do not invent OKS sigmas — record the source of any sigma used, or state that none is established.
- [x] 4.4 Wire the AP-10K pose backend and retire the COCO-human analogy keypoint map in `src/lhv/perception/pose.py`.
- [x] 4.5 Map AP-10K's emitted points onto the profile skeleton. Emit unmapped profile keypoints as not-visible — mid-thoracic in particular, which AP-10K does not supply.

## 5. Feature set version 1

- [x] 5.1 Declare all nine features in `cattle.yaml` at `feature_set` version 1, each naming the view its definition holds under.
- [x] 5.2 Mark the six unavailable features with a reason naming what is missing: hoof ground-contact detection for five, the mid-dorsal keypoint for `back_posture`.
- [x] 5.3 Retire `lateral_sway`, `head_lateral_offset`, `spine_lateral_curvature`, `stride_frequency_front`/`_back`, `step_asymmetry_front`/`_back` and `speed_variability`, with `requires_sampling_hz`.
- [x] 5.4 Move `tracking_jitter` out of the feature set and onto the pass as observation-quality metadata.
- [x] 5.5 Record priors: measured anchors with a direction for `stride_duration`, `stance_duration`, `speed` and `stride_length` under a provenance distinct from `unvalidated-p0-prior`; bands stay unvalidated. Do not record a group mean as a low/high band.

## 6. Feature computation

- [x] 6.1 Refuse to compute a feature whose declared view differs from the skeleton's, naming the feature and both views.
- [x] 6.2 Add an explicit unavailable state on the feature record, distinct from the reduced-quality flag.
- [x] 6.3 Exclude unavailable features from the pass validity decision, so a property of the configuration does not reject every pass.
- [x] 6.4 Rewrite `_compute` in `src/lhv/phenotype/features.py` for `stride_length`, `speed` and `head_bob` against the lateral skeleton; remove the retired computations.
- [x] 6.5 Update `min_usable_features` in `src/lhv/config.py` for a set of three implemented features, and the `PhenotypeConfig` comment that reasons about a top-down view.
- [x] 6.6 Tests: foreign-view feature is refused rather than reinterpreted; unavailable feature carries its reason, yields no value downstream, and does not invalidate the pass.

## 8. Retain the top-down profile (added during implementation)

- [x] 8.1 Copy the version-0 profile to `cattle-topdown.yaml`, frozen, with every feature declaring `view: top-down` and the pose weights declared a placeholder.
- [x] 8.2 Declare `default: true` on `cattle.yaml`, and resolve `default_profile_name` from that declaration rather than from there being one file.
- [x] 8.3 Declare the body axis as a skeleton role in both profiles, so extraction never names a keypoint.
- [x] 8.4 Drive `_compute` from a per-feature implementation registry, each implementation taking its points from the feature's own `depends_on`.
- [x] 8.5 Refuse, by name, a feature declared available with no implementation.
- [x] 8.6 Point the real-dataset tests at `cattle-topdown`, and add a test that the lateral profile refuses that footage.

## 7. Close the change

- [x] 7.1 Run `./tools/check.sh` — lint, species-seam check and tests.
- [x] 7.2 Confirm a run over CattleEyeView now aborts on the view mismatch rather than producing feature values, and that the abort names both views.
- [x] 7.3 Confirm feature-set version 0 records remain readable and remain distinguishable by their version field. They are not migrated.
- [x] 7.4 Update `docs/P1-MODEL-DECISION.md` to record what executed and what did not — `back_posture` slipping to the mid-dorsal keypoint is a decision the document does not yet carry.

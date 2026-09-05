## Purpose

Detects animals, maintains tracklets across frames and estimates keypoints behind stable interfaces, so that pretrained weights can be replaced without altering any downstream contract.

## ADDED Requirements

### Requirement: Detection output contract
Detection SHALL emit, for each detected animal in a frame, a bounding region, a class label, a confidence score, and the provenance of the originating frame.

#### Scenario: Detection carries frame provenance
- **WHEN** a frame is processed by detection
- **THEN** each detection references the provenance of that frame, so a detection can be traced to source, camera, day and site without a separate lookup

#### Scenario: Frame with no animals yields an empty result, not an absent one
- **WHEN** a frame contains no detectable animal
- **THEN** an empty detection result is emitted for that frame rather than the frame being omitted from the stream

### Requirement: Tracklet formation
Tracking SHALL group detections across consecutive frames into tracklets. Each tracklet SHALL carry a tracklet identifier, its first and last frame, the ordered sequence of detections, and a termination reason when it ends.

#### Scenario: Tracklet records why it ended
- **WHEN** a tracklet terminates because the animal left the field of view
- **THEN** the tracklet records a termination reason distinguishing exit from occlusion loss and from detection failure

#### Scenario: Tracklet identifiers are unique within a source
- **WHEN** multiple tracklets are formed from one source
- **THEN** no two tracklets in that source share a tracklet identifier

### Requirement: Pose output contract
Pose estimation SHALL emit named keypoints with a per-keypoint confidence and a per-keypoint visibility state, against a declared and versioned skeleton definition.

#### Scenario: Skeleton definition is versioned on output
- **WHEN** pose estimation emits keypoints
- **THEN** the output records the skeleton definition identifier and version used to produce it

#### Scenario: Occluded keypoint is marked, not interpolated
- **WHEN** a keypoint is not observable in a frame
- **THEN** it is emitted with a not-visible state rather than an interpolated coordinate presented as an observation

### Requirement: Model identity recorded on every output
Every detection, tracklet and pose output SHALL record the identity and version of the model that produced it.

#### Scenario: Swapping weights changes recorded identity
- **WHEN** the detector weights are replaced and the same source is reprocessed
- **THEN** the new outputs record the new model identity and version, and outputs from the two runs are distinguishable by that field alone

### Requirement: Explicit degradation
When detection or pose confidence falls below the configured operating threshold, the result SHALL be emitted marked as low confidence rather than silently discarded.

#### Scenario: Low-confidence result remains visible downstream
- **WHEN** a detection scores below the operating threshold
- **THEN** it is emitted with a low-confidence mark, and the count of such results is reported for the run

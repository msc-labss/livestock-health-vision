## Purpose

Converts a single lane pass into named locomotion features with explicit per-feature quality, so that health inference consumes measurements with known reliability rather than raw pose output.

## ADDED Requirements

### Requirement: Pass segmentation
The system SHALL segment a tracklet into discrete lane passes with a defined entry and exit, and SHALL classify a pass that lacks either boundary as partial.

#### Scenario: Complete pass is segmented
- **WHEN** a tracklet crosses the configured entry boundary and later the exit boundary
- **THEN** one complete pass is emitted spanning those two crossings

#### Scenario: Partial pass is labelled, not extrapolated
- **WHEN** a tracklet begins after the entry boundary or ends before the exit boundary
- **THEN** the pass is emitted labelled partial, and its boundaries are not extrapolated to the missing crossing

### Requirement: Versioned named feature set
Locomotion features SHALL be emitted as a versioned record in which each feature has a declared name and unit.

#### Scenario: Feature record declares its version
- **WHEN** a feature record is emitted
- **THEN** it names the feature-set version used, so records produced under different feature definitions are distinguishable

#### Scenario: Undeclared feature is refused
- **WHEN** an extraction produces a value whose name is not in the declared feature set
- **THEN** extraction fails for that pass and reports the undeclared feature name

### Requirement: Per-feature quality flags
Each emitted feature SHALL carry a quality flag derived from the confidence and coverage of the keypoints it was computed from.

#### Scenario: Feature computed from sparse keypoints is flagged
- **WHEN** a feature depends on a keypoint that was not visible for part of the pass
- **THEN** that feature carries a reduced quality flag naming the limiting keypoint

### Requirement: Explicit pass validity decision
Each pass SHALL receive an overall validity decision with a reason. Only valid passes SHALL enter the per-animal time series; invalid passes SHALL be retained for audit.

#### Scenario: Invalid pass is excluded but retained
- **WHEN** a pass is judged invalid because too many features are of reduced quality
- **THEN** the pass does not enter the time series, and it remains retrievable with its invalidity reason

#### Scenario: Invalid pass does not emit usable features
- **WHEN** a pass is judged invalid
- **THEN** no feature values from that pass are presented to downstream consumers as measurements

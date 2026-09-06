## Purpose

Converts a single lane pass into named locomotion features with explicit per-feature quality, so that health inference consumes measurements with known reliability rather than raw pose output.

## Requirements

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

### Requirement: Features declare the view they are valid under
Each feature in the declared feature set SHALL name the skeleton view under which its definition holds, and extraction SHALL refuse to compute a feature whose declared view differs from the view of the skeleton in use.

A feature computed in image coordinates measures a different physical quantity under a different camera geometry while its name, unit and priors stay the same. Refusing is the only outcome that does not silently substitute one quantity for another.

#### Scenario: Feature from a foreign view is refused, not reinterpreted
- **WHEN** a feature declared valid only under a top-down view is present in a profile whose skeleton declares a lateral view
- **THEN** extraction fails naming the feature, its declared view and the skeleton's view, rather than computing an image-plane quantity whose physical meaning has changed

#### Scenario: Feature valid under the skeleton's view is computed
- **WHEN** a feature's declared view matches the view of the skeleton in use
- **THEN** the feature is computed and its record names the view it was computed under

### Requirement: Declared but unavailable features
The feature set MAY declare a feature that cannot be computed under the current skeleton or backend. Such a feature SHALL be reported as unavailable with a reason naming what is missing, and SHALL NOT be omitted from the record, given a default value, or presented downstream as a measurement.

Declaring a decided feature that cannot yet be computed keeps the decision visible and keeps the reason for its absence recorded, rather than leaving a reader to infer that it was never wanted.

#### Scenario: Unavailable feature is recorded with its reason
- **WHEN** a declared feature depends on a capability the current skeleton or backend does not provide
- **THEN** the feature record carries that feature as unavailable, naming the missing capability

#### Scenario: Unavailable feature yields no value downstream
- **WHEN** a feature is marked unavailable
- **THEN** no numeric value for it is presented to downstream consumers, and no default is substituted

#### Scenario: Source-wide unavailability does not invalidate every pass
- **WHEN** a feature is unavailable for an entire source because of a property of the skeleton or backend rather than of the pass
- **THEN** its unavailability does not count toward the pass validity decision, so passes are not rejected for a property of the configuration

### Requirement: Prior provenance is recorded per feature
Each feature's normal-range prior SHALL record the source it came from, and a prior derived from published group statistics SHALL be recorded as an anchor with a direction rather than as a population range.

A group mean with a standard error describes where a group sat, not how far a population spreads. Recording one as a low/high band would produce a plausibility check that is confidently wrong.

#### Scenario: Measured anchor is distinguishable from a guess
- **WHEN** a report or profile listing presents a feature's prior
- **THEN** the prior names its source, and a prior sourced to published group statistics is distinguishable from one recorded as unvalidated

#### Scenario: Group statistic is not presented as a range
- **WHEN** a prior is derived from a published group mean
- **THEN** it is recorded as an anchor value with the direction lameness moves it, and not as a low and high bound

## Purpose

Maintains a per-animal phenotype time series and produces deviation-based risk scores against both the animal's own history and the contemporaneous herd, with explicit uncertainty and an explicit cold-start state.

## Requirements

### Requirement: Time series append
Valid passes SHALL be appended to a per-animal time series keyed by animal identity and observation timestamp.

#### Scenario: Valid pass is appended
- **WHEN** a pass is valid and its identity is resolved
- **THEN** its feature record is appended to that animal's time series at the pass timestamp

#### Scenario: Unresolved identity does not enter any series
- **WHEN** a valid pass has an unresolved identity
- **THEN** no time series is modified, and the pass is recorded in an unattributed store

### Requirement: Own-history baseline
The system SHALL compute a baseline for each animal from that animal's prior observations over a declared lookback window, excluding the observation being scored.

#### Scenario: Baseline excludes the observation under test
- **WHEN** an observation is scored against the animal's own history
- **THEN** the baseline is computed from prior observations only, and the scored observation does not contribute to its own baseline

### Requirement: Herd baseline
The system SHALL compute a contemporaneous herd baseline over a declared population and window, so an animal's deviation can be separated from a herd-wide shift.

#### Scenario: Herd-wide shift is distinguishable from individual deviation
- **WHEN** every animal's feature values shift together within a window
- **THEN** own-history deviation is elevated while herd-relative deviation is not, and both are reported

### Requirement: Explicit cold-start state
An animal with fewer prior observations than the declared minimum SHALL produce an explicit insufficient-history state. A risk score SHALL NOT be emitted for such an animal.

#### Scenario: New animal yields insufficient history, not a low score
- **WHEN** an animal has fewer observations than the configured minimum
- **THEN** the output is an insufficient-history state naming the shortfall, and no numeric risk score is produced

### Requirement: Risk score carries uncertainty and its inputs
An emitted risk score SHALL carry an uncertainty measure and SHALL name the baselines and window definitions it was computed from.

#### Scenario: Score is reproducible from its recorded inputs
- **WHEN** a risk score is emitted
- **THEN** it names the own-history baseline, the herd baseline and the windows used, sufficient to recompute the score from the stored series

### Requirement: Anomaly injection is supported and marked
The system SHALL support injecting synthetic deviations into a time series for evaluation, and SHALL mark every downstream output derived from injected data.

#### Scenario: Injected deviation is detectable end to end
- **WHEN** a synthetic deviation of declared magnitude is injected into an animal's series
- **THEN** the resulting risk score reflects it, and the score is marked as derived from injected data

#### Scenario: Injected data cannot be mistaken for observation
- **WHEN** a report includes any output derived from injected data
- **THEN** those outputs are marked, and the report states that injected data was present

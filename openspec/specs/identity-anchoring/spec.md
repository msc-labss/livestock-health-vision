## Purpose

Resolves each tracklet to a stable animal identity, preferring an external identifier over visual re-identification, so that longitudinal baselines are not built on the least reliable link in the chain.

## Requirements

### Requirement: External anchor is the primary path
When an external animal identifier is available for the observation window of a tracklet, identity anchoring SHALL use it and SHALL record the anchor source.

#### Scenario: External identifier resolves the tracklet
- **WHEN** a tracklet's observation window matches an external identifier record
- **THEN** the tracklet is assigned that animal identity, with the assignment method recorded as external anchor and the anchor source named

#### Scenario: Ambiguous external match is not resolved by guessing
- **WHEN** a tracklet's observation window matches more than one external identifier record
- **THEN** the assignment is left unresolved and the competing candidates are recorded

### Requirement: Visual fallback is marked as fallback
When no external identifier is available, identity MAY be assigned by visual re-identification, and such an assignment SHALL be marked as a fallback assignment carrying its own confidence.

#### Scenario: Fallback assignment is distinguishable
- **WHEN** identity is assigned by visual re-identification
- **THEN** the assignment records method as visual fallback with a confidence value, and consumers can filter on that method

### Requirement: Unresolved identity is a valid outcome
Identity anchoring SHALL emit an explicit unresolved state when neither path yields an assignment above the configured confidence floor. It SHALL NOT assign an arbitrary or provisional identity.

#### Scenario: No confident assignment yields unresolved
- **WHEN** no external identifier is available and visual confidence is below the floor
- **THEN** the tracklet is marked unresolved and is excluded from per-animal time series while remaining available for audit

### Requirement: Assignment provenance
Every identity assignment SHALL carry the method used, the confidence, the time of assignment, and a reference to the evidence it was derived from.

#### Scenario: Assignment can be audited after the fact
- **WHEN** an assignment is later reviewed
- **THEN** its method, confidence, assignment time and evidence reference are retrievable without reprocessing the video

### Requirement: Identity conflict is surfaced
When two tracklets overlapping in time are assigned the same animal identity, the conflict SHALL be flagged and both assignments SHALL be withheld from per-animal time series until resolved.

#### Scenario: Concurrent duplicate identity is flagged
- **WHEN** two tracklets whose observation windows overlap resolve to the same animal identity
- **THEN** a conflict is recorded naming both tracklets, and neither contributes to that animal's time series

## Purpose

Turns recorded video into a deterministic, resumable frame stream in which every frame carries the provenance and split keys that downstream stages and the evaluation harness depend on, so no later stage has to infer where a frame came from.

## Requirements

### Requirement: Frame provenance
Every frame emitted by ingest SHALL carry a provenance record containing source identifier, camera identifier, capture timestamp, frame index within the source, and the split keys used for evaluation: animal set, calendar day, and site.

#### Scenario: Frame carries complete provenance
- **WHEN** a registered source is ingested
- **THEN** each emitted frame carries source identifier, camera identifier, capture timestamp, frame index, animal-set key, day key and site key

#### Scenario: Incomplete provenance is rejected at registration
- **WHEN** a source is registered without a site key or camera identifier
- **THEN** registration fails with an error naming the missing field, and no frames are emitted from that source

### Requirement: Deterministic iteration
Given the same source material and the same ingest configuration, ingest SHALL emit an identical sequence of frames with identical provenance records.

#### Scenario: Repeated ingest is byte-identical in metadata
- **WHEN** the same source is ingested twice with the same configuration
- **THEN** the two runs emit the same number of frames in the same order with equal provenance records

### Requirement: Resumable iteration
Ingest SHALL support resuming from a recorded position so that an interrupted run continues without re-emitting already-processed frames.

#### Scenario: Resume after interruption
- **WHEN** an ingest run is interrupted after emitting N frames and is then resumed from its recorded position
- **THEN** the resumed run emits frame N+1 onward and does not re-emit the first N frames

### Requirement: Timestamp integrity
Ingest SHALL NOT substitute a default value for a missing or implausible capture timestamp. Such frames SHALL be emitted with the timestamp marked unreliable, or excluded, according to configuration.

#### Scenario: Missing timestamp is flagged rather than defaulted
- **WHEN** a frame has no recoverable capture timestamp
- **THEN** the frame's provenance marks the timestamp unreliable and downstream stages can filter on that mark

#### Scenario: Non-monotonic timestamps are reported
- **WHEN** capture timestamps within a single source decrease between consecutive frames
- **THEN** ingest reports the anomaly with the affected frame indices

### Requirement: Derived media isolation
Ingest SHALL write decoded frames, clips and any other derived media only beneath configured output roots that are excluded from version control.

#### Scenario: Output root outside the excluded area is refused
- **WHEN** ingest is configured with an output root that is tracked by version control
- **THEN** ingest fails before writing any media and reports the offending path

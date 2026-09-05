## Purpose

Constructs leakage-free evaluation splits and produces reproducible reports that keep perception, phenotype and operational metrics separate, with the change's declared limitations carried into every report.

## Requirements

### Requirement: Split construction from provenance
The harness SHALL construct frame-level, animal-disjoint, day-disjoint and site-disjoint splits using the split keys carried in frame provenance.

#### Scenario: Animal-disjoint split separates animals
- **WHEN** an animal-disjoint split is requested
- **THEN** no animal identity appears in both the training and the test partition

#### Scenario: Site-disjoint split separates sites
- **WHEN** a site-disjoint split is requested
- **THEN** no site key appears in both partitions

### Requirement: Leakage is a failure, not a warning
When a requested split cannot be satisfied without overlap on its disjointness key, the harness SHALL fail and report the overlapping keys. It SHALL NOT emit a report from a leaking split.

#### Scenario: Overlapping animals abort the run
- **WHEN** an animal-disjoint split is requested and an animal appears in both partitions
- **THEN** the harness fails naming the overlapping animal identities, and no metrics are produced

### Requirement: Separated metric families
The harness SHALL report perception, phenotype and operational metrics as separate families and SHALL NOT combine them into a single headline figure.

#### Scenario: Report keeps families separate
- **WHEN** a report is produced
- **THEN** perception, phenotype and operational metrics appear under distinct sections with no aggregate score across families

### Requirement: Operational metrics are first class
Where the evaluated configuration produces alerts, the harness SHALL report alarm burden expressed per one thousand animal-days and detection lead time relative to the reference event.

#### Scenario: Alarm burden is reported with the threshold that produced it
- **WHEN** an alerting configuration is evaluated
- **THEN** the report states alarm burden per thousand animal-days alongside the threshold policy identity used

#### Scenario: Lead time is reported against a stated reference
- **WHEN** lead time is reported
- **THEN** the report names the reference event that lead time is measured against

### Requirement: Reproducible report
Every report SHALL record the dataset version, model identities and versions, split definition and configuration digest, and the same inputs SHALL produce the same report.

#### Scenario: Report is reproducible from its recorded inputs
- **WHEN** a report is regenerated from the recorded dataset version, model identities, split definition and configuration digest
- **THEN** the regenerated report matches the original

### Requirement: Declared limitations appear in the report
Every report SHALL surface the limitations that apply to it, including stubbed inference and the number of distinct sites the evaluation covers.

#### Scenario: Stubbed inference is stated in the report
- **WHEN** a report covers a configuration whose health inference is stubbed
- **THEN** the report states that health inference is stubbed and that its results are not clinical evidence

#### Scenario: Single-site evaluation is stated as such
- **WHEN** an evaluation covers only one site
- **THEN** the report states the site count and notes that site-disjoint validation was not exercised

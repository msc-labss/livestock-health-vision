## MODIFIED Requirements

### Requirement: Declared limitations appear in the report
Every report SHALL surface the limitations that apply to it, including stubbed inference, the number of distinct sites the evaluation covers, and any metric family whose output came from weights the profile declares as a placeholder.

#### Scenario: Stubbed inference is stated in the report
- **WHEN** a report covers a configuration whose health inference is stubbed
- **THEN** the report states that health inference is stubbed and that its results are not clinical evidence

#### Scenario: Single-site evaluation is stated as such
- **WHEN** an evaluation covers only one site
- **THEN** the report states the site count and notes that site-disjoint validation was not exercised

#### Scenario: Placeholder weights are stated in the report
- **WHEN** a metric family's output came from weights the profile declares as a placeholder
- **THEN** the report names those weights and states that the family's metrics measure the placeholder rather than an achievable result

## ADDED Requirements

### Requirement: Skeleton view must match the source view
Pose estimation SHALL refuse to emit keypoints when the skeleton definition's declared view disagrees with the view declared for the source. A skeleton's view determines what its keypoints mean spatially, so pose emitted against a mismatched view is a measurement of the wrong quantity rather than a noisy measurement of the right one.

#### Scenario: Mismatched view aborts rather than emitting keypoints
- **WHEN** a source whose declared view is lateral is processed with a profile whose skeleton declares a top-down view
- **THEN** pose estimation fails for that source, naming both the skeleton's view and the source's view, and no keypoints are emitted

#### Scenario: Undeclared view is refused rather than assumed
- **WHEN** a source carries no declared view
- **THEN** pose estimation fails naming that source, so that recording a view is an explicit act rather than a default

#### Scenario: Matching view proceeds and is recorded
- **WHEN** the skeleton's declared view matches the source's declared view
- **THEN** pose estimation proceeds and the emitted records name the view they were produced under

### Requirement: Placeholder weights are identifiable on their output
Weights that the profile declares as a placeholder — not trained for the task, the species, or the skeleton they are mapped onto — SHALL be recorded as placeholder on every output they produce, so that no consumer can treat their values as model output without being told otherwise.

#### Scenario: Placeholder pose output carries the mark
- **WHEN** pose output is produced by weights the profile declares as a placeholder
- **THEN** each emitted pose record carries a placeholder mark naming the weights it came from

#### Scenario: Replacing placeholder weights removes the mark
- **WHEN** placeholder weights are replaced with weights not declared as a placeholder and the same source is reprocessed
- **THEN** the new outputs carry no placeholder mark, and outputs from the two runs are distinguishable by that field alone

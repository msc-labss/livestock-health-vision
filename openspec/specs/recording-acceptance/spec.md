## Purpose

Judges whether a candidate recording can support the measurements the species profile declares, before a study commits to it. A requirement this stage checks makes some measurement impossible rather than merely noisy, so its verdicts decide whether footage is worth gathering at all — and a verdict it could not actually reach must say so rather than read as a pass.

## Requirements

### Requirement: A check that could not be run reports unknown
An acceptance check SHALL report a pass only when it completed the test the specification states for it. A check that could not be attempted, or that could attempt only part of its test, SHALL report unknown and name what it could not do.

A check that quietly passes when it could not run is worse than no check, because a reader takes a verdict at face value and a missing check announces itself.

#### Scenario: A partially attempted check does not report a pass
- **WHEN** a requirement's test has more than one part and only some parts could be attempted
- **THEN** the check reports unknown, names the part it could not attempt, and reports the results of the parts it did

#### Scenario: An unattempted check is distinguishable from a failing one
- **WHEN** a check could not be attempted at all
- **THEN** its verdict is unknown rather than fail, and the report states separately how many requirements failed and how many could not be judged

### Requirement: Every threshold names its source
Each threshold an acceptance check applies SHALL record where it came from, and the report SHALL name the source of any threshold that decided a verdict.

A number with no stated origin cannot be revised with confidence, because nothing says what would justify changing it.

#### Scenario: A verdict names the bound that produced it
- **WHEN** a check passes or fails against a threshold
- **THEN** the report names the threshold's value and its source

#### Scenario: A profile requirement stricter than the specification's takes precedence
- **WHEN** the species profile declares a sampling requirement for an available feature that exceeds the recording specification's floor
- **THEN** the stricter bound is applied and the report names which of the two it used

### Requirement: The analysis covers the whole recording
Checks derived from the frames SHALL be computed over material sampled across the whole recording, not over a single leading segment.

Sampling must be of contiguous windows, because tracklet formation needs continuity, and those windows must be spread across the duration, because occupancy of a lane is not uniform and an empty opening minute is not evidence about the recording.

#### Scenario: A recording whose opening is empty is still judged
- **WHEN** a recording contains no animals in its first analysed window but does later
- **THEN** the frame-derived checks are computed from the later material rather than reporting that nothing was detected

#### Scenario: The report states how much was analysed
- **WHEN** frame-derived checks are reported
- **THEN** the report states the duration analysed and the duration of the recording

### Requirement: Frame rate is judged constant as well as sufficient
The frame-rate check SHALL judge both that the rate meets the required floor and that it is constant, and SHALL report the two separately.

A rate reported as an average says nothing about whether the interval between frames is stable, and a gait measurement made across an unstable interval is timed against an assumption rather than a clock.

#### Scenario: A variable frame rate fails even when its average is sufficient
- **WHEN** a recording's average frame rate meets the floor but its frame rate is not constant
- **THEN** the frame-rate check does not report a pass, and names variability rather than rate as the cause

#### Scenario: Constancy that cannot be established is not assumed
- **WHEN** constancy cannot be determined from the container
- **THEN** that part of the check reports unknown rather than assuming a constant rate

### Requirement: Traversal is measured as displacement
The transit check SHALL measure how far an animal travelled through the measured section, not how long its tracklet persisted.

An animal that halts in the lane persists in frame indefinitely and produces no strides; a tracklet that fragments produces a short persistence from a complete traverse. Neither is what the requirement is about.

#### Scenario: A stationary animal does not satisfy the transit requirement
- **WHEN** a tracklet persists for longer than the required duration without displacing along the direction of travel
- **THEN** it is not counted as a satisfying transit, and the report distinguishes persistence from traversal

#### Scenario: Transit is reported with the displacement it rests on
- **WHEN** the transit check reports a duration
- **THEN** it also reports the displacement over which that duration was measured

### Requirement: A verdict names the configuration it was reached under
Every acceptance report SHALL name the species profile, feature-set version and camera view it judged against.

The same recording is acceptable against one geometry and not another, so a verdict without its configuration is not interpretable later.

#### Scenario: The report identifies what it judged against
- **WHEN** an acceptance report is produced
- **THEN** it names the profile identifier, the feature-set version and the view the checks assumed

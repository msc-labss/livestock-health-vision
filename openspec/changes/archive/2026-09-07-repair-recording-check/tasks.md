## 1. Thresholds carry their provenance

- [x] 1.1 Give each entry in `Requirements` a source string, and remove the frame-rate comment that reasons about the stride-frequency prior feature-set version 1 retired.
- [x] 1.2 Take the effective frame-rate floor as the maximum of the specification's figure and the highest `requires_sampling_hz` among the profile's **available** features; an unavailable feature's requirement describes a measurement this configuration cannot make.
- [x] 1.3 Name the applied bound and its source in every check that a threshold decided.
- [x] 1.4 Tests: the specification's floor applies when the profile declares nothing; a stricter profile requirement wins and is named; an unavailable feature's requirement is ignored.

## 2. Incompleteness is visible

- [x] 2.1 Give `Check` an explicit notion of a partially attempted test, so a requirement whose test has several parts can report which ran.
- [x] 2.2 R6 reports unknown rather than pass: the container timestamp is one half of its test and the clock-offset comparison against the identity feed is the other, and the tool cannot attempt the second.
- [x] 2.3 Report failed and unjudged counts separately in `describe`, and make the exit status reflect only failures.
- [x] 2.4 Tests: a partially attempted check never reports a pass and names the missing part; an unattempted check is unknown, not fail.

## 3. Coverage of the recording

- [x] 3.1 Replace the single leading window with several contiguous windows spread evenly across the duration, keeping frames within a window consecutive so tracklets still form.
- [x] 3.2 Report the duration analysed alongside the duration of the recording.
- [x] 3.3 Tests: a recording whose opening window is empty but which contains animals later is still judged from the later material.

## 4. R2 judges constancy

- [x] 4.1 Read the container's average and nominal frame rates via `ffprobe` and treat a disagreement as variable rate.
- [x] 4.2 Report rate and constancy as separate parts of R2, so a variable-rate file with a sufficient average does not pass.
- [x] 4.3 Report constancy unknown where `ffprobe` is unavailable or the container carries only one of the two rates.
- [x] 4.4 Tests: sufficient but variable does not pass and names variability; undeterminable constancy is unknown, not assumed.

## 5. R4 measures traversal

- [x] 5.1 Measure a transit as displacement along the principal direction of travel across the measured section, not as tracklet persistence.
- [x] 5.2 Report the displacement the duration was measured over.
- [x] 5.3 Report a tracklet that persists without displacing as a stalled tracklet, separately from transits — it is a finding about the lane, not a measurement to discard.
- [x] 5.4 Tests: a stationary tracklet does not satisfy the transit requirement however long it persists.

## 6. R6 gains the filename fallback

- [x] 6.1 Read a capture time from the filename when the container carries none, as the recording specification already permits.
- [x] 6.2 Name which source the time came from, so a container time and a filename time are distinguishable in the report.
- [x] 6.3 Tests: a file with a parseable name and no container tag yields a time and names its source.

## 7. The verdict names its configuration

- [x] 7.1 Record the profile identifier, feature-set version and assumed view on the report, and print them.
- [x] 7.2 Tests: the report names all three.

## 8. Close the change

- [x] 8.1 Amend the acceptance procedure in `docs/P1-RECORDING-SPECIFICATION.md` to describe what the tool now reports, including that R6 is only half checkable without the identity feed.
- [x] 8.2 Run `./tools/check.sh`.
- [x] 8.3 Confirm `lhv recording-check` runs end to end on a synthetic file and that its report names every threshold source.

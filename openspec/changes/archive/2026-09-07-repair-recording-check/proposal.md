## Why

`lhv recording-check` is what the pilot acceptance procedure runs on, and that
procedure is the first thing that happens when a cooperating dairy is found. It
is also the only stage in this project with no specification, and it has drifted
exactly as far as that would predict.

It now judges a recording against a stride-frequency prior the profile no longer
declares — feature-set version 1 retired both the band and the
`requires_sampling_hz` mechanism that read it — so its frame-rate check reasons
about something that does not exist. It reads the first sixty seconds of the
one-hour recording the specification asks for. And it reports two requirements
as met while having checked half of each.

The module's own docstring states the standard it is failing: *a check that
quietly passes when it could not run is worse than no check*. This change gives
it a specification and brings it back to that standard.

## What Changes

- **New capability `recording-acceptance`.** The stage that decides whether a
  candidate recording can support measurement at all has never had one, and its
  drift is the argument for it.
- Every threshold names where it came from. The frame-rate floor belongs to the
  recording specification, not to the species profile; where a profile declares
  a sampling requirement for an available feature, the stricter of the two
  applies and the report says which bound it used.
- **BREAKING for callers reading the report as text**: a check that could not be
  completed reports unknown rather than pass. R6 currently passes on the
  container timestamp alone while its clock-offset half is never attempted.
- The analysis covers the whole recording rather than its first minute, by
  sampling contiguous windows spread across the duration — contiguous because
  tracklet formation needs continuity, spread because a lane is often empty at
  the top of the hour.
- R6 gains the filename fallback the specification already allows.
- R2 judges the frame rate constant as well as sufficient. The specification
  says "constant" and nothing checks it; a variable-rate file passes today.
- R4 measures traversal as displacement along the direction of travel rather
  than as tracklet persistence. An animal that stops in the race currently
  scores well and yields no strides at all.
- The report names the profile, feature-set version and view it judged against,
  so a verdict can be read back against the configuration that produced it.

## Capabilities

### New Capabilities

- `recording-acceptance`: judging whether a candidate recording can support the
  measurements the profile declares, before a study commits to it. Covers
  threshold provenance, the incompleteness rule, coverage of the recording, and
  what a verdict must carry.

### Modified Capabilities

None. This is adjacent to `video-ingest` but distinct from it: ingest turns a
recording into a frame stream, whereas this decides whether the recording should
be made at all.

## Impact

- `src/lhv/recording.py` — thresholds, their provenance, the sampling window,
  and the R2, R4 and R6 checks.
- `src/lhv/cli.py` — the `recording-check` command's reporting.
- `tests/test_recording.py`. **Correction to this proposal as first written:** it
  claimed the module had no tests. It has seventeen, and two of them encoded the
  behaviour this change corrects — one asserted the frame-rate note reasoning
  about the retired stride band, the other looked for a check under its old
  name. Both are updated rather than deleted, and sixteen more are added for the
  behaviour the specification now states.
- `docs/P1-RECORDING-SPECIFICATION.md` — the acceptance procedure describes what
  the tool reports, and that description changes.

### Out of scope

- **Moving the frame-rate figure itself.** L5 argues 15 fps is too low and 25 is
  the floor, but that rests on a capture rate in Kang et al. 2020 that this
  project has not been able to read. The number stays where the recording
  specification puts it; this change makes it a number with a stated source
  rather than a magic constant, so moving it later is an edit in one place.
- **R4's three-body-length requirement.** The same paper challenges it, for the
  same unread reason.

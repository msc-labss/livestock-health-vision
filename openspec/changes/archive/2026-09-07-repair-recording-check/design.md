## Context

`lhv recording-check` judges a candidate recording before a study commits to it.
It was written alongside `docs/P1-RECORDING-SPECIFICATION.md` and never given a
specification of its own, which is why it is the one stage that drifted when the
profile changed underneath it.

Three drifts are now visible. Its frame-rate threshold carries the comment
*"samples per stride at the fast end of the profile's prior band"*, and
feature-set version 1 retired both that band and the `requires_sampling_hz`
mechanism that read it, so the rationale points at nothing. Its analysis window
is sixty seconds against an acceptance procedure that asks for an hour. And its
R6 check reports a pass on the container timestamp while the clock-offset half
of that requirement's test is never attempted.

The module's own docstring already states the standard: *"It reports what it
measured alongside the verdict, and says 'unknown' where it cannot tell, because
a check that quietly passes when it could not run is worse than no check."* The
code does not meet it. This change is mostly a matter of making it do so.

## Goals / Non-Goals

**Goals:**

- Give the acceptance stage a specification, so the next profile change cannot
  silently invalidate it.
- Make incompleteness visible: a check reports a pass only when it ran.
- Make thresholds traceable, so the ones the literature disputes can be moved in
  one place when the evidence arrives.
- Judge the recording rather than its first minute.

**Non-Goals:**

- Moving the frame-rate floor or R4's field-of-view requirement. Both are
  disputed by a paper this project has not been able to read, and both stay
  where the recording specification puts them. This change makes them
  attributable, not different.
- Adding checks the specification does not ask for. Exposure and motion blur are
  a real gap, recorded under L9 as *no usable evidence found*, and inventing a
  threshold for them here would be exactly the failure the rest of this project
  has been avoiding.

## Decisions

### Where the frame-rate floor lives

**Decision.** The floor belongs to the recording specification, and the profile
may raise it but not lower it. `Requirements` records each threshold with a
source string; the frame-rate entry names the recording specification. Before
judging, the check takes the maximum of that floor and the highest
`requires_sampling_hz` among the profile's *available* features, and the report
names whichever bound it applied.

**Alternative considered:** deriving the floor entirely from the profile, which
is stale-proof but wrong. Feature-set version 1 declares no sampling requirement
at all, so a purely derived floor would be zero and the check would pass any
recording — a silent regression dressed as principle. The specification is the
authority for what a study needs; the profile is the authority for what a
feature needs; the stricter wins.

Note that only *available* features count. An unavailable feature's sampling
requirement describes a measurement this configuration cannot make, and holding
a recording to it would reject footage for a property of the backend.

### Sampling across the recording

**Decision.** Analyse several contiguous windows spread evenly across the
duration, rather than one leading window.

Contiguous because tracklet formation needs consecutive frames, so a sparse
frame-level sample would destroy the transit and one-animal-at-a-time checks.
Spread because lane occupancy is not uniform: an empty opening minute is a fact
about when recording started, not about the recording.

**Alternative considered:** analysing every frame. Rejected on cost — an hour at
25 fps is 90,000 frames through a detector, for a check whose purpose is to be
run early and often while a mounting is being adjusted.

### Distinguishing persistence from traversal

**Decision.** A transit is a tracklet whose displacement along the principal
direction of travel spans the measured section. Duration is reported for that
displacement, and persistence without displacement is reported separately as a
stalled tracklet.

This is the same distinction the phenotype stage already draws between a
complete and a partial pass, and it should have been drawn here from the start.
A stalled animal is worth reporting rather than discarding: it says the lane
design lets animals stop, which is a finding about the site.

### Constancy

**Decision.** Compare the container's average and nominal frame rates via
`ffprobe`; a disagreement means variable rate. Where `ffprobe` is unavailable or
the container does not carry both, report that half unknown.

This is a weaker test than examining packet timestamps and it is the one
available without decoding the file. Its limits are stated in the report rather
than glossed, which is the rule this change exists to enforce.

## Risks / Trade-offs

- **More checks now report unknown than before.** → That is the correction, not
  a regression. A reader who previously saw "R6 ok" was being told the
  requirement was met when half its test had not been run.

- **The sampling windows could miss a rare event.** → They can, and the report
  states the fraction of the recording analysed so a reader can judge that. The
  alternative, a leading window, misses far more and says nothing about it.

- **`ffprobe` is an external dependency for the constancy test.** → Already true
  for the timestamp check, and its absence is already handled by reporting the
  reason rather than failing. The new test follows the same path.

- **Giving this stage a capability adds a spec to maintain.** → It is the stage
  whose lack of one allowed a threshold's rationale to outlive the thing it
  referred to. The maintenance is the point.

## Migration Plan

1. The specification, the thresholds and their sources, and the checks land
   together — a threshold that names its source is only useful if the report
   prints it.
2. No stored artefact changes format. `recording-check` writes nothing durable;
   it prints a verdict and exits with a status.
3. The acceptance procedure in `docs/P1-RECORDING-SPECIFICATION.md` describes
   what the tool reports, so it is amended in the same change.

## Open Questions

- Does Kang et al. 2020 move the frame-rate floor to 25, and R4's field of view
  below three body lengths? Both are held, and both are now single-value edits.
- Should a stalled tracklet fail R4 outright, or be reported as a distinct
  finding about the lane? This change reports it; whether it should also fail is
  a question the first real pilot recording will answer better than reasoning
  will.

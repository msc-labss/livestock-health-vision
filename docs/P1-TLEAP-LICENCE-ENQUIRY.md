# T-LEAP trajectory release: licence enquiry

The outreach behind outstanding items 2 and 4 of
[docs/P1-LITERATURE-FINDINGS.md](P1-LITERATURE-FINDINGS.md), recorded here for
the same reason the literature prompt is: an answer is only as traceable as the
question that produced it.

**Status: sent 2026-09-07** to both recipients, Gmail message id
`1a07895a196026d3`. No reply yet. Record the reply here when it arrives, or
record its absence: a question that went unanswered is a finding too, and the
decisions below stay open either way.

## Recipients

| | |
|---|---|
| helena.russello@wur.nl | Helena Russello, Agricultural Biosystems Engineering, Wageningen University & Research |
| gert.kootstra@wur.nl | Gert Kootstra, same group |

Both are named corresponding authors with these addresses printed in the arXiv
HTML of *Video-based automatic lameness detection of dairy cows using pose
estimation and multiple locomotion traits* (arXiv:2401.05202). Taken from the
paper itself rather than inferred from an institutional naming convention,
because a guessed address sends this to a stranger.

## What it unblocks

- **The licence question.** `github.com/hrussel/lstm-lameness-detection` holds
  272 T-LEAP keypoint trajectories with per-trajectory lameness scores and
  states no licence for code or data. That release is the only identified route
  to testing feature-to-score correlation — P1's gate, in its cross-sectional
  form — before any farm recording exists. This project records the licence and
  access route of every dataset it uses, so it cannot be registered until the
  terms are known.
- **The keypoint count.** The literature review reports T-LEAP/CoWalk as 17
  landmarks; Russello et al. 2024 states that T-LEAP extracted motion from nine.
  The difference is roughly half the labelling bill, and it sets whether the
  provisional nine-point skeleton in `cattle.yaml` is right. One line from the
  authors settles what would otherwise cost a library trip.

## The message

**Subject:** Licence for the keypoint trajectories in lstm-lameness-detection

---

Dear Dr Russello and Dr Kootstra,

I am working on an MSc project on camera-based lameness detection in dairy
cattle. Your work has shaped several of its decisions — the locomotion traits in
your 2024 paper are the basis of the feature set I am building, in preference to
the top-down features I started with.

I have two short questions.

First, the repository github.com/hrussel/lstm-lameness-detection provides 272
T-LEAP keypoint trajectories together with their lameness scores, but I could
not find a licence statement for either the code or the data. Would you be able
to say under what terms the trajectory data may be used? I would like to use it
for method development and validation in an academic project, citing the
relevant papers, before any farm recording of my own exists. My project records
the licence and access route of every dataset it uses, so I would rather ask
than assume.

Second, a small factual point I could not settle from the papers available to
me: the 2024 paper states that T-LEAP extracted motion from nine keypoints.
Could you confirm how many keypoints the released model predicts, and which
anatomical landmarks they are? It determines how much keypoint labelling my own
recording will need, so it is a useful number to have right before committing to
a mounting.

Thank you for making the trajectories public — they are the only cattle data I
have found that pairs pose with a scored clinical reference, and their
availability makes a real difference to what a project at this scale can
validate.

With thanks,

Muhammed Said Çakır

---

No affiliation line is included. The body already establishes that this is an
MSc project, and naming an institution was left to the sender rather than
assumed. Add one under the name if you want it there.

## When a reply arrives

The licence answer goes to `src/lhv/datasets/registrations/` as a registration
in the same form as the existing two, or is recorded as a refusal. The keypoint
answer goes to the skeleton in `src/lhv/profiles/definitions/cattle.yaml`, which
is marked provisional pending exactly this. Both then update their rows in
`docs/P1-LITERATURE-FINDINGS.md`.

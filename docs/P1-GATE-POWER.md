# P1 gate: how many animals, and one correction to the gate itself

[docs/P1-LITERATURE-FINDINGS.md](P1-LITERATURE-FINDINGS.md) established that a
single human locomotion score is too noisy to be P1's reference, and the gate
was restated as *the system agrees with a scorer consensus as well as an
individual scorer does*. That is a non-inferiority claim. Non-inferiority claims
need a sample size decided before the study, and this is that arithmetic.

It is arithmetic under stated assumptions, not a measurement. Every assumption
is a named constant in [tools/gate_power.py](../tools/gate_power.py), so
changing one and re-running is a minute's work rather than an argument.

---

## The correction, which matters more than the numbers

**The gate as first restated is unpassable, and a system that measured
locomotion perfectly would fail it.**

The problem is that the scorers are inside the consensus and the system is not.
A scorer agrees with a reference they helped form far better than an outsider
does, purely by having voted in it. Simulated at 20% prevalence with scorers as
noisy as Thomsen et al. measured:

| comparison | Cohen's kappa | Gwet's AC1 |
|---|---|---|
| a scorer against the consensus **they helped form** | 0.739 | 0.859 |
| the same scorer against the consensus of **the others** | 0.473 | 0.762 |
| a model exactly as good, against the consensus | 0.547 | 0.756 |
| **bias against the model** | **−0.192** | **−0.103** |

A bias of 0.19 on kappa exceeds any plausible non-inferiority margin. The gate
would have rejected a perfect system, which is the same failure L6 was written
to fix, arriving from the opposite direction: the old gate could not be failed
and the new one could not be passed.

**The fix is a matched comparison.** Hold each scorer out in turn and compare
them against the consensus of the others; compare the system against that same
consensus, on the same animals. Animals whose remaining scorers disagree carry
no reference and are dropped from both sides equally. With that, the measured
bias falls to −0.002 on kappa and −0.001 on AC1 — a system as good as a scorer
now scores like one.

This has to appear in R8, because a scoring protocol that does not say the
comparison is leave-one-out will not produce a passable gate.

---

## The second correction: the gate must name its statistic

Kappa and AC1 disagree by more than a factor of four about how many animals are
needed, and they disagree about which herd is easier to study.

Animals required for 80% power, one-sided 95%, three blinded scorers, scorers
agreeing at kappa 0.50:

### Cohen's kappa

| prevalence | margin 0.05 | margin 0.10 | margin 0.15 |
|---|---|---|---|
| 10% | >480 | >480 | >480 |
| 15% | >480 | >480 | 480 |
| 20% | >480 | >480 | 480 |
| 30% | >480 | >480 | 320 |

### Gwet's AC1

| prevalence | margin 0.05 | margin 0.10 | margin 0.15 |
|---|---|---|---|
| 10% | 320 | **80** | 40 |
| 15% | 480 | **120** | 60 |
| 20% | >480 | **160** | 80 |
| 30% | >480 | **320** | 160 |

Two things fall out of the shape.

**Under kappa the study is barely feasible on one farm.** Kappa's
chance-agreement term grows as the marginals become unbalanced, so it is
depressed and noisy exactly where lameness prevalence sits. At any margin
tighter than 0.15 no herd size in the range reaches power.

**Under AC1 the study is an ordinary dairy herd.** 80 to 160 animals at a 0.10
margin, which a single cooperating farm can supply. This is the statistic Wilson
et al. 2026 chose, for the same reason.

**They disagree about which farm to prefer.** Under AC1 a *lower*-prevalence
herd needs fewer animals; under kappa a higher-prevalence herd needs fewer.
A gate that does not name its statistic therefore does not determine its own
sample size, and could not say which candidate farm is the better one.

---

## What to ask a farm for

On AC1 with a 0.10 margin, and reading the prevalence a farm reports from its
own records:

> **80 to 160 cows passing the lane, scored by two trained scorers plus an
> adjudicator on disagreement, with the comparison made leave-one-out.**

That is a materially more answerable request than "we need a cooperating dairy",
and it is the reason this calculation was worth doing before one was found
rather than after.

---

## What this does not establish

- **It is a simulation, not a measurement.** The scorers are a latent-severity
  model calibrated to reproduce one published kappa. Real scorers are correlated
  in ways this does not capture — two people trained together share biases, and
  shared bias inflates their agreement without improving the reference.
- **Binary, not ordinal.** Lame against sound, which is what the agreement
  literature reports. The Sprecher 1–5 scale the profile records would need
  weighted statistics, and the numbers here would move.
- **The system is simulated as exactly as good as a scorer.** The question asked
  is whether a study of a given size could *demonstrate* non-inferiority when it
  is true — not whether it is true.
- **Prevalence is the input the project has least hold of.** The 7–20% range is
  from Anagnostopoulos et al. 2023 across three farms; Wilson et al. 2026 had
  ten observers estimate 36.2–57.0% on the same 207 cows, which is a fact about
  observers rather than about herds. Whichever farm is found should supply its
  own figure, and this should be re-run against it.
- **Nothing here sizes the longitudinal endpoint.** Lead time needs repeated
  scoring and cases that develop within the window, which is a different
  calculation and rests on L8, whose supporting citation remains unread.

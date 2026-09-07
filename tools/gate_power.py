#!/usr/bin/env python3
"""How many animals P1's gate needs, and how hard prevalence makes it.

The gate is no longer "gait features correlate with human locomotion score".
docs/P1-LITERATURE-FINDINGS.md established that a single scorer is too noisy a
reference to carry a threshold, so the gate became: the system agrees with a
prespecified scorer consensus about as well as an individual trained scorer
does. That is a non-inferiority claim, and non-inferiority claims need a sample
size before the study starts rather than after.

This is arithmetic under stated assumptions, not a measurement. Every assumption
is a named constant below and every one of them is wrong to some degree; the
script exists so that changing one and re-running is a minute's work rather than
an argument.

Run:  python tools/gate_power.py
"""

from __future__ import annotations

import numpy as np

# -- what the literature supplies -------------------------------------------
#
# Both figures are recorded as confirmed in docs/P1-LITERATURE-FINDINGS.md.

# Thomsen et al. 2008, J Dairy Sci 91:119-126. Interobserver weighted kappa on a
# five-point field scale ranged 0.24-0.68, mean 0.48 before training and 0.52
# after. The midpoint of the trained figure is used here.
HUMAN_KAPPA = 0.50

# Anagnostopoulos et al. 2023 observed 7-20% human-scored lameness prevalence
# across visits on three farms. Wilson et al. 2026 had ten observers estimate
# 36.2-57.0% on the same 207 cows, which is a fact about observers rather than
# about herds. The lower band is the planning range: a herd that a farm is
# willing to host a study on is unlikely to be at the top of it.
PREVALENCE = (0.10, 0.15, 0.20, 0.30)

# -- design ------------------------------------------------------------------

SCORERS = 3  # two plus an adjudicator on disagreement behaves much like three
MARGINS = (0.05, 0.10, 0.15)  # non-inferiority margin on the agreement statistic
POWER_TARGET = 0.80
SIMULATIONS = 120
BOOTSTRAPS = 250
HERD_SIZES = (40, 60, 80, 120, 160, 240, 320, 480)


def _kappa(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's kappa for two binary raters.

    Kappa is prevalence-sensitive by construction: its chance-agreement term
    grows as the marginals become unbalanced, so the same raters score lower on
    a herd with less lameness. That is not a defect of the raters and it is the
    reason the alternative statistic below is also reported.
    """
    observed = float(np.mean(a == b))
    pa, pb = float(np.mean(a)), float(np.mean(b))
    expected = pa * pb + (1 - pa) * (1 - pb)
    return (observed - expected) / (1 - expected) if expected < 1 else 0.0


def _gwet_ac1(a: np.ndarray, b: np.ndarray) -> float:
    """Gwet's AC1, which Wilson et al. 2026 used for the same reason."""
    observed = float(np.mean(a == b))
    pi = (float(np.mean(a)) + float(np.mean(b))) / 2.0
    expected = 2.0 * pi * (1.0 - pi)
    return (observed - expected) / (1 - expected) if expected < 1 else 0.0


def _calibrate_noise(prevalence: float, target_kappa: float, rng) -> float:
    """The rater noise that reproduces the literature's rater-rater agreement.

    A latent-severity model: each animal has a true severity, each rater sees it
    with independent noise and calls lame above the herd threshold. The noise is
    solved for rather than assumed, so the simulated scorers disagree as much as
    real ones did at this prevalence.
    """
    threshold = float(np.quantile(rng.standard_normal(200_000), 1 - prevalence))

    def kappa_at(sigma: float) -> float:
        severity = rng.standard_normal(60_000)
        a = (severity + sigma * rng.standard_normal(60_000)) > threshold
        b = (severity + sigma * rng.standard_normal(60_000)) > threshold
        return _kappa(a.astype(int), b.astype(int))

    low, high = 0.01, 6.0
    for _ in range(40):
        mid = (low + high) / 2
        if kappa_at(mid) > target_kappa:
            low = mid  # too much agreement, add noise
        else:
            high = mid
    return (low + high) / 2


def _matched_difference(scores: np.ndarray, model: np.ndarray, statistic) -> float:
    """Model minus human agreement, with both judged the same way.

    THE POINT OF THIS FUNCTION. Comparing a human to a consensus they helped
    form against a model excluded from it is not a comparison: the human agrees
    better by construction, because their own vote is inside the thing they are
    scored against. Simulated at 20% prevalence with scorers as noisy as Thomsen
    et al. measured, that inclusion is worth about 0.19 of kappa — more than any
    plausible non-inferiority margin, so a system that measured locomotion
    perfectly would still fail.

    The fix is to judge both the same way. Each scorer is held out in turn and
    compared against the consensus of the others; the model is compared against
    that identical consensus, on the identical animals. Animals where the
    remaining scorers do not agree carry no consensus to be judged against and
    are dropped from both sides equally.
    """
    human_scores, model_scores = [], []
    for i in range(len(scores)):
        others = np.delete(scores, i, axis=0)
        settled = (others == others[0]).all(axis=0)
        if settled.sum() < 10:
            continue
        reference = others[0][settled]
        human_scores.append(statistic(scores[i][settled], reference))
        model_scores.append(statistic(model[settled], reference))
    if not human_scores:
        return 0.0
    return float(np.mean(model_scores)) - float(np.mean(human_scores))


def _one_study(n: int, prevalence: float, sigma: float, rng, statistic):
    """One simulated study.

    The model is simulated as exactly as good as a human scorer, because the
    question is not whether it is better — it is whether a study of this size
    could *demonstrate* non-inferiority when non-inferiority is true.
    """
    threshold = float(np.quantile(rng.standard_normal(200_000), 1 - prevalence))
    severity = rng.standard_normal(n)

    scores = np.stack(
        [(severity + sigma * rng.standard_normal(n)) > threshold for _ in range(SCORERS)]
    ).astype(int)
    model = ((severity + sigma * rng.standard_normal(n)) > threshold).astype(int)
    return model, scores, _matched_difference(scores, model, statistic)


def _lower_bound(n: int, prevalence: float, sigma: float, rng, statistic) -> float:
    """The one-sided 95% lower bound this study would put on the difference.

    Computed once per simulated study and compared against every margin, because
    the margins do not change the study — only what counts as passing it.
    """
    model, scores, _ = _one_study(n, prevalence, sigma, rng, statistic)
    differences = np.empty(BOOTSTRAPS)
    for b in range(BOOTSTRAPS):
        pick = rng.integers(0, n, n)
        differences[b] = _matched_difference(scores[:, pick], model[pick], statistic)
    return float(np.quantile(differences, 0.05))


def main() -> int:
    rng = np.random.default_rng(20260907)

    print("P1 gate: animals needed to demonstrate non-inferiority")
    print(f"  reference      : consensus of {SCORERS} blinded scorers")
    print(f"  human agreement: kappa {HUMAN_KAPPA} between scorers (Thomsen et al. 2008)")
    print(f"  power target   : {POWER_TARGET:.0%}, one-sided 95%")
    print("  model simulated as exactly as good as one human scorer\n")

    for name, statistic in (("Cohen's kappa", _kappa), ("Gwet's AC1", _gwet_ac1)):
        print(f"== {name} " + "=" * (58 - len(name)))
        header = "  prevalence  " + "".join(f"  margin {m:.2f}" for m in MARGINS)
        print(header)
        for prevalence in PREVALENCE:
            sigma = _calibrate_noise(prevalence, HUMAN_KAPPA, rng)
            needed = dict.fromkeys(MARGINS)
            for n in HERD_SIZES:
                if all(needed.values()):
                    break
                bounds = np.array(
                    [_lower_bound(n, prevalence, sigma, rng, statistic) for _ in range(SIMULATIONS)]
                )
                for margin in MARGINS:
                    if needed[margin] is None and np.mean(bounds > -margin) >= POWER_TARGET:
                        needed[margin] = n
            cells = [
                f"{needed[m]:>13}" if needed[m] else f"{'>' + str(HERD_SIZES[-1]):>13}"
                for m in MARGINS
            ]
            print(f"  {prevalence:>9.0%}  " + "".join(cells))
        print()

    print("Read the columns, not the cells: the shape is the finding.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

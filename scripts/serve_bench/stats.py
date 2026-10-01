"""Small-sample statistics for eval comparisons (stdlib only).

RoboDojo runs are 2-20 episodes, so a success-rate gap means nothing
without its uncertainty. These are the three checks every comparison table
now carries:

- wilson_ci: 95% interval for one success rate (well behaved at 0/n, n/n).
- fisher_exact_p: two-sided p-value that two success counts differ.
- mcnemar_exact_p: paired version for two runs over the SAME layouts
  (same seed): only scenes where the strategies disagree carry signal.
- min_detectable_gap: the smallest true gap n episodes per arm can detect
  (80% power, alpha 0.05), so a "no difference" is read as "too few
  episodes to tell" when it is.
"""
from __future__ import annotations

import math


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _log_comb(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def fisher_exact_p(s1: int, n1: int, s2: int, n2: int) -> float:
    """Two-sided Fisher exact test on the 2x2 table [[s1, n1-s1], [s2, n2-s2]]."""
    total_s = s1 + s2
    lo, hi = max(0, total_s - n2), min(total_s, n1)

    def prob(k: int) -> float:
        return math.exp(_log_comb(n1, k) + _log_comb(n2, total_s - k)
                        - _log_comb(n1 + n2, total_s))

    observed = prob(s1)
    return min(1.0, sum(prob(k) for k in range(lo, hi + 1)
                        if prob(k) <= observed * (1 + 1e-9)))


def mcnemar_exact_p(only_a: int, only_b: int) -> float:
    """Two-sided exact McNemar: discordant pairs ~ Binomial(n, 1/2)."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired_episodes(a: list, b: list) -> dict:
    """Match [layout_id, success, score] rows of two runs by layout_id."""
    first = {row[0]: row for row in a}
    pairs = [(first[row[0]], row) for row in b if row[0] in first]
    only_a = sum(1 for x, y in pairs if x[1] and not y[1])
    only_b = sum(1 for x, y in pairs if y[1] and not x[1])
    diffs = [float(y[2] or 0) - float(x[2] or 0) for x, y in pairs]
    mean = sum(diffs) / len(diffs) if diffs else 0.0
    sd = (math.sqrt(sum((d - mean) ** 2 for d in diffs) / (len(diffs) - 1))
          if len(diffs) > 1 else 0.0)
    half = 1.96 * sd / math.sqrt(len(diffs)) if diffs else 0.0
    return {"pairs": len(pairs), "only_a": only_a, "only_b": only_b,
            "mcnemar_p": mcnemar_exact_p(only_a, only_b),
            "score_diff": mean, "score_diff_ci": (mean - half, mean + half)}


def min_detectable_gap(n: int, base: float = 0.5) -> float:
    """Approximate smallest detectable success-rate gap, n episodes per arm."""
    if n <= 0:
        return 1.0
    z_alpha, z_beta = 1.96, 0.8416
    return (z_alpha + z_beta) * math.sqrt(2 * base * (1 - base) / n)


def median(vals: list[float]) -> float | None:
    if not vals:
        return None
    s = sorted(vals)
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2

"""Shared statistical utility functions (paired t-test, bootstrap CI, Holm), used by analysis/p1_analysis.py."""
import numpy as np
from scipy import stats

def welch_ttest(mean_a, std_a, mean_b, std_b, n=5):
    """Welch's t-test from summary statistics."""
    se = np.sqrt(std_a**2/n + std_b**2/n)
    if se == 0:
        return 0.0, 1.0
    t_stat = (mean_a - mean_b) / se
    df = (std_a**2/n + std_b**2/n)**2 / ((std_a**2/n)**2/(n-1) + (std_b**2/n)**2/(n-1))
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
    return t_stat, p_value


def paired_ttest(a, b):
    """Paired t-test on per-seed values. Returns (t, p, mean_diff, n_pairs).

    a and b must be same-seed-aligned. Falls back to (nan, nan, nan, 0) if the two
    series cannot be paired, so callers can drop back to the unpaired Welch test rather
    than silently comparing mismatched runs.
    """
    n = min(len(a), len(b))
    if n < 2:
        return float('nan'), float('nan'), float('nan'), n
    d = np.asarray(a[:n], dtype=float) - np.asarray(b[:n], dtype=float)
    if np.allclose(d, d[0]):
        # zero variance in the differences: deterministic offset, p is degenerate
        return (float('inf') if d[0] else 0.0), (0.0 if d[0] else 1.0), float(d.mean()), n
    t, p = stats.ttest_rel(a[:n], b[:n])
    return float(t), float(p), float(d.mean()), n


def t_ci(d, alpha=0.05):
    """t-based CI of the mean of the paired differences d (the reported interval since 2026-11-16:
    a percentile bootstrap over 2 to 6 seeds under-covers)."""
    d = np.asarray(d, dtype=float)
    n = d.size
    if n < 2:
        return float('nan'), float('nan')
    half = stats.t.ppf(1 - alpha / 2, n - 1) * d.std(ddof=1) / np.sqrt(n)
    return float(d.mean() - half), float(d.mean() + half)


def welch_ci(x, y, alpha=0.05):
    """Welch t-based CI of mean(x) - mean(y) for two independent samples."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.size < 2 or y.size < 2:
        return float('nan'), float('nan')
    vx, vy = x.var(ddof=1) / x.size, y.var(ddof=1) / y.size
    se = np.sqrt(vx + vy)
    df = (vx + vy) ** 2 / (vx ** 2 / (x.size - 1) + vy ** 2 / (y.size - 1)) if se > 0 else 1.0
    half = stats.t.ppf(1 - alpha / 2, df) * se
    m = x.mean() - y.mean()
    return float(m - half), float(m + half)


def bootstrap_ci(a, b=None, statistic='mean_diff', n_boot=10000, alpha=0.05, seed=0):
    """Percentile bootstrap CI, for the paired difference by default.

    Reported alongside the t-interval because at n = 3-10 the normal-theory interval
    leans on an assumption the sample cannot check. Resamples pairs, not the two series
    independently, so the pairing is preserved.
    """
    rng = np.random.default_rng(seed)
    if b is not None:
        n = min(len(a), len(b))
        if n < 2:
            return float('nan'), float('nan')
        d = np.asarray(a[:n], dtype=float) - np.asarray(b[:n], dtype=float)
    else:
        d = np.asarray(a, dtype=float)
        n = len(d)
        if n < 2:
            return float('nan'), float('nan')
    idx = rng.integers(0, n, size=(n_boot, n))
    stat = d[idx].mean(axis=1)
    lo, hi = np.percentile(stat, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def hedges_g(mean_a, std_a, mean_b, std_b, n=5):
    """Hedges' g: Cohen's d with the small-sample bias correction.

    Cohen's d overestimates the population effect at the sample sizes used here (n = 3-10),
    which is exactly where this study lives. The correction factor J = 1 - 3/(4*df - 1)
    removes most of that bias. Reported alongside d rather than instead of it, so the two
    can be compared against the published values.
    """
    d = cohens_d(mean_a, std_a, mean_b, std_b, n=n)
    df = 2 * n - 2
    if df <= 1:
        return d
    J = 1.0 - 3.0 / (4.0 * df - 1.0)
    return d * J


def tost_equivalence(mean_a, std_a, mean_b, std_b, margin, n=5, alpha=0.05):
    """Two one-sided tests (TOST) for equivalence within +/- margin.

    Non-significance of a difference is NOT evidence of equivalence — with n = 3-5 a test
    can fail to reject simply for want of power. TOST inverts the burden of proof: the null
    is 'the difference is at least `margin`', so rejecting it is positive evidence that the
    two methods are practically the same.

    Returns (equivalent, p_tost, ci_low, ci_high) where p_tost is the larger of the two
    one-sided p-values and the interval is the (1-2*alpha) CI of the difference a - b.
    Equivalence is declared when p_tost < alpha, i.e. when the whole CI sits inside
    (-margin, +margin).
    """
    se = np.sqrt(std_a**2 / n + std_b**2 / n)
    diff = mean_a - mean_b
    if se == 0:
        return abs(diff) < margin, 0.0 if abs(diff) < margin else 1.0, diff, diff
    df = (std_a**2/n + std_b**2/n)**2 / ((std_a**2/n)**2/(n-1) + (std_b**2/n)**2/(n-1))
    # H01: diff <= -margin   H02: diff >= +margin ; reject both => equivalent
    p_lower = stats.t.sf((diff + margin) / se, df)   # tests diff > -margin
    p_upper = stats.t.cdf((diff - margin) / se, df)  # tests diff < +margin
    p_tost = max(p_lower, p_upper)
    crit = stats.t.ppf(1 - alpha, df)
    return p_tost < alpha, p_tost, diff - crit * se, diff + crit * se


def cohens_d(mean_a, std_a, mean_b, std_b, n=5):
    """Cohen's d effect size with pooled standard deviation.

    Expects sample (ddof=1) standard deviations, matching the CSVs written by
    ExperimentTracker.get_statistics (CODICE-6).
    """
    pooled = np.sqrt(((n-1)*std_a**2 + (n-1)*std_b**2) / (2*n - 2))
    if pooled == 0:
        return 0.0
    return (mean_a - mean_b) / pooled


# ---------------------------------------------------------------------------
# Multiple-comparison correction (STAT-1)
#
# The study runs ~112 t-tests (τ(w) vs 7 competitors × 16 benchmarks). Counting
# raw significances as a scoreboard (e.g. "22 wins / 6 losses") without correction
# over-states the signal: under a global null at α=0.05, ~5.6 of 112 tests are
# significant by chance. The functions below let callers apply ONE correction
# symmetrically to wins AND losses, instead of narrating them asymmetrically.
# ---------------------------------------------------------------------------

def benjamini_hochberg(pvals, q=0.05):
    """Benjamini-Hochberg FDR control.

    Args:
        pvals: 1-D iterable of p-values for the whole comparison family
               (both directions — wins and losses — together).
        q: target false-discovery rate.

    Returns:
        Boolean numpy array; True where H0 is rejected at FDR q.
    """
    p = np.asarray(list(pvals), dtype=float)
    m = p.size
    if m == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p)
    ranked = p[order]
    thresh = q * (np.arange(1, m + 1) / m)
    below = ranked <= thresh
    if not below.any():
        return np.zeros(m, dtype=bool)
    kmax = np.max(np.where(below)[0])      # largest rank passing the BH line
    cutoff = ranked[kmax]
    return p <= cutoff


def bonferroni(pvals, alpha=0.05):
    """Bonferroni FWER control over a family of m p-values.

    Returns a boolean numpy array; True where p < alpha / m.
    """
    p = np.asarray(list(pvals), dtype=float)
    m = p.size
    if m == 0:
        return np.zeros(0, dtype=bool)
    return p < (alpha / m)


def holm(pvals, alpha=0.05):
    """Holm step-down FWER control over a family of m p-values.

    Sort ascending; the k-th smallest (1-based) is compared with alpha / (m - k + 1); the first
    failure stops the procedure. Uniformly more powerful than Bonferroni at the same FWER.
    Returns a boolean numpy array; True where H0 is rejected.
    """
    p = np.asarray(list(pvals), dtype=float)
    m = p.size
    if m == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p)
    reject = np.zeros(m, dtype=bool)
    for k, idx in enumerate(order):
        if p[idx] <= alpha / (m - k):
            reject[idx] = True
        else:
            break
    return reject

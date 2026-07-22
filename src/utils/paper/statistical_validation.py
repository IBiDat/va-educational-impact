"""
Between-group comparison of the IMPROVEMENT / outcome for each figure.

The question is never "did scores rise pre→post" (that is assumed). It is always:
**does the size of the improvement differ BETWEEN the groups the figure shows, how big is
that difference, and which contrast is the largest?**  Every figure therefore reduces to a
between-group comparison of one metric:

  * pre_post_*      -> raw gain      (post - pre)
  * hake_gain_*     -> Hake gain     (already a column)
  * self_confidence -> self-eff gain (ta_post - ta_pre)
  * trad_scale      -> grade-level change (ordinal)
  * tcc_*           -> post load     (no pre exists for cognitive load)
  * mejora_*        -> improvement CATEGORY  (Improve / Not / Worsen)

Tests:
  * 2 groups      -> Mann-Whitney U            + rank-biserial r
  * k > 2 groups  -> Kruskal-Wallis (+ Dunn-style pairwise Mann-Whitney, Holm)
                     -> reports the LARGEST pairwise difference and the top group
  * categorical   -> Chi-square                + Cramér's V (+ best/worst group)

Every result stores a common 0–1 `strength` (|rank-biserial| of the biggest contrast, or
Cramér's V) so all figures can be ranked against each other at the end.
"""
from __future__ import annotations
import itertools
import numpy as np
import polars as pl
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.contingency_tables import SquareTable
ALPHA = 0.05
RESULTS: list[dict] = []
TRAD = ['Excellent', 'Very Good', 'Good', 'Pass', 'Fail']            # best → worst
GRADE_RANK = {c: (len(TRAD) - 1 - i) for i, c in enumerate(TRAD)}    # Excellent=4 … Fail=0

def reset_results():
    RESULTS.clear()


# ------------------------------------------------------------------ utilities
def _with_metric(df, metric=None, pre=None, post=None):
    if pre is not None and post is not None:
        return df.with_columns(
            (pl.col(post).cast(pl.Float64, strict=False) -
             pl.col(pre).cast(pl.Float64, strict=False)).alias("_m"))
    return df.with_columns(pl.col(metric).cast(pl.Float64, strict=False).alias("_m"))


def _num(df, col, filt=None):
    d = df if filt is None else df.filter(filt)
    a = d.select(pl.col(col).cast(pl.Float64, strict=False)).to_series().to_numpy()
    return a[~np.isnan(a)]


def _sig(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return " "
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def _mag_r(r):
    a = abs(r)
    return "negligible" if a < 0.1 else "small" if a < 0.3 else "medium" if a < 0.5 else "large"


def _mag_eps(e):
    return "negligible" if e < 0.01 else "small" if e < 0.06 else "medium" if e < 0.14 else "large"


def _rb(a, b):
    """rank-biserial for (a vs b): >0 means a stochastically larger than b."""
    U, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return 2 * U / (len(a) * len(b)) - 1, p


def _reg(**kw):
    RESULTS.append(kw)


# =====================================================================
# 2 groups  (e.g. experimental vs control)
# =====================================================================
def diff_2(figure, grouping, metric_label, df, group, g1, g2,
           metric=None, pre=None, post=None, baseline=False):
    d = _with_metric(df, metric, pre, post)
    a = _num(d, "_m", pl.col(group) == g1)
    b = _num(d, "_m", pl.col(group) == g2)
    if len(a) < 3 or len(b) < 3:
        print(f"█ {figure}: [skip] insufficient data (n={len(a)},{len(b)})\n")
        return
    rb, p = _rb(a, b)
    m1, m2 = float(np.median(a)), float(np.median(b))
    higher = g1 if rb > 0 else g2
    mag = _mag_r(rb)
    print(f"█ {figure}   ·  grouping: {group}   ·  metric: {metric_label}")
    if baseline:
        print(f"   Q: are {g1} and {g2} balanced at baseline? (want NO difference)")
    else:
        print(f"   Q: is the improvement different between {g1} and {g2}?")
    print(f"   Mann-Whitney U   n={len(a)+len(b)}   "
          f"median {g1}={m1:+.3f} vs {g2}={m2:+.3f}  (Δ={m1-m2:+.3f})")
    print(f"   p = {p:.4f} {_sig(p)}   rank-biserial r = {rb:+.3f} ({mag})")
    if baseline:
        verdict = ("⚠ groups DIFFER at baseline — potential confound." if p < ALPHA
                   else "baseline balanced (no significant difference).")
    else:
        verdict = (f"{higher} is higher on {metric_label} — difference is {mag}." if p < ALPHA
                   else f"no significant between-group difference ({mag} effect).")
    print(f"   → {verdict}\n")
    _reg(figure=figure, grouping=group, metric=metric_label, test="Mann-Whitney U",
         n=len(a) + len(b), p=p, effect_name="rank-biserial r", effect_value=rb,
         magnitude=mag, strength=abs(rb), biggest_contrast=f"{g1} vs {g2}",
         top_group=higher)


# =====================================================================
# k > 2 groups  (v4, v3, mejora-as-group ...)  + post-hoc "biggest difference"
# =====================================================================
def diff_k(figure, grouping, metric_label, df, group, order=None,
           metric=None, pre=None, post=None):
    d = _with_metric(df, metric, pre, post)
    if order is not None:
        present = set(d.select(pl.col(group)).drop_nulls().unique().to_series().to_list())
        cats = [c for c in order if c in present]
    else:
        cats = d.select(pl.col(group)).drop_nulls().unique().to_series().to_list()
    groups, labels = [], []
    for c in cats:
        arr = _num(d, "_m", pl.col(group) == c)
        if len(arr) >= 3:
            groups.append(arr)
            labels.append(c)
    if len(groups) < 2:
        print(f"█ {figure}: [skip] <2 usable groups\n")
        return
    H, p = stats.kruskal(*groups)
    n = sum(len(g) for g in groups)
    k = len(groups)
    eps = (H - k + 1) / (n - k) if (n - k) > 0 else np.nan
    meds = {l: float(np.median(g)) for l, g in zip(labels, groups)}
    top = max(meds, key=meds.get)
    bot = min(meds, key=meds.get)

    # pairwise Mann-Whitney (Holm) -> find the biggest contrast by |rank-biserial|
    pairs = list(itertools.combinations(range(k), 2))
    rows, praw = [], []
    for i, j in pairs:
        rb, pp = _rb(groups[i], groups[j])
        rows.append((labels[i], labels[j], rb, pp))
        praw.append(pp)
    padj = multipletests(praw, method="holm")[1] if praw else []
    biggest = max(range(len(rows)), key=lambda t: abs(rows[t][2])) if rows else None

    print(f"█ {figure}   ·  grouping: {group}   ·  metric: {metric_label}")
    print(f"   Q: does {metric_label} differ across the {k} groups?")
    print(f"   Kruskal-Wallis   H={H:.3f}, df={k-1}, n={n}   "
          f"p = {p:.4f} {_sig(p)}   ε² = {eps:.3f} ({_mag_eps(eps)})")
    print(f"   group medians: " + ", ".join(f"{l}={v:+.3f}" for l, v in meds.items()))
    print(f"   highest: {top} ({meds[top]:+.3f})   |   lowest: {bot} ({meds[bot]:+.3f})")
    if biggest is not None:
        c1, c2, rb, _ = rows[biggest]
        print(f"   BIGGEST pairwise difference: {c1} vs {c2}  "
              f"r={rb:+.3f} ({_mag_r(rb)}), p-Holm={padj[biggest]:.4f} {_sig(padj[biggest])}")
        sig_pairs = [(rows[t][0], rows[t][1], rows[t][2], padj[t])
                     for t in range(len(rows)) if padj[t] < ALPHA]
        if sig_pairs:
            print("   significant pairs (Holm): " +
                  "; ".join(f"{a}>{b} r={r:+.2f}" if r > 0 else f"{b}>{a} r={-r:+.2f}"
                            for a, b, r, _ in sig_pairs))
        else:
            print("   (no pair survives Holm correction)")
    print()
    c1, c2, rb, _ = rows[biggest]
    _reg(figure=figure, grouping=group, metric=metric_label, test="Kruskal-Wallis",
         n=n, p=p, effect_name="epsilon^2", effect_value=eps, magnitude=_mag_eps(eps),
         strength=abs(rb), biggest_contrast=f"{c1} vs {c2} (r={rb:+.2f})", top_group=top)


# =====================================================================
# categorical improvement (mejora)  -> proportions differ between groups?
# =====================================================================
def _cramers_v(chi2, table):
    n = table.sum()
    r, c = table.shape
    denom = n * (min(r, c) - 1)
    return np.sqrt(chi2 / denom) if denom > 0 else np.nan


def diff_cat(figure, grouping, df, cat, group, positive="Improve", order=None):
    sub = df.select([pl.col(group).cast(pl.Utf8).alias("_g"),
                     pl.col(cat).cast(pl.Utf8).alias("_c")]).drop_nulls()
    g_levels = sorted(sub["_g"].unique().to_list())
    c_levels = ([c for c in order if c in set(sub["_c"].to_list())]
                if order else sorted(sub["_c"].unique().to_list()))
    if sub.height == 0 or len(g_levels) < 2 or len(c_levels) < 2:
        print(f"█ {figure}: [skip] degenerate table\n")
        return
    counts = sub.group_by(["_g", "_c"]).len()
    look = {(r["_g"], r["_c"]): r["len"] for r in counts.iter_rows(named=True)}
    table = np.array([[look.get((g, c), 0) for c in c_levels] for g in g_levels], float)
    chi2, p, dof, exp = stats.chi2_contingency(table)
    v = _cramers_v(chi2, table)
    # % of `positive` category per group -> best / worst improving group
    pos_rate = {}
    if positive in c_levels:
        pi = c_levels.index(positive)
        for gi, g in enumerate(g_levels):
            tot = table[gi].sum()
            pos_rate[g] = table[gi, pi] / tot if tot else np.nan
    best = max(pos_rate, key=pos_rate.get) if pos_rate else "-"
    worst = min(pos_rate, key=pos_rate.get) if pos_rate else "-"
    small = (exp < 5).mean()

    print(f"█ {figure}   ·  grouping: {group}   ·  metric: {cat} (proportions)")
    print(f"   Q: does the improvement-category mix differ between groups?")
    print(f"   Chi-square   χ²={chi2:.3f}, df={dof}, n={int(table.sum())}   "
          f"p = {p:.4f} {_sig(p)}   Cramér's V = {v:.3f} ({_mag_r(v)})")
    if pos_rate:
        print("   %" + positive + " by group: " +
              ", ".join(f"{g}={pos_rate[g]*100:.0f}%" for g in g_levels))
        print(f"   most improving: {best} ({pos_rate[best]*100:.0f}%)   |   "
              f"least: {worst} ({pos_rate[worst]*100:.0f}%)")
    if small > 0:
        print(f"   note: {small*100:.0f}% of cells expected <5 (χ² approximate)")
    verdict = ("Proportions differ between groups." if p < ALPHA
               else "No significant difference in the improvement mix.")
    print(f"   → {verdict}\n")
    _reg(figure=figure, grouping=group, metric=f"{cat} proportions",
         test="Chi-square", n=int(table.sum()), p=p, effect_name="Cramér's V",
         effect_value=v, magnitude=_mag_r(v), strength=v,
         biggest_contrast=f"{best} vs {worst} (%{positive})", top_group=best)


# =====================================================================
# Matthew: within an arm, is improvement different for High vs Low starters?
# =====================================================================
def matthew(figure, df, pre, post, low_max=0.5, high_min=0.7):
    d = df.with_columns(
        pl.when(pl.col(pre) <= low_max).then(pl.lit("Low"))
          .when(pl.col(pre) >= high_min).then(pl.lit("High")).alias("_lvl"),
        (pl.col(post) - pl.col(pre)).alias("_m"))
    gh = _num(d, "_m", pl.col("_lvl") == "High")
    gl = _num(d, "_m", pl.col("_lvl") == "Low")
    if len(gh) < 3 or len(gl) < 3:
        print(f"█ {figure}: [skip] insufficient High/Low data\n")
        return
    rb, p = _rb(gh, gl)   # >0 => High gained more
    gap_pre = float(np.median(_num(d, pre, pl.col("_lvl") == "High")) -
                    np.median(_num(d, pre, pl.col("_lvl") == "Low")))
    gap_post = float(np.median(_num(d, post, pl.col("_lvl") == "High")) -
                     np.median(_num(d, post, pl.col("_lvl") == "Low")))
    mag = _mag_r(rb)
    if p < ALPHA:
        verdict = ("High starters gain MORE → gap widens (Matthew effect)." if rb > 0
                   else "Low starters gain MORE → gap narrows (compensatory).")
    else:
        verdict = "No significant difference in gain between High and Low starters."
    print(f"█ {figure}   ·  grouping: High vs Low baseline   ·  metric: raw gain")
    print(f"   Q: is the improvement different for High vs Low starters?")
    print(f"   Mann-Whitney U   n={len(gh)+len(gl)}   "
          f"gain High={np.median(gh):+.3f} vs Low={np.median(gl):+.3f}")
    print(f"   gap High−Low: pre={gap_pre:+.3f} → post={gap_post:+.3f}")
    print(f"   p = {p:.4f} {_sig(p)}   rank-biserial r = {rb:+.3f} ({mag})")
    print(f"   → {verdict}\n")
    _reg(figure=figure, grouping="High vs Low", metric="raw gain",
         test="Mann-Whitney U", n=len(gh) + len(gl), p=p,
         effect_name="rank-biserial r", effect_value=rb, magnitude=mag,
         strength=abs(rb), biggest_contrast="High vs Low",
         top_group="High" if rb > 0 else "Low")


# =====================================================================
# cross-figure summary — ranked by effect size ("biggest difference")
# =====================================================================
def summary():
    import pandas as pd
    if not RESULTS:
        print("No results.")
        return None
    df = pd.DataFrame(RESULTS)
    df["p_holm"] = multipletests(df["p"].values, method="holm")[1]
    df = df.sort_values("strength", ascending=False).reset_index(drop=True)
    show = df[["figure", "grouping", "test", "n", "p", "p_holm", "effect_name",
               "effect_value", "magnitude", "strength", "biggest_contrast", "top_group"]]
    with pd.option_context("display.max_rows", None, "display.width", 240,
                           "display.max_colwidth", 40,
                           "display.float_format", lambda v: f"{v:.4f}"):
        print("Ranked by between-group effect size (strength = |r| of biggest contrast, or V):\n")
        print(show.to_string(index=False))
    sig = df[df["p"] < ALPHA]
    print("\n" + "=" * 70)
    print("BIGGEST between-group differences (significant, by effect size):")
    for _, r in sig.head(8).iterrows():
        print(f"  • {r['figure']:<42} {r['magnitude']:>10}  "
              f"strength={r['strength']:.3f}  [{r['biggest_contrast']}]")
    if sig.empty:
        print("  (none reached significance)")
    return df

# =====================================================================
# PAIRED within-subject tests  (same students, pre vs post)
# =====================================================================
def _matched_rc(diff):
    """matched-pairs rank-biserial: >0 => post > pre."""
    nz = diff[diff != 0]
    if len(nz) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(nz))
    return float(np.sum(np.sign(nz) * ranks) / np.sum(ranks))

# =====================================================================
# analyze if the distribution has changed pre→post (paired categorical / ordinal)
# =====================================================================

def paired_distribution(figure, grouping, df, pre, post, order, filt=None):

    """Stuart-Maxwell marginal homogeneity: did the DISTRIBUTION move pre→post?
    (paired categorical / ordinal — same students measured twice)."""
    d = df if filt is None else df.filter(filt)
    d = d.select([pl.col(pre).cast(pl.Utf8).alias("_p"),
                  pl.col(post).cast(pl.Utf8).alias("_q")]).drop_nulls()
    if d.height < 5:
        print(f"█ {figure}: [skip] insufficient paired data (n={d.height})\n")
        return
    pl_, ql_ = d["_p"].to_list(), d["_q"].to_list()
    present = set(pl_) | set(ql_)
    cats = [c for c in order if c in present] + sorted(present - set(order))
    idx = {c: i for i, c in enumerate(cats)}
    k = len(cats)
    table = np.zeros((k, k))
    for x, y in zip(pl_, ql_):
        table[idx[x], idx[y]] += 1
    try:
        sm = SquareTable(table).homogeneity()
        stat_v, p, dfree = sm.statistic, sm.pvalue, sm.df
    except Exception:
        stat_v, p, dfree = np.nan, np.nan, np.nan
    # ordinal effect: matched rank-biserial on mapped levels (first level = best = high)
    val = {c: (len(order) - 1 - order.index(c)) if c in order else 0 for c in cats}
    diff = np.array([val[y] for y in ql_], float) - np.array([val[x] for x in pl_], float)
    rc = _matched_rc(diff)
    mag = _mag_r(rc)
    n = int(table.sum())
    # marginal proportions pre vs post
    pre_marg = table.sum(1) / n
    post_marg = table.sum(0) / n
    print(f"█ {figure}   ·  {grouping}   ·  metric: grade distribution  (PAIRED)")
    print(f"   Q: did the grade DISTRIBUTION change from pre to post?")
    print(f"   {'band':>12} | pre%   post%   Δ")
    for c, pm, qm in zip(cats, pre_marg, post_marg):
        print(f"   {c:>12} | {pm*100:5.1f}  {qm*100:5.1f}  {(qm-pm)*100:+5.1f}")
    print(f"   Stuart-Maxwell (marginal homogeneity)   n={n}   "
          f"χ²={stat_v:.3f}, df={dfree}   p = {p:.4f} {_sig(p)}")
    print(f"   ordinal shift r_c = {rc:+.3f} ({mag})")
    verdict = (f"distribution shifted {'toward better' if rc > 0 else 'toward worse'} grades ({mag})."
               if (p is not None and not np.isnan(p) and p < ALPHA)
               else "no significant change in the grade distribution.")
    print(f"   → {verdict}\n")
    _reg(figure=figure, grouping=grouping, metric="grade distribution",
         test="Stuart-Maxwell", n=n, p=p, effect_name="rank-biserial r_c",
         effect_value=rc, magnitude=mag, strength=abs(rc),
         biggest_contrast="pre-dist vs post-dist", top_group=("post" if rc > 0 else "pre"), design="within-subject")
    

def _mag(v):
    a = abs(v)
    return "negligible" if a < 0.1 else "small" if a < 0.3 else "medium" if a < 0.5 else "large"


def _contingency(df, band_col, group_col, order):
    sub = df.select([pl.col(group_col).cast(pl.Utf8).alias("_g"),
                     pl.col(band_col).cast(pl.Utf8).alias("_b")]).drop_nulls()
    g_levels = sorted(sub["_g"].unique().to_list())
    b_levels = [c for c in order if c in set(sub["_b"].to_list())]
    counts = sub.group_by(["_g", "_b"]).len()
    look = {(r["_g"], r["_b"]): r["len"] for r in counts.iter_rows(named=True)}
    table = np.array([[look.get((g, b), 0) for b in b_levels] for g in g_levels], float)
    return table, g_levels, b_levels


def _print_pct(table, g_levels, b_levels, title):
    print(f"   {title} (row % within group)")
    print("     " + " " * 14 + "  ".join(f"{b:>10}" for b in b_levels))
    for i, g in enumerate(g_levels):
        tot = table[i].sum()
        pct = (table[i] / tot * 100) if tot else table[i]
        cells = "  ".join(f"{v:>9.1f}%" for v in pct)
        print(f"     {g:>14}  {cells}   n={int(tot)}")


def test_expa_trad_scale(df, target="ExpA",
                        group_col="grupo_segmented_v4",
                        pre_col="score_tc_cat_trad_scale_pre",
                        post_col="score_tc_cat_trad_scale_post"):
    print("=" * 78)
    print(f"pre_post_trad_scale.pdf — does '{target}' improve more than the other "
          f"{group_col} categories?")
    print("=" * 78)

    dat = df.filter(pl.col(group_col).is_not_null()
                    & pl.col(pre_col).is_not_null()
                    & pl.col(post_col).is_not_null())

    # (1) PRE homogeneity
    print("\n(1) PRE — is the grade distribution homogeneous across the 4 groups?")
    print("      test : Chi-square of independence   (want NON-significant p)")
    tbl_pre, g_levels, b_levels = _contingency(dat, pre_col, group_col, TRAD)
    if target not in g_levels:
        raise ValueError(f"target '{target}' not found in {group_col} (available: {g_levels})")
    chi2_pre, p_pre, dof_pre, exp_pre = stats.chi2_contingency(tbl_pre)
    v_pre = _cramers_v(chi2_pre, tbl_pre)
    _print_pct(tbl_pre, g_levels, b_levels, "PRE distribution")
    print(f"     Chi² = {chi2_pre:.3f}, df = {dof_pre}, n = {int(tbl_pre.sum())}   "
          f"p = {p_pre:.4f} {_sig(p_pre)}   Cramer's V = {v_pre:.3f} ({_mag(v_pre)})")
    if (exp_pre < 5).mean() > 0:
        print(f"     note: {(exp_pre<5).mean()*100:.0f}% of cells expected <5 (χ² approximate)")
    print("     → " + ("baseline HOMOGENEOUS ✓" if p_pre >= ALPHA
                       else "⚠ baseline NOT homogeneous — groups start with different distributions"))

    # (2a) POST omnibus
    print("\n(2a) POST — do the 4 groups differ overall at post?")
    print("      test : Chi-square of independence   (want significant p)")
    tbl_post, g_levels_p, b_levels_p = _contingency(dat, post_col, group_col, TRAD)
    chi2_post, p_post, dof_post, exp_post = stats.chi2_contingency(tbl_post)
    v_post = _cramers_v(chi2_post, tbl_post)
    _print_pct(tbl_post, g_levels_p, b_levels_p, "POST distribution")
    print(f"     Chi² = {chi2_post:.3f}, df = {dof_post}, n = {int(tbl_post.sum())}   "
          f"p = {p_post:.4f} {_sig(p_post)}   Cramer's V = {v_post:.3f} ({_mag(v_post)})")
    if (exp_post < 5).mean() > 0:
        print(f"     note: {(exp_post<5).mean()*100:.0f}% of cells expected <5")
    print("     → " + ("post distribution DIFFERS across groups ✓" if p_post < ALPHA
                       else "no significant overall difference at post"))

    # (2b) POST: ExpA vs each other v4 level on the ordinal grade rank
    print(f"\n(2b) POST — does '{target}' hold BETTER grades than each other group?")
    print("      test : Mann-Whitney U on grade rank (Excellent = high),")
    print("             positive rank-biserial ⇒ target holds better grades   (Holm-adjusted)")

    dat_r = dat.with_columns(
        pl.col(post_col).replace_strict(GRADE_RANK, default=None).alias("_gr")
    ).drop_nulls("_gr")

    tgt = dat_r.filter(pl.col(group_col) == target)["_gr"].to_numpy()
    others = [g for g in g_levels_p if g != target]
    raw_p, rows = [], []
    for g in others:
        comp = dat_r.filter(pl.col(group_col) == g)["_gr"].to_numpy()
        if len(tgt) < 3 or len(comp) < 3:
            rows.append((g, len(tgt), len(comp), np.nan, np.nan, np.nan, np.nan))
            raw_p.append(1.0); continue
        U, p = stats.mannwhitneyu(tgt, comp, alternative="two-sided")
        rb = 2 * U / (len(tgt) * len(comp)) - 1
        rows.append((g, len(tgt), len(comp), rb, p, float(np.median(tgt)), float(np.median(comp))))
        raw_p.append(p)
    padj = multipletests(raw_p, method="holm")[1]

    print(f"\n     {'contrast':<32} {'n₁':>4} {'n₂':>4}  {'med(rank)':>13}  "
          f"{'r':>7}  {'p':>8}  {'p-Holm':>8}   effect")
    wins = 0
    for (g, n1, n2, rb, p, m1, m2), pa in zip(rows, padj):
        med_str = f"{m1:.1f} vs {m2:.1f}" if not np.isnan(m1) else "n/a"
        rb_s = f"{rb:+.3f}" if not np.isnan(rb) else "   n/a"
        p_s = f"{p:.4f}" if not np.isnan(p) else "   n/a"
        pa_s = f"{pa:.4f}" if not np.isnan(pa) else "   n/a"
        mag = _mag(rb) if not np.isnan(rb) else "-"
        direction = ("better" if (not np.isnan(rb) and rb > 0)
                     else "worse" if (not np.isnan(rb) and rb < 0) else "-")
        marker = _sig(pa) if not np.isnan(pa) else " "
        print(f"     {target} vs {g:<20} {n1:>4} {n2:>4}  {med_str:>13}  "
              f"{rb_s:>7}  {p_s:>8}  {pa_s:>8} {marker}  {mag} {direction}")
        if direction == "better" and not np.isnan(pa) and pa < ALPHA:
            wins += 1

    print(f"\n     → '{target}' significantly holds BETTER grades than "
          f"{wins}/{len(others)} of the other groups at post (Holm-adjusted).")

    # overall verdict
    print("\n" + "-" * 78)
    if p_pre >= ALPHA and p_post < ALPHA and wins >= 1:
        print(f"CLAIM SUPPORTED  ✓  pre homogeneous, post differs, and '{target}' "
              f"is significantly better than {wins} other group(s).")
    elif p_pre >= ALPHA and wins >= 1:
        print(f"partially supported: pre homogeneous and '{target}' is better than "
              f"{wins} group(s), but the omnibus post χ² is not significant.")
    elif p_pre < ALPHA:
        print("baseline NOT homogeneous — a post-hoc claim about '"
              + target + "' improving more must control for the starting difference.")
    else:
        print(f"'{target}' does not show a significant advantage at post.")
    print("-" * 78)

def _target_vs_others(df, group_col, target, value_col=None, ordinal_map=None):
    if ordinal_map is not None:
        d = df.with_columns(
                pl.col(value_col)
                  .replace_strict(ordinal_map, default=None, return_dtype=pl.Int64)
                  .alias("_v")
            ).filter(pl.col(group_col).is_not_null() & pl.col("_v").is_not_null())
    else:
        d = df.filter(pl.col(group_col).is_not_null() & pl.col(value_col).is_not_null()) \
              .with_columns(pl.col(value_col).cast(pl.Float64, strict=False).alias("_v"))

    levels = sorted(d.select(pl.col(group_col)).drop_nulls().unique().to_series().to_list())
    if target not in levels:
        raise ValueError(f"target '{target}' not in {group_col} (available: {levels})")

    tgt = d.filter(pl.col(group_col) == target)["_v"].to_numpy()
    others = [g for g in levels if g != target]

    raw_p, rows = [], []
    for g in others:
        comp = d.filter(pl.col(group_col) == g)["_v"].to_numpy()
        if len(tgt) < 3 or len(comp) < 3:
            rows.append(dict(comparator=g, n_t=len(tgt), n_c=len(comp),
                             med_t=(float(np.median(tgt)) if len(tgt) else np.nan),
                             med_c=(float(np.median(comp)) if len(comp) else np.nan),
                             r=np.nan, p=np.nan, p_holm=np.nan))
            raw_p.append(1.0); continue
        U, p = stats.mannwhitneyu(tgt, comp, alternative="two-sided")
        rb = 2 * U / (len(tgt) * len(comp)) - 1
        rows.append(dict(comparator=g, n_t=len(tgt), n_c=len(comp),
                         med_t=float(np.median(tgt)), med_c=float(np.median(comp)),
                         r=rb, p=p, p_holm=np.nan))
        raw_p.append(p)
    padj = multipletests(raw_p, method="holm")[1]
    for r, pa in zip(rows, padj):
        r["p_holm"] = pa
    return rows

def _print_contrasts(target, rows, value_label):
    print(f"     {'contrast':<32} {'n_t':>4} {'n_c':>4}  {'med(t)':>7} {'med(c)':>7}"
          f"  {'r':>7}  {'p':>8}  {'p-Holm':>8}   effect")
    wins = 0
    for r in rows:
        rb, p, pa = r["r"], r["p"], r["p_holm"]
        rb_s = f"{rb:+.3f}" if not np.isnan(rb) else "   n/a"
        p_s = f"{p:.4f}" if not np.isnan(p) else "   n/a"
        pa_s = f"{pa:.4f}" if not np.isnan(pa) else "   n/a"
        m1_s = f"{r['med_t']:.2f}" if not np.isnan(r["med_t"]) else "n/a"
        m2_s = f"{r['med_c']:.2f}" if not np.isnan(r["med_c"]) else "n/a"
        mag = _mag(rb) if not np.isnan(rb) else "-"
        direction = ("higher" if (not np.isnan(rb) and rb > 0)
                     else "lower" if (not np.isnan(rb) and rb < 0) else "-")
        print(f"     {target} vs {r['comparator']:<20} {r['n_t']:>4} {r['n_c']:>4}  "
              f"{m1_s:>7} {m2_s:>7}  {rb_s:>7}  {p_s:>8}  {pa_s:>8} {_sig(pa)}  "
              f"{mag} {direction}")
        if direction == "higher" and not np.isnan(pa) and pa < ALPHA:
            wins += 1
    print(f"\n     -> '{target}' is significantly HIGHER on {value_label} than "
          f"{wins}/{len(rows)} of the other groups (Holm-adjusted).")
    return wins

def test_expaa_pre_v3(
    df,
    target="ExpAA",
    group_col="grupo_segmented_v3",
    v3_col="score_tc_pre",
):
    print("=" * 82)
    print(
        f"TEST 1 - Does '{target}' have a higher baseline ({v3_col}) "
        f"than the other {group_col} groups?"
    )
    print("=" * 82)
    print(f"   Metric : Continuous baseline score ({v3_col})")
    print("   Test   : Mann-Whitney U comparing the target group against each")
    print("            remaining group (Holm-adjusted p-values).")
    print("            Positive rank-biserial correlation (r) indicates that")
    print("            the target group has a higher baseline.\n")

    rows = _target_vs_others(
        df,
        group_col,
        target,
        value_col=v3_col,
    )  # no ordinal_map

    wins = _print_contrasts(target, rows, f"baseline ({v3_col})")

    return {
        "target": target,
        "contrasts": rows,
        "wins": wins,
    }

def test_expa_ta_post(df, target="ExpA",
                     group_col="grupo_segmented_v4",
                     value_col="score_ta_post"):
    print("=" * 82)
    print(f"TEST 2  -  does '{target}' hold HIGHER post self-efficacy ({value_col}) "
          f"than the other {group_col} groups?")
    print("=" * 82)
    print("   metric : continuous post self-efficacy score")
    print("   test   : Mann-Whitney U on score_ta_post, target vs each other group,")
    print("            Holm-adjusted    (positive r => target scores HIGHER)\n")
    rows = _target_vs_others(df, group_col, target, value_col=value_col)
    wins = _print_contrasts(target, rows, "self-efficacy (post)")
    return dict(target=target, contrasts=rows, wins=wins)

def test_mejora_across_baseline_strata(
        df,
        stratum_col="score_tc_cat_pre",
        outcome_col="mejora_hake_gain_v2",
        stratum_order=("Low", "Medium", "High"),
        outcome_order=("Improve", "Not Improve", "Worsen")):
    print("=" * 82)
    print("TEST 3  -  is the improvement mix (Improve / Not Improve / Worsen) the SAME")
    print(f"           across the three baseline panels ({stratum_col} = Low / Medium / High)")
    print("           of mejora_nota_pre.pdf, collapsing groups within each panel?")
    print("=" * 82)
    print("   metric : proportion in each mejora category")
    print("   test   : Chi-square of independence on the 3 (baseline) x 3 (mejora) table")
    print("            H0: the mejora mix is the same across the three baseline strata\n")

    sub = df.select([pl.col(stratum_col).cast(pl.Utf8).alias("_s"),
                     pl.col(outcome_col).cast(pl.Utf8).alias("_o")]).drop_nulls()
    s_levels = [c for c in stratum_order if c in set(sub["_s"].to_list())]
    o_levels = [c for c in outcome_order if c in set(sub["_o"].to_list())]
    counts = sub.group_by(["_s", "_o"]).len()
    look = {(r["_s"], r["_o"]): r["len"] for r in counts.iter_rows(named=True)}
    table = np.array([[look.get((s, o), 0) for o in o_levels] for s in s_levels], float)

    print(f"     {'stratum':>10}   " + "  ".join(f"{o:>12}" for o in o_levels) + "     n")
    for i, s in enumerate(s_levels):
        tot = table[i].sum()
        pct = (table[i] / tot * 100) if tot else table[i]
        cells = "  ".join(f"{v:>11.1f}%" for v in pct)
        print(f"     {s:>10}   {cells}   {int(tot):>5}")

    chi2, p, dof, exp = stats.chi2_contingency(table)
    v = _cramers_v(chi2, table)
    small = (exp < 5).mean()
    print(f"\n     Chi2 = {chi2:.3f}, df = {dof}, n = {int(table.sum())}   "
          f"p = {p:.4f} {_sig(p)}   Cramer's V = {v:.3f} ({_mag(v)})")
    if small > 0:
        print(f"     note: {small*100:.0f}% of cells expected <5 (chi2 approximation)")
    verdict = ("mejora mix DIFFERS across baseline strata"
               if p < ALPHA else
               "no significant difference in the mejora mix across strata (H0 not rejected)")
    print(f"     -> {verdict}")
    return dict(chi2=chi2, dof=dof, p=p, cramers_v=v, table=table.tolist(),
                strata=s_levels, outcomes=o_levels)
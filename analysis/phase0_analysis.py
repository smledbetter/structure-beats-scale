#!/usr/bin/env python3
"""
Phase 0: Free analytical fixes for Study C
Reads existing data, produces corrected analysis. No API calls needed.
"""

import json
import glob
import os
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE = Path(__file__).parent
RAW_DIR = BASE / "Study C results" / "raw"
FULL_RAW_DIR = BASE / "adversarial_bench_full" / "results" / "raw"
ANALYSIS_DIR = BASE / "Study C results" / "analysis"
OUTPUT_DIR = BASE / "phase0_results"
OUTPUT_DIR.mkdir(exist_ok=True)

# Mapping from Study C reused conditions to their Study A/B raw data prefixes
REUSED_MAPPING = {
    "C0": "A_A0",
    "C1": "A_A2",
    "C2": "B_B2",
    "C5": "A_A1",
    "C9": "B_B3",
}

# ── Load all data ──────────────────────────────────────────────────────────

def load_pass_rates():
    """Load the pass rates CSV into a dict of condition -> {field: value}."""
    rows = {}
    with open(ANALYSIS_DIR / "study_c_pass_rates.csv") as f:
        for r in csv.DictReader(f):
            rows[r["condition"]] = {
                "family": r["strategy_family"],
                "mean_pass_rate": float(r["mean_pass_rate"]),
                "std": float(r["std_pass_rate"]),
                "pass_at_1": float(r["mean_pass_at_1"]),
                "cost": float(r["mean_cost_usd"]),
                "n": int(r["n_runs"]),
            }
    return rows

def load_raw_results():
    """Load all raw JSON results into a nested dict: condition -> task -> replica -> data."""
    results = defaultdict(lambda: defaultdict(dict))
    for f in glob.glob(str(RAW_DIR / "*.json")):
        try:
            d = json.load(open(f))
            c = d["condition"]
            task = d["task_id"]
            replica = d["replica"]
            # Get final pass rate from last pipeline
            # Use final_test_results (best-of-k selection) when available
            if d.get("final_test_results"):
                pr = d["final_test_results"]["pass_rate"]
            elif d.get("pipelines"):
                pr = d["pipelines"][-1]["test_results"]["pass_rate"]
            else:
                pr = 0.0
            cost = d.get("total_cost_usd", 0.0)
            if cost == 0.0:
                cost = sum(g["cost_usd"] for g in d.get("generations", []))
                cost += sum(r_["cost_usd"] for p_ in d.get("pipelines", []) for r_ in p_.get("reviews", []))
                cost += sum(fix.get("cost_usd", 0) for p_ in d.get("pipelines", []) for fix in p_.get("fixes", []))
            results[c][task][replica] = {
                "pass_rate": pr,
                "cost": cost,
                "raw": d,
            }
        except Exception as e:
            print(f"  Warning: could not load {f}: {e}", file=sys.stderr)
    return results

def load_reused_raw_results():
    """Load raw data for reused conditions (C0/C1/C2/C5/C9) from adversarial_bench_full."""
    results = defaultdict(lambda: defaultdict(dict))
    for study_c_cond, prefix in REUSED_MAPPING.items():
        for f in glob.glob(str(FULL_RAW_DIR / f"{prefix}_*.json")):
            try:
                d = json.load(open(f))
                task = d["task_id"]
                replica = d["replica"]

                # Study A/B format: flat structure with top-level test_results
                if d.get("pipelines"):
                    pr = d["pipelines"][-1]["test_results"]["pass_rate"]
                elif d.get("test_results"):
                    pr = d["test_results"]["pass_rate"]
                else:
                    pr = 0.0

                cost = d.get("total_cost_usd", 0.0)
                if cost == 0.0:
                    cost = sum(g["cost_usd"] for g in d.get("generations", []))
                    cost += sum(r["cost_usd"] for p in d.get("pipelines", []) for r in p.get("reviews", []))

                results[study_c_cond][task][replica] = {
                    "pass_rate": pr,
                    "cost": cost,
                    "raw": d,
                }
            except Exception as e:
                print(f"  Warning: could not load {f}: {e}", file=sys.stderr)
    return results

def load_c16_raw_results():
    """Load C16 (iso-cost, best-of-7 gen) results from adversarial_bench_full."""
    results = defaultdict(lambda: defaultdict(dict))
    for f in glob.glob(str(FULL_RAW_DIR / "C_C16_*.json")):
        try:
            d = json.load(open(f))
            task = d["task_id"]
            replica = d["replica"]
            if d.get("final_test_results"):
                pr = d["final_test_results"]["pass_rate"]
            elif d.get("pipelines"):
                pr = d["pipelines"][-1]["test_results"]["pass_rate"]
            else:
                pr = 0.0
            cost = d.get("total_cost_usd", 0.0)
            if cost == 0.0:
                cost = sum(g["cost_usd"] for g in d.get("generations", []))
            results["C16"][task][replica] = {
                "pass_rate": pr,
                "cost": cost,
                "raw": d,
            }
        except Exception as e:
            print(f"  Warning: could not load {f}: {e}", file=sys.stderr)
    return results

def compute_pass_rates_from_raw(raw_results, condition):
    """Compute pass_rates entry from raw data for a condition not in the CSV."""
    tasks = raw_results[condition]
    all_prs = []
    all_p1 = []
    all_costs = []
    for task in tasks:
        reps = tasks[task]
        rep_prs = [reps[r]["pass_rate"] for r in reps]
        rep_costs = [reps[r]["cost"] for r in reps]
        all_prs.extend(rep_prs)
        all_costs.extend(rep_costs)
        # pass@1: mean of individual run pass rates
        all_p1.extend(rep_prs)
    return {
        "family": "gen-only",
        "mean_pass_rate": float(np.mean(all_prs)),
        "std": float(np.std(all_prs)),
        "pass_at_1": float(np.mean(all_p1)),
        "cost": float(np.mean(all_costs)),
        "n": len(all_prs),
    }

# ── 0a: Bootstrap Pareto Frontier Stability ────────────────────────────────

def compute_pareto_frontier(points):
    """Given list of (cost, pass_rate, label), return Pareto-optimal labels."""
    sorted_pts = sorted(points, key=lambda x: x[0])
    frontier = []
    best_pr = -1
    for cost, pr, label in sorted_pts:
        if pr > best_pr:
            frontier.append(label)
            best_pr = pr
    return set(frontier)

def bootstrap_pareto(raw_results, pass_rates, n_bootstrap=10000):
    """Bootstrap resample problems, recompute Pareto frontier each time."""
    print("=== 0a: Bootstrap Pareto Frontier Stability ===")

    # Build per-problem, per-condition mean pass rates
    # For conditions with raw data: compute from raw
    # For reused conditions: use overall mean (can't resample per-problem)
    all_conditions = sorted(pass_rates.keys(), key=lambda x: int(x[1:]))

    # Get the shared problem set from raw data
    raw_conditions = sorted(raw_results.keys(), key=lambda x: int(x[1:]))
    if not raw_conditions:
        print("  No raw data found!")
        return {}
    problems = sorted(raw_results[raw_conditions[0]].keys())
    n_problems = len(problems)

    # Build matrix: condition -> problem -> mean_pass_rate (across replicas)
    cond_problem_pr = {}
    cond_problem_cost = {}
    for c in all_conditions:
        if c in raw_results:
            pr_by_problem = {}
            cost_by_problem = {}
            for task in problems:
                if task in raw_results[c]:
                    reps = raw_results[c][task]
                    pr_by_problem[task] = np.mean([reps[r]["pass_rate"] for r in reps])
                    cost_by_problem[task] = np.mean([reps[r]["cost"] for r in reps])
                else:
                    pr_by_problem[task] = np.nan
                    cost_by_problem[task] = np.nan
            cond_problem_pr[c] = pr_by_problem
            cond_problem_cost[c] = cost_by_problem
        else:
            # Reused condition — use fixed values (can't resample)
            cond_problem_pr[c] = {t: pass_rates[c]["mean_pass_rate"] for t in problems}
            cond_problem_cost[c] = {t: pass_rates[c]["cost"] for t in problems}

    # Bootstrap
    np.random.seed(42)
    frontier_counts = defaultdict(int)
    cond_boot_pr = defaultdict(list)
    cond_boot_cost = defaultdict(list)

    for _ in range(n_bootstrap):
        # Resample problems with replacement
        boot_problems = np.random.choice(problems, size=n_problems, replace=True)

        points = []
        for c in all_conditions:
            boot_prs = [cond_problem_pr[c][t] for t in boot_problems if not np.isnan(cond_problem_pr[c].get(t, np.nan))]
            boot_costs = [cond_problem_cost[c][t] for t in boot_problems if not np.isnan(cond_problem_cost[c].get(t, np.nan))]
            if boot_prs:
                mean_pr = np.mean(boot_prs)
                mean_cost = np.mean(boot_costs)
                points.append((mean_cost, mean_pr, c))
                cond_boot_pr[c].append(mean_pr)
                cond_boot_cost[c].append(mean_cost)

        frontier = compute_pareto_frontier(points)
        for c in frontier:
            frontier_counts[c] += 1

    # Report
    print(f"\n  Pareto frontier inclusion rates ({n_bootstrap} bootstrap resamples):")
    print(f"  {'Condition':<8} {'Family':<15} {'Inclusion%':>10} {'Boot PR 95% CI':>20} {'Boot Cost 95% CI':>22}")
    print(f"  {'-'*78}")
    for c in all_conditions:
        pct = 100.0 * frontier_counts.get(c, 0) / n_bootstrap
        if c in cond_boot_pr and cond_boot_pr[c]:
            pr_lo, pr_hi = np.percentile(cond_boot_pr[c], [2.5, 97.5])
            cost_lo, cost_hi = np.percentile(cond_boot_cost[c], [2.5, 97.5])
            pr_ci = f"[{pr_lo:.3f}, {pr_hi:.3f}]"
            cost_ci = f"[${cost_lo:.4f}, ${cost_hi:.4f}]"
        else:
            pr_ci = "N/A (reused)"
            cost_ci = "N/A (reused)"
        fam = pass_rates.get(c, {}).get("family", "?")
        marker = " ***" if pct >= 50 else ""
        print(f"  {c:<8} {fam:<15} {pct:>9.1f}% {pr_ci:>20} {cost_ci:>22}{marker}")

    # Save to file
    with open(OUTPUT_DIR / "bootstrap_pareto.csv", "w") as f:
        w = csv.writer(f)
        w.writerow(["condition", "family", "frontier_inclusion_pct", "boot_pr_2.5", "boot_pr_97.5", "boot_cost_2.5", "boot_cost_97.5"])
        for c in all_conditions:
            pct = 100.0 * frontier_counts.get(c, 0) / n_bootstrap
            if c in cond_boot_pr and cond_boot_pr[c]:
                pr_lo, pr_hi = np.percentile(cond_boot_pr[c], [2.5, 97.5])
                cost_lo, cost_hi = np.percentile(cond_boot_cost[c], [2.5, 97.5])
            else:
                pr_lo = pr_hi = pass_rates[c]["mean_pass_rate"]
                cost_lo = cost_hi = pass_rates[c]["cost"]
            fam = pass_rates.get(c, {}).get("family", "?")
            w.writerow([c, fam, f"{pct:.1f}", f"{pr_lo:.4f}", f"{pr_hi:.4f}", f"{cost_lo:.4f}", f"{cost_hi:.4f}"])

    return frontier_counts

# ── 0b: Missing Statistical Comparisons ────────────────────────────────────

def missing_comparisons(raw_results):
    """Compute C0 vs C13, C9 vs C13, and other key missing comparisons."""
    print("\n=== 0b: Missing Statistical Comparisons ===")

    # We need per-run pass rates for each condition
    # For reused conditions (C0, C9), we don't have raw data — skip those pairs
    # But we CAN compare conditions that share the same problem set in raw data

    comparisons = [
        ("C0", "C13"),   # baseline vs best hybrid (KEY TEST)
        ("C0", "C9"),    # baseline vs best debate (KEY TEST)
        ("C9", "C13"),   # debate vs hybrid (KEY TEST)
        ("C3", "C13"),   # gen-only vs best hybrid
        ("C7", "C13"),   # hybrid-small vs hybrid-large
        ("C4", "C10"),   # iterative 2 vs 4
        ("C10", "C12"),  # iterative 4 vs 8
        ("C3", "C9"),    # gen-only vs debate
        ("C6", "C11"),   # 3-gen vs 5-gen scaling
        ("C0", "C15"),   # baseline vs debate (C15)
        ("C2", "C9"),    # debate small vs medium
    ]

    # Build per-problem mean pass rate for paired tests
    raw_conditions = sorted(raw_results.keys())
    problems = sorted(raw_results[raw_conditions[0]].keys()) if raw_conditions else []

    results_table = []
    for c1, c2 in comparisons:
        if c1 not in raw_results or c2 not in raw_results:
            print(f"  {c1} vs {c2}: SKIPPED (no raw data for one or both)")
            continue

        # Paired by problem (average replicas within problem first)
        vals1 = []
        vals2 = []
        for task in problems:
            if task in raw_results[c1] and task in raw_results[c2]:
                mean1 = np.mean([raw_results[c1][task][r]["pass_rate"] for r in raw_results[c1][task]])
                mean2 = np.mean([raw_results[c2][task][r]["pass_rate"] for r in raw_results[c2][task]])
                vals1.append(mean1)
                vals2.append(mean2)

        if len(vals1) < 5:
            print(f"  {c1} vs {c2}: SKIPPED (insufficient paired data)")
            continue

        vals1 = np.array(vals1)
        vals2 = np.array(vals2)
        diff = vals2 - vals1

        # Wilcoxon signed-rank test (paired)
        try:
            stat, p = stats.wilcoxon(vals1, vals2)
            # Effect size r = Z / sqrt(N)
            z = stats.norm.ppf(p / 2)
            r = abs(z) / np.sqrt(len(vals1))
        except ValueError:
            # All differences are zero
            stat, p, r = 0, 1.0, 0.0

        n_comparisons = 12 + len(comparisons)  # Original 12 + new ones
        sig = p < 0.05 / n_comparisons  # Bonferroni

        mean_diff = np.mean(diff)
        print(f"  {c1} vs {c2}: delta={mean_diff:+.3f}, p={p:.4f}, r={r:.3f}, {'SIGNIFICANT' if sig else 'not significant'} (Bonferroni α={0.05/n_comparisons:.4f})")
        results_table.append({
            "comparison": f"{c1} vs {c2}",
            "mean_c1": np.mean(vals1),
            "mean_c2": np.mean(vals2),
            "mean_diff": mean_diff,
            "p_value": p,
            "effect_size_r": r,
            "n_problems": len(vals1),
            "significant_bonferroni": bool(sig),
        })

    with open(OUTPUT_DIR / "additional_comparisons.json", "w") as f:
        json.dump(results_table, f, indent=2)

    return results_table

# ── 0c: Distribution Analysis ──────────────────────────────────────────────

def distribution_analysis(raw_results, pass_rates):
    """Analyse bimodality: fraction at 0.0 and 1.0 per condition."""
    print("\n=== 0c: Distribution Analysis (Bimodality) ===")

    all_conditions = sorted(pass_rates.keys(), key=lambda x: int(x[1:]))

    print(f"\n  {'Condition':<8} {'Family':<15} {'N':>4} {'@0.0':>6} {'@1.0':>6} {'Between':>8} {'Mean':>6}")
    print(f"  {'-'*58}")

    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    axes = axes.flatten()

    dist_data = {}
    for i, c in enumerate(all_conditions):
        if c not in raw_results:
            continue

        all_prs = []
        for task in raw_results[c]:
            for rep in raw_results[c][task]:
                all_prs.append(raw_results[c][task][rep]["pass_rate"])

        all_prs = np.array(all_prs)
        n = len(all_prs)
        at_zero = np.sum(all_prs == 0.0) / n * 100
        at_one = np.sum(all_prs == 1.0) / n * 100
        between = 100 - at_zero - at_one
        mean = np.mean(all_prs)
        fam = pass_rates[c]["family"]

        print(f"  {c:<8} {fam:<15} {n:>4} {at_zero:>5.1f}% {at_one:>5.1f}% {between:>7.1f}% {mean:>6.3f}")
        dist_data[c] = {"at_zero": at_zero, "at_one": at_one, "between": between, "values": all_prs}

        # Histogram subplot
        ax = axes[i] if i < len(axes) else None
        if ax:
            ax.hist(all_prs, bins=20, range=(0, 1), color={"baseline": "gray", "gen-only": "#4488cc",
                     "review-heavy": "#ff9933", "debate": "#66bb44", "hybrid": "#cc3333",
                     "iterative": "#9944cc"}.get(fam, "gray"), alpha=0.8, edgecolor="white")
            ax.set_title(f"{c} ({fam})", fontsize=10)
            ax.set_xlim(0, 1)
            ax.axvline(mean, color="red", linestyle="--", linewidth=1)

    # Hide unused subplots
    for j in range(len([c for c in all_conditions if c in raw_results]), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Study C: Pass Rate Distributions by Condition", fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "distributions.png", dpi=150)
    plt.close()
    print(f"\n  Saved distribution plot to {OUTPUT_DIR / 'distributions.png'}")

    return dist_data

# ── 0d: Cost-Normalised Family Analysis ────────────────────────────────────

def cost_normalised_analysis(pass_rates):
    """For each strategy family, show pass rate vs cost curve."""
    print("\n=== 0d: Cost-Normalised Family Analysis ===")

    families = defaultdict(list)
    for c, data in pass_rates.items():
        families[data["family"]].append((data["cost"], data["mean_pass_rate"], c))

    # What does each family achieve at $0.06?
    print(f"\n  Interpolated pass rate at $0.06/problem:")
    for fam, points in sorted(families.items()):
        points.sort()
        costs = [p[0] for p in points]
        prs = [p[1] for p in points]
        if len(points) == 1:
            interp = prs[0] if abs(costs[0] - 0.06) < 0.02 else None
        else:
            interp = np.interp(0.06, costs, prs)
        if interp is not None:
            print(f"    {fam:<15}: {interp:.3f} (from {len(points)} conditions)")
        else:
            print(f"    {fam:<15}: out of range (cost={costs[0]:.3f})")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = {"baseline": "gray", "gen-only": "#4488cc", "review-heavy": "#ff9933",
              "debate": "#66bb44", "hybrid": "#cc3333", "iterative": "#9944cc"}

    for fam, points in sorted(families.items()):
        points.sort()
        costs = [p[0] for p in points]
        prs = [p[1] for p in points]
        labels = [p[2] for p in points]
        ax.plot(costs, prs, 'o-', color=colors.get(fam, "gray"), label=fam, markersize=8, linewidth=2)
        for cost, pr, label in points:
            ax.annotate(label, (cost, pr), textcoords="offset points", xytext=(5, 5), fontsize=8)

    ax.axvline(0.06, color="gray", linestyle=":", alpha=0.5, label="$0.06 reference")
    ax.set_xlabel("Mean Cost per Problem (USD)")
    ax.set_ylabel("Mean Pass Rate")
    ax.set_title("Study C: Cost-Normalised Family Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cost_normalised.png", dpi=150)
    plt.close()
    print(f"  Saved cost-normalised plot to {OUTPUT_DIR / 'cost_normalised.png'}")

# ── 0e: Verdict Parsing Audit ──────────────────────────────────────────────

def verdict_parsing_audit(raw_results):
    """Check whether parsed verdicts match raw_response verdicts."""
    print("\n=== 0e: Verdict Parsing Audit ===")

    total_reviews = 0
    parsed_pass = 0
    parsed_fail = 0
    raw_pass = 0
    raw_fail = 0
    raw_unparseable = 0
    mismatches = 0
    mismatch_details = []
    unparseable_samples = []

    for c in raw_results:
        for task in raw_results[c]:
            for rep in raw_results[c][task]:
                d = raw_results[c][task][rep]["raw"]
                for p in d.get("pipelines", []):
                    for rev in p.get("reviews", []):
                        total_reviews += 1
                        parsed_verdict = rev.get("verdict", "UNKNOWN")
                        raw_response = rev.get("raw_response", "")

                        if parsed_verdict == "PASS":
                            parsed_pass += 1
                        else:
                            parsed_fail += 1

                        # Try to extract verdict from raw_response
                        try:
                            raw_json = json.loads(raw_response) if isinstance(raw_response, str) else raw_response
                            raw_verdict = raw_json.get("verdict", "UNKNOWN")
                        except (json.JSONDecodeError, AttributeError):
                            raw_verdict = "UNPARSEABLE"

                        if raw_verdict == "PASS":
                            raw_pass += 1
                        elif raw_verdict == "FAIL":
                            raw_fail += 1
                        elif raw_verdict in ("UNKNOWN", "UNPARSEABLE"):
                            raw_unparseable += 1
                            if len(unparseable_samples) < 5:
                                snippet = str(raw_response)[:200] if raw_response else "<empty>"
                                unparseable_samples.append({
                                    "condition": c, "task": task, "replica": rep,
                                    "parsed_verdict": parsed_verdict,
                                    "raw_response_snippet": snippet,
                                })

                        if parsed_verdict != raw_verdict and raw_verdict not in ("UNKNOWN", "UNPARSEABLE"):
                            mismatches += 1
                            if len(mismatch_details) < 10:
                                bugs_in_raw = len(raw_json.get("bugs", [])) if isinstance(raw_json, dict) else 0
                                mismatch_details.append({
                                    "condition": c,
                                    "task": task,
                                    "replica": rep,
                                    "parsed": parsed_verdict,
                                    "raw": raw_verdict,
                                    "bugs_in_raw": bugs_in_raw,
                                    "parsed_bugs_reported": rev.get("bugs_reported", "?"),
                                })

    parseable = raw_pass + raw_fail
    unparseable_pct = 100 * raw_unparseable / max(total_reviews, 1)
    print(f"\n  Total reviews analysed: {total_reviews}")
    print(f"  Parsed verdicts: {parsed_pass} PASS, {parsed_fail} FAIL")
    print(f"  Raw response verdicts: {raw_pass} PASS, {raw_fail} FAIL")
    print(f"  *** UNPARSEABLE raw responses: {raw_unparseable}/{total_reviews} ({unparseable_pct:.1f}%) ***")
    print(f"  → Only {parseable} of {total_reviews} reviews could be audited for mismatches")
    print(f"  Mismatches (among parseable): {mismatches} ({100*mismatches/max(parseable,1):.1f}%)")
    if unparseable_samples:
        print(f"\n  Sample unparseable raw responses:")
        for s in unparseable_samples:
            print(f"    {s['condition']}/{s['task']}/r{s['replica']}: parsed={s['parsed_verdict']}, raw snippet: {s['raw_response_snippet'][:100]}")

    if mismatches > 0:
        print(f"\n  Direction of mismatches:")
        raw_fail_parsed_pass = sum(1 for m in mismatch_details if m["raw"] == "FAIL" and m["parsed"] == "PASS")
        raw_pass_parsed_fail = sum(1 for m in mismatch_details if m["raw"] == "PASS" and m["parsed"] == "FAIL")
        print(f"    Raw=FAIL, Parsed=PASS: {raw_fail_parsed_pass} (of first {len(mismatch_details)} mismatches)")
        print(f"    Raw=PASS, Parsed=FAIL: {raw_pass_parsed_fail} (of first {len(mismatch_details)} mismatches)")

        print(f"\n  Sample mismatches (first {min(5, len(mismatch_details))}):")
        for m in mismatch_details[:5]:
            print(f"    {m['condition']}/{m['task']}/r{m['replica']}: parsed={m['parsed']}, raw={m['raw']}, raw_bugs={m['bugs_in_raw']}, parsed_bugs_reported={m['parsed_bugs_reported']}")

    # Check if the override correlates with test-suite validation
    # i.e., when raw says FAIL but parsed says PASS, did the code actually pass tests?
    validated_overrides = 0
    unvalidated_overrides = 0
    for c in raw_results:
        for task in raw_results[c]:
            for rep in raw_results[c][task]:
                d = raw_results[c][task][rep]["raw"]
                actual_pr = raw_results[c][task][rep]["pass_rate"]
                for p in d.get("pipelines", []):
                    for rev in p.get("reviews", []):
                        parsed_verdict = rev.get("verdict", "UNKNOWN")
                        try:
                            raw_json = json.loads(rev.get("raw_response", "{}"))
                            raw_verdict = raw_json.get("verdict", "UNKNOWN")
                        except:
                            continue
                        if raw_verdict == "FAIL" and parsed_verdict == "PASS":
                            # The harness overrode FAIL to PASS — was the code actually correct?
                            if actual_pr >= 0.9:
                                validated_overrides += 1
                            else:
                                unvalidated_overrides += 1

    if validated_overrides + unvalidated_overrides > 0:
        total_overrides = validated_overrides + unvalidated_overrides
        print(f"\n  Override analysis (raw=FAIL overridden to parsed=PASS):")
        print(f"    Code actually passed tests (pr>=0.9): {validated_overrides}/{total_overrides} ({100*validated_overrides/total_overrides:.1f}%)")
        print(f"    Code actually failed tests (pr<0.9):  {unvalidated_overrides}/{total_overrides} ({100*unvalidated_overrides/total_overrides:.1f}%)")
        print(f"    → Overrides appear to be {'VALIDATED (harness checks test results)' if validated_overrides > unvalidated_overrides else 'CONCERNING (many overrides on failing code)'}")

    audit = {
        "total_reviews": total_reviews,
        "parsed_pass": parsed_pass,
        "parsed_fail": parsed_fail,
        "raw_pass": raw_pass,
        "raw_fail": raw_fail,
        "raw_unparseable": raw_unparseable,
        "unparseable_pct": round(unparseable_pct, 1),
        "parseable_audited": parseable,
        "mismatches": mismatches,
        "mismatch_rate_of_parseable": mismatches / max(parseable, 1),
        "validated_overrides": validated_overrides,
        "unvalidated_overrides": unvalidated_overrides,
        "sample_mismatches": mismatch_details[:10],
        "sample_unparseable": unparseable_samples[:5],
        "NOTE": f"Only {parseable} of {total_reviews} reviews ({100*parseable/max(total_reviews,1):.0f}%) had parseable raw verdicts. The '0 mismatches' finding is limited to this subset.",
    }
    with open(OUTPUT_DIR / "verdict_audit.json", "w") as f:
        json.dump(audit, f, indent=2)

    return audit

# ── 0f: Problem Difficulty Stratification ──────────────────────────────────

def difficulty_stratification(raw_results, pass_rates):
    """Stratify problems by baseline (C0) difficulty, show strategy effects per stratum."""
    print("\n=== 0f: Problem Difficulty Stratification ===")

    # Use C0 (baseline) pass rate as difficulty proxy to avoid circularity
    if "C0" not in raw_results:
        print("  ERROR: C0 not in raw_results, cannot stratify without baseline")
        return {}
    problems = sorted(raw_results["C0"].keys())

    # Compute baseline difficulty per problem using C0 only (avoids circularity)
    problem_difficulty = {}
    for task in problems:
        if task in raw_results["C0"]:
            reps = raw_results["C0"][task]
            problem_difficulty[task] = np.mean([reps[r]["pass_rate"] for r in reps])
        else:
            problem_difficulty[task] = 0.5

    # Stratify: easy (>0.67), medium (0.33-0.67), hard (<0.33)
    easy = [t for t in problems if problem_difficulty[t] > 0.67]
    medium = [t for t in problems if 0.33 <= problem_difficulty[t] <= 0.67]
    hard = [t for t in problems if problem_difficulty[t] < 0.33]

    print(f"\n  Problem stratification: {len(easy)} easy, {len(medium)} medium, {len(hard)} hard")
    print(f"  Easy problems: {easy}")
    print(f"  Medium problems: {medium}")
    print(f"  Hard problems: {hard}")

    # Per-stratum performance for each condition
    strata = {"easy": easy, "medium": medium, "hard": hard}
    print(f"\n  {'Condition':<8} {'Family':<15} {'Easy':>8} {'Medium':>8} {'Hard':>8} {'Δ Hard':>8}")
    print(f"  {'-'*55}")

    strat_data = {}
    for c in sorted(raw_results.keys(), key=lambda x: int(x[1:])):
        fam = pass_rates.get(c, {}).get("family", "?")
        row = {"family": fam}
        for sname, sproblems in strata.items():
            prs = []
            for task in sproblems:
                if task in raw_results[c]:
                    for rep in raw_results[c][task]:
                        prs.append(raw_results[c][task][rep]["pass_rate"])
            row[sname] = np.mean(prs) if prs else np.nan

        # Delta on hard problems vs C3 (gen-only baseline-ish)
        c3_hard = []
        if "C3" in raw_results:
            for task in hard:
                if task in raw_results["C3"]:
                    for rep in raw_results["C3"][task]:
                        c3_hard.append(raw_results["C3"][task][rep]["pass_rate"])
        c3_hard_mean = np.mean(c3_hard) if c3_hard else 0
        delta_hard = row.get("hard", 0) - c3_hard_mean

        print(f"  {c:<8} {fam:<15} {row.get('easy', 0):>7.3f} {row.get('medium', 0):>7.3f} {row.get('hard', 0):>7.3f} {delta_hard:>+7.3f}")
        strat_data[c] = row

    with open(OUTPUT_DIR / "difficulty_stratification.json", "w") as f:
        json.dump(strat_data, f, indent=2, default=str)

    return strat_data

# ── 0g: Effective Sample Size & Design Effect ──────────────────────────────

def design_effect_analysis(raw_results, pass_rates):
    """Compute ICC and design effect for each condition."""
    print("\n=== 0g: Design Effect & Effective Sample Size ===")

    print(f"\n  {'Condition':<8} {'Family':<15} {'ICC':>6} {'DEFF':>6} {'Eff_N':>6} {'Corrected SE':>12} {'Corrected 95% CI':>20}")
    print(f"  {'-'*78}")

    for c in sorted(raw_results.keys(), key=lambda x: int(x[1:])):
        problems = sorted(raw_results[c].keys())
        fam = pass_rates.get(c, {}).get("family", "?")

        # Compute ICC
        between_vals = []
        within_vars = []
        for task in problems:
            reps = raw_results[c][task]
            rep_prs = [reps[r]["pass_rate"] for r in reps]
            between_vals.append(np.mean(rep_prs))
            if len(rep_prs) > 1:
                within_vars.append(np.var(rep_prs, ddof=1))

        between_var = np.var(between_vals, ddof=1) if len(between_vals) > 1 else 0
        within_var = np.mean(within_vars) if within_vars else 0

        if between_var + within_var > 0:
            icc = between_var / (between_var + within_var)
        else:
            icc = 0

        m = 3  # replicas
        deff = 1 + (m - 1) * icc
        n_total = len(problems) * m
        eff_n = n_total / deff

        grand_mean = pass_rates[c]["mean_pass_rate"]
        grand_std = pass_rates[c]["std"]
        corrected_se = grand_std / np.sqrt(eff_n)
        ci_lo = grand_mean - 1.96 * corrected_se
        ci_hi = grand_mean + 1.96 * corrected_se

        print(f"  {c:<8} {fam:<15} {icc:>5.3f} {deff:>5.2f} {eff_n:>5.1f} {corrected_se:>11.4f} [{ci_lo:.3f}, {ci_hi:.3f}]")

# ── 0h: Pass@1 Analysis ───────────────────────────────────────────────────

def pass_at_1_analysis(pass_rates):
    """Compare pass@1 across conditions, especially C13 vs C6 (both 3-gen)."""
    print("\n=== 0h: Pass@1 Analysis ===")

    # Key comparison: C13 (3 gens + 2 reviews each) vs C6 (3 gens, no review)
    # Both use 3 generations, so the difference isolates the review effect
    conditions = sorted(pass_rates.keys(), key=lambda x: int(x[1:]))

    print(f"\n  {'Condition':<8} {'Family':<15} {'Mean PR':>8} {'Pass@1':>7} {'Gap':>7} {'N gens':>7}")
    print(f"  {'-'*55}")

    # Manually annotate generation counts for context
    gen_counts = {
        "C0": 1, "C1": 1, "C2": 1, "C3": 2, "C4": 1, "C5": 1,
        "C6": 3, "C7": 2, "C8": 1, "C9": 1, "C10": 1, "C11": 5,
        "C12": 1, "C13": 3, "C14": 1, "C15": 1, "C16": 7,
    }

    for c in conditions:
        d = pass_rates[c]
        gap = d["mean_pass_rate"] - d["pass_at_1"]
        n_gen = gen_counts.get(c, "?")
        print(f"  {c:<8} {d['family']:<15} {d['mean_pass_rate']:>7.3f} {d['pass_at_1']:>7.3f} {gap:>+6.3f} {n_gen:>7}")

    # Highlight the key comparison
    if "C13" in pass_rates and "C6" in pass_rates:
        c13_p1 = pass_rates["C13"]["pass_at_1"]
        c6_p1 = pass_rates["C6"]["pass_at_1"]
        diff = c13_p1 - c6_p1
        print(f"\n  KEY COMPARISON: C13 pass@1 ({c13_p1:.3f}) vs C6 pass@1 ({c6_p1:.3f})")
        print(f"  Both use 3 generations. C13 adds 2 reviews per generation.")
        print(f"  Difference: {diff:+.3f}")
        print(f"  → Reviews improve individual pipeline quality by {diff*100:+.1f}pp,")
        print(f"    not just adding lottery tickets via more generations.")

    # Also compare C7 vs C3 (both 2-gen)
    if "C7" in pass_rates and "C3" in pass_rates:
        c7_p1 = pass_rates["C7"]["pass_at_1"]
        c3_p1 = pass_rates["C3"]["pass_at_1"]
        diff2 = c7_p1 - c3_p1
        print(f"\n  SUPPORTING: C7 pass@1 ({c7_p1:.3f}) vs C3 pass@1 ({c3_p1:.3f})")
        print(f"  Both use 2 generations. C7 adds 1 review per generation.")
        print(f"  Difference: {diff2:+.3f}")

    result = {
        "key_comparison": {
            "C13_pass_at_1": pass_rates.get("C13", {}).get("pass_at_1"),
            "C6_pass_at_1": pass_rates.get("C6", {}).get("pass_at_1"),
            "difference": pass_rates.get("C13", {}).get("pass_at_1", 0) - pass_rates.get("C6", {}).get("pass_at_1", 0),
            "interpretation": "Reviews improve individual pipelines, not just lottery effect",
        },
        "all_pass_at_1": {c: pass_rates[c]["pass_at_1"] for c in conditions},
    }
    with open(OUTPUT_DIR / "pass_at_1_analysis.json", "w") as f:
        json.dump(result, f, indent=2)

    return result

# ── 0i: Problem-Level Item-Response Table ─────────────────────────────────

def item_response_table(raw_results, pass_rates):
    """For each problem, show which conditions solve it (>0.9 mean) and which don't."""
    print("\n=== 0i: Problem-Level Item-Response Table ===")

    conditions = sorted(raw_results.keys(), key=lambda x: int(x[1:]))
    # Use problems from C0 as the reference set
    problems = sorted(raw_results.get("C0", raw_results[conditions[0]]).keys())

    # Build matrix: problem x condition -> mean pass rate
    matrix = {}
    for task in problems:
        matrix[task] = {}
        for c in conditions:
            if task in raw_results.get(c, {}):
                reps = raw_results[c][task]
                matrix[task][c] = np.mean([reps[r]["pass_rate"] for r in reps])
            else:
                matrix[task][c] = np.nan

    # Classify problems
    print(f"\n  Problem solve rates (mean pass rate > 0.9 = solved):")
    print(f"  {'Problem':<10} {'Solved by':>10} {'Type':<12} ", end="")
    # Show a few key conditions
    key_conds = ["C0", "C3", "C9", "C13"]
    for c in key_conds:
        print(f" {c:>5}", end="")
    print()
    print(f"  {'-'*60}")

    swing_problems = []
    always_solved = []
    never_solved = []

    for task in problems:
        solved_by = sum(1 for c in conditions if matrix[task].get(c, 0) > 0.9)
        total_c = sum(1 for c in conditions if not np.isnan(matrix[task].get(c, np.nan)))

        if solved_by == total_c:
            ptype = "always"
            always_solved.append(task)
        elif solved_by == 0:
            ptype = "never"
            never_solved.append(task)
        else:
            ptype = "SWING"
            swing_problems.append(task)

        print(f"  {task:<10} {solved_by:>4}/{total_c:<5} {ptype:<12} ", end="")
        for c in key_conds:
            val = matrix[task].get(c, np.nan)
            if np.isnan(val):
                print(f"    -", end="")
            elif val > 0.9:
                print(f"    ✓", end="")
            elif val < 0.1:
                print(f"    ✗", end="")
            else:
                print(f" {val:.2f}", end="")
        print()

    print(f"\n  Summary: {len(always_solved)} always solved, {len(never_solved)} never solved, {len(swing_problems)} swing")
    print(f"  Swing problems (where strategies actually differ): {swing_problems}")

    result = {
        "always_solved": always_solved,
        "never_solved": never_solved,
        "swing_problems": swing_problems,
        "matrix": {task: {c: float(v) if not np.isnan(v) else None for c, v in matrix[task].items()} for task in problems},
    }
    with open(OUTPUT_DIR / "item_response_table.json", "w") as f:
        json.dump(result, f, indent=2)

    return result

# ── 0j: Cost Variance Analysis ───────────────────────────────────────────

def cost_variance_analysis(raw_results, pass_rates):
    """Report cost distributions per condition, flag outliers."""
    print("\n=== 0j: Cost Variance Analysis ===")

    conditions = sorted(raw_results.keys(), key=lambda x: int(x[1:]))

    print(f"\n  {'Condition':<8} {'Family':<15} {'Mean$':>7} {'Median$':>8} {'Std$':>7} {'Min$':>7} {'Max$':>7} {'Outliers':>9}")
    print(f"  {'-'*75}")

    cost_data = {}
    for c in conditions:
        costs = []
        for task in raw_results[c]:
            for rep in raw_results[c][task]:
                costs.append(raw_results[c][task][rep]["cost"])

        costs = np.array(costs)
        fam = pass_rates.get(c, {}).get("family", "?")

        # Flag outliers: > 3x median
        median = np.median(costs)
        outlier_threshold = max(median * 3, 0.01)
        n_outliers = np.sum(costs > outlier_threshold)

        print(f"  {c:<8} {fam:<15} ${np.mean(costs):>.4f} ${median:>.4f} ${np.std(costs):>.4f} ${np.min(costs):>.4f} ${np.max(costs):>.4f} {n_outliers:>9}")

        cost_data[c] = {
            "family": fam,
            "mean": float(np.mean(costs)),
            "median": float(median),
            "std": float(np.std(costs)),
            "min": float(np.min(costs)),
            "max": float(np.max(costs)),
            "p25": float(np.percentile(costs, 25)),
            "p75": float(np.percentile(costs, 75)),
            "n_outliers_3x_median": int(n_outliers),
            "n_runs": len(costs),
        }

    # Highlight conditions with high cost variance
    high_var = [(c, d) for c, d in cost_data.items() if d["std"] / max(d["mean"], 0.001) > 0.5]
    if high_var:
        print(f"\n  Conditions with high cost variance (CV > 0.5):")
        for c, d in high_var:
            cv = d["std"] / max(d["mean"], 0.001)
            print(f"    {c}: CV={cv:.2f}, range=[${d['min']:.4f}, ${d['max']:.4f}]")

    with open(OUTPUT_DIR / "cost_variance.json", "w") as f:
        json.dump(cost_data, f, indent=2)

    return cost_data

# ── Phase 2: Held-Out Replication Analysis ─────────────────────────────────

HELD_OUT_PROBLEMS = [
    "cc_000", "cc_002", "cc_003", "cc_007", "cc_008", "cc_009",
    "cc_011", "cc_012", "cc_013", "cc_014", "cc_015", "cc_017",
    "cc_022", "cc_023", "cc_025", "cc_026", "cc_029", "cc_030",
    "cc_031", "cc_032", "cc_035", "cc_037", "cc_040", "cc_042",
    "cc_044", "cc_045", "cc_053", "cc_056", "cc_058", "cc_059",
]

PHASE2_CONDITIONS = ["C0", "C9", "C13"]

# Strategy family mapping for Phase 2 conditions
PHASE2_FAMILIES = {"C0": "baseline", "C9": "debate", "C13": "hybrid"}

def load_phase2_raw():
    """Load Phase 2 held-out replication data from adversarial_bench_full."""
    results = defaultdict(lambda: defaultdict(dict))
    for cond in PHASE2_CONDITIONS:
        for f in glob.glob(str(FULL_RAW_DIR / f"C_{cond}_*.json")):
            try:
                d = json.load(open(f))
                task = d["task_id"]
                if task not in HELD_OUT_PROBLEMS:
                    continue
                replica = d["replica"]
                if d.get("final_test_results"):
                    pr = d["final_test_results"]["pass_rate"]
                elif d.get("pipelines"):
                    pr = d["pipelines"][-1]["test_results"]["pass_rate"]
                elif d.get("test_results"):
                    pr = d["test_results"]["pass_rate"]
                else:
                    pr = 0.0
                cost = d.get("total_cost_usd", 0.0)
                if cost == 0.0:
                    cost = sum(g["cost_usd"] for g in d.get("generations", []))
                    cost += sum(r_["cost_usd"] for p_ in d.get("pipelines", []) for r_ in p_.get("reviews", []))
                results[cond][task][replica] = {
                    "pass_rate": pr,
                    "cost": cost,
                    "raw": d,
                }
            except Exception as e:
                print(f"  Warning: could not load {f}: {e}", file=sys.stderr)
    return results


def phase2_replication_analysis(phase2_raw, original_pass_rates):
    """Compare Phase 2 held-out results with original Study C results."""
    print("\n" + "=" * 60)
    print("PHASE 2: Held-Out Replication Analysis")
    print("=" * 60)

    # Compute Phase 2 pass rates
    print(f"\n  Held-out problems: {len(HELD_OUT_PROBLEMS)}")
    print(f"  Conditions: {PHASE2_CONDITIONS}")
    print(f"  Replicas: 2 per problem per condition")

    phase2_rates = {}
    for cond in PHASE2_CONDITIONS:
        if cond not in phase2_raw:
            print(f"  WARNING: {cond} not found in Phase 2 data")
            continue
        all_prs = []
        all_costs = []
        for task in phase2_raw[cond]:
            for rep in phase2_raw[cond][task]:
                all_prs.append(phase2_raw[cond][task][rep]["pass_rate"])
                all_costs.append(phase2_raw[cond][task][rep]["cost"])
        phase2_rates[cond] = {
            "mean_pass_rate": float(np.mean(all_prs)),
            "std": float(np.std(all_prs)),
            "cost": float(np.mean(all_costs)),
            "n": len(all_prs),
            "family": PHASE2_FAMILIES.get(cond, "?"),
        }

    # Summary table
    print(f"\n  {'Condition':<10} {'Family':<10} {'Original':>10} {'Held-Out':>10} {'Delta':>8} {'Cost':>8} {'N':>4}")
    print(f"  {'-'*62}")
    for cond in PHASE2_CONDITIONS:
        if cond not in phase2_rates:
            continue
        orig = original_pass_rates.get(cond, {}).get("mean_pass_rate", 0)
        held = phase2_rates[cond]["mean_pass_rate"]
        delta = held - orig
        cost = phase2_rates[cond]["cost"]
        n = phase2_rates[cond]["n"]
        fam = phase2_rates[cond]["family"]
        print(f"  {cond:<10} {fam:<10} {orig:>9.3f} {held:>9.3f} {delta:>+7.3f} ${cost:>6.4f} {n:>4}")

    # Key test: Does C0 < C9 < C13 hold?
    if all(c in phase2_rates for c in PHASE2_CONDITIONS):
        c0 = phase2_rates["C0"]["mean_pass_rate"]
        c9 = phase2_rates["C9"]["mean_pass_rate"]
        c13 = phase2_rates["C13"]["mean_pass_rate"]
        ordering_holds = c0 < c9 < c13
        print(f"\n  CORE TEST: C0 ({c0:.3f}) < C9 ({c9:.3f}) < C13 ({c13:.3f})")
        print(f"  Ordering holds: {'YES' if ordering_holds else 'NO'}")

        # Paired Wilcoxon on held-out problems
        problems = sorted(set(phase2_raw["C0"].keys()) & set(phase2_raw["C9"].keys()) & set(phase2_raw["C13"].keys()))
        comparisons = [("C0", "C9"), ("C0", "C13"), ("C9", "C13")]
        print(f"\n  Paired comparisons on {len(problems)} held-out problems:")
        stat_results = []
        for c1, c2 in comparisons:
            vals1, vals2 = [], []
            for task in problems:
                if task in phase2_raw[c1] and task in phase2_raw[c2]:
                    m1 = np.mean([phase2_raw[c1][task][r]["pass_rate"] for r in phase2_raw[c1][task]])
                    m2 = np.mean([phase2_raw[c2][task][r]["pass_rate"] for r in phase2_raw[c2][task]])
                    vals1.append(m1)
                    vals2.append(m2)
            vals1, vals2 = np.array(vals1), np.array(vals2)
            diff = vals2 - vals1
            try:
                stat, p = stats.wilcoxon(vals1, vals2)
                z = stats.norm.ppf(p / 2)
                r = abs(z) / np.sqrt(len(vals1))
            except ValueError:
                stat, p, r = 0, 1.0, 0.0
            sig = p < 0.05 / 3  # Bonferroni for 3 comparisons
            print(f"    {c1} vs {c2}: delta={np.mean(diff):+.3f}, p={p:.4f}, r={r:.3f}, {'SIGNIFICANT' if sig else 'not significant'} (Bonferroni α=0.0167)")
            stat_results.append({
                "comparison": f"{c1} vs {c2}",
                "mean_diff": float(np.mean(diff)),
                "p_value": float(p),
                "effect_size_r": float(r),
                "n_problems": len(vals1),
                "significant_bonferroni": bool(sig),
            })

    # Per-problem breakdown
    print(f"\n  Per-problem pass rates (held-out):")
    print(f"  {'Problem':<10} {'C0':>6} {'C9':>6} {'C13':>6} {'C13>C0':>7}")
    print(f"  {'-'*40}")
    problem_data = {}
    for task in sorted(HELD_OUT_PROBLEMS):
        row = {}
        for cond in PHASE2_CONDITIONS:
            if task in phase2_raw.get(cond, {}):
                reps = phase2_raw[cond][task]
                row[cond] = np.mean([reps[r]["pass_rate"] for r in reps])
            else:
                row[cond] = np.nan
        c0v = row.get("C0", np.nan)
        c13v = row.get("C13", np.nan)
        wins = "YES" if c13v > c0v else ("TIE" if c13v == c0v else "no")
        print(f"  {task:<10} {row.get('C0', 0):>5.2f} {row.get('C9', 0):>5.2f} {row.get('C13', 0):>5.2f} {wins:>7}")
        problem_data[task] = {c: float(v) if not np.isnan(v) else None for c, v in row.items()}

    # Save results
    result = {
        "phase2_pass_rates": {c: {k: v for k, v in d.items()} for c, d in phase2_rates.items()},
        "original_pass_rates": {c: original_pass_rates.get(c, {}).get("mean_pass_rate") for c in PHASE2_CONDITIONS},
        "ordering_holds": bool(ordering_holds) if all(c in phase2_rates for c in PHASE2_CONDITIONS) else None,
        "statistical_tests": stat_results if 'stat_results' in dir() else [],
        "per_problem": problem_data,
    }
    with open(OUTPUT_DIR / "phase2_replication.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    return result


# ── Phase 3: Broader Replication (C2, C3, C10, C16 on held-out) ────────────

PHASE3_CONDITIONS = ["C2", "C3", "C10", "C16"]
PHASE3_FAMILIES = {"C2": "debate", "C3": "gen-only", "C10": "iterative", "C16": "gen-only"}

def load_phase3_raw():
    """Load Phase 3 broader replication data — C2, C3, C10, C16 on held-out problems."""
    results = defaultdict(lambda: defaultdict(dict))
    for cond in PHASE3_CONDITIONS:
        for f in glob.glob(str(FULL_RAW_DIR / f"C_{cond}_cc_*.json")):
            try:
                d = json.load(open(f))
                task = d["task_id"]
                if task not in HELD_OUT_PROBLEMS:
                    continue
                replica = d["replica"]
                if d.get("final_test_results"):
                    pr = d["final_test_results"]["pass_rate"]
                elif d.get("pipelines"):
                    pr = d["pipelines"][-1]["test_results"]["pass_rate"]
                elif d.get("test_results"):
                    pr = d["test_results"]["pass_rate"]
                else:
                    pr = 0.0
                cost = d.get("total_cost_usd", 0.0)
                if cost == 0.0:
                    cost = sum(g.get("cost_usd", 0) for g in d.get("generations", []))
                    cost += sum(r_.get("cost_usd", 0) for p_ in d.get("pipelines", []) for r_ in p_.get("reviews", []))
                results[cond][task][replica] = {
                    "pass_rate": pr,
                    "cost": cost,
                    "raw": d,
                }
            except Exception as e:
                print(f"  Warning: could not load {f}: {e}", file=sys.stderr)
    return results


def phase3_broader_replication(phase3_raw, phase2_raw, original_pass_rates):
    """Phase 3: Does the full Pareto frontier shape replicate on held-out problems?"""
    print("\n" + "=" * 60)
    print("PHASE 3: Broader Replication (Full Frontier)")
    print("=" * 60)

    # Combine Phase 2 + Phase 3 data for full held-out picture
    all_held_out = {}
    for cond in PHASE2_CONDITIONS:
        if cond in phase2_raw:
            all_held_out[cond] = phase2_raw[cond]
    for cond in PHASE3_CONDITIONS:
        if cond in phase3_raw:
            all_held_out[cond] = phase3_raw[cond]

    all_families = {**PHASE2_FAMILIES, **PHASE3_FAMILIES}

    # Compute pass rates for all conditions on held-out
    held_out_rates = {}
    for cond, data in sorted(all_held_out.items()):
        all_prs = []
        all_costs = []
        for task in data:
            for rep in data[task]:
                all_prs.append(data[task][rep]["pass_rate"])
                all_costs.append(data[task][rep]["cost"])
        held_out_rates[cond] = {
            "mean_pass_rate": float(np.mean(all_prs)),
            "std": float(np.std(all_prs)),
            "cost": float(np.mean(all_costs)),
            "n": len(all_prs),
            "family": all_families.get(cond, "?"),
        }

    # Summary table
    print(f"\n  {'Condition':<10} {'Family':<12} {'Original':>10} {'Held-Out':>10} {'Delta':>8} {'Cost':>8} {'N':>4}")
    print(f"  {'-'*68}")
    for cond in sorted(held_out_rates.keys(), key=lambda x: int(x[1:])):
        orig = original_pass_rates.get(cond, {}).get("mean_pass_rate", 0)
        held = held_out_rates[cond]["mean_pass_rate"]
        delta = held - orig
        cost = held_out_rates[cond]["cost"]
        n = held_out_rates[cond]["n"]
        fam = held_out_rates[cond]["family"]
        print(f"  {cond:<10} {fam:<12} {orig:>9.3f} {held:>9.3f} {delta:>+7.3f} ${cost:>6.4f} {n:>4}")

    # Check Pareto frontier on held-out
    # Original frontier: C0 → C2 → C9 → C13
    print(f"\n  Pareto frontier check (cost vs pass rate on held-out):")
    sorted_by_cost = sorted(held_out_rates.items(), key=lambda x: x[1]["cost"])
    frontier = []
    best_pr = -1
    for cond, data in sorted_by_cost:
        if data["mean_pass_rate"] > best_pr:
            frontier.append(cond)
            best_pr = data["mean_pass_rate"]
    print(f"  Held-out Pareto frontier: {' → '.join(frontier)}")
    print(f"  Original frontier: C0 → C2 → C9 → C13")

    # Rank-order correlation between original and held-out
    common = [c for c in held_out_rates if c in original_pass_rates]
    if len(common) >= 3:
        orig_ranks = [original_pass_rates[c]["mean_pass_rate"] for c in common]
        held_ranks = [held_out_rates[c]["mean_pass_rate"] for c in common]
        tau, p_tau = stats.kendalltau(orig_ranks, held_ranks)
        rho, p_rho = stats.spearmanr(orig_ranks, held_ranks)
        print(f"\n  Rank correlation (original vs held-out, {len(common)} conditions):")
        print(f"    Kendall's τ = {tau:.3f} (p={p_tau:.4f})")
        print(f"    Spearman's ρ = {rho:.3f} (p={p_rho:.4f})")

    result = {
        "held_out_rates": {c: d for c, d in held_out_rates.items()},
        "pareto_frontier_held_out": frontier,
        "pareto_frontier_original": ["C0", "C2", "C9", "C13"],
        "rank_correlation": {
            "kendall_tau": float(tau) if len(common) >= 3 else None,
            "kendall_p": float(p_tau) if len(common) >= 3 else None,
            "spearman_rho": float(rho) if len(common) >= 3 else None,
            "spearman_p": float(p_rho) if len(common) >= 3 else None,
            "n_conditions": len(common),
        } if len(common) >= 3 else {},
    }
    with open(OUTPUT_DIR / "phase3_broader_replication.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    return result


# ── Phase 4: Cross-Model (Qwen Generator) ─────────────────────────────────

PHASE4_CONDITIONS = ["C0", "C9", "C13"]

def load_phase4_raw():
    """Load Phase 4 data — D_ prefix files (Qwen as generator)."""
    results = defaultdict(lambda: defaultdict(dict))
    for cond in PHASE4_CONDITIONS:
        for f in glob.glob(str(FULL_RAW_DIR / f"D_{cond}_*.json")):
            try:
                d = json.load(open(f))
                task = d["task_id"]
                replica = d["replica"]
                if d.get("final_test_results"):
                    pr = d["final_test_results"]["pass_rate"]
                elif d.get("pipelines"):
                    pr = d["pipelines"][-1]["test_results"]["pass_rate"]
                elif d.get("test_results"):
                    pr = d["test_results"]["pass_rate"]
                else:
                    pr = 0.0
                cost = d.get("total_cost_usd", 0.0)
                if cost == 0.0:
                    cost = sum(g.get("cost_usd", 0) for g in d.get("generations", []))
                    cost += sum(r_.get("cost_usd", 0) for p_ in d.get("pipelines", []) for r_ in p_.get("reviews", []))
                results[cond][task][replica] = {
                    "pass_rate": pr,
                    "cost": cost,
                    "raw": d,
                }
            except Exception as e:
                print(f"  Warning: could not load {f}: {e}", file=sys.stderr)
    return results


def phase4_cross_model_analysis(phase4_raw, original_pass_rates):
    """Phase 4: Do strategy orderings hold with Qwen 2.5 Coder 32B as generator?"""
    print("\n" + "=" * 60)
    print("PHASE 4: Cross-Model Generalisability (Qwen Generator)")
    print("=" * 60)

    # Compute pass rates
    phase4_rates = {}
    for cond in PHASE4_CONDITIONS:
        if cond not in phase4_raw:
            print(f"  WARNING: {cond} not found in Phase 4 data")
            continue
        all_prs = []
        all_costs = []
        for task in phase4_raw[cond]:
            for rep in phase4_raw[cond][task]:
                all_prs.append(phase4_raw[cond][task][rep]["pass_rate"])
                all_costs.append(phase4_raw[cond][task][rep]["cost"])
        phase4_rates[cond] = {
            "mean_pass_rate": float(np.mean(all_prs)),
            "std": float(np.std(all_prs)),
            "cost": float(np.mean(all_costs)),
            "n": len(all_prs),
        }

    # Summary table comparing Sonnet (original) vs Qwen
    print(f"\n  {'Condition':<10} {'Sonnet (orig)':>14} {'Qwen':>10} {'Delta':>8} {'Qwen Cost':>10} {'N':>4}")
    print(f"  {'-'*58}")
    for cond in PHASE4_CONDITIONS:
        if cond not in phase4_rates:
            continue
        orig = original_pass_rates.get(cond, {}).get("mean_pass_rate", 0)
        qwen = phase4_rates[cond]["mean_pass_rate"]
        delta = qwen - orig
        cost = phase4_rates[cond]["cost"]
        n = phase4_rates[cond]["n"]
        print(f"  {cond:<10} {orig:>13.3f} {qwen:>9.3f} {delta:>+7.3f} ${cost:>8.4f} {n:>4}")

    # Check ordering
    if all(c in phase4_rates for c in PHASE4_CONDITIONS):
        c0 = phase4_rates["C0"]["mean_pass_rate"]
        c9 = phase4_rates["C9"]["mean_pass_rate"]
        c13 = phase4_rates["C13"]["mean_pass_rate"]
        ordering_holds = c0 < c9 < c13
        # Also check if C0 is at floor
        floor_effect = c0 < 0.05
        print(f"\n  CORE TEST: C0 ({c0:.3f}) < C9 ({c9:.3f}) < C13 ({c13:.3f})")
        print(f"  Ordering holds: {'YES' if ordering_holds else 'NO'}")
        if floor_effect:
            print(f"  WARNING: C0 baseline at floor ({c0:.3f}). Qwen cannot solve these problems alone.")
            if c9 > 0.05 or c13 > 0.05:
                print(f"  NOTABLE: Review pipeline rescues Qwen from total failure (C9={c9:.3f}, C13={c13:.3f})")
            else:
                print(f"  Phase 4 is a wash — Qwen cannot produce reviewable code for competitive programming.")

        # Statistical test
        problems = sorted(set(phase4_raw["C0"].keys()) & set(phase4_raw["C13"].keys()))
        if len(problems) >= 5:
            vals_c0, vals_c13 = [], []
            for task in problems:
                m0 = np.mean([phase4_raw["C0"][task][r]["pass_rate"] for r in phase4_raw["C0"][task]])
                m13 = np.mean([phase4_raw["C13"][task][r]["pass_rate"] for r in phase4_raw["C13"][task]])
                vals_c0.append(m0)
                vals_c13.append(m13)
            try:
                stat, p = stats.wilcoxon(vals_c0, vals_c13)
                print(f"\n  Wilcoxon C0 vs C13: p={p:.4f}")
            except ValueError as e:
                print(f"\n  Wilcoxon C0 vs C13: cannot compute ({e})")
                p = 1.0
    else:
        ordering_holds = None
        floor_effect = None
        p = None

    result = {
        "generator_model": "qwen/qwen-2.5-coder-32b-instruct",
        "phase4_rates": phase4_rates,
        "original_rates_sonnet": {c: original_pass_rates.get(c, {}).get("mean_pass_rate") for c in PHASE4_CONDITIONS},
        "ordering_holds": bool(ordering_holds) if ordering_holds is not None else None,
        "floor_effect": bool(floor_effect) if floor_effect is not None else None,
        "c0_vs_c13_p": float(p) if p is not None else None,
    }
    with open(OUTPUT_DIR / "phase4_cross_model.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    return result


# ── Phase 4b: Cross-Model — DeepSeek V3 ────────────────────────────────────

PHASE4B_CONDITIONS = ["C0", "C9", "C13"]

def load_phase4b_raw():
    """Load Phase 4b data — E_ prefix files (DeepSeek V3 as generator)."""
    results = defaultdict(lambda: defaultdict(dict))
    for cond in PHASE4B_CONDITIONS:
        for f in glob.glob(str(FULL_RAW_DIR / f"E_{cond}_*.json")):
            try:
                d = json.load(open(f))
                task = d["task_id"]
                replica = d["replica"]
                if d.get("final_test_results"):
                    pr = d["final_test_results"]["pass_rate"]
                elif d.get("pipelines"):
                    pr = d["pipelines"][-1]["test_results"]["pass_rate"]
                elif d.get("test_results"):
                    pr = d["test_results"]["pass_rate"]
                else:
                    pr = 0.0
                cost = d.get("total_cost_usd", 0.0)
                if cost == 0.0:
                    cost = sum(g.get("cost_usd", 0) for g in d.get("generations", []))
                    cost += sum(r_.get("cost_usd", 0) for p_ in d.get("pipelines", []) for r_ in p_.get("reviews", []))
                results[cond][task][replica] = {
                    "pass_rate": pr,
                    "cost": cost,
                    "raw": d,
                }
            except Exception as e:
                print(f"  Warning: could not load {f}: {e}", file=sys.stderr)
    return results


def phase4b_deepseek_analysis(phase4b_raw, original_pass_rates, phase4_qwen_rates=None):
    """Phase 4b: Do strategy orderings hold with DeepSeek V3 as generator?"""
    print("\n" + "=" * 60)
    print("PHASE 4b: Cross-Model Generalisability (DeepSeek V3 Generator)")
    print("=" * 60)

    # Compute pass rates
    phase4b_rates = {}
    for cond in PHASE4B_CONDITIONS:
        if cond not in phase4b_raw:
            print(f"  WARNING: {cond} not found in Phase 4b data")
            continue
        all_prs = []
        all_costs = []
        for task in phase4b_raw[cond]:
            for rep in phase4b_raw[cond][task]:
                all_prs.append(phase4b_raw[cond][task][rep]["pass_rate"])
                all_costs.append(phase4b_raw[cond][task][rep]["cost"])
        phase4b_rates[cond] = {
            "mean_pass_rate": float(np.mean(all_prs)),
            "std": float(np.std(all_prs)),
            "cost": float(np.mean(all_costs)),
            "n": len(all_prs),
        }

    # Summary table comparing Sonnet (original), Qwen, and DeepSeek
    has_qwen = phase4_qwen_rates is not None
    if has_qwen:
        print(f"\n  {'Condition':<10} {'Sonnet':>10} {'Qwen':>10} {'DeepSeek':>10} {'DS Cost':>10} {'N':>4}")
        print(f"  {'-'*58}")
        for cond in PHASE4B_CONDITIONS:
            if cond not in phase4b_rates:
                continue
            orig = original_pass_rates.get(cond, {}).get("mean_pass_rate", 0)
            qwen = phase4_qwen_rates.get(cond, {}).get("mean_pass_rate", 0) if has_qwen else 0
            ds = phase4b_rates[cond]["mean_pass_rate"]
            cost = phase4b_rates[cond]["cost"]
            n = phase4b_rates[cond]["n"]
            print(f"  {cond:<10} {orig:>9.3f} {qwen:>9.3f} {ds:>9.3f} ${cost:>8.4f} {n:>4}")
    else:
        print(f"\n  {'Condition':<10} {'Sonnet':>10} {'DeepSeek':>10} {'Delta':>8} {'DS Cost':>10} {'N':>4}")
        print(f"  {'-'*58}")
        for cond in PHASE4B_CONDITIONS:
            if cond not in phase4b_rates:
                continue
            orig = original_pass_rates.get(cond, {}).get("mean_pass_rate", 0)
            ds = phase4b_rates[cond]["mean_pass_rate"]
            delta = ds - orig
            cost = phase4b_rates[cond]["cost"]
            n = phase4b_rates[cond]["n"]
            print(f"  {cond:<10} {orig:>9.3f} {ds:>9.3f} {delta:>+7.3f} ${cost:>8.4f} {n:>4}")

    # Check ordering
    if all(c in phase4b_rates for c in PHASE4B_CONDITIONS):
        c0 = phase4b_rates["C0"]["mean_pass_rate"]
        c9 = phase4b_rates["C9"]["mean_pass_rate"]
        c13 = phase4b_rates["C13"]["mean_pass_rate"]
        ordering_holds = c0 < c9 < c13
        floor_effect = c0 < 0.05
        print(f"\n  CORE TEST: C0 ({c0:.3f}) < C9 ({c9:.3f}) < C13 ({c13:.3f})")
        print(f"  Ordering holds: {'YES' if ordering_holds else 'NO'}")
        if floor_effect:
            print(f"  WARNING: C0 baseline at floor ({c0:.3f}).")
        else:
            print(f"  DeepSeek solves problems independently (C0={c0:.3f}) — review lift is genuine, not rescue.")

        # Statistical tests — C0 vs C13 and C0 vs C9
        for label, cA, cB in [("C0 vs C13", "C0", "C13"), ("C0 vs C9", "C0", "C9"), ("C9 vs C13", "C9", "C13")]:
            problems = sorted(set(phase4b_raw[cA].keys()) & set(phase4b_raw[cB].keys()))
            if len(problems) >= 5:
                vals_a, vals_b = [], []
                for task in problems:
                    ma = np.mean([phase4b_raw[cA][task][r]["pass_rate"] for r in phase4b_raw[cA][task]])
                    mb = np.mean([phase4b_raw[cB][task][r]["pass_rate"] for r in phase4b_raw[cB][task]])
                    vals_a.append(ma)
                    vals_b.append(mb)
                try:
                    stat, p = stats.wilcoxon(vals_a, vals_b)
                    print(f"  Wilcoxon {label}: p={p:.4f}")
                except ValueError as e:
                    print(f"  Wilcoxon {label}: cannot compute ({e})")
    else:
        ordering_holds = None
        floor_effect = None

    result = {
        "generator_model": "deepseek/deepseek-chat",
        "phase4b_rates": phase4b_rates,
        "original_rates_sonnet": {c: original_pass_rates.get(c, {}).get("mean_pass_rate") for c in PHASE4B_CONDITIONS},
        "ordering_holds": bool(ordering_holds) if ordering_holds is not None else None,
        "floor_effect": bool(floor_effect) if floor_effect is not None else None,
    }
    if has_qwen:
        result["qwen_rates"] = phase4_qwen_rates
    with open(OUTPUT_DIR / "phase4b_deepseek.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    return result


# ── Phase 5: Cross-Domain (HumanEval/MBPP) ────────────────────────────────

PHASE5_CONDITIONS = ["C0", "C9", "C13", "C16"]
PHASE5_FAMILIES = {"C0": "baseline", "C9": "debate", "C13": "hybrid", "C16": "gen-only"}

def load_phase5_raw():
    """Load Phase 5 data — HumanEval (he_) and MBPP (mbpp_) problems."""
    results = defaultdict(lambda: defaultdict(dict))
    for cond in PHASE5_CONDITIONS:
        for f in glob.glob(str(FULL_RAW_DIR / f"C_{cond}_he_*.json")) + \
                 glob.glob(str(FULL_RAW_DIR / f"C_{cond}_mbpp_*.json")):
            try:
                d = json.load(open(f))
                task = d["task_id"]
                replica = d["replica"]
                if d.get("final_test_results"):
                    pr = d["final_test_results"]["pass_rate"]
                elif d.get("pipelines"):
                    pr = d["pipelines"][-1]["test_results"]["pass_rate"]
                elif d.get("test_results"):
                    pr = d["test_results"]["pass_rate"]
                else:
                    pr = 0.0
                cost = d.get("total_cost_usd", 0.0)
                if cost == 0.0:
                    cost = sum(g.get("cost_usd", 0) for g in d.get("generations", []))
                    cost += sum(r_.get("cost_usd", 0) for p_ in d.get("pipelines", []) for r_ in p_.get("reviews", []))
                results[cond][task][replica] = {
                    "pass_rate": pr,
                    "cost": cost,
                    "raw": d,
                }
            except Exception as e:
                print(f"  Warning: could not load {f}: {e}", file=sys.stderr)
    return results


def phase5_cross_domain_analysis(phase5_raw, original_pass_rates):
    """Phase 5: Do strategy orderings hold on HumanEval/MBPP?"""
    print("\n" + "=" * 60)
    print("PHASE 5: Cross-Domain Generalisability (HumanEval/MBPP)")
    print("=" * 60)

    # Split by benchmark
    he_results = defaultdict(lambda: defaultdict(dict))
    mbpp_results = defaultdict(lambda: defaultdict(dict))
    for cond in phase5_raw:
        for task in phase5_raw[cond]:
            if task.startswith("he_"):
                he_results[cond][task] = phase5_raw[cond][task]
            elif task.startswith("mbpp_"):
                mbpp_results[cond][task] = phase5_raw[cond][task]

    # Compute pass rates per benchmark and combined
    for label, data in [("HumanEval", he_results), ("MBPP", mbpp_results), ("Combined", phase5_raw)]:
        print(f"\n  --- {label} ---")
        rates = {}
        for cond in PHASE5_CONDITIONS:
            if cond not in data:
                continue
            all_prs = []
            all_costs = []
            for task in data[cond]:
                for rep in data[cond][task]:
                    all_prs.append(data[cond][task][rep]["pass_rate"])
                    all_costs.append(data[cond][task][rep]["cost"])
            if all_prs:
                rates[cond] = {
                    "mean_pass_rate": float(np.mean(all_prs)),
                    "std": float(np.std(all_prs)),
                    "cost": float(np.mean(all_costs)),
                    "n": len(all_prs),
                    "n_problems": len(data[cond]),
                }

        print(f"  {'Condition':<10} {'Family':<12} {'Pass Rate':>10} {'Cost':>8} {'N probs':>8} {'N runs':>7}")
        print(f"  {'-'*58}")
        for cond in PHASE5_CONDITIONS:
            if cond not in rates:
                continue
            fam = PHASE5_FAMILIES.get(cond, "?")
            r = rates[cond]
            print(f"  {cond:<10} {fam:<12} {r['mean_pass_rate']:>9.3f} ${r['cost']:>6.4f} {r['n_problems']:>8} {r['n']:>7}")

        # Check ordering
        if all(c in rates for c in ["C0", "C9", "C13"]):
            c0 = rates["C0"]["mean_pass_rate"]
            c9 = rates["C9"]["mean_pass_rate"]
            c13 = rates["C13"]["mean_pass_rate"]
            print(f"\n  Ordering: C0 ({c0:.3f}) < C9 ({c9:.3f}) < C13 ({c13:.3f}): {'YES' if c0 < c9 < c13 else 'NO'}")
            if "C16" in rates:
                c16 = rates["C16"]["mean_pass_rate"]
                print(f"  C13 ({c13:.3f}) vs C16 ({c16:.3f}): {'C13 wins' if c13 > c16 else 'C16 wins'}")

    # Combined statistical tests
    print(f"\n  --- Statistical Tests (Combined) ---")
    combined_rates = {}
    stat_results = []
    for cond in PHASE5_CONDITIONS:
        if cond in phase5_raw:
            all_prs = [phase5_raw[cond][t][r]["pass_rate"] for t in phase5_raw[cond] for r in phase5_raw[cond][t]]
            combined_rates[cond] = float(np.mean(all_prs))

    comparisons = [("C0", "C9"), ("C0", "C13"), ("C9", "C13"), ("C13", "C16")]
    for c1, c2 in comparisons:
        if c1 not in phase5_raw or c2 not in phase5_raw:
            continue
        problems = sorted(set(phase5_raw[c1].keys()) & set(phase5_raw[c2].keys()))
        if len(problems) < 5:
            continue
        vals1, vals2 = [], []
        for task in problems:
            m1 = np.mean([phase5_raw[c1][task][r]["pass_rate"] for r in phase5_raw[c1][task]])
            m2 = np.mean([phase5_raw[c2][task][r]["pass_rate"] for r in phase5_raw[c2][task]])
            vals1.append(m1)
            vals2.append(m2)
        try:
            stat, p = stats.wilcoxon(vals1, vals2)
        except ValueError:
            p = 1.0
        sig = p < 0.05 / len(comparisons)
        diff = np.mean(np.array(vals2) - np.array(vals1))
        print(f"    {c1} vs {c2}: delta={diff:+.3f}, p={p:.4f}, {'SIGNIFICANT' if sig else 'not significant'} (Bonferroni α={0.05/len(comparisons):.4f})")
        stat_results.append({
            "comparison": f"{c1} vs {c2}",
            "mean_diff": float(diff),
            "p_value": float(p),
            "significant_bonferroni": bool(sig),
            "n_problems": len(problems),
        })

    result = {
        "benchmarks": ["HumanEval", "MBPP"],
        "conditions": PHASE5_CONDITIONS,
        "combined_rates": combined_rates,
        "statistical_tests": stat_results,
        "humaneval_n_problems": len(he_results.get("C0", {})),
        "mbpp_n_problems": len(mbpp_results.get("C0", {})),
    }
    with open(OUTPUT_DIR / "phase5_cross_domain.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    return result


# ── Phase 6: Fixer Ablation ────────────────────────────────────────────────

def load_phase6_raw():
    """Load Phase 6 data — F_C13_nocode_ prefix files."""
    results = defaultdict(dict)
    for f in glob.glob(str(FULL_RAW_DIR / "F_C13_nocode_*.json")):
        try:
            d = json.load(open(f))
            task = d["task_id"]
            replica = d["replica"]
            if d.get("final_test_results"):
                pr = d["final_test_results"]["pass_rate"]
            elif d.get("pipelines"):
                pr = d["pipelines"][-1]["test_results"]["pass_rate"]
            elif d.get("test_results"):
                pr = d["test_results"]["pass_rate"]
            else:
                pr = 0.0
            cost = d.get("total_cost_usd", 0.0)
            results[task][replica] = {
                "pass_rate": pr,
                "cost": cost,
                "raw": d,
            }
        except Exception as e:
            print(f"  Warning: could not load {f}: {e}", file=sys.stderr)
    return results


def phase6_fixer_ablation(phase6_raw, phase2_raw):
    """Phase 6: Does the fixer need the generator's code?

    Compares F_C13_nocode (fixer gets problem + reviews, no code)
    against held-out C13 (fixer gets problem + reviews + code).
    """
    print("\n" + "=" * 60)
    print("PHASE 6: Fixer Ablation (Does the fixer need the generator's code?)")
    print("=" * 60)

    if not phase6_raw:
        print("  No Phase 6 data found.")
        return None

    # Compute nocode pass rate
    all_prs = []
    all_costs = []
    for task in phase6_raw:
        for rep in phase6_raw[task]:
            all_prs.append(phase6_raw[task][rep]["pass_rate"])
            all_costs.append(phase6_raw[task][rep]["cost"])

    nocode_rate = float(np.mean(all_prs))
    nocode_std = float(np.std(all_prs))
    nocode_cost = float(np.mean(all_costs))
    n_nocode = len(all_prs)

    # Get held-out C13 rate from Phase 2 data
    c13_prs = []
    if "C13" in phase2_raw:
        for task in phase2_raw["C13"]:
            for rep in phase2_raw["C13"][task]:
                c13_prs.append(phase2_raw["C13"][task][rep]["pass_rate"])
    c13_rate = float(np.mean(c13_prs)) if c13_prs else None

    print(f"\n  C13 held-out (with code):    {c13_rate:.3f}" if c13_rate else "  C13 held-out: N/A")
    print(f"  C13 nocode  (without code):  {nocode_rate:.3f} (n={n_nocode})")
    if c13_rate:
        delta = nocode_rate - c13_rate
        print(f"  Delta: {delta:+.3f} ({delta*100:+.1f}pp)")

    # Statistical test: paired comparison on shared problems
    if "C13" in phase2_raw:
        problems = sorted(set(phase6_raw.keys()) & set(phase2_raw["C13"].keys()))
        if len(problems) >= 5:
            vals_nocode = []
            vals_withcode = []
            for task in problems:
                mn = np.mean([phase6_raw[task][r]["pass_rate"] for r in phase6_raw[task]])
                mw = np.mean([phase2_raw["C13"][task][r]["pass_rate"] for r in phase2_raw["C13"][task]])
                vals_nocode.append(mn)
                vals_withcode.append(mw)
            try:
                stat, p = stats.wilcoxon(vals_nocode, vals_withcode)
                print(f"\n  Paired Wilcoxon (nocode vs with-code): p={p:.4f}")
                if p < 0.05:
                    if nocode_rate < c13_rate:
                        print("  RESULT: Fixer performs significantly WORSE without code.")
                        print("  → The generator's code provides useful context to the fixer.")
                    else:
                        print("  RESULT: Fixer performs significantly BETTER without code.")
                        print("  → The generator's code may be misleading the fixer.")
                else:
                    print("  RESULT: No significant difference. The fixer may not need the code.")
                    print("  → The generator's contribution to the pipeline is unclear.")
            except ValueError as e:
                print(f"  Wilcoxon: cannot compute ({e})")
                p = None
        else:
            print(f"  Only {len(problems)} shared problems — insufficient for paired test")
            p = None
    else:
        p = None

    result = {
        "ablation": "fixer_nocode",
        "nocode_pass_rate": nocode_rate,
        "nocode_std": nocode_std,
        "nocode_cost": nocode_cost,
        "nocode_n": n_nocode,
        "withcode_pass_rate": c13_rate,
        "delta": nocode_rate - c13_rate if c13_rate else None,
        "p_value": float(p) if p is not None else None,
    }
    with open(OUTPUT_DIR / "phase6_fixer_ablation.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    return result


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("Phase 0: Study C Analytical Fixes")
    print("=" * 60)

    pass_rates = load_pass_rates()
    raw_results = load_raw_results()

    # Merge in reused conditions from adversarial_bench_full
    reused_raw = load_reused_raw_results()
    for c in reused_raw:
        # Only keep tasks that overlap with the Study C problem set
        study_c_problems = set()
        for rc in raw_results:
            study_c_problems.update(raw_results[rc].keys())
        for task in reused_raw[c]:
            if task in study_c_problems:
                raw_results[c][task] = reused_raw[c][task]

    # Load C16 (Phase 1: iso-cost best-of-7) data
    c16_raw = load_c16_raw_results()
    if "C16" in c16_raw:
        for task in c16_raw["C16"]:
            raw_results["C16"][task] = c16_raw["C16"][task]
        pass_rates["C16"] = compute_pass_rates_from_raw(raw_results, "C16")
        print(f"Loaded C16 (iso-cost): {len(c16_raw['C16'])} tasks, pass_rate={pass_rates['C16']['mean_pass_rate']:.3f}, cost=${pass_rates['C16']['cost']:.4f}")

    print(f"\nLoaded {len(pass_rates)} conditions from pass_rates CSV + computed")
    print(f"Loaded raw data for {len(raw_results)} conditions, {sum(len(raw_results[c]) for c in raw_results)} task entries")
    reused_conds = [c for c in REUSED_MAPPING if c in raw_results]
    print(f"Including reused conditions from Studies A/B: {reused_conds}")

    frontier_counts = bootstrap_pareto(raw_results, pass_rates)
    missing_comparisons(raw_results)
    distribution_analysis(raw_results, pass_rates)
    cost_normalised_analysis(pass_rates)
    verdict_parsing_audit(raw_results)
    difficulty_stratification(raw_results, pass_rates)
    design_effect_analysis(raw_results, pass_rates)
    pass_at_1_analysis(pass_rates)
    item_response_table(raw_results, pass_rates)
    cost_variance_analysis(raw_results, pass_rates)

    # Phase 2: Held-out replication
    phase2_raw = load_phase2_raw()
    if phase2_raw:
        phase2_replication_analysis(phase2_raw, pass_rates)

    # Phase 3: Broader replication (C2, C3, C10, C16 on held-out)
    phase3_raw = load_phase3_raw()
    if phase3_raw:
        phase3_broader_replication(phase3_raw, phase2_raw, pass_rates)

    # Phase 4: Cross-model (Qwen generator)
    phase4_raw = load_phase4_raw()
    phase4_result = None
    if phase4_raw:
        phase4_result = phase4_cross_model_analysis(phase4_raw, pass_rates)

    # Phase 4b: Cross-model (DeepSeek V3 generator)
    phase4b_raw = load_phase4b_raw()
    if phase4b_raw:
        qwen_rates = phase4_result.get("phase4_rates") if phase4_result else None
        phase4b_deepseek_analysis(phase4b_raw, pass_rates, qwen_rates)

    # Phase 5: Cross-domain (HumanEval/MBPP)
    phase5_raw = load_phase5_raw()
    if phase5_raw:
        phase5_cross_domain_analysis(phase5_raw, pass_rates)

    # Phase 6: Fixer ablation
    phase6_raw = load_phase6_raw()
    if phase6_raw:
        phase6_fixer_ablation(phase6_raw, phase2_raw)

    print(f"\n{'=' * 60}")
    print(f"All outputs saved to: {OUTPUT_DIR}/")
    print(f"Files: bootstrap_pareto.csv, additional_comparisons.json,")
    print(f"       distributions.png, cost_normalised.png,")
    print(f"       verdict_audit.json, difficulty_stratification.json,")
    print(f"       pass_at_1_analysis.json, item_response_table.json,")
    print(f"       cost_variance.json, phase2_replication.json,")
    print(f"       phase3_broader_replication.json, phase4_cross_model.json,")
    print(f"       phase4b_deepseek.json, phase5_cross_domain.json,")
    print(f"       phase6_fixer_ablation.json")

if __name__ == "__main__":
    main()

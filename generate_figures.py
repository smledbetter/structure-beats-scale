#!/usr/bin/env python3
"""Generate publication-quality figures for 'Structure Beats Scale' paper.

Reads phase0_results data and produces PDF vector figures for LaTeX inclusion.
"""

import json
import csv
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.ticker as mticker

# Paths
PHASE0 = os.path.join(os.path.dirname(__file__), '..', 'research', 'adversary', 'phase0_results')
FIGURES = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGURES, exist_ok=True)

# Style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

FAMILY_COLORS = {
    'baseline': '#333333',
    'gen-only': '#e74c3c',
    'review-heavy': '#3498db',
    'debate': '#2ecc71',
    'iterative': '#f39c12',
    'hybrid': '#9b59b6',
}

FAMILY_MARKERS = {
    'baseline': 's',
    'gen-only': 'D',
    'review-heavy': '^',
    'debate': 'o',
    'iterative': 'v',
    'hybrid': '*',
}

FAMILY_ORDER = ['baseline', 'gen-only', 'review-heavy', 'debate', 'iterative', 'hybrid']

# Standard errors from paper tables (SE = std of per-problem pass rates / sqrt(n_problems))
CALIBRATION_SE = {
    'C0': 0.082, 'C1': 0.074, 'C2': 0.075, 'C3': 0.080,
    'C4': 0.080, 'C5': 0.079, 'C6': 0.076, 'C7': 0.069,
    'C8': 0.071, 'C9': 0.071, 'C10': 0.074, 'C11': 0.071,
    'C12': 0.074, 'C13': 0.057, 'C14': 0.077, 'C15': 0.071,
    'C16': 0.056,
}

CROSS_MODEL_SE = {
    'Sonnet':   {'C0': 0.088, 'C9': 0.086, 'C13': 0.087},
    'Qwen':     {'C0': 0.000, 'C9': 0.084, 'C13': 0.085},
    'DeepSeek': {'C0': 0.071, 'C9': 0.082, 'C13': 0.082},
}


def load_bootstrap():
    rows = []
    with open(os.path.join(PHASE0, 'bootstrap_pareto.csv')) as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['frontier_inclusion_pct'] = float(row['frontier_inclusion_pct'])
            row['boot_pr_2.5'] = float(row['boot_pr_2.5'])
            row['boot_pr_97.5'] = float(row['boot_pr_97.5'])
            row['boot_cost_2.5'] = float(row['boot_cost_2.5'])
            row['boot_cost_97.5'] = float(row['boot_cost_97.5'])
            rows.append(row)
    return rows


# Calibration data from Table 2 in the paper
CALIBRATION = {
    'C0':  {'family': 'baseline',     'pr': 0.520, 'cost': 0.038, 'pass1': 0.322, 'pareto': True},
    'C1':  {'family': 'review-heavy', 'pr': 0.720, 'cost': 0.063, 'pass1': 0.568, 'pareto': False},
    'C2':  {'family': 'debate',       'pr': 0.699, 'cost': 0.054, 'pass1': 0.544, 'pareto': True},
    'C3':  {'family': 'gen-only',     'pr': 0.635, 'cost': 0.056, 'pass1': 0.433, 'pareto': False},
    'C4':  {'family': 'iterative',    'pr': 0.650, 'cost': 0.057, 'pass1': 0.544, 'pareto': False},
    'C5':  {'family': 'review-heavy', 'pr': 0.666, 'cost': 0.143, 'pass1': 0.533, 'pareto': False},
    'C6':  {'family': 'gen-only',     'pr': 0.689, 'cost': 0.094, 'pass1': 0.522, 'pareto': False},
    'C7':  {'family': 'hybrid',       'pr': 0.746, 'cost': 0.089, 'pass1': 0.489, 'pareto': False},
    'C8':  {'family': 'review-heavy', 'pr': 0.739, 'cost': 0.067, 'pass1': 0.573, 'pareto': False},
    'C9':  {'family': 'debate',       'pr': 0.748, 'cost': 0.058, 'pass1': 0.600, 'pareto': True},
    'C10': {'family': 'iterative',    'pr': 0.725, 'cost': 0.059, 'pass1': 0.618, 'pareto': False},
    'C11': {'family': 'gen-only',     'pr': 0.733, 'cost': 0.162, 'pass1': 0.500, 'pareto': False},
    'C12': {'family': 'iterative',    'pr': 0.720, 'cost': 0.064, 'pass1': 0.596, 'pareto': False},
    'C13': {'family': 'hybrid',       'pr': 0.851, 'cost': 0.141, 'pass1': 0.689, 'pareto': True},
    'C14': {'family': 'iterative',    'pr': 0.689, 'cost': 0.212, 'pass1': 0.556, 'pareto': False},
    'C15': {'family': 'debate',       'pr': 0.738, 'cost': 0.059, 'pass1': 0.578, 'pareto': False},
    'C16': {'family': 'gen-only',     'pr': 0.828, 'cost': 0.221, 'pass1': 0.674, 'pareto': False},
}


def fig1_pareto_frontier():
    """Figure 1: Pareto frontier scatter plot (cost vs pass rate)."""
    boot = {r['condition']: r for r in load_bootstrap()}
    fig, ax = plt.subplots(figsize=(5.5, 4))

    # Plot each family
    plotted_families = set()
    for cond, d in CALIBRATION.items():
        fam = d['family']
        label = fam.replace('-', ' ').title() if fam not in plotted_families else None
        plotted_families.add(fam)

        b = boot.get(cond, {})
        # Error bars from bootstrap
        xerr_lo = d['cost'] - b.get('boot_cost_2.5', d['cost'])
        xerr_hi = b.get('boot_cost_97.5', d['cost']) - d['cost']
        yerr_lo = d['pr'] - b.get('boot_pr_2.5', d['pr'])
        yerr_hi = b.get('boot_pr_97.5', d['pr']) - d['pr']

        ax.errorbar(d['cost'], d['pr'],
                     xerr=[[xerr_lo], [xerr_hi]],
                     yerr=[[yerr_lo], [yerr_hi]],
                     fmt='none', ecolor=FAMILY_COLORS[fam], alpha=0.5, linewidth=1.0)
        ax.scatter(d['cost'], d['pr'],
                   c=FAMILY_COLORS[fam], marker=FAMILY_MARKERS[fam],
                   s=80 if d['pareto'] else 50,
                   edgecolors='black' if d['pareto'] else 'none',
                   linewidths=1.2 if d['pareto'] else 0,
                   zorder=5 if d['pareto'] else 3,
                   label=label)

    # Pareto frontier line
    pareto_conds = ['C0', 'C2', 'C9', 'C13']
    px = [CALIBRATION[c]['cost'] for c in pareto_conds]
    py = [CALIBRATION[c]['pr'] for c in pareto_conds]
    ax.plot(px, py, 'k-', linewidth=1.5, alpha=0.6, zorder=4)

    # Label key conditions
    offsets = {
        'C0': (6, -12), 'C2': (-8, -14), 'C9': (-8, 8),
        'C13': (-12, 8), 'C16': (6, -4),
    }
    for cond, (dx, dy) in offsets.items():
        d = CALIBRATION[cond]
        ax.annotate(cond, (d['cost'], d['pr']),
                    xytext=(dx, dy), textcoords='offset points',
                    fontsize=7, fontweight='bold',
                    arrowprops=dict(arrowstyle='-', color='gray', lw=0.5) if abs(dx) > 8 else None)

    ax.set_xlabel('Cost per Problem (USD)')
    ax.set_ylabel('Pass Rate')
    ax.set_ylim(0.45, 0.95)
    ax.set_xlim(-0.005, 0.25)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    ax.legend(loc='lower right', framealpha=0.9, edgecolor='gray')
    ax.grid(True, alpha=0.15)
    ax.set_title('')

    fig.savefig(os.path.join(FIGURES, 'pareto_frontier.pdf'), format='pdf')
    plt.close(fig)
    print('  [1/8] pareto_frontier.pdf')


def fig2_difficulty_stratification():
    """Figure 2: Difficulty stratification grouped bar chart."""
    with open(os.path.join(PHASE0, 'difficulty_stratification.json')) as f:
        strat = json.load(f)

    conds = ['C0', 'C9', 'C13', 'C16']
    labels = ['Baseline\n(C0)', 'Debate\n(C9)', 'Hybrid\n(C13)', 'Gen×7\n(C16)']
    difficulties = ['easy', 'medium', 'hard']
    diff_labels = ['Easy (n=7)', 'Medium (n=17)', 'Hard (n=6)']
    diff_colors = ['#2ecc71', '#f39c12', '#e74c3c']
    diff_n = {'easy': 7, 'medium': 17, 'hard': 6}  # problems per stratum

    x = np.arange(len(conds))
    width = 0.25

    fig, ax = plt.subplots(figsize=(5, 3.5))
    for i, (diff, dlabel, color) in enumerate(zip(difficulties, diff_labels, diff_colors)):
        vals = [strat[c][diff] for c in conds]
        # SE = sqrt(p*(1-p)/n) where n = problems in stratum
        n = diff_n[diff]
        ses = [np.sqrt(v * (1 - v) / n) for v in vals]
        ax.bar(x + (i - 1) * width, vals, width, yerr=ses, label=dlabel,
               color=color, edgecolor='white', linewidth=0.5,
               capsize=2, error_kw={'linewidth': 0.7, 'alpha': 0.6})

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Pass Rate')
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylim(0, 1.08)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, axis='y', alpha=0.15)

    fig.savefig(os.path.join(FIGURES, 'difficulty_stratification.pdf'), format='pdf')
    plt.close(fig)
    print('  [2/8] difficulty_stratification.pdf')


def fig3_cross_model():
    """Figure 3: Cross-model grouped bar chart (Sonnet vs Qwen vs DeepSeek)."""
    # Held-out data from draft-4 Table 4
    data = {
        'C0':  {'Sonnet': 0.387, 'Qwen': 0.000, 'DeepSeek': 0.256},
        'C9':  {'Sonnet': 0.515, 'Qwen': 0.405, 'DeepSeek': 0.403},
        'C13': {'Sonnet': 0.556, 'Qwen': 0.532, 'DeepSeek': 0.573},
    }

    conds = ['C0', 'C9', 'C13']
    labels = ['Baseline (C0)', 'Debate (C9)', 'Hybrid (C13)']
    models = ['Sonnet', 'Qwen', 'DeepSeek']
    model_colors = ['#9b59b6', '#e74c3c', '#3498db']

    x = np.arange(len(conds))
    width = 0.25

    fig, ax = plt.subplots(figsize=(5, 3.5))
    for i, (model, color) in enumerate(zip(models, model_colors)):
        vals = [data[c][model] for c in conds]
        ses = [CROSS_MODEL_SE[model][c] for c in conds]
        bars = ax.bar(x + (i - 1) * width, vals, width, yerr=ses,
                      label=model, color=color, edgecolor='white', linewidth=0.5,
                      capsize=2, error_kw={'linewidth': 0.8, 'alpha': 0.7})
        # Add value labels
        for bar, val, se in zip(bars, vals, ses):
            if val > 0.02:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + se + 0.015,
                        f'{val:.0%}', ha='center', va='bottom', fontsize=6.5)
            else:
                ax.text(bar.get_x() + bar.get_width()/2, 0.02,
                        '0%', ha='center', va='bottom', fontsize=6.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Pass Rate (Held-Out Problems)')
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylim(0, 0.78)
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, axis='y', alpha=0.15)

    fig.savefig(os.path.join(FIGURES, 'cross_model.pdf'), format='pdf')
    plt.close(fig)
    print('  [3/8] cross_model.pdf')


def fig4_pass_at_1():
    """Figure 4: Pass@1 comparison showing review improves individual pipelines."""
    with open(os.path.join(PHASE0, 'pass_at_1_analysis.json')) as f:
        p1 = json.load(f)['all_pass_at_1']

    # Key comparison: gen-only vs reviewed multi-gen
    conds = ['C0', 'C3', 'C6', 'C7', 'C13', 'C16']
    labels = ['C0\n1 gen\nno review', 'C3\n2 gen\nno review', 'C6\n3 gen\nno review',
              'C7\n2 gen\n+review', 'C13\n3 gen\n+review', 'C16\n7 gen\nno review']
    vals = [p1[c] for c in conds]
    colors = [FAMILY_COLORS[CALIBRATION[c]['family']] for c in conds]
    # Binomial SE: sqrt(p*(1-p)/n), n=30 problems (the independent unit;
    # replicas within a problem are correlated)
    n_problems = 30
    ses = [np.sqrt(v * (1 - v) / n_problems) for v in vals]

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    bars = ax.bar(range(len(conds)), vals, yerr=ses, color=colors,
                  edgecolor='white', linewidth=0.5, capsize=3,
                  error_kw={'linewidth': 0.8, 'alpha': 0.7})

    # Annotate the key comparison
    ax.annotate('', xy=(4, vals[4] + ses[4] + 0.02), xytext=(2, vals[2] + ses[2] + 0.02),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.text(3, max(vals[2] + ses[2], vals[4] + ses[4]) + 0.04, '+16.7pp\n(review + fix effect)',
            ha='center', fontsize=7, fontweight='bold')

    for bar, val, se in zip(bars, vals, ses):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + se + 0.008,
                f'{val:.1%}', ha='center', va='bottom', fontsize=7)

    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel('Pass@1 (Individual Pipeline Success)')
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylim(0, 0.88)
    ax.grid(True, axis='y', alpha=0.15)

    fig.savefig(os.path.join(FIGURES, 'pass_at_1.pdf'), format='pdf')
    plt.close(fig)
    print('  [4/8] pass_at_1.pdf')


def fig5_bootstrap_stability():
    """Figure 5: Bootstrap Pareto frontier inclusion rates."""
    boot = load_bootstrap()
    boot.sort(key=lambda r: r['frontier_inclusion_pct'], reverse=True)

    conds = [r['condition'] for r in boot]
    inclusions = [r['frontier_inclusion_pct'] for r in boot]
    families = [r['family'] for r in boot]
    colors = [FAMILY_COLORS[f] for f in families]

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    bars = ax.barh(range(len(conds)), inclusions, color=colors, edgecolor='white', linewidth=0.5)

    ax.set_yticks(range(len(conds)))
    ax.set_yticklabels(conds, fontsize=7)
    ax.set_xlabel('Pareto Frontier Inclusion (%)')
    ax.set_xlim(0, 105)
    ax.invert_yaxis()
    ax.grid(True, axis='x', alpha=0.15)

    # Value labels
    for i, (bar, val) in enumerate(zip(bars, inclusions)):
        if val > 3:
            ax.text(val + 1, i, f'{val:.1f}%', va='center', fontsize=6.5)

    # Legend by family
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=FAMILY_COLORS[f], label=f.replace('-', ' ').title())
                       for f in FAMILY_ORDER]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=7, framealpha=0.9)

    fig.savefig(os.path.join(FIGURES, 'bootstrap_stability.pdf'), format='pdf')
    plt.close(fig)
    print('  [5/8] bootstrap_stability.pdf')


def fig6_gen_only_scaling():
    """Figure 6: Generation-only scaling curve with C13 comparison."""
    gen_only = {
        1: {'cond': 'C0',  'pr': 0.520, 'cost': 0.038, 'se': CALIBRATION_SE['C0']},
        2: {'cond': 'C3',  'pr': 0.635, 'cost': 0.056, 'se': CALIBRATION_SE['C3']},
        3: {'cond': 'C6',  'pr': 0.689, 'cost': 0.094, 'se': CALIBRATION_SE['C6']},
        5: {'cond': 'C11', 'pr': 0.733, 'cost': 0.162, 'se': CALIBRATION_SE['C11']},
        7: {'cond': 'C16', 'pr': 0.828, 'cost': 0.221, 'se': CALIBRATION_SE['C16']},
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 3))

    # Left: pass rate vs num generations
    # Separate C0 (baseline) from gen-only conditions (C3+)
    go_gens = [g for g in sorted(gen_only.keys()) if g > 1]
    go_prs = [gen_only[g]['pr'] for g in go_gens]
    go_ses = [gen_only[g]['se'] for g in go_gens]
    ax1.errorbar(go_gens, go_prs, yerr=go_ses, fmt='D-', color=FAMILY_COLORS['gen-only'],
                 markersize=6, capsize=3, linewidth=1, elinewidth=0.8, label='Gen-only')
    # C0 as separate baseline point (not connected to gen-only line)
    ax1.errorbar([1], [gen_only[1]['pr']], yerr=[gen_only[1]['se']],
                 fmt='s', color=FAMILY_COLORS['baseline'], markersize=6, capsize=3,
                 linewidth=1, elinewidth=0.8, label='Baseline (C0)')
    ax1.axhline(y=0.851, color=FAMILY_COLORS['hybrid'], linestyle='--', alpha=0.7, label='C13 (hybrid)')
    ax1.axhline(y=0.748, color=FAMILY_COLORS['debate'], linestyle=':', alpha=0.7, label='C9 (debate)')

    gens = sorted(gen_only.keys())
    for g in gens:
        lbl = gen_only[g]['cond']
        ax1.annotate(lbl, (g, gen_only[g]['pr']),
                     xytext=(5, -10), textcoords='offset points', fontsize=6.5)

    ax1.set_xlabel('Number of Generations')
    ax1.set_ylabel('Pass Rate')
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    ax1.set_ylim(0.40, 0.95)
    ax1.legend(fontsize=6, loc='lower right')
    ax1.grid(True, alpha=0.15)

    # Right: pass rate vs cost — same separation
    prs = [gen_only[g]['pr'] for g in gens]
    ses = [gen_only[g]['se'] for g in gens]
    go_costs = [gen_only[g]['cost'] for g in go_gens]
    ax2.errorbar(go_costs, go_prs, yerr=go_ses, fmt='D-', color=FAMILY_COLORS['gen-only'],
                 markersize=6, capsize=3, linewidth=1, elinewidth=0.8, label='Gen-only')
    ax2.errorbar([gen_only[1]['cost']], [gen_only[1]['pr']], yerr=[gen_only[1]['se']],
                 fmt='s', color=FAMILY_COLORS['baseline'], markersize=6, capsize=3,
                 linewidth=1, elinewidth=0.8, label='Baseline (C0)')
    ax2.scatter([0.141], [0.851], c=FAMILY_COLORS['hybrid'], marker='*', s=150,
                edgecolors='black', linewidths=1, zorder=5, label='C13 (hybrid)')
    ax2.scatter([0.058], [0.748], c=FAMILY_COLORS['debate'], marker='o', s=80,
                edgecolors='black', linewidths=1, zorder=5, label='C9 (debate)')

    ax2.annotate('C13', (0.141, 0.851), xytext=(-15, 8), textcoords='offset points',
                 fontsize=7, fontweight='bold')
    ax2.annotate('C16', (0.221, 0.828), xytext=(5, -10), textcoords='offset points',
                 fontsize=7, fontweight='bold')
    # Annotate non-significant difference
    ax2.annotate('', xy=(0.141, 0.851 + 0.057 + 0.01), xytext=(0.221, 0.828 + 0.056 + 0.01),
                 arrowprops=dict(arrowstyle='-', color='gray', lw=0.8))
    ax2.text(0.181, 0.93, 'p=0.53\n(n.s.)', ha='center', fontsize=6, color='gray')

    ax2.set_xlabel('Cost per Problem (USD)')
    ax2.set_ylabel('Pass Rate')
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    ax2.set_ylim(0.40, 0.95)
    ax2.legend(fontsize=7, loc='lower right')
    ax2.grid(True, alpha=0.15)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, 'gen_only_scaling.pdf'), format='pdf')
    plt.close(fig)
    print('  [6/8] gen_only_scaling.pdf')


def fig7_latency():
    """Figure 7: Latency vs pass rate scatter (mirrors Pareto frontier but with time)."""
    RAW_DIR = os.path.join(os.path.dirname(__file__), '..', 'research', 'adversary',
                           'adversarial_bench_full', 'results', 'raw')

    # Collect per-condition latency from study C
    from collections import defaultdict
    latencies = defaultdict(list)
    for fname in os.listdir(RAW_DIR):
        if not fname.endswith('.json'):
            continue
        fpath = os.path.join(RAW_DIR, fname)
        with open(fpath) as f:
            data = json.load(f)
        if data.get('study') != 'C':
            continue
        cond = data.get('condition', '')
        lat = data.get('total_latency_ms')
        if lat is not None:
            latencies[cond].append(lat / 1000.0)  # convert to seconds

    fig, ax = plt.subplots(figsize=(5.5, 4))

    plotted_families = set()
    for cond in sorted(CALIBRATION.keys(), key=lambda c: int(c[1:])):
        d = CALIBRATION[cond]
        fam = d['family']
        if cond not in latencies or len(latencies[cond]) == 0:
            continue
        med_lat = np.median(latencies[cond])

        label = fam.replace('-', ' ').title() if fam not in plotted_families else None
        plotted_families.add(fam)

        ms = 120 if d.get('pareto') else 50
        ax.scatter(med_lat, d['pr'],
                   c=FAMILY_COLORS[fam], marker=FAMILY_MARKERS[fam],
                   s=ms, edgecolors='black', linewidths=0.5,
                   label=label, zorder=5)

    # Label key conditions
    key_conds = {'C0': (8, -12), 'C9': (8, -12), 'C13': (8, 5), 'C16': (5, -12)}
    for cond, (dx, dy) in key_conds.items():
        if cond in latencies:
            med_lat = np.median(latencies[cond])
            pr = CALIBRATION[cond]['pr']
            ax.annotate(cond, (med_lat, pr), xytext=(dx, dy),
                        textcoords='offset points', fontsize=7, fontweight='bold')

    ax.set_xlabel('Median Per-Problem Latency (seconds)')
    ax.set_ylabel('Pass Rate')
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylim(0.45, 0.90)
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(True, alpha=0.15)
    ax.set_title('Pass Rate vs. Serial Per-Problem Latency (Calibration Set)', fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, 'latency.pdf'), format='pdf')
    plt.close(fig)
    print('  [7/8] latency.pdf')


def fig8_cross_domain():
    """Figure 8: Cross-domain validation (HumanEval vs MBPP grouped bars)."""
    # Per-benchmark data from Table 5
    humaneval = {'C0': 0.834, 'C9': 0.934, 'C13': 0.954, 'C16': 0.834}
    mbpp =      {'C0': 0.400, 'C9': 0.820, 'C13': 0.907, 'C16': 0.460}
    # Binomial SE, n=25 problems per benchmark (the independent unit;
    # replicas within a problem are correlated)
    n_bench = 25

    conds = ['C0', 'C9', 'C13', 'C16']
    labels = ['Baseline\n(C0)', 'Debate\n(C9)', 'Hybrid\n(C13)', 'Gen×7\n(C16)']
    benchmarks = ['HumanEval', 'MBPP']
    bench_data = [humaneval, mbpp]
    bench_colors = ['#3498db', '#e74c3c']

    x = np.arange(len(conds))
    width = 0.35

    fig, ax = plt.subplots(figsize=(5, 3.5))
    for i, (bench, bdata, color) in enumerate(zip(benchmarks, bench_data, bench_colors)):
        vals = [bdata[c] for c in conds]
        ses = [np.sqrt(v * (1 - v) / n_bench) for v in vals]
        bars = ax.bar(x + (i - 0.5) * width, vals, width, yerr=ses,
                      label=bench, color=color, edgecolor='white', linewidth=0.5,
                      capsize=2, error_kw={'linewidth': 0.8, 'alpha': 0.7})
        for bar, val, se in zip(bars, vals, ses):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + se + 0.015,
                    f'{val:.0%}', ha='center', va='bottom', fontsize=6.5)

    # Annotate combined p-value for C13 vs C16
    ax.annotate('', xy=(2.175, 0.97), xytext=(3.175, 0.56),
                arrowprops=dict(arrowstyle='-', color='black', lw=1.0))
    ax.text(2.7, 0.99, 'combined\np=0.0002', ha='center', fontsize=6.5, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Pass Rate')
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylim(0, 1.12)
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, axis='y', alpha=0.15)

    fig.savefig(os.path.join(FIGURES, 'cross_domain.pdf'), format='pdf')
    plt.close(fig)
    print('  [8/8] cross_domain.pdf')


if __name__ == '__main__':
    print('Generating figures...')
    fig1_pareto_frontier()
    fig2_difficulty_stratification()
    fig3_cross_model()
    fig4_pass_at_1()
    fig5_bootstrap_stability()
    fig6_gen_only_scaling()
    fig7_latency()
    fig8_cross_domain()
    print('Done. Figures saved to', FIGURES)

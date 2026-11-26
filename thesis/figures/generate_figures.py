"""
Generate all thesis/paper figures as PDFs.
Run from repo root:  python3 thesis/figures/generate_figures.py

Reads data from CGRSP/Outputs/phase3/*.json (C_max=5 runs).
Saves to thesis/figures/ and copies to paper/ automatically.
"""
import csv
import json
import shutil
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

# Resolve paths relative to this file so the script works from anywhere
HERE   = Path(__file__).resolve().parent          # thesis/figures/
PAPER  = HERE.parent.parent / 'paper'             # paper/
CGRSP  = HERE.parent.parent / 'CGRSP'             # CGRSP/
DATA   = CGRSP / 'Outputs' / 'phase3'
RAW_CSV = HERE.parent.parent / 'data' / 'rendering_performance_data.csv'

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 150,
})

SCHED_COLORS = {
    'FIFO': '#95A5A6', 'SPT': '#E74C3C', 'EDF': '#2ECC71',
    'LCF': '#3498DB', 'Balanced': '#9B59B6', 'Random': '#E67E22',
    'SPT+Rescue': '#1ABC9C', 'RH': '#34495E', 'CADR': '#F39C12',
    'Adaptive': '#E91E63',
}


CHART_WIDTH = 8.6   # inches
CHART_FIGS = {
    'cmax_sensitivity.pdf', 'ttest_effects.pdf', 'phase2_comparison.pdf',
    'cr_sensitivity.pdf', 'reservation_sensitivity.pdf', 'cost_comparison.pdf',
    'diurnal_model.pdf', 'phase3_ci.pdf', 'weight_sensitivity.pdf',
    'threshold_sensitivity.pdf', 'deadline_sensitivity.pdf',
}


def _save(fig, name: str):
    """Save to thesis/figures/ and copy to paper/ (created if absent)."""
    out = HERE / name
    if name in CHART_FIGS:
        # Charts drawn on wide canvases print at about half scale; refit to a
        # print-sized canvas (text stays the same point size, so it prints larger).
        w, h = fig.get_size_inches()
        if w > CHART_WIDTH:
            k = CHART_WIDTH / w
            fig.set_size_inches(CHART_WIDTH, h * k ** 0.7)
            try:
                fig.tight_layout(rect=[0, 0, 1, 0.95] if fig._suptitle is not None else None)
            except Exception:
                pass
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    PAPER.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out, PAPER / name)
    print(f'  Saved {out}  ->  {PAPER / name}')


# ── Figure 1: System Architecture Block Diagram ──────────────────────────────

def fig_architecture():
    """CGRSP simulator architecture: restrained thesis-style block diagram.

    Layout grid (canvas 10 x 5.8):
      top row:    Scheduler (highlighted, component under study) | Fleet Manager
      middle row: Job Generator | DES Core (event loop spine) | Metrics Collector
      bottom:     GPU fleet container (C_max slots) under the Fleet Manager
    Every edge is labelled with what flows along it.
    """
    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5.8); ax.axis('off')

    EDGE = '#1A1A1A'
    FILL = '#F2F2F2'          # neutral component fill
    HILT = '#D9D9D9'          # scheduler highlight (component under study)

    def box(x, y, w, h, title, sub='', fill=FILL, lw=1.2, ls='-'):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.06',
                                    linewidth=lw, edgecolor=EDGE,
                                    facecolor=fill, linestyle=ls))
        cy = y + h / 2 + (0.16 if sub else 0)
        ax.text(x + w/2, cy, title, ha='center', va='center',
                fontsize=10.5, fontweight='bold', color='#1a1a1a')
        if sub:
            ax.text(x + w/2, cy - 0.42, sub, ha='center', va='center',
                    fontsize=8.5, color='#555')

    def edge(x1, y1, x2, y2, label='', lx=None, ly=None, ha='center',
             style='->', rad=0.0):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color=EDGE, lw=1.2,
                                   shrinkA=4, shrinkB=4,
                                   connectionstyle=f'arc3,rad={rad}'))
        if label:
            ax.text(lx if lx is not None else (x1+x2)/2,
                    ly if ly is not None else (y1+y2)/2,
                    label, ha=ha, va='center', fontsize=8.2,
                    color='#555', style='italic',
                    bbox=dict(boxstyle='round,pad=0.12', facecolor='white',
                              edgecolor='none', alpha=0.9))

    # ── middle row: event-loop spine ─────────────────────────────────────────
    box(0.3, 3.0, 2.1, 1.0, 'Job Generator', 'Poisson arrivals')
    box(3.6, 3.0, 3.1, 1.0, 'DES Core', 'event loop (min-heap)')
    box(7.1, 3.0, 1.8, 1.0, 'Metrics\nCollector')

    # ── top row: decision components ─────────────────────────────────────────
    box(3.6, 4.7, 3.1, 0.9, 'Scheduler', 'policy under study', fill=HILT, lw=1.8)
    box(7.8, 4.7, 1.95, 0.9, 'Fleet Manager', 'slot state')

    # ── bottom: GPU fleet container ──────────────────────────────────────────
    ax.add_patch(FancyBboxPatch((3.0, 0.35), 6.75, 1.5,
                                boxstyle='round,pad=0.06', linewidth=1.0,
                                edgecolor='#666666', facecolor='none',
                                linestyle='--'))
    ax.text(3.2, 1.62, 'GPU fleet ($C_{\\max}$ on-demand slots, RunPod Secure Cloud)',
            ha='left', va='center', fontsize=8.5, color='#555', style='italic')
    gpus = ['RTX 3090', 'RTX A5000', 'RTX A4500', 'RTX A4000']
    for i, g in enumerate(gpus):
        gx = 3.25 + i * 1.6
        ax.add_patch(FancyBboxPatch((gx, 0.55), 1.45, 0.75,
                                    boxstyle='round,pad=0.04', linewidth=1.0,
                                    edgecolor=EDGE, facecolor='white'))
        ax.text(gx + 0.725, 0.925, g, ha='center', va='center', fontsize=8.8)

    # ── labelled edges ────────────────────────────────────────────────────────
    # Job Generator -> DES Core
    edge(2.4, 3.5, 3.6, 3.5, 'arrival\nevents', lx=3.0, ly=3.82)
    # DES Core <-> Scheduler (trigger up, assignments down)
    edge(4.5, 4.0, 4.5, 4.7, 'schedule trigger:\nqueue + fleet state',
         lx=4.32, ly=4.36, ha='right')
    edge(5.8, 4.7, 5.8, 4.0, 'assignments\n$(j \\to m)$',
         lx=5.98, ly=4.36, ha='left')
    # Fleet Manager -> Scheduler (observability)
    edge(7.8, 5.15, 6.7, 5.15, 'idle slots +\nstock status', lx=7.25, ly=5.5)
    # DES Core -> Metrics Collector
    edge(6.7, 3.5, 7.1, 3.5, 'completion\nrecords', lx=7.15, ly=4.28)
    # Fleet Manager <-> fleet container (provision / release)
    edge(9.3, 4.7, 9.3, 1.85, 'provision /\nrelease', lx=9.42, ly=3.3,
         ha='left')
    # DES Core -> fleet (job execution) and back (completions)
    edge(4.6, 3.0, 4.6, 1.85, 'dispatch job', lx=4.42, ly=2.42, ha='right')
    edge(5.7, 1.85, 5.7, 3.0, 'JOB_COMPLETE', lx=5.88, ly=2.42, ha='left')

    fig.tight_layout(pad=0.3)
    _save(fig, 'system_architecture.pdf')


# ── Figure 2: Phase 2 Benchmark Comparison ───────────────────────────────────

def fig_phase2_comparison():
    """Phase 2 comparison: normal day vs hectic day, seed=42."""
    import matplotlib.colors as mc

    with open(CGRSP / 'Outputs/runpod_scheduler/report.json') as f:
        d_hectic = json.load(f)
    with open(CGRSP / 'Outputs/runpod_normal/report.json') as f:
        d_normal = json.load(f)

    order = ['SPT+Rescue', 'SPT', 'EDF', 'Balanced', 'FIFO', 'LCF', 'Random']

    def _extract(d):
        wait = [d['results'][s]['timing_metrics']['avg_waiting_time'] / 60 for s in order]
        miss = [d['results'][s]['job_counts']['deadline_miss_rate_pct'] for s in order]
        return wait, miss

    wait_n, miss_n = _extract(d_normal)
    wait_h, miss_h = _extract(d_hectic)

    cmax = d_hectic['metadata']['max_concurrency']
    seed = d_hectic['metadata']['seed']
    n_normal = d_normal['metadata']['num_jobs']
    n_hectic = d_hectic['metadata']['num_jobs']
    lam_normal = d_normal['metadata']['arrival_rate_jobs_per_sec']
    lam_hectic = d_hectic['metadata']['arrival_rate_jobs_per_sec']

    def lighten(hex_color, f=0.55):
        c = mc.to_rgb(hex_color)
        return tuple(ci + (1 - ci) * f for ci in c)

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.0))
    (ax_nw, ax_nm), (ax_hw, ax_hm) = axes

    scenarios = [
        (ax_nw, ax_nm, wait_n, miss_n, n_normal, lam_normal, 'Normal',   lighten),
        (ax_hw, ax_hm, wait_h, miss_h, n_hectic, lam_hectic, 'Hectic',   lambda c, f=0: c),
    ]

    x = np.arange(len(order))

    for ax_w, ax_m, wait, miss, n_jobs, lam, label, cfn in scenarios:
        colors = [cfn(SCHED_COLORS.get(s, '#95A5A6')) for s in order]
        miss_colors = ['#E74C3C' if m > 0 else cfn('#95A5A6') for m in miss]

        # Wait time
        ax_w.bar(x, wait, color=colors, edgecolor='white')
        ax_w.set_xticks(x); ax_w.set_xticklabels(order, fontsize=8.5, rotation=30, ha='right')
        ax_w.set_ylabel('Avg Waiting Time (min)')
        ax_w.set_title(f'({("a" if label=="Normal" else "c")}) {label} Day: Wait Time\n'
                       f'({n_jobs} jobs/day, $\\lambda$={lam} j/s)')
        # zero baseline: truncated bar axes overstate the differences; the
        # value labels keep small gaps readable
        ax_w.set_ylim(0, max(wait) * 1.12)
        for xi, v in zip(x, wait):
            ax_w.text(xi, v + max(wait) * 0.015, f'{v:.1f}',
                      ha='center', va='bottom', fontsize=8)
        ax_w.grid(True, alpha=0.3, axis='y')

        # Miss rate
        ax_m.bar(x, miss, color=miss_colors, edgecolor='white')
        ax_m.set_xticks(x); ax_m.set_xticklabels(order, fontsize=8.5, rotation=30, ha='right')
        ax_m.set_ylabel('Deadline Miss Rate (%)')
        ax_m.set_title(f'({("b" if label=="Normal" else "d")}) {label} Day: Miss Rate\n'
                       f'({n_jobs} jobs/day, $\\lambda$={lam} j/s)')
        ymax_m = max(1.0, max(miss) * 1.15)
        ax_m.set_ylim(0, ymax_m)
        for xi, v in zip(x, miss):
            ax_m.text(xi, v + ymax_m * 0.01, f'{v:.1f}%',
                      ha='center', va='bottom', fontsize=8)
        ax_m.grid(True, alpha=0.3, axis='y')

    fig.suptitle(
        f'Phase 2 Benchmark: Normal vs Hectic Day  ($C_{{\\max}}$={cmax}, seed={seed})',
        fontsize=10.5, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, 'phase2_comparison.pdf')


# ── Figure 3: Sensitivity, Day-Type Comparison at C_max=5 ───────────────────

def fig_sensitivity_heatmap():
    """Day-type sensitivity at C_max=5: wait time and miss rate per scheduler."""
    with open(DATA / 'sensitivity.json') as f:
        d = json.load(f)

    day_types  = ['quiet', 'normal', 'hectic', 'surge']
    all_scheds = ['FIFO', 'SPT', 'EDF', 'LCF', 'Balanced', 'Random', 'SPT+Rescue', 'RH', 'CADR', 'Adaptive']
    # Keep only schedulers present in the JSON (guards against missing entries)
    sample_key = f'fleet5_type{day_types[0]}'
    available  = set(d['sensitivity'][sample_key]['schedulers'].keys())
    scheds     = [s for s in all_scheds if s in available]

    # Build labels dynamically from JSON so job counts stay in sync with simulation params
    dt_labels = []
    for dt in day_types:
        key = f'fleet5_type{dt}'
        entry = d['sensitivity'][key]
        n = entry['n_jobs']
        lam = entry['arrival_rate']
        if lam >= 0.05:
            dt_labels.append(f'{dt.capitalize()}\n({n}/day\nλ={lam:.3f})')
        elif lam >= 0.01:
            dt_labels.append(f'{dt.capitalize()}\n({n}/day\nλ={lam:.3f})')
        else:
            dt_labels.append(f'{dt.capitalize()}\n({n}/day)')

    wait_data = {s: [] for s in scheds}
    miss_data = {s: [] for s in scheds}
    for dt in day_types:
        key = f'fleet5_type{dt}'
        for s in scheds:
            row = d['sensitivity'][key]['schedulers'][s]
            wait_data[s].append(row['avg_wait_min'])
            miss_data[s].append(row.get('avg_miss_pct', 0))

    # 9 schedulers: use a wider canvas and narrower bars so groups don't overlap
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.0, 3.6))
    n_dt = len(day_types)
    n_s  = len(scheds)
    bw   = 0.085
    x    = np.arange(n_dt)
    offsets = np.linspace(-(n_s - 1) / 2 * bw, (n_s - 1) / 2 * bw, n_s)

    for s, off in zip(scheds, offsets):
        c = SCHED_COLORS.get(s, '#888')
        # hatch the two proposed schedulers so they survive a greyscale print
        hatch = '////' if s in ('RH', 'CADR') else None
        ax1.bar(x + off, wait_data[s], bw, label=s, color=c, edgecolor='white', linewidth=0.4, hatch=hatch)
        ax2.bar(x + off, miss_data[s], bw, label=s, color=c, edgecolor='white', linewidth=0.4, hatch=hatch)

    ax1.set_xticks(x); ax1.set_xticklabels(dt_labels, fontsize=9)
    ax1.set_ylabel('Avg Waiting Time (min)')
    ax1.set_title('(a) Wait Time by Day Type')
    ax1.set_ylim(top=ax1.get_ylim()[1] * 1.08)
    ax1.grid(True, alpha=0.3, axis='y')

    ax2.set_xticks(x); ax2.set_xticklabels(dt_labels, fontsize=9)
    ax2.set_ylabel('Deadline Miss Rate (%)')
    ax2.set_title('(b) Deadline Miss Rate by Day Type')
    all_miss = [v for vals in miss_data.values() for v in vals]
    ax2.set_ylim(bottom=0, top=max(1.0, max(all_miss) * 1.08))
    ax2.grid(True, alpha=0.3, axis='y')

    fig.suptitle(r'Day-Type Sensitivity at $C_{\max}=5$ (30 seeds each)',
                 fontsize=11, fontweight='bold')
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=5, fontsize=8.5, loc='lower center', frameon=False)
    fig.tight_layout(rect=[0, 0.13, 1, 0.95])
    _save(fig, 'sensitivity_heatmap.pdf')


# ── Figure 4: Deadline Tightness Sensitivity ──────────────────────────────────

def fig_deadline_sensitivity():
    with open(DATA / 'deadline.json') as f:
        d = json.load(f)

    tight_pcts = [10, 30, 50, 70, 90]
    scheds     = ['FIFO', 'SPT', 'EDF', 'LCF', 'Balanced', 'Random', 'SPT+Rescue']
    styles     = ['-o', '-s', '-^', '-D', '-v', '-x', '-P']
    concurrency_configs = d['metadata']['concurrency_configs']   # [5]

    fig, axes = plt.subplots(1, len(concurrency_configs), figsize=(8.5, 3.8), sharey=False)
    if len(concurrency_configs) == 1:
        axes = [axes]

    for ax, cmax in zip(axes, concurrency_configs):
        fleet_key = f'fleet{cmax}'
        results   = d['results'][fleet_key]
        frac_keys = [f'tight{p}pct' for p in tight_pcts]
        for sched, sty in zip(scheds, styles):
            vals = [results[k]['schedulers'][sched].get('avg_miss_pct',
                    results[k]['schedulers'][sched].get('deadline_miss_pct', 0))
                    for k in frac_keys]
            ax.plot(tight_pcts, vals, sty, label=sched,
                    color=SCHED_COLORS.get(sched, '#888'),
                    lw=1.8, markersize=5)
        ax.set_xlabel('Tight Deadline Fraction (%)')
        ax.set_ylabel('Deadline Miss Rate (%)')
        ax.set_title(f'$C_{{\\max}}={cmax}$')
        ax.set_xticks(tight_pcts)
        # Clamp to at least 1 % range so all-zero panels read cleanly
        ax.set_ylim(bottom=0, top=max(1.0, ax.get_ylim()[1]))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8.5, ncol=2)

    fig.suptitle(r'Deadline Tightness Sensitivity (Surge Day, $C_{\max}=5$, 30 seeds)',
                 fontsize=11, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, 'deadline_sensitivity.pdf')


# ── Figure 5: Phase 4 Monthly Simulation ─────────────────────────────────────

def fig_phase4_monthly():
    """Phase 4 monthly simulation: wait by day type + monthly deadline miss %."""
    with open(CGRSP / 'Outputs/phase4/monthly_results.json') as f:
        d = json.load(f)

    all_scheds = ['FIFO', 'SPT', 'EDF', 'LCF', 'Balanced', 'Random', 'SPT+Rescue', 'RH', 'CADR', 'Adaptive']
    # Keep only schedulers present in the JSON summary
    available  = set(d['summary'].keys())
    scheds     = [s for s in all_scheds if s in available]
    day_types  = ['surge', 'hectic', 'normal', 'quiet']
    # Realised day-type shares over the simulated month, not the nominal targets.
    stats = d.get('schedule_stats', {})
    dt_labels  = [f"{dt.capitalize()}\n({stats[dt]['pct']:.0f}%)" if dt in stats
                  else dt.capitalize() for dt in day_types]

    wait_by_dt = {dt: [] for dt in day_types}
    miss_monthly = []
    for s in scheds:
        r = d['summary'][s]
        for dt in day_types:
            wait_by_dt[dt].append(r[f'{dt}_wait_min']['mean'])
        miss_monthly.append(r['deadline_miss_pct']['mean'])

    # 9 schedulers: wider canvas, narrower bars
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.3, 3.3))

    # Left: grouped bars, day type on x-axis, one bar per scheduler
    n_dt = len(day_types)
    n_s  = len(scheds)
    bw   = 0.09
    x    = np.arange(n_dt)
    offsets = np.linspace(-(n_s - 1) / 2 * bw, (n_s - 1) / 2 * bw, n_s)
    for i, (s, off) in enumerate(zip(scheds, offsets)):
        vals = [wait_by_dt[dt][i] for dt in day_types]
        color = SCHED_COLORS.get(s, '#95A5A6')
        # hatch the two proposed schedulers so they survive a greyscale print
        hatch = '////' if s in ('RH', 'CADR') else None
        ax1.bar(x + off, vals, bw, label=s, color=color, edgecolor='white', linewidth=0.4, hatch=hatch)

    ax1.set_xticks(x)
    ax1.set_xticklabels(dt_labels, fontsize=8.5)
    ax1.set_ylabel('Avg Waiting Time (min)')
    ax1.set_xlabel('Day type (share of simulated days)', fontsize=9)
    ax1.set_title('(a) Wait Time by Day Type')
    ax1.set_ylim(top=ax1.get_ylim()[1] * 1.05)
    ax1.grid(True, alpha=0.3, axis='y')

    # Right: monthly deadline miss % per scheduler (use each sched's own colour)
    colors2 = [SCHED_COLORS.get(s, '#95A5A6') for s in scheds]
    x2 = np.arange(n_s)
    ax2.bar(x2, miss_monthly, color=colors2, edgecolor='white')
    for bar, s_ in zip(ax2.patches, scheds):
        if s_ in ('RH', 'CADR'):
            bar.set_hatch('////')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(scheds, fontsize=8.5, rotation=35, ha='right')
    ax2.set_ylabel('Monthly Deadline Miss Rate (%)')
    ax2.set_title('(b) Monthly Deadline Miss Rate')
    ax2.set_ylim(0, max(0.5, max(miss_monthly) * 1.22))
    for xi, v in zip(x2, miss_monthly):
        ax2.text(xi, v + max(miss_monthly) * 0.03, f'{v:.1f}',
                 ha='center', va='bottom', fontsize=7.5)
    ax2.grid(True, alpha=0.3, axis='y')

    fig.suptitle(
        r'Phase 4 Monthly Simulation: 30 days, $C_{\max}=5$, 10 replications',
        fontsize=11, fontweight='bold')
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=5, fontsize=8.5, loc='lower center', frameon=False)
    fig.tight_layout(rect=[0, 0.12, 1, 0.95])
    _save(fig, 'phase4_monthly.pdf')


# ── Figure 6: Miss Rate vs Tardiness Tradeoff (Phase 4) ──────────────────────

def fig_tardiness_tradeoff():
    """Scatter: deadline miss rate vs mean tardiness across all 9 schedulers (Phase 4)."""
    with open(CGRSP / 'Outputs/phase4/monthly_results.json') as f:
        d = json.load(f)

    all_scheds = ['FIFO', 'SPT', 'EDF', 'LCF', 'Balanced', 'Random', 'SPT+Rescue', 'RH', 'CADR', 'Adaptive']
    available  = set(d['summary'].keys())
    scheds     = [s for s in all_scheds if s in available]

    miss_vals = []
    tard_vals = []
    for s in scheds:
        r = d['summary'][s]
        miss_vals.append(r['deadline_miss_pct']['mean'])
        tard_vals.append(r['avg_tardiness_min']['mean'])

    # Pareto-efficient front: schedulers not dominated on both axes
    # A point is Pareto-efficient if no other point has both lower miss and lower tardiness
    pareto_idx = []
    for i, (m_i, t_i) in enumerate(zip(miss_vals, tard_vals)):
        dominated = any(
            (miss_vals[j] <= m_i and tard_vals[j] <= t_i and
             (miss_vals[j] < m_i or tard_vals[j] < t_i))
            for j in range(len(scheds)) if j != i
        )
        if not dominated:
            pareto_idx.append(i)

    # Sort Pareto front by miss rate for the connecting line
    pareto_sorted = sorted(pareto_idx, key=lambda i: miss_vals[i])

    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    # Plot all points
    for i, s in enumerate(scheds):
        c = SCHED_COLORS.get(s, '#888')
        marker_size = 120 if i in pareto_idx else 70
        ax.scatter(miss_vals[i], tard_vals[i], color=c, s=marker_size,
                   zorder=3, edgecolors='white', linewidths=0.8)

    # Dashed line connecting Pareto-efficient points (lower-left frontier)
    px = [miss_vals[i] for i in pareto_sorted]
    py = [tard_vals[i] for i in pareto_sorted]
    ax.plot(px, py, linestyle='--', color='#7F8C8D', linewidth=1.0,
            zorder=2, alpha=0.7)

    # Annotate each point; offset label to avoid overlap
    offsets_ann = {
        'FIFO':       (-46, -20),  # far-right point: label to the lower left, not clipped
        'SPT':        (4,  -10),
        'EDF':        (6,  -12),   # below-right, clear of Adaptive
        'LCF':        (-46, 12),
        'Balanced':   (8,  10),
        'Random':     (4,   4),
        'SPT+Rescue': (4,  -10),
        'RH':         (10,  6),
        'CADR':       (4,   4),
        'Adaptive':   (-62,  4),   # left of its point, clear of EDF
    }
    ax.margins(x=0.08)
    for i, s in enumerate(scheds):
        ox, oy = offsets_ann.get(s, (4, 4))
        ax.annotate(
            s,
            (miss_vals[i], tard_vals[i]),
            xytext=(ox, oy),
            textcoords='offset points',
            fontsize=8.5,
            color=SCHED_COLORS.get(s, '#333'),
            fontweight='bold' if i in pareto_idx else 'normal',
            arrowprops=dict(arrowstyle='-', lw=0.5, color='#999'),
        )

    ax.set_xlabel('Deadline miss rate (%)')
    ax.set_ylabel('Mean tardiness (min)')
    ax.set_title('Miss rate vs tardiness (Phase 4 monthly, 10 replications)')
    ax.grid(True, alpha=0.3)

    # Legend patch per scheduler
    patches = [mpatches.Patch(color=SCHED_COLORS.get(s, '#888'), label=s) for s in scheds]
    ax.legend(handles=patches, fontsize=7.5, ncol=3, loc='lower right')

    fig.tight_layout()
    _save(fig, 'tardiness_tradeoff.pdf')


# ── Figure 0b: Numerical Example Visual Aid ──────────────────────────────────

def fig_numerical_example():
    """Gantt-style diagram for the 3-job 2-GPU numerical example in Section 3.7."""
    # Problem setup
    jobs = [
        dict(name='$j_1$', gpu='RTX 3090 ($g_1$, $m_1$)', start=0, exec=59.5,
             deadline=3600, complexity='Low ($\\kappa=1$)', tight=True,
             cost=59.5 * 0.46 / 3600, color='#1A1A1A'),
        dict(name='$j_2$', gpu='RTX A4000 ($g_2$, $m_2$)', start=0, exec=72.1,
             deadline=28800, complexity='Medium ($\\kappa=2$)', tight=False,
             cost=72.1 * 0.25 / 3600, color='#4D4D4D'),
        dict(name='$j_3$', gpu='RTX A4000 ($g_2$, $m_3$)', start=0, exec=63.3,
             deadline=28800, complexity='Low ($\\kappa=1$)', tight=False,
             cost=63.3 * 0.25 / 3600, color='#737373'),
    ]

    # Use a two-panel layout: left = execution zoom (0-170s); right = deadline view (schematic)
    fig = plt.figure(figsize=(9.75, 3.3))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1.0], wspace=0.08)
    ax_zoom = fig.add_subplot(gs[0])
    ax_dead = fig.add_subplot(gs[1])

    lane_y = [2.0, 1.0, 0.0]
    bar_h = 0.6

    for ax, xlim, show_deadline_bar in [
        (ax_zoom, (0, 170), False),
        (ax_dead, (0, 30000), True),
    ]:
        for i, job in enumerate(jobs):
            y = lane_y[i]
            # Execution bar
            ax.barh(y, job['exec'], left=job['start'], height=bar_h,
                    color=job['color'], alpha=0.85, edgecolor='white', linewidth=0.8)
            # Deadline marker
            if show_deadline_bar or job['deadline'] <= xlim[1]:
                ax.axvline(job['deadline'], color=job['color'],
                           linestyle='--', linewidth=1.2, alpha=0.8)

        ax.set_xlim(*xlim)
        ax.set_ylim(-0.5, 2.9)
        ax.set_yticks(lane_y)
        ax.set_yticklabels([j['gpu'] for j in jobs], fontsize=9)
        ax.grid(True, alpha=0.2, axis='x')
        ax.axvline(0, color='#333', linewidth=1.0)

    # Annotations on zoom panel
    for i, job in enumerate(jobs):
        y = lane_y[i]
        mid_x = job['start'] + job['exec'] / 2
        ax_zoom.text(mid_x, y, f"{job['name']}\n{job['exec']:.1f}s",
                     ha='center', va='center', fontsize=9, fontweight='bold', color='white')
        ax_zoom.text(job['start'] + job['exec'] + 3, y,
                     f"${job['cost']:.5f}",
                     ha='left', va='center', fontsize=8.5, color='#555')

    # Deadline labels on zoom panel (only j1 is visible)
    ax_zoom.text(168, lane_y[0] + 0.42, '$d_{j_1}$ = 3,600 s (off scale)', fontsize=8.5,
                 color='#1A1A1A', ha='right')

    # Annotations on deadline panel
    for i, job in enumerate(jobs):
        y = lane_y[i]
        ax_dead.barh(y, job['exec'], left=job['start'], height=bar_h,
                     color=job['color'], alpha=0.85, edgecolor='white', linewidth=0.8)
        tight_str = "tight" if job["tight"] else "loose"
        dead_label = f'$d={job["deadline"]:,}$s\n({tight_str})'
        if job['tight']:
            ax_dead.text(job['deadline'] + 300, y, dead_label,
                         ha='left', va='center', fontsize=9.5, color=job['color'])
        else:
            ax_dead.text(job['deadline'] - 300, y, dead_label,
                         ha='right', va='center', fontsize=9.5, color=job['color'])

    ax_dead.tick_params(labelleft=False)
    ax_dead.set_xlabel('Time (seconds)')
    ax_zoom.set_xlabel('Time (seconds)')
    ax_zoom.set_title('(a) Execution window (0 to 170 s)', fontsize=11)
    ax_dead.set_title('(b) Full deadline window (0 to 28,800 s)', fontsize=11)

    # Break marks between panels
    d = 0.015
    kwargs = dict(transform=ax_zoom.transAxes, color='k', clip_on=False, lw=1.2)
    ax_zoom.plot([1 - d, 1 + d], [-d, +d], **kwargs)
    ax_zoom.plot([1 - d, 1 + d], [1 - d, 1 + d], **kwargs)

    # Legend
    legend_items = [
        mpatches.Patch(color=j['color'], label=f"{j['name']}: {j['complexity']}, {'tight 1h' if j['tight'] else 'loose 8h'} deadline")
        for j in jobs
    ]
    fig.legend(handles=legend_items, fontsize=10, loc='upper center',
               bbox_to_anchor=(0.5, 0.86), ncol=3)

    fig.suptitle(
        'Numerical example: three-job lexicographic assignment. '
        'All jobs start at $t=0$; $j_1$ takes the RTX 3090 by tie-break;\n'
        '$j_2$, $j_3$ take the RTX A4000 (cost saving, slack permits).',
        fontsize=10, y=1.0)
    fig.subplots_adjust(left=0.19, right=0.985, top=0.68, bottom=0.17)
    _save(fig, 'numerical_example.pdf')


# ── Figure 7: Diurnal Availability Model ────────────────────────────────────

def fig_diurnal_model():
    """Two panels: (a) m(t) step function with snapshot rug; (b) real job arrival pattern."""
    import glob

    # ── m(t) definition (matches cgrsp/data/runpod_availability.py TimeOfDay) ─
    # runpod_availability.py's get_time_of_day() uses datetime.fromtimestamp,
    # i.e. the machine's local clock; the simulator is calibrated so sim time 0
    # falls in the MORNING band (DECISIONS.md #8) — that local clock is Taiwan
    # time (UTC+8) throughout this project. Band boundaries (local/Taiwan hours):
    #   Off-peak: 00:00-06:00  x1.3
    #   Morning:  06:00-12:00  x0.9
    #   Peak:     12:00-18:00  x0.5
    #   Evening:  18:00-24:00  x1.0
    band_defs = [
        (0,  6,  1.3, 'Off-peak', '#2ECC71'),
        (6,  12, 0.9, 'Morning',  '#E67E22'),
        (12, 18, 0.5, 'Peak',     '#E74C3C'),
        (18, 24, 1.0, 'Evening',  '#3498DB'),
    ]

    # Build hour-resolution step array
    m_vals = [0.0] * 24
    for start, end, mult, _, _ in band_defs:
        for h in range(start, end):
            m_vals[h] = mult

    # ── Parse 77 RunPod pricing snapshots for snapshot hour rug ──────────────
    # generated_at is tagged UTC by the scraper's own timezone.utc call, but the
    # scraping machine's system clock was actually running on GMT+7; shift by
    # +1h to Taiwan time (UTC+8) for display, consistent with the other panel.
    pricing_dir = HERE.parent.parent / 'data' / 'runpod_pricing'
    summary_hours = HERE.parent.parent / 'data' / 'runpod_pricing_summary' / 'snapshot_hours.csv'
    snapshot_hours = []
    if pricing_dir.exists():
        for f in sorted(pricing_dir.glob('distributions_*.json')):
            try:
                with open(f) as fh:
                    d = json.load(fh)
                ts = datetime.fromisoformat(d['generated_at'])
                snapshot_hours.append((ts.hour + ts.minute / 60.0 + 1) % 24)
            except Exception:
                pass
    elif summary_hours.exists():
        with open(summary_hours, newline='') as fh:
            for row in csv.DictReader(fh):
                try:
                    ts = datetime.fromisoformat(row['generated_at'])
                    snapshot_hours.append((ts.hour + ts.minute / 60.0 + 1) % 24)
                except (KeyError, ValueError):
                    pass
    else:
        # No archive and no compact summary: keep the committed PDF rather than
        # silently redraw this figure without its snapshot rug.
        print(f'  Skipping diurnal_model: neither {pricing_dir} nor {summary_hours} found')
        plt.close('all')
        return

    # ── Real job arrival pattern from CSV ────────────────────────────────────
    hours_real = defaultdict(int)
    if RAW_CSV.exists():
        with open(RAW_CSV, newline='') as f:
            for row in csv.DictReader(f):
                if row.get('is_success') != 'true':
                    continue
                try:
                    dt = datetime.strptime(row['timestamp'], '%Y:%m:%dT%H:%M:%S')
                    # source timestamps are UTC (verified against live
                    # Lambda/CloudWatch logs); shift to Taiwan time (UTC+8)
                    hours_real[(dt.hour + 8) % 24] += 1
                except ValueError:
                    pass

    # ── Per-type high-stock probability p_g^H(t) = min(0.95, beta_g * m(t)) ──
    # beta_g values match cgrsp/data/runpod_availability.py BASE_HIGH_STOCK_PROB
    beta_g = {
        'RTX 3090':  0.50,
        'RTX A5000': 0.65,
        'RTX A4500': 0.70,
        'RTX A4000': 0.75,
    }
    gpu_colors = {
        'RTX A5000': '#2ECC71',
        'RTX 3090':  '#E74C3C',
        'RTX A4500': '#3498DB',
        'RTX A4000': '#9B59B6',
    }
    p_high = {
        name: [min(0.95, b * m_vals[h]) for h in range(24)]
        for name, b in beta_g.items()
    }

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    x_h = np.arange(24)

    # Panel (a): m(t) step function (bars, left axis) and per-type p_g^H(t) (lines, right axis)
    bar_colors_m = []
    for h in range(24):
        for start, end, _, _, c in band_defs:
            if start <= h < end:
                bar_colors_m.append(c)
                break
    ax1.bar(x_h, m_vals, color=bar_colors_m, edgecolor='white', linewidth=0.4, alpha=0.35)
    ax1.set_ylabel('Availability multiplier $\\phi(t)$')
    ax1.set_title('(a) Diurnal availability model: multiplier $\\phi(t)$ and per-type high-stock probability $p_g^H(t)$')
    ax1.set_ylim(0, 2.0)
    ax1.set_yticks([0, 0.5, 1.0, 1.5])
    ax1.grid(True, alpha=0.25, axis='y')
    # sharex hides the upper panel's tick labels by default; re-enable them so
    # panel (a) is readable on its own
    ax1.set_xticks(range(0, 24, 2))
    ax1.set_xticklabels([f'{h:02d}:00' for h in range(0, 24, 2)],
                        rotation=30, ha='right', fontsize=8)
    ax1.tick_params(axis='x', labelbottom=True)
    ax1.set_xlabel('Hour of day (Taiwan time, UTC+8)', fontsize=9)

    ax1_twin = ax1.twinx()
    for name, b in beta_g.items():
        ax1_twin.step(x_h, p_high[name], where='post', linewidth=1.8,
                      color=gpu_colors[name], marker='o', markersize=3,
                      label=f'{name} ($\\beta_g$={b:.2f})')
    ax1_twin.set_ylabel('High-stock probability $p_g^H(t)$')
    ax1_twin.set_ylim(0, 1.29)
    ax1_twin.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax1_twin.legend(fontsize=7.5, loc='upper center', ncol=4)

    # Rug marks for snapshot hours
    if snapshot_hours:
        ax1.scatter(snapshot_hours, [-0.05] * len(snapshot_hours),
                    s=42, color='#566573', marker='|', zorder=5,
                    clip_on=False)
        ax1.text(0.995, 0.02, f'{len(snapshot_hours)} API snapshots (rug, bottom)',
                 transform=ax1.transAxes, ha='right', va='bottom',
                 fontsize=7, color='#566573')

    # Panel (b): real arrival pattern
    counts_real = [hours_real.get(h, 0) for h in range(24)]
    bar_colors_real = []
    for h in range(24):
        for start, end, _, _, c in band_defs:
            if start <= h < end:
                bar_colors_real.append(c)
                break
    ax2.bar(x_h, counts_real, color=bar_colors_real, edgecolor='white', linewidth=0.4, alpha=0.8)
    ax2.set_xlabel('Hour of day (Taiwan time, UTC+8)')
    ax2.set_ylabel('Job count (6,234 jobs)')
    ax2.set_title('(b) Real job arrival pattern from production data (Jul 2025 to Jun 2026) coloured by time band')
    ax2.set_xticks(range(0, 24, 2))
    ax2.set_xticklabels([f'{h:02d}:00' for h in range(0, 24, 2)], rotation=30, ha='right', fontsize=8)
    ax2.grid(True, alpha=0.25, axis='y')

    # Shared band legend
    legend_patches = [
        mpatches.Patch(color=c, alpha=0.8, label=f'{label} ({s:02d}:00-{e:02d}:00, $\\times${mult})')
        for s, e, mult, label, c in band_defs
    ]
    ax2.set_ylim(0, max(counts_real) * 1.32)   # headroom so legend clears bars
    ax2.legend(handles=legend_patches, fontsize=7.5, ncol=2, loc='upper right')

    fig.suptitle('RunPod GPU availability model and real job arrival pattern',
                 fontsize=11, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, 'diurnal_model.pdf')


# ── Figure 8: Historical Dataset Overview ────────────────────────────────────

def fig_data_overview():
    """Two panels: (a) jobs per hour of day; (b) execution-time box plots by GPU."""
    if not RAW_CSV.exists():
        print(f'  Skipping data_overview: {RAW_CSV} not found')
        return

    # Parse CSV
    hours = defaultdict(int)
    gpu_times = defaultdict(list)
    gpu_label_map = {
        'NVIDIA GeForce RTX 3090': 'RTX 3090',
        'NVIDIA RTX A5000':        'RTX A5000',
        'NVIDIA RTX A4500':        'RTX A4500',
        'NVIDIA RTX A4000':        'RTX A4000',
    }
    gpu_order = ['RTX A5000', 'RTX 3090', 'RTX A4500', 'RTX A4000']

    with open(RAW_CSV, newline='') as f:
        for row in csv.DictReader(f):
            if row.get('is_success') != 'true':
                continue
            ts = row.get('timestamp', '')
            try:
                dt = datetime.strptime(ts, '%Y:%m:%dT%H:%M:%S')
                # source timestamps are UTC (verified against live Lambda/CloudWatch
                # logs); shift to Taiwan time (UTC+8) for display
                hours[(dt.hour + 8) % 24] += 1
            except ValueError:
                pass
            label = gpu_label_map.get(row.get('gpu_used', ''))
            if label:
                try:
                    gpu_times[label].append(float(row['render_time_s']))
                except (ValueError, KeyError):
                    pass

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.4, 5.6))

    # Panel (a): arrivals by hour of day
    x_h = np.arange(24)
    counts = [hours.get(h, 0) for h in x_h]
    bar_colors = ['#4D4D4D' if 14 <= h <= 17 else '#D9D9D9' for h in x_h]
    ax1.bar(x_h, counts, color=bar_colors, edgecolor='white', linewidth=0.4)
    ax1.set_xlabel('Hour of day (Taiwan time, UTC+8)')
    ax1.set_ylabel('Number of jobs')
    ax1.set_title('(a) Arrivals by hour of day (6,234 jobs)')
    ax1.set_xticks(range(0, 24, 4))
    ax1.set_xticklabels([f'{h:02d}:00' for h in range(0, 24, 4)], fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')
    peak_patch = mpatches.Patch(color='#4D4D4D', label='Peak hours (14:00-17:00 Taiwan time)')
    other_patch = mpatches.Patch(color='#D9D9D9', label='Off-peak')
    ax1.set_ylim(top=max(counts) * 1.32)
    ax1.legend(handles=[peak_patch, other_patch], fontsize=9, loc='upper left')

    # Panel (b): execution-time box plots by GPU type
    plot_data = [gpu_times[g] for g in gpu_order]
    gpu_colors = ['#E6E6E6', '#BFBFBF', '#808080', '#4D4D4D']
    bp = ax2.boxplot(
        plot_data,
        patch_artist=True,
        notch=False,
        vert=True,
        widths=0.5,
        medianprops=dict(color='white', linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker='o', markersize=2.5, alpha=0.4),
    )
    for patch, color in zip(bp['boxes'], gpu_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    for flier, color in zip(bp['fliers'], gpu_colors):
        flier.set(markerfacecolor=color, markeredgecolor=color)

    n_labels = [f'{g}\n(n={len(gpu_times[g])})' for g in gpu_order]
    ax2.set_xticks(range(1, len(gpu_order) + 1))
    ax2.set_xticklabels(n_labels, fontsize=9)
    ax2.set_ylabel('Render time (s)')
    ax2.set_title('(b) Execution-time distribution by GPU type')
    ax2.grid(True, alpha=0.3, axis='y')

    # Annotate medians
    for i, g in enumerate(gpu_order):
        median_val = np.median(gpu_times[g])
        ax2.text(i + 1 + 0.3, median_val, f'{median_val:.0f}s',
                 ha='left', va='center', fontsize=9, color='#333',
                 bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='none', alpha=0.85))

    fig.suptitle('Historical rendering job dataset: arrival pattern and execution times',
                 fontsize=11, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, 'data_overview.pdf')


# ── Figure 8: Phase 4 Day-by-Day Timeline ────────────────────────────────────

def fig_phase4_timeline():
    """Daily deadline miss rate over 30 days for a representative replication (Rep 0)."""
    with open(CGRSP / 'Outputs/phase4/monthly_results.json') as f:
        d = json.load(f)

    focal_scheds = ['FIFO', 'SPT', 'EDF', 'SPT+Rescue', 'RH', 'CADR']
    available = set(d['replications'].keys())
    scheds = [s for s in focal_scheds if s in available]

    # Use replication 0 as the representative example
    rep_idx = 0
    days_data = {}
    for s in scheds:
        daily = d['replications'][s][rep_idx]['daily_results']
        days_data[s] = daily

    n_days = len(days_data[scheds[0]])
    day_indices = list(range(1, n_days + 1))

    # Day-type sequence from replication 0 (any scheduler; all share the same sequence)
    day_types = [days_data[scheds[0]][i]['day_type'] for i in range(n_days)]

    dt_colors = {
        'hectic': ('#E74C3C', 0.12),
        'surge':  ('#E67E22', 0.12),
        'normal': ('#2ECC71', 0.06),
        'quiet':  ('#3498DB', 0.08),
    }

    fig, ax = plt.subplots(figsize=(9.6, 3.9))

    # Background shading by day type
    for i, dt in enumerate(day_types):
        color, alpha = dt_colors.get(dt, ('#CCCCCC', 0.06))
        ax.axvspan(i + 0.5, i + 1.5, color=color, alpha=alpha, linewidth=0)

    # Plot miss rate line for each scheduler
    for s in scheds:
        miss_vals = [days_data[s][i]['deadline_miss_pct'] for i in range(n_days)]
        color = SCHED_COLORS.get(s, '#888')
        lw = 2.0 if s in ('RH', 'CADR') else 1.4
        ls = '-' if s in ('RH', 'CADR', 'SPT+Rescue') else '--'
        ax.plot(day_indices, miss_vals, color=color, linewidth=lw,
                linestyle=ls, label=s, alpha=0.9)

    ax.set_xlabel('Day of simulation')
    ax.set_ylabel('Daily deadline miss rate (%)')
    ax.set_title(
        r'Phase 4 daily miss rate over 30 days (replication 1 of 10, $C_{\max}=5$, '
        '20% tight / 80% loose deadlines)'
    )
    ax.set_xlim(0.5, n_days + 0.5)
    ax.set_xticks(range(1, n_days + 1, 2))
    ax.grid(True, alpha=0.25, axis='y')

    # Scheduler legend
    sched_handles = [
        mpatches.Patch(color=SCHED_COLORS.get(s, '#888'), label=s) for s in scheds
    ]
    # Day-type legend
    dt_legend = [
        mpatches.Patch(color=c, alpha=a * 5, label=dt.capitalize())
        for dt, (c, a) in dt_colors.items()
    ]
    leg1 = ax.legend(handles=sched_handles, fontsize=9, loc='upper left', ncol=3)
    ax.add_artist(leg1)
    ax.legend(handles=dt_legend, fontsize=8.5, loc='upper right', title='Day type', title_fontsize=9)

    fig.tight_layout()
    _save(fig, 'phase4_timeline.pdf')


# ── Figure 9: Scheduling Decision Diagram ───────────────────────────────────

def fig_scheduling_decision():
    """
    Conceptual diagram of the CGRSP scheduling decision at a single event.

    Left column:  job queue (3 jobs with complexity/deadline properties).
    Right column: GPU fleet (5 slots: 3 idle, 2 busy).
    Center:       scheduler decision box with lexicographic objective.
    Bottom:       resulting assignment, with the provisioning delay tau_prov_j
                  made explicit on the dispatch-to-start step.
    """
    # Restrained palette: neutral slate structure, GPU-type colour only as a
    # thin accent (matching the gpu_colors mapping used in data_overview.pdf /
    # diurnal_model.pdf), a single red accent reserved for deadline urgency.
    INK      = '#1A1A1A'
    MUTED    = '#4D4D4D'
    LIGHT_BG = '#F2F2F2'
    URGENT   = '#1A1A1A'
    OK       = '#4D4D4D'
    GPU_COLOR = {
        'RTX A5000': '#D9D9D9', 'RTX 3090': '#A6A6A6',
        'RTX A4500': '#808080', 'RTX A4000': '#BFBFBF',
    }

    fig = plt.figure(figsize=(10.4, 6.7))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 13); ax.set_ylim(0, 8.4); ax.axis('off')

    # ── helpers ──────────────────────────────────────────────────────────────
    def rbox(x, y, w, h, facecolor, edgecolor=INK, lw=1.0, alpha=None,
             linestyle='-', radius=0.045):
        rect = FancyBboxPatch((x, y), w, h,
                              boxstyle=f'round,pad={radius}',
                              linewidth=lw, edgecolor=edgecolor,
                              facecolor=facecolor, alpha=alpha,
                              linestyle=linestyle)
        ax.add_patch(rect)

    def txt(x, y, s, size=9, weight='normal', color='#1a1a1a', ha='center',
            va='center', style='normal'):
        ax.text(x, y, s, fontsize=size * 1.4, fontweight=weight, color=color,
                ha=ha, va=va, fontstyle=style)

    def arrow(x1, y1, x2, y2, color=INK, lw=1.3, style='->'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color=color,
                                   lw=lw, shrinkA=3, shrinkB=3))

    # ── Section labels ────────────────────────────────────────────────────────
    txt(3.2, 8.15, 'Job Queue', size=10.5, weight='bold', color=INK)
    txt(9.8, 8.15, r'GPU Fleet  ($C_{\max}$ slots)', size=10.5, weight='bold',
        color=INK)

    # ── Job cards (left column) ───────────────────────────────────────────────
    jobs = [
        dict(label='$j_1$', kappa=r'$\kappa=1$ (low)',
             deadline='Tight (1 h)', tight=True),
        dict(label='$j_2$', kappa=r'$\kappa=2$ (medium)',
             deadline='Loose (8 h)', tight=False),
        dict(label='$j_3$', kappa=r'$\kappa=1$ (low)',
             deadline='Loose (8 h)', tight=False),
    ]
    jx, jw, jh, jy0, jgap = 0.35, 2.55, 1.05, 5.35, 0.32
    for i, j in enumerate(jobs):
        y = jy0 - i * (jh + jgap)
        edge = URGENT if j['tight'] else MUTED
        rbox(jx, y, jw, jh, facecolor='white', edgecolor=edge, lw=1.3)
        txt(jx + jw / 2, y + jh * 0.78, j['label'],
            size=11, weight='bold', color=INK)
        txt(jx + jw / 2, y + jh * 0.46, j['kappa'], size=9.5, color=INK)
        txt(jx + jw / 2, y + jh * 0.16,
            j['deadline'], size=9.5, weight='bold',
            color=URGENT if j['tight'] else OK)

    # Arrival label
    txt(jx + jw / 2, jy0 + jh + 0.25,
        r'Arrival stream  $\lambda$ (Poisson)',
        size=8.5, color=MUTED, style='italic')

    # ── GPU slot cards (right column) ─────────────────────────────────────────
    slots = [
        dict(gpu='RTX A5000', status='IDLE', stock='High stock',
             price='\\$0.27/hr', idle=True),
        dict(gpu='RTX 3090',  status='IDLE', stock='High stock',
             price='\\$0.46/hr', idle=True),
        dict(gpu='RTX A4000', status='IDLE', stock='High stock',
             price='\\$0.25/hr', idle=True),
        dict(gpu='RTX A4000', status='BUSY', stock='',
             price='',           idle=False),
        dict(gpu='RTX A4500', status='BUSY', stock='',
             price='',           idle=False),
    ]
    sx, sw, sh, sy0, sgap = 7.1, 2.55, 0.82, 7.05, 0.28
    fleet_top = sy0 + sh
    fleet_bottom = sy0 - (len(slots) - 1) * (sh + sgap)
    for i, s in enumerate(slots):
        y = sy0 - i * (sh + sgap)
        accent = GPU_COLOR[s['gpu']]
        face = LIGHT_BG if s['idle'] else 'white'
        alpha = 1.0 if s['idle'] else 0.6
        rbox(sx, y, sw, sh, facecolor=face,
             edgecolor=accent if s['idle'] else '#BFBFBF',
             lw=1.3 if s['idle'] else 0.9, alpha=alpha)
        # thin accent bar identifying GPU type, consistent with other figures
        ax.add_patch(plt.Rectangle((sx, y), 0.09, sh, facecolor=accent,
                                    edgecolor='none',
                                    alpha=1.0 if s['idle'] else 0.5))
        ec = INK if s['idle'] else '#808080'
        st_col = OK if s['status'] == 'IDLE' else URGENT
        txt(sx + sw / 2 + 0.05, y + sh * 0.72, s['gpu'],
            size=9.5, weight='bold', color=ec)
        txt(sx + sw * 0.30, y + sh * 0.30, s['status'],
            size=8.5, weight='bold', color=st_col, ha='center')
        if s['stock']:
            txt(sx + sw * 0.75, y + sh * 0.37, s['stock'],
                size=8, color=MUTED, ha='center')
        if s['price']:
            txt(sx + sw * 0.74, y + sh * 0.28 if not s['stock'] else y + sh * 0.08,
                s['price'], size=8, color=MUTED, ha='center')

    # ── Scheduler box (centre gap between the two columns) ───────────────────
    # width kept clear of the fleet column (sx=7.1) so the fleet->scheduler
    # arrow has an unobstructed horizontal channel
    schx, schy, schw, schh = 3.3, 3.95, 3.2, 1.9
    rbox(schx, schy, schw, schh, facecolor=LIGHT_BG,
         edgecolor=INK, lw=1.6)
    txt(schx + schw / 2, schy + schh * 0.80,
        'Scheduler  (policy $\\pi$)',
        size=10.5, weight='bold', color=INK)
    txt(schx + schw / 2, schy + schh * 0.55,
        r'Observe: $Q$, fleet, $s_g(t)$',
        size=9.5, color=INK)
    txt(schx + schw / 2, schy + schh * 0.32,
        r'Decide: $x_j^m \in \{0,1\}$',
        size=9.5, color=INK)
    txt(schx + schw / 2, schy + schh * 0.11,
        r'Target: $C_{\mathrm{QoS}}$, then $C_{\mathrm{Cost}}$',
        size=9.5, color=INK)

    # ── Arrows: queue + fleet -> scheduler ───────────────────────────────────
    # horizontal, in the clear channels either side of the scheduler box
    # (0.05 stand-off past the rounded-corner pad so arrowheads touch edges,
    # not boxes)
    arrow(jx + jw + 0.05, schy + schh * 0.65,
          schx - 0.05, schy + schh * 0.65, color=MUTED)
    arrow(sx - 0.06, schy + schh * 0.65,
          schx + schw + 0.05, schy + schh * 0.65, color=MUTED)

    # ── Dispatch step: scheduler -> provisioning delay -> assignment cards ────
    assign = [
        dict(label='$j_1 \\to$ RTX 3090', gpu='RTX 3090',
             note='Low-complexity tie\nwith $j_3$'),
        dict(label='$j_2 \\to$ RTX A4000', gpu='RTX A4000',
             note='Medium job; lowest\ntotal rental cost'),
        dict(label='$j_3 \\to$ RTX A5000', gpu='RTX A5000',
             note='Low-complexity tie\nwith $j_1$'),
    ]
    xs = [1.0, 4.7, 8.4]
    aschx_center = schx + schw / 2
    line_y = 2.2   # below the j3 card's bottom edge (~2.57), clear of all boxes
    arrow(aschx_center, schy - 0.06, aschx_center, line_y, color=MUTED)
    ax.plot([xs[0] + 1.6, xs[-1] + 1.6], [line_y, line_y], color=MUTED, lw=1.3)

    # Provisioning-delay annotation beside the dispatch arrow (no box, so it
    # never sits in an arrow's path), tau_prov_j explicit
    txt(aschx_center + 0.15, 3.15,
        'dispatch\n$\\to$ provisioning\ndelay $\\tau^{\\mathrm{prov}}_j$\n$\\to$ start',
        size=8.5, color=INK, style='italic', ha='left')

    aw, ah = 3.2, 1.15
    for x, a in zip(xs, assign):
        accent = GPU_COLOR[a['gpu']]
        arrow(x + 1.6, line_y, x + 1.6, 0.35 + ah + 0.07, color=accent)
        rbox(x, 0.35, aw, ah, facecolor=LIGHT_BG, edgecolor=accent, lw=1.3)
        ax.add_patch(plt.Rectangle((x, 0.35), aw, 0.09, facecolor=accent,
                                    edgecolor='none'))
        txt(x + aw / 2, 0.35 + ah * 0.75,
            a['label'], size=9, weight='bold', color=INK)
        txt(x + aw / 2, 0.35 + ah * 0.32,
            a['note'], size=8.5, color=MUTED)

    # ── Outcome label bar ─────────────────────────────────────────────────────
    txt(6.5, 0.04, r'Outcomes measured: $w_j$ (wait) $\cdot$ $\delta_j$ (deadline met) $\cdot$ $C_{\mathrm{Cost}}$ (rental)',
        size=10, color=MUTED, style='italic')

    # ── Title ──────────────────────────────────────────────────────────────
    txt(6.5, 8.62, 'CGRSP Scheduling Decision  (single event at time $t$)',
        size=12, weight='bold', color='#1a1a1a')

    fig.tight_layout(pad=0.1)
    _save(fig, 'scheduling_decision.pdf')


# ── Main ──────────────────────────────────────────────────────────────────────

# ── Figure 13: C_max Sensitivity Line Chart ──────────────────────────────────

def fig_cmax_sensitivity():
    with open(DATA / 'cmax_sensitivity.json') as f:
        raw = json.load(f)
    cmax_vals = raw['metadata']['cmax_values']
    scheds = ['FIFO', 'SPT', 'EDF', 'SPT+Rescue', 'RH', 'CADR', 'Adaptive']
    data_miss = {s: [raw['results'][f'cmax{c}']['schedulers'][s]['avg_miss_pct']
                     for c in cmax_vals] for s in scheds}
    data_wait = {s: [raw['results'][f'cmax{c}']['schedulers'][s]['avg_wait_min']
                     for c in cmax_vals] for s in scheds}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    lss = {'FIFO': ':', 'SPT': '-', 'EDF': '--', 'SPT+Rescue': '-.',
           'RH': '-', 'CADR': '-', 'Adaptive': '--'}
    mks = {'FIFO': 'o', 'SPT': 's', 'EDF': '^', 'SPT+Rescue': 'D',
           'RH': '*', 'CADR': 'P', 'Adaptive': 'v'}

    for sched in ['FIFO', 'SPT', 'EDF', 'SPT+Rescue', 'Adaptive', 'CADR', 'RH']:
        ax1.plot(cmax_vals, data_miss[sched], color=SCHED_COLORS[sched],
                 ls=lss[sched], marker=mks[sched], ms=7 if sched in ('RH','CADR') else 5,
                 lw=2 if sched in ('RH','CADR') else 1.3, label=sched)
        ax2.plot(cmax_vals, data_wait[sched], color=SCHED_COLORS[sched],
                 ls=lss[sched], marker=mks[sched], ms=7 if sched in ('RH','CADR') else 5,
                 lw=2 if sched in ('RH','CADR') else 1.3, label=sched)

    ax1.axvline(5, color='#7F8C8D', lw=0.8, ls='--', alpha=0.6)
    ax1.set_xlabel('Concurrency limit $C_{\\max}$'); ax1.set_ylabel('Miss rate (%)')
    ax1.set_xticks(cmax_vals); ax1.set_title('(a) Deadline miss rate vs $C_{\\max}$')
    ax1.legend(fontsize=8, ncol=2); ax1.grid(True, alpha=0.3)

    ax2.axvline(5, color='#7F8C8D', lw=0.8, ls='--', alpha=0.6)
    ax2.set_xlabel('Concurrency limit $C_{\\max}$'); ax2.set_ylabel('Avg wait (min)')
    ax2.set_xticks(cmax_vals); ax2.set_title('(b) Average wait vs $C_{\\max}$')
    ax2.legend(fontsize=8, ncol=2); ax2.grid(True, alpha=0.3)

    ax1.annotate('baseline\n$C_{\\max}=5$', xy=(5, data_miss['RH'][cmax_vals.index(5)]), xytext=(6.5, 25),
                 fontsize=7.5, color='#7F8C8D',
                 arrowprops=dict(arrowstyle='->', color='#7F8C8D', lw=0.8))

    fig.suptitle('Concurrency Limit Sensitivity ($n=10$ seeds, hectic day)', fontsize=11)
    fig.tight_layout()
    _save(fig, 'cmax_sensitivity.pdf')


# ── Figure 14: Ablation Waterfall ────────────────────────────────────────────

def fig_ablation_waterfall():
    with open(DATA / 'statistical.json') as f:
        raw = json.load(f)
    chain = ['FIFO', 'SPT', 'EDF', 'SPT+Rescue', 'Adaptive', 'RH', 'CADR']
    wait = [raw['schedulers'][s]['wait_min']['mean'] for s in chain]
    miss = [raw['schedulers'][s]['miss_pct']['mean'] for s in chain]
    tard = [raw['schedulers'][s]['avg_tardiness_min']['mean'] for s in chain]

    x = np.arange(len(chain))
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.8))
    titles = ['(a) Avg wait (min)', '(b) Miss rate (%)', '(c) Mean tardiness (min)']
    datasets = [wait, miss, tard]
    ylabels = ['Avg wait (min)', 'Miss rate (%)', 'Mean tardiness (min)']

    for ax, vals, title, ylabel in zip(axes, datasets, titles, ylabels):
        colors = [SCHED_COLORS.get(s, '#999') for s in chain]
        bars = ax.bar(x, vals, color=colors, edgecolor='white', linewidth=0.7, width=0.6)
        ax.plot(x, vals, 'k--o', ms=4, lw=1.0, zorder=5, alpha=0.6)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals) * 0.035,
                    f'{v:.1f}', ha='center', va='bottom', fontsize=9, zorder=6,
                    bbox=dict(boxstyle='round,pad=0.12', fc='white', ec='none', alpha=0.85))
        ax.set_xticks(x); ax.set_xticklabels(chain, rotation=30, ha='right', fontsize=9.5)
        ax.set_ylabel(ylabel); ax.set_title(title)
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_ylim(0, max(vals) * 1.22)

    fig.suptitle('Ablation: evolutionary chain, Phase 3, hectic day ($n=30$ seeds, $C_{\\max}=5$)',
                 fontsize=10)
    fig.tight_layout()
    _save(fig, 'ablation_waterfall.pdf')


# ── Figure 15: Phase 3 CI Forest Plot ────────────────────────────────────────

def fig_phase3_ci():
    with open(DATA / 'statistical.json') as f:
        raw = json.load(f)
    scheds = list(raw['schedulers'].keys())
    def col(metric, field):
        return [raw['schedulers'][s][metric][field] for s in scheds]
    wait_mean, wait_lo, wait_hi = (col('wait_min', k) for k in ('mean', 'ci95_lo', 'ci95_hi'))
    miss_mean, miss_lo, miss_hi = (col('miss_pct', k) for k in ('mean', 'ci95_lo', 'ci95_hi'))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.4, 6.3))
    y = np.arange(len(scheds))

    for ax, means, los, his, xlabel, title in [
        (ax1, wait_mean, wait_lo, wait_hi, 'Avg wait (min)', '(a) Average wait with 95% CI'),
        (ax2, miss_mean, miss_lo, miss_hi, 'Miss rate (%)',  '(b) Miss rate with 95% CI'),
    ]:
        # Sort schedulers by mean within each panel (best at the bottom)
        order = sorted(range(len(scheds)), key=lambda i: means[i], reverse=True)
        for row, idx in enumerate(order):
            c = SCHED_COLORS.get(scheds[idx], '#999')
            ax.plot([los[idx], his[idx]], [row, row], color=c, lw=2.5,
                    solid_capstyle='round')
            ax.plot(means[idx], row, 'o', color=c, ms=7, zorder=5)
        ax.set_yticks(y); ax.set_yticklabels([scheds[i] for i in order], fontsize=9)
        ax.set_xlabel(xlabel); ax.set_title(title)
        ax.grid(True, axis='x', alpha=0.3)
        best = order[-1]
        ax.axvline(means[best], color=SCHED_COLORS.get(scheds[best], '#999'),
                   lw=0.7, ls='--', alpha=0.4)

    fig.suptitle('Phase 3 statistical validation: 95% confidence intervals ($n=30$ seeds, hectic day)',
                 fontsize=10)
    fig.tight_layout()
    _save(fig, 'phase3_ci.pdf')


# ── Figure 16: Scheduler Design Lineage DAG ──────────────────────────────────

def fig_scheduler_lineage():
    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    ax.set_xlim(0, 12); ax.set_ylim(0, 7.3); ax.axis('off')

    def node(x, y, label, sub='', color='#E6E6E6', w=2.2, h=0.85, lw=1.2):
        rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                              boxstyle='round,pad=0.08', lw=lw,
                              edgecolor='#1A1A1A', facecolor=color)
        ax.add_patch(rect)
        ax.text(x, y + (0.18 if sub else 0), label, ha='center', va='center',
                fontsize=10.5, fontweight='bold')
        if sub:
            ax.text(x, y - 0.22, sub, ha='center', va='center',
                    fontsize=9.5, color='#4D4D4D', style='italic')

    def edge(x1, y1, x2, y2, label='', color='#4D4D4D', side='right', dx=0.12, dy=0.0):
        ax.annotate('', xy=(x2, y2 + 0.43), xytext=(x1, y1 - 0.43),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.3,
                                   shrinkA=2, shrinkB=2))
        if label:
            mx, my = (x1 + x2)/2, (y1 + y2)/2
            if side == 'left':
                ax.text(mx - dx, my + dy, label, ha='right', va='center',
                        fontsize=9.5, color='#4D4D4D', style='italic')
            else:
                ax.text(mx + dx, my + dy, label, ha='left', va='center',
                        fontsize=9.5, color='#4D4D4D', style='italic')

    # Layout: y levels 5.5 (FIFO), 4 (SPT/EDF), 2.5 (SPT+Rescue/Adaptive), 1 (RH/CADR)
    node(6, 5.5, 'FIFO', 'First-in first-out baseline', '#D9D9D9')

    node(3, 4.0, 'SPT',  'Shortest processing time', '#E6E6E6')
    node(9, 4.0, 'EDF',  'Earliest deadline first',  '#CCCCCC')

    node(3, 2.5, 'SPT+Rescue', 'Laxity rescue tier',  '#BFBFBF')
    node(9, 2.5, 'Adaptive',   'Queue-pressure widening\n+ hopeless demotion', '#CCCCCC')

    node(1.7, 1.0, 'RH',   'Rolling-Horizon', '#A6A6A6', w=2.2, lw=2.4)
    node(4.6, 1.0, 'CADR', 'Cost-Aware\nDeadline-Risk', '#B3B3B3', w=2.2, lw=2.4)

    edge(6, 5.5, 3, 4.0, '+ SPT ordering', side='left', dx=0.1, dy=0.3)
    edge(6, 5.5, 9, 4.0, '+ deadline ordering', dx=0.1, dy=0.3)
    edge(3, 4.0, 3, 2.5, '+ laxity rescue')
    edge(9, 4.0, 9, 2.5, 'EDF-style\n  urgent ordering')
    # SPT+Rescue -> Adaptive (horizontal): Adaptive extends the rescue idea
    ax.annotate('', xy=(9 - 1.25, 2.5), xytext=(3 + 1.25, 2.5),
                arrowprops=dict(arrowstyle='->', color='#555', lw=1.3,
                               shrinkA=2, shrinkB=2))
    ax.text(6, 2.72, '+ pressure widening  + hopeless tier', ha='center',
            va='bottom', fontsize=9.5, color='#555', style='italic')
    edge(3, 2.5, 1.7, 1.0, '+ timeline projection\n+ anticipation  ', '#1A1A1A', side='left', dx=0.5, dy=0.15)
    edge(3, 2.5, 4.6, 1.0, '+ CR* triage\n  + cost-aware placement', '#1A1A1A', dx=0.65, dy=0.3)

    ax.text(6, 7.0, 'Scheduler Design Lineage', ha='center', fontsize=12,
            fontweight='bold')
    ax.text(6, 6.6, 'Each arrow labels the additional algorithmic ingredient',
            ha='center', fontsize=9.5, color='#4D4D4D')

    # Legend: border weight distinguishes proposed schedulers from baselines
    for i, (lbl, lw_) in enumerate([('Baseline / existing', 1.2),
                                     ('Proposed (this work)', 2.4)]):
        rect = FancyBboxPatch((7.5 + i*2.3, 0.1), 2.1, 0.55,
                              boxstyle='round,pad=0.06', lw=lw_,
                              edgecolor='#1A1A1A', facecolor='white')
        ax.add_patch(rect)
        ax.text(7.5 + i*2.3 + 1.05, 0.375, lbl, ha='center', va='center', fontsize=9.5)

    fig.tight_layout()
    _save(fig, 'scheduler_lineage.pdf')


# ── Figure 17: Job Lifecycle Timeline ────────────────────────────────────────

def fig_job_lifecycle():
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 4.0))

    for ax, (met, missed) in zip(axes, [(False, False), (True, True)]):
        # Per ch3: t_start = t_dispatch + tau_prov, and w_j = t_start - t_submit,
        # so the provisioning delay is a component of the waiting time.
        t_submit   = 0
        t_dispatch = 45                  # slot acquired, rental request issued
        t_start    = t_dispatch + 15     # provisioning done, execution begins
        t_comp     = t_start + 80        # 80 s execution
        d_j        = 160 if not missed else 125   # deadline

        ax.set_xlim(-10, 210); ax.set_ylim(-0.5, 2.35); ax.axis('off')

        # Timeline axis
        ax.annotate('', xy=(200, 0), xytext=(-5, 0),
                    arrowprops=dict(arrowstyle='->', color='#1A1A1A', lw=1.5))
        ax.text(200, -0.3, 'time (s)', ha='right', fontsize=8.5, color='#1A1A1A')

        def brace(x0, x1, y, label, color):
            ax.annotate('', xy=(x1, y), xytext=(x0, y),
                        arrowprops=dict(arrowstyle='<->', color=color, lw=1.5))
            ax.text((x0+x1)/2, y + 0.22, label, ha='center', va='bottom',
                    fontsize=10, color=color, fontweight='bold')

        # Brackets: wait includes provisioning (w_j ends at execution start)
        brace(t_submit,   t_start, 1.3, '$w_j$ (wait)', '#1A1A1A')
        brace(t_dispatch, t_start, 0.55, '$\\tau^{\\mathrm{prov}}_j$', '#666666')
        brace(t_start,    t_comp,  1.3,  '$W(\\kappa,g)$ (exec)', '#4D4D4D')

        # Events
        for xt, lbl, col in [(t_submit,   '$t^{\\rm submit}_j$',   '#1A1A1A'),
                              (t_dispatch, '$t^{\\rm dispatch}_j$', '#4D4D4D'),
                              (t_start,    '$t^{\\rm start}_j$',    '#666666'),
                              (t_comp,     '$t^{\\rm complete}_j$', '#333333')]:
            ax.plot(xt, 0, '|', ms=14, mew=2, color=col)
            ax.text(xt, -0.3, lbl, ha='center', va='top', fontsize=9.5, color=col)

        # Deadline
        dl_color = '#1A1A1A'
        dl_ls = '--' if not missed else ':'
        ax.vlines(d_j, -0.5, 1.85, colors=dl_color, linestyles=dl_ls, lw=1.5)
        ax.text(d_j, 1.9, '$d_j$', ha='center', fontsize=10, color=dl_color, fontweight='bold')

        if missed:
            brace(d_j, t_comp, 0.55, '', '#1A1A1A')
            ax.text(t_comp + 4, 0.55, '$\\tau_j$ (tardiness)', ha='left', va='center',
                    fontsize=10, color='#1A1A1A', fontweight='bold')
            ax.text(195, 1.7, r'$\delta_j=0$ (MISSED)', ha='right', fontsize=9,
                    color='#1A1A1A', fontweight='bold')
        else:
            ax.text(195, 1.7, r'$\delta_j=1$ (ON TIME)', ha='right', fontsize=9,
                    color='#4D4D4D', fontweight='bold')

        subtitle = '(a) Job meets deadline' if not missed else '(b) Job misses deadline'
        ax.text(-5, 2.2, subtitle, ha='left', fontsize=9.5, fontweight='bold')

    fig.suptitle('Single-job lifecycle: metric intervals on the time axis', fontsize=11, y=1.01)
    fig.tight_layout()
    _save(fig, 'job_lifecycle.pdf')


# ── Figure 18: GPU Performance Heat Map ──────────────────────────────────────

def fig_gpu_heatmap():
    # Load W(kappa, g) from the simulator's calibrated lookup (single source
    # of truth), bypassing the package to avoid pulling in heavy deps.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'execution_times', CGRSP / 'cgrsp' / 'data' / 'execution_times.py')
    et = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(et)

    gpu_enums = [et.GPUType.RTX_A5000, et.GPUType.RTX_3090,
                 et.GPUType.RTX_A4500, et.GPUType.RTX_A4000]
    gpus  = ['RTX A5000', 'RTX 3090', 'RTX A4500', 'RTX A4000']
    kaps  = ['Low ($\\kappa=1$)', 'Medium ($\\kappa=2$)', 'High ($\\kappa=3$)']
    times = np.array([[et.EXECUTION_TIME_LOOKUP[(k, g)] for k in (1, 2, 3)]
                      for g in gpu_enums])
    # Secure-cloud prices: fixed deterministic constants pinned to the
    # 11 March 2026 snapshot (see tab:gpu_specs); deliberately not live data.
    prices = [0.27, 0.46, 0.25, 0.25]   # $/hr

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))

    im1 = ax1.imshow(times, aspect='auto', cmap='YlOrRd')
    ax1.set_xticks(range(3)); ax1.set_xticklabels(kaps, fontsize=9)
    ax1.set_yticks(range(4)); ax1.set_yticklabels(gpus, fontsize=9)
    ax1.set_title('(a) Mean execution time (s)', fontsize=10)
    plt.colorbar(im1, ax=ax1, shrink=0.85, label='seconds')
    # hatch the three cells estimated rather than measured (n < 5)
    for i, j in [(2, 0), (3, 0), (3, 1)]:
        ax1.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                    hatch='///', edgecolor='#555', lw=0))
    ax1.set_xlabel('Hatched cells: estimated, not measured', fontsize=8)
    for i in range(4):
        for j in range(3):
            ax1.text(j, i, f'{times[i,j]:.1f}', ha='center', va='center',
                     fontsize=9, color='black' if times[i,j] < 90 else 'white', zorder=5,
                     bbox=(dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.85)
                           if (i, j) in [(2, 0), (3, 0), (3, 1)] else None))

    price_arr = np.array(prices).reshape(-1, 1)
    im2 = ax2.imshow(price_arr, aspect='auto', cmap='Blues')
    ax2.set_xticks([0]); ax2.set_xticklabels(['$/hr'], fontsize=9)
    ax2.set_yticks(range(4)); ax2.set_yticklabels(gpus, fontsize=9)
    ax2.set_title('(b) Hourly rental price', fontsize=10)
    plt.colorbar(im2, ax=ax2, shrink=0.85, label='USD/hr')
    for i, p in enumerate(prices):
        ax2.text(0, i, f'${p:.2f}', ha='center', va='center', fontsize=10,
                 color='black' if p < 0.4 else 'white')

    fig.tight_layout()
    _save(fig, 'gpu_heatmap.pdf')


# ── Figure 19: Rescue Threshold Line Chart ───────────────────────────────────

def fig_threshold_sensitivity():
    with open(DATA / 'threshold_sensitivity.json') as f:
        raw = json.load(f)
    theta = raw['metadata']['theta_values']
    wait  = [raw['results'][f'theta{t}']['avg_wait_min'] for t in theta]
    miss  = [raw['results'][f'theta{t}']['avg_miss_pct'] for t in theta]
    xs    = range(len(theta))       # equal category spacing (no tick crowding)
    xdef  = theta.index(600)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    ax1.plot(xs, miss, 'o-', color=SCHED_COLORS['SPT+Rescue'], lw=2, ms=7)
    ax1.axvline(xdef, color='#999', lw=1, ls='--')
    ax1.set_xlabel('Laxity threshold $\\theta$ (s)')
    ax1.set_ylabel('Miss rate (%)')
    ax1.set_title('(a) Miss rate vs $\\theta$')
    ax1.set_xticks(list(xs)); ax1.set_xticklabels([str(t) for t in theta])
    ax1.grid(True, alpha=0.3)
    # Data-driven limits: fixed limits once hid every point of panel (b).
    ax1.set_ylim(min(miss) - 0.8, max(miss) + 1.2)
    ax1.annotate('default\n$\\theta=600$s', xy=(xdef, miss[xdef]),
                 xytext=(xdef + 0.55, max(miss) + 0.4),
                 fontsize=8, color='#666',
                 arrowprops=dict(arrowstyle='->', color='#666', lw=0.8))

    ax2.plot(xs, wait, 's-', color=SCHED_COLORS['SPT+Rescue'], lw=2, ms=7)
    ax2.axvline(xdef, color='#999', lw=1, ls='--')
    ax2.set_xlabel('Laxity threshold $\\theta$ (s)')
    ax2.set_ylabel('Avg wait (min)')
    ax2.set_title('(b) Avg wait vs $\\theta$')
    ax2.set_xticks(list(xs)); ax2.set_xticklabels([str(t) for t in theta])
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(min(wait) - 1.0, max(wait) + 1.0)

    fig.suptitle('SPT+Rescue threshold sensitivity ($n=10$ seeds, hectic day, $C_{\\max}=5$)',
                 fontsize=10)
    fig.tight_layout()
    _save(fig, 'threshold_sensitivity.pdf')


# ── Figure 20: CR* Sensitivity Line Chart ────────────────────────────────────

def fig_cr_sensitivity():
    with open(DATA / 'cr_sensitivity.json') as f:
        raw = json.load(f)
    cr    = raw['metadata']['cr_star_values']
    wait  = [raw['results'][str(c)]['avg_wait_min'] for c in cr]
    miss  = [raw['results'][str(c)]['avg_miss_pct'] for c in cr]
    xs    = range(len(cr))          # equal category spacing (no tick collisions)
    xdef  = cr.index(3.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    ax1.plot(xs, miss, 'o-', color=SCHED_COLORS['CADR'], lw=2, ms=7)
    ax1.axvline(xdef, color='#999', lw=1, ls='--')
    ax1.set_xlabel('Critical-ratio threshold $CR^*$')
    ax1.set_ylabel('Miss rate (%)')
    ax1.set_title('(a) Miss rate vs $CR^*$')
    ax1.set_xticks(list(xs)); ax1.set_xticklabels([f'{c:g}' for c in cr])
    ax1.grid(True, alpha=0.3)
    # Anchor on the plotted point: a fixed anchor below the axis was never drawn.
    ax1.set_ylim(min(miss) - 0.8, max(miss) + 1.2)
    ax1.annotate('default\n$CR^*=3$', xy=(xdef, miss[xdef]),
                 xytext=(xdef + 1.2, max(miss) + 0.4),
                 fontsize=8, color='#666',
                 arrowprops=dict(arrowstyle='->', color='#666', lw=0.8))

    ax2.plot(xs, wait, 's-', color=SCHED_COLORS['CADR'], lw=2, ms=7)
    ax2.axvline(xdef, color='#999', lw=1, ls='--')
    ax2.set_xlabel('Critical-ratio threshold $CR^*$')
    ax2.set_ylabel('Avg wait (min)')
    ax2.set_title('(b) Avg wait vs $CR^*$')
    ax2.set_xticks(list(xs)); ax2.set_xticklabels([f'{c:g}' for c in cr])
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(min(wait) - 0.5, max(wait) + 0.5)

    fig.suptitle('CADR critical-ratio threshold sensitivity ($n=10$ seeds, hectic day, $C_{\\max}=5$)',
                 fontsize=10)
    fig.tight_layout()
    _save(fig, 'cr_sensitivity.pdf')


# ── Figure 21: Anticipation Reservation Grouped Bar Chart ────────────────────

def fig_reservation_sensitivity():
    with open(DATA / 'reservation_sensitivity.json') as f:
        raw = json.load(f)
    rho_vals = raw['metadata']['rho_res_values']
    def day(dname, field):
        return [raw['results'][dname][f'r{r}'][field] for r in rho_vals]
    surge_miss_overall  = day('surge',  'avg_miss_pct')
    surge_miss_tight    = day('surge',  'tight_miss_pct')
    hectic_miss_overall = day('hectic', 'avg_miss_pct')
    hectic_miss_tight   = day('hectic', 'tight_miss_pct')

    x = np.arange(len(rho_vals))
    w = 0.35
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # Surge day
    ax1.bar(x - w/2, surge_miss_overall, w, label='Overall miss %',
            color='#3498DB', alpha=0.85)
    ax1.bar(x + w/2, surge_miss_tight, w, label='Tight-deadline miss %',
            color='#E74C3C', alpha=0.85)
    ax1.set_xticks(x); ax1.set_xticklabels([f'$\\rho_{{\\rm res}}={r}$' for r in rho_vals])
    ax1.set_ylabel('Miss rate (%)'); ax1.set_title('(a) Surge day ($\\rho < 1$)')
    ax1.legend(fontsize=8.5, loc='upper left'); ax1.grid(True, axis='y', alpha=0.3)
    ax1.set_ylim(0, 37)
    for xi, (ov, ti) in enumerate(zip(surge_miss_overall, surge_miss_tight)):
        ax1.text(xi - w/2, ov + 0.5, f'{ov:.1f}', ha='center', fontsize=8)
        ax1.text(xi + w/2, ti + 0.5, f'{ti:.1f}', ha='center', fontsize=8)

    # Hectic day
    ax2.bar(x - w/2, hectic_miss_overall, w, label='Overall miss %',
            color='#3498DB', alpha=0.85)
    ax2.bar(x + w/2, hectic_miss_tight, w, label='Tight-deadline miss %',
            color='#E74C3C', alpha=0.85)
    ax2.set_xticks(x); ax2.set_xticklabels([f'$\\rho_{{\\rm res}}={r}$' for r in rho_vals])
    ax2.set_ylabel('Miss rate (%)'); ax2.set_title('(b) Hectic day ($\\rho > 1$, load gate active)')
    ax2.legend(fontsize=8.5, loc='upper left'); ax2.grid(True, axis='y', alpha=0.3)
    ax2.set_ylim(0, 58)
    for xi, (ov, ti) in enumerate(zip(hectic_miss_overall, hectic_miss_tight)):
        ax2.text(xi - w/2, ov + 0.5, f'{ov:.1f}', ha='center', fontsize=8)
        ax2.text(xi + w/2, ti + 0.5, f'{ti:.1f}', ha='center', fontsize=8)
    ax2.text(2.2, 49, 'Load gate active:\nresult unchanged\nacross all $\\rho_{\\rm res}$',
             ha='center', fontsize=9, color='#555',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#FEF9E7', edgecolor='#999'))

    fig.suptitle('Anticipation reservation sensitivity: RH scheduler\n'
                 '($n=30$ seeds, $C_{\\max}=5$; surge and hectic day types)',
                 fontsize=10)
    fig.tight_layout()
    _save(fig, 'reservation_sensitivity.pdf')


# ── Figure 22: SPT+Rescue Decision Flowchart ─────────────────────────────────

def fig_rescue_flow():
    fig, ax = plt.subplots(figsize=(8.6, 3.61))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5); ax.axis('off')

    def fbox(x, y, w, h, label, color='#F2F2F2', fs=10):
        rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                              boxstyle='round,pad=0.08', lw=1.2,
                              edgecolor='#1A1A1A', facecolor=color)
        ax.add_patch(rect)
        ax.text(x, y, label, ha='center', va='center', fontsize=fs)

    def diamond(x, y, w, h, label, fs=10):
        pts = [(x, y + h/2), (x + w/2, y), (x, y - h/2), (x - w/2, y)]
        ax.add_patch(mpatches.Polygon(pts, closed=True, lw=1.2,
                                       edgecolor='#1A1A1A', facecolor='#F2F2F2'))
        ax.text(x, y, label, ha='center', va='center', fontsize=fs)

    def farrow(x1, y1, x2, y2, label='', lab_dx=0.12, lab_dy=0.0):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#1A1A1A', lw=1.3,
                                   shrinkA=4, shrinkB=4))
        if label:
            ax.text((x1+x2)/2 + lab_dx, (y1+y2)/2 + lab_dy, label,
                    ha='left', va='center', fontsize=9, color='#4D4D4D',
                    fontweight='bold')

    fbox(1.5, 2.5, 2.4, 1.1, 'Job $j$ in queue\nat event time $t$', '#E6E6E6')
    fbox(4.6, 2.5, 2.6, 1.1,
         'Compute laxity\n$L_j = d_j - t - e_j$', '#EBEBEB')
    diamond(7.6, 2.5, 1.9, 1.7, '$L_j < \\theta$?')
    fbox(10.3, 3.9, 2.6, 1.0, 'Urgent tier\nEDF order (by $d_j$)', '#BFBFBF')
    fbox(10.3, 1.1, 2.6, 1.0, 'Normal tier\nSPT order (by $e_j$)', '#E6E6E6')

    farrow(2.7, 2.5, 3.3, 2.5)
    farrow(5.9, 2.5, 6.65, 2.5)
    farrow(7.6, 3.35, 8.95, 3.9, 'yes', lab_dx=-0.9, lab_dy=0.30)
    farrow(7.6, 1.65, 8.95, 1.1, 'no',  lab_dx=-0.9, lab_dy=-0.30)

    ax.text(10.3, 2.5,
            'All urgent jobs dispatch\nbefore any normal job',
            ha='center', va='center', fontsize=8, color='#4D4D4D', style='italic')

    ax.text(6, 4.6, 'SPT+Rescue: laxity rescue decision per job '
            '(rescue threshold $\\theta = 600$ s)',
            ha='center', fontsize=11, fontweight='bold')
    fig.tight_layout()
    _save(fig, 'rescue_flow.pdf')


# ── Figure 23: RH Slot-Timeline Classification ───────────────────────────────

def fig_rh_timeline():
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    ax.set_xlim(-38, 185); ax.set_ylim(-0.8, 7.6); ax.axis('off')

    t_free = 60   # soonest slot to free for a job not dispatched now

    # Time axis
    ax.annotate('', xy=(160, 0), xytext=(-2, 0),
                arrowprops=dict(arrowstyle='->', color='#1A1A1A', lw=1.5))
    ax.text(160, -0.42, 'time', ha='right', fontsize=9, color='#1A1A1A')
    ax.plot(0, 0, '|', ms=14, mew=2, color='#1A1A1A')
    ax.text(0, -0.42, '$t$ (now)', ha='center', fontsize=9, color='#1A1A1A')
    ax.axvline(t_free, color='#4D4D4D', lw=1.2, ls='--', ymin=0.08, ymax=0.95)
    ax.text(t_free, 7.25, '$t_{\\rm free}$ (next slot frees)',
            ha='center', fontsize=9, color='#4D4D4D')

    def slot_row(y, label, busy_until=None):
        ax.text(-6, y, label, ha='right', va='center', fontsize=9,
                fontweight='bold')
        if busy_until:
            ax.add_patch(mpatches.Rectangle((0, y-0.28), busy_until, 0.56,
                         facecolor='#BFBFBF', edgecolor='#4D4D4D', lw=0.8))
            ax.text(busy_until/2, y, 'busy', ha='center', va='center',
                    fontsize=8, color='#1A1A1A')
        else:
            ax.add_patch(mpatches.Rectangle((0, y-0.28), 155, 0.56,
                         facecolor='#F2F2F2', edgecolor='#808080', lw=0.8))
            ax.text(20, y, 'idle now', ha='center', va='center',
                    fontsize=8, color='#1A1A1A')

    slot_row(6.6, 'Slot 1')                 # idle
    slot_row(5.8, 'Slot 2', busy_until=60)  # frees at t_free
    slot_row(5.0, 'Slot 3', busy_until=100)

    def job_row(y, name, e, d, tier, tier_color):
        ax.text(-6, y + 0.18, name, ha='right', va='center', fontsize=10,
                fontweight='bold')
        ax.text(-6, y - 0.22, f'$e_j={e}$', ha='right', va='center',
                fontsize=9, color='#555')
        # if dispatched now
        ax.add_patch(mpatches.Rectangle((0, y), e, 0.30,
                     facecolor=tier_color, edgecolor='#1A1A1A', lw=0.8))
        ax.text(e + 2, y + 0.15, 'now', ha='left', va='center',
                fontsize=8.5, color='#555')
        # if deferred to t_free
        ax.add_patch(mpatches.Rectangle((t_free, y - 0.40), e, 0.30,
                     facecolor='none', edgecolor='#4D4D4D', lw=1.0,
                     linestyle='--'))
        ax.text(t_free + e + 2, y - 0.25, 'deferred', ha='left', va='center',
                fontsize=8.5, color='#4D4D4D')
        # deadline marker
        ax.plot(d, y - 0.05, 'v', ms=9, color='#1A1A1A', zorder=6)
        ax.text(d, y + 0.44, f'$d_j={d}$', ha='center', fontsize=9,
                color='#1A1A1A')
        # tier label
        ax.text(163, y - 0.05, tier, ha='left', va='center', fontsize=9,
                fontweight='bold', color='#1A1A1A')

    # Urgent: t+e <= d < t_free+e  (40 <= 80 < 100)
    job_row(3.7, 'Job A', 40, 80,  'Urgent',   '#4D4D4D')
    # Normal: t_free+e <= d        (90 <= 150)
    job_row(2.3, 'Job B', 30, 150, 'Normal',   '#A6A6A6')
    # Hopeless: t+e > d            (50 > 30)
    job_row(0.9, 'Job C', 50, 30,  'Hopeless', '#D9D9D9')

    ax.text(128, 5.8,
            'Classification\n'
            'Urgent:  $t{+}e_j \\leq d_j < t_{\\rm free}{+}e_j$\n'
            'Normal:  $t_{\\rm free}{+}e_j \\leq d_j$\n'
            'Hopeless:  $t{+}e_j > d_j$',
            ha='left', va='center', fontsize=10, color='#1A1A1A',
            bbox=dict(boxstyle='round,pad=0.45', facecolor='#F2F2F2',
                      edgecolor='#4D4D4D', lw=0.8))

    ax.set_title('Rolling-Horizon: slot timelines and job classification at one event',
                 fontsize=11, fontweight='bold', pad=6)
    fig.tight_layout()
    _save(fig, 'rh_timeline.pdf')


# ── Figure 24: CADR Critical-Ratio Triage ────────────────────────────────────

def fig_cadr_triage():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.6, 5.0),
                                    gridspec_kw={'height_ratios': [1, 1.25]})

    # Panel (a): CR number line with tier regions
    ax1.set_xlim(0, 6.4); ax1.set_ylim(0, 1.6); ax1.axis('off')
    regions = [(0, 1, '#BFBFBF', 'Doomed\n$\\mathrm{CR}_j \\leq 1$\n(dispatch last)'),
               (1, 3, '#D9D9D9', 'At-risk\n$1 < \\mathrm{CR}_j \\leq \\mathrm{CR}^*$\n(EDF order)'),
               (3, 6.4, '#F2F2F2', 'Safe\n$\\mathrm{CR}_j > \\mathrm{CR}^*$\n(SPT order)')]
    for x0, x1, c, lbl in regions:
        ax1.add_patch(mpatches.Rectangle((x0, 0.35), x1-x0, 0.5,
                      facecolor=c, edgecolor='#4D4D4D', lw=0.8))
        ax1.text((x0+x1)/2, 1.25, lbl, ha='center', va='center', fontsize=9.5)
    ax1.annotate('', xy=(6.4, 0.35), xytext=(0, 0.35),
                 arrowprops=dict(arrowstyle='->', color='#1A1A1A', lw=1.3))
    for xv, lbl in [(1, '1'), (3, '$\\mathrm{CR}^*{=}3$')]:
        ax1.plot(xv, 0.35, '|', ms=14, mew=2, color='#1A1A1A')
        ax1.text(xv, 0.12, lbl, ha='center', fontsize=9, color='#1A1A1A')
    ax1.text(6.25, 0.12, '$\\mathrm{CR}_j = (d_j - t)/e_j$', ha='right',
             fontsize=10, color='#4D4D4D')
    # example jobs
    for cr, name in [(0.6, '$j_1$'), (1.8, '$j_2$'), (2.5, '$j_3$'), (4.6, '$j_4$')]:
        ax1.plot(cr, 0.6, 'o', ms=8, color='#1A1A1A', zorder=5)
        ax1.text(cr, 0.72, name, ha='center', fontsize=8, color='#1A1A1A')
    ax1.set_title('(a) Critical-ratio triage: dispatch order is at-risk, then safe, then doomed',
                  fontsize=9.5)

    # Panel (b): cost-aware node-selection flow
    ax2.set_xlim(0, 12); ax2.set_ylim(0, 4); ax2.axis('off')

    def fbox(x, y, w, h, label, color='#F2F2F2', fs=8.5):
        rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                              boxstyle='round,pad=0.07', lw=1.1,
                              edgecolor='#1A1A1A', facecolor=color)
        ax2.add_patch(rect)
        ax2.text(x, y, label, ha='center', va='center', fontsize=fs)

    def diamond(x, y, w, h, label, fs=8.5):
        pts = [(x, y + h/2), (x + w/2, y), (x, y - h/2), (x - w/2, y)]
        ax2.add_patch(mpatches.Polygon(pts, closed=True, lw=1.1,
                                        edgecolor='#1A1A1A', facecolor='#F2F2F2'))
        ax2.text(x, y, label, ha='center', va='center', fontsize=fs)

    def farrow(x1, y1, x2, y2, label='', dx=0.1, dy=0.0):
        ax2.annotate('', xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle='->', color='#1A1A1A', lw=1.2,
                                    shrinkA=4, shrinkB=4))
        if label:
            ax2.text((x1+x2)/2 + dx, (y1+y2)/2 + dy, label, ha='left',
                     va='center', fontsize=8, color='#4D4D4D', fontweight='bold')

    fbox(1.6, 2.0, 2.6, 1.3, 'Capable idle nodes\nfor job $j$', '#E6E6E6')
    fbox(4.7, 2.0, 2.6, 1.3, 'Prefer nodes not\nlow on stock', '#EBEBEB')
    diamond(7.8, 2.0, 2.1, 1.9, 'any node with\n$t + e_j(m) \\leq d_j$?', fs=8)
    fbox(10.7, 3.1, 2.3, 1.1, 'Cheapest\ndeadline-feasible\nnode', '#BFBFBF', fs=8)
    fbox(10.7, 0.9, 2.3, 1.1, 'Fastest\navailable node\n(fallback)', '#D9D9D9', fs=8)

    farrow(2.9, 2.0, 3.4, 2.0)
    farrow(6.0, 2.0, 6.75, 2.0)
    farrow(7.8, 2.95, 10.7, 3.1 - 0.6, 'yes', dx=-1.7, dy=0.25)
    farrow(7.8, 1.05, 10.7, 0.9 + 0.6, 'no',  dx=-1.7, dy=-0.25)

    ax2.set_title('(b) Cost-aware node selection within a placement', fontsize=9.5)

    fig.suptitle('CADR: critical-ratio triage and cost-aware placement',
                 fontsize=11, fontweight='bold')
    fig.tight_layout()
    _save(fig, 'cadr_triage.pdf')


# ── Figure 26: Objective-Weight Sensitivity ──────────────────────────────────

def fig_weight_sensitivity():
    with open(DATA / 'weight_sensitivity.json') as f:
        raw = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.5), sharey=True)

    def sweep_panel(ax, sweep, title):
        ratios = sorted(sweep.keys(), key=float)
        xs = [float(r) for r in ratios]
        scheds = list(sweep[ratios[0]]['composite'].keys())
        for s in scheds:
            ys = [sweep[r]['composite'][s] for r in ratios]
            ax.plot(xs, ys, 'o-', color=SCHED_COLORS.get(s, '#999'),
                    lw=1.8 if s in ('RH', 'CADR') else 1.2,
                    ms=6 if s in ('RH', 'CADR') else 4, label=s)
        # mark the winner at each ratio
        for r, x in zip(ratios, xs):
            win = sweep[r]['winner']
            ax.plot(x, sweep[r]['composite'][win], '*', ms=13,
                    color=SCHED_COLORS.get(win, '#999'),
                    markeredgecolor='black', markeredgewidth=0.6, zorder=6)
        ax.set_xscale('log')
        ax.set_xticks(xs); ax.set_xticklabels([f'{x:g}' for x in xs])
        ax.set_xlabel('Deadline penalty ratio $\\alpha_2/\\alpha_1$', fontsize=12)
        ax.set_title(title, fontsize=12)
        ax.tick_params(labelsize=11)
        ax.grid(True, alpha=0.3)

    sweep_panel(ax1, raw['sweep_alpha2'],
                '(a) Miss-indicator objective composite')
    sweep_panel(ax2, raw['tardiness_objective'],
                '(b) Tardiness objective composite')
    ax1.set_ylabel('Normalised composite score (lower is better)', fontsize=12)
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=11, ncol=6, loc='lower center',
               bbox_to_anchor=(0.5, -0.05), frameon=False)

    fig.suptitle('Objective-weight sensitivity: composite score vs deadline penalty ratio\n'
                 '(stars mark the winner at each ratio; $n=10$ seeds, hectic day)',
                 fontsize=12)
    fig.tight_layout()
    _save(fig, 'weight_sensitivity.pdf')


# ── Figure 27: Cost Comparison ───────────────────────────────────────────────

def fig_cost_comparison():
    with open(DATA / 'statistical.json') as f:
        p3 = json.load(f)
    with open(CGRSP / 'Outputs' / 'phase4' / 'monthly_results.json') as f:
        p4 = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))

    def cost_panel(ax, names, means, los, his, ylabel, title):
        order = sorted(range(len(names)), key=lambda i: means[i])
        x = np.arange(len(names))
        for xi, idx in enumerate(order):
            c = SCHED_COLORS.get(names[idx], '#999')
            ax.bar(xi, means[idx], color=c, width=0.62,
                   edgecolor='white', lw=0.5)
            ax.errorbar(xi, means[idx],
                        yerr=[[means[idx]-los[idx]], [his[idx]-means[idx]]],
                        fmt='none', ecolor='#333', capsize=3, lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels([names[i] for i in order], rotation=35,
                           ha='right', fontsize=9.5)
        ax.set_ylabel(ylabel); ax.set_title(title, fontsize=10)
        ax.grid(True, axis='y', alpha=0.3)
        lo, hi = min(means), max(means)
        pad = (hi - lo) * 0.6 + 1e-9
        ax.set_ylim(max(0, lo - pad), hi + pad)
        spread = 100 * (hi - lo) / lo
        ax.text(0.03, 0.95, f'spread: {spread:.1f}%', transform=ax.transAxes,
                fontsize=9, va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#FEF9E7',
                          edgecolor='#B7950B', lw=0.7))

    names3 = list(p3['schedulers'].keys())
    cost_panel(ax1, names3,
               [p3['schedulers'][s]['cost_usd']['mean'] for s in names3],
               [p3['schedulers'][s]['cost_usd']['ci95_lo'] for s in names3],
               [p3['schedulers'][s]['cost_usd']['ci95_hi'] for s in names3],
               'Cost per 24-hour run (\\$)',
               '(a) Phase 3 hectic day ($n=30$ seeds, 95% CI)')

    names4 = list(p4['summary'].keys())
    cost_panel(ax2, names4,
               [p4['summary'][s]['total_cost']['mean'] for s in names4],
               [p4['summary'][s]['total_cost']['ci95_lo'] for s in names4],
               [p4['summary'][s]['total_cost']['ci95_hi'] for s in names4],
               'Cost per 30-day month (\\$)',
               '(b) Phase 4 monthly (10 replications, 95% CI)')

    fig.suptitle('Operational cost by scheduler (y-axes zoomed to the observed range)',
                 fontsize=11)
    fig.tight_layout()
    _save(fig, 'cost_comparison.pdf')


# ── Figure 28: Pairwise t-test Effect Sizes ──────────────────────────────────

def fig_ttest_matrix():
    with open(DATA / 'statistical.json') as f:
        raw = json.load(f)

    metric_names = {'avg_wait_min': 'wait', 'deadline_miss_pct': 'miss'}
    # comparison -> {metric: test}
    comps = {}
    for row in raw['t_tests']:
        comps.setdefault(row['comparison'], {})[row['metric']] = row['test']
    labels = list(comps.keys())

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    y = np.arange(len(labels))

    for ax, metric in zip(axes, metric_names):
        for i, lbl in enumerate(labels):
            t = comps[lbl].get(metric)
            if t is None:
                ax.text(0, i, 'not tested', ha='center', va='center',
                        fontsize=8.5, color='#999', style='italic')
                continue
            d, sig = t['cohen_d'], t['significant_p05']
            first = lbl.split(' vs ')[0]
            c = SCHED_COLORS.get(first, '#666')
            ax.barh(i, d, color=c if sig else 'white',
                    edgecolor=c, lw=1.4, height=0.6,
                    hatch='' if sig else '///')
            ax.text(d + (0.12 if d >= 0 else -0.12), i,
                    f'{d:.2f}' + ('' if sig else ' (n.s.)'),
                    ha='left' if d >= 0 else 'right', va='center', fontsize=9)
        ax.axvline(0, color='#2C3E50', lw=1)
        ax.set_title(f"Cohen's $d$: {metric_names[metric]}", fontsize=10)
        ax.grid(True, axis='x', alpha=0.3)
        ax.set_xlabel("Cohen's $d$ (negative favours first-named)")
        lim = max(abs(v['cohen_d']) for c_ in comps.values()
                  for v in c_.values()) * 1.35
        ax.set_xlim(-lim, lim)

    axes[0].set_yticks(y); axes[0].set_yticklabels(labels, fontsize=8.5)
    fig.suptitle('Paired $t$-test effect sizes, Phase 3 ($n=30$ seeds, hectic day). '
                 'Hatched bars are not significant at the raw, uncorrected $p<0.05$ level.',
                 fontsize=10)
    fig.tight_layout()
    _save(fig, 'ttest_effects.pdf')


# ── Figure 29: Per-Seed Distributions ────────────────────────────────────────

def fig_seed_distributions():
    with open(DATA / 'statistical.json') as f:
        raw = json.load(f)
    scheds = list(raw['schedulers'].keys())

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.4, 6.0))

    def dist_panel(ax, field, ylabel, title):
        order = sorted(scheds,
                       key=lambda s: np.mean(raw['schedulers'][s][field]))
        data = [raw['schedulers'][s][field] for s in order]
        bp = ax.boxplot(data, patch_artist=True, widths=0.55,
                        medianprops=dict(color='black', lw=1.2),
                        flierprops=dict(marker='o', ms=3, alpha=0.6))
        for patch, s in zip(bp['boxes'], order):
            patch.set_facecolor(SCHED_COLORS.get(s, '#999'))
            patch.set_alpha(0.75)
        ax.set_xticklabels(order, rotation=40, ha='right', fontsize=9)
        ax.set_ylabel(ylabel); ax.set_title(title, fontsize=10)
        ax.grid(True, axis='y', alpha=0.3)

    dist_panel(ax1, 'raw_wait_min',  'Avg wait (min)',
               '(a) Per-seed average wait')
    dist_panel(ax2, 'raw_miss_pct', 'Miss rate (%)',
               '(b) Per-seed miss rate')

    n = raw['schedulers'][scheds[0]]['wait_min']['n']
    fig.suptitle(f'Phase 3 distributions across {n} matched seeds\n'
                 '(schedulers sorted by mean within each panel)',
                 fontsize=10)
    fig.tight_layout()
    _save(fig, 'seed_distributions.pdf')


# ── Figure 30: CGRSP Problem Overview (Chapter 1) ────────────────────────────

def fig_problem_overview():
    fig, ax = plt.subplots(figsize=(9.6, 4.16))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis('off')

    def box(x, y, w, h, label, sub='', color='#F2F2F2', fs=9.5):
        rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                              boxstyle='round,pad=0.1', lw=1.2,
                              edgecolor='#1A1A1A', facecolor=color)
        ax.add_patch(rect)
        ax.text(x, y + (0.40 if sub else 0), label, ha='center', va='center',
                fontsize=fs * 1.25, fontweight='bold')
        if sub:
            ax.text(x, y - 0.38, sub, ha='center', va='center',
                    fontsize=12, color='#4D4D4D')

    def arrow(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#1A1A1A', lw=1.5,
                                   shrinkA=4, shrinkB=4))

    box(1.5, 3.0, 2.4, 1.9, 'Rendering jobs',
        'stochastic arrivals\ncomplexity $\\kappa$\ndeadline $d_j$', '#E6E6E6')
    box(4.6, 3.0, 2.2, 1.5, 'Queue', 'jobs waiting\nfor a slot', '#F2F2F2')
    box(7.5, 3.0, 2.2, 1.9, 'Scheduler',
        'which job?\nwhich GPU?\nwhen?', '#D9D9D9')
    box(10.6, 4.3, 2.3, 1.5, 'Heterogeneous\nGPU fleet',
        '4 types, different\nspeed and price', '#CCCCCC')
    box(10.6, 1.6, 2.3, 1.5, 'Outcomes',
        'wait $w_j$ · miss $(1-\\delta_j)$\ncost $c_j$', '#BFBFBF')

    arrow(2.7, 3.0, 3.5, 3.0)
    arrow(5.7, 3.0, 6.4, 3.0)
    arrow(8.6, 3.35, 9.45, 4.15)
    arrow(10.6, 3.55, 10.6, 2.35)

    # Challenge annotations under the flow
    for x, txt in [(1.5,  'jobs arrive at\nunknown times'),
                   (7.5,  'availability is stochastic\nand only coarsely visible'),
                   (10.6, 'speed and price vary:\nQoS vs cost tension')]:
        ax.text(x, 0.3, txt, ha='center', va='center', fontsize=12,
                color='#4D4D4D', style='italic')

    ax.set_title('The Cloud GPU Rendering Scheduling Problem (CGRSP) at a glance',
                 fontsize=12, fontweight='bold', pad=10)
    fig.tight_layout()
    _save(fig, 'problem_overview.pdf')


# ── Figure 25: Observed RunPod Availability History ──────────────────────────

def fig_snapshot_decision():
    """Frozen ten-policy decision snapshot at one queue state (illustrative)."""
    sys.path.insert(0, str(CGRSP))
    from cgrsp.config import ResourceType
    from cgrsp.simulator.fleet import FleetManager, NodeState
    from cgrsp.simulator.queue import Job
    from cgrsp.data.runpod_availability import StockStatus
    from cgrsp.data.execution_times import ExecutionTimeModel, GPUType
    from cgrsp.schedulers import (
        FIFOScheduler, SPTScheduler, EDFScheduler, LCFScheduler,
        BalancedScheduler, RandomScheduler, LaxityRescueScheduler,
        AdaptiveScheduler, RollingHorizonScheduler,
        CostAwareDeadlineRiskScheduler,
    )

    T = 600.0
    # id, kappa, deadline, submit, class
    specs = [
        (1, 3, 700,   450, 'tight-feasible'),
        (2, 2, 660,   520, 'tight-infeasible'),
        (3, 1, 29400, 300, 'loose'),
        (4, 2, 29400, 330, 'loose'),
        (5, 3, 29400, 360, 'loose'),
        (6, 1, 29400, 390, 'loose'),
        (7, 3, 620,   500, 'tight-infeasible'),
        (8, 1, 680,   550, 'tight-feasible'),
        (9, 2, 28800, 580, 'loose'),
    ]
    jobs = [Job(jid, k, d, s, 0.0) for jid, k, d, s, _ in specs]

    fleet = FleetManager([
        (ResourceType.RTX_A4000, 1),
        (ResourceType.RTX_A4500, 1),
        (ResourceType.RTX_A5000, 1),
        (ResourceType.RTX_3090, 1),
        (ResourceType.RTX_A4000, 1),
    ])
    stock = {
        ResourceType.RTX_A4000: StockStatus.HIGH,
        ResourceType.RTX_A4500: StockStatus.HIGH,
        ResourceType.RTX_A5000: StockStatus.MEDIUM,
        ResourceType.RTX_3090: StockStatus.LOW,
    }
    fleet.get_stock_status = lambda rt: stock.get(rt, StockStatus.HIGH)
    for nid in (3, 4):
        fleet.nodes[nid].state = NodeState.RUNNING
        fleet.nodes[nid].current_job_id = 100 + nid
        fleet.nodes[nid].state_entry_time = T - 30
        fleet.nodes[nid].job_start_time = T - 30

    scheds = [
        ('FIFO', FIFOScheduler(fleet, seed=42)),
        ('SPT', SPTScheduler(fleet, seed=42)),
        ('EDF', EDFScheduler(fleet, seed=42)),
        ('LCF', LCFScheduler(fleet, seed=42)),
        ('Balanced', BalancedScheduler(fleet, seed=42)),
        ('Random', RandomScheduler(fleet, seed=42)),
        ('SPT+Rescue', LaxityRescueScheduler(fleet, seed=42)),
        ('Adaptive', AdaptiveScheduler(fleet, seed=42)),
        ('RH', RollingHorizonScheduler(fleet, seed=42, arrival_rate=0.1)),
        ('CADR', CostAwareDeadlineRiskScheduler(fleet, seed=42)),
    ]
    assignments = {name: s.schedule(jobs, T) for name, s in scheds}
    idle_ids = [0, 1, 2]
    idle_labels = ['A4000\n(High)', 'A4500\n(High)', 'A5000\n(Medium)']
    class_of = {jid: cls for jid, _, _, _, cls in specs}
    feasible = {jid for jid, cls in class_of.items() if cls == 'tight-feasible'}
    infeasible = {jid for jid, cls in class_of.items() if cls == 'tight-infeasible'}

    rows = []
    for name, _ in scheds:
        by_node = {}
        for jid, nid in assignments[name].items():
            if nid in idle_ids:
                by_node[nid] = jid
        cells = [f'J{by_node[n]}' if n in by_node else '—' for n in idle_ids]
        served = sum(1 for jid in by_node.values() if jid in feasible)
        wasted = sum(1 for jid in by_node.values() if jid in infeasible)
        rows.append([name] + cells + [f'{served}/2', str(wasted)])

    fig = plt.figure(figsize=(7.2, 9.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.5], hspace=0.22)
    axq = fig.add_subplot(gs[0])
    axq.set_xlim(0, 12); axq.set_ylim(0, 5.4); axq.axis('off')
    axq.text(0.1, 5.15, 'Queue state at $t = 600$ s',
             ha='left', va='top', fontsize=11, fontweight='bold')
    axq.text(0.1, 4.78,
             'Nine waiting jobs.  Three idle slots: A4000 (High), A4500 (High), '
             'A5000 (Medium).',
             ha='left', va='top', fontsize=8.6, color='#333333')
    axq.text(0.1, 4.50, 'Two busy slots: RTX 3090 and A4000.',
             ha='left', va='top', fontsize=8.6, color='#333333')

    for i, (jid, kappa, deadline, _, cls) in enumerate(specs):
        row, col = divmod(i, 3)
        x = 0.15 + col * 3.95
        y = 3.12 - row * 1.18
        w, h = 3.70, 1.00
        if cls == 'tight-feasible':
            face, hatch, lw, ls = '#E6E6E6', None, 2.2, '-'
        elif cls == 'tight-infeasible':
            face, hatch, lw, ls = '#D9D9D9', None, 1.2, '--'
        else:
            face, hatch, lw, ls = '#F2F2F2', None, 0.8, '-'
        axq.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.03',
                                     edgecolor='#1A1A1A', facecolor=face,
                                     hatch=hatch, lw=lw, linestyle=ls))
        axq.text(x + 0.12, y + h - 0.27, f'J{jid}   $\\kappa = {kappa}$   '
                 f'$d = {deadline:,}$ s', ha='left', va='center',
                 fontsize=9.0, fontweight='bold')
        if cls == 'tight-feasible':
            desc = 'Tight, feasible'
        elif cls == 'tight-infeasible':
            desc = 'Tight, infeasible'
        else:
            desc = 'Loose deadline'
        axq.text(x + 0.12, y + 0.26, desc, ha='left', va='center',
                 fontsize=8.4, color='#333333', style='italic')

    # queue legend
    lx = 0.15
    for label, face, hatch, lw, ls in [
        ('Tight, feasible', '#E6E6E6', None, 2.2, '-'),
        ('Tight, infeasible', '#D9D9D9', None, 1.2, '--'),
        ('Loose deadline', '#F2F2F2', None, 0.8, '-'),
    ]:
        axq.add_patch(FancyBboxPatch((lx, 0.08), 0.35, 0.25,
                                     boxstyle='round,pad=0.02',
                                     edgecolor='#1A1A1A', facecolor=face,
                                     hatch=hatch, lw=lw, linestyle=ls))
        axq.text(lx + 0.50, 0.20, label, ha='left', va='center', fontsize=8.2)
        lx += 3.2

    axt = fig.add_subplot(gs[1])
    axt.axis('off')
    axt.text(0.5, 0.97, 'Policy decisions at this event', ha='center',
             va='bottom', fontsize=10.5, fontweight='bold',
             transform=axt.transAxes)
    col_labels = ['Policy', 'A4000', 'A4500', 'A5000',
                  'Feasible\ntight served', 'Infeasible\ndispatched']
    table = axt.table(cellText=rows, colLabels=col_labels,
                      loc='center', cellLoc='center',
                      bbox=[0.0, 0.08, 1.0, 0.84],
                      colWidths=[0.19, 0.13, 0.13, 0.13, 0.22, 0.20])
    table.auto_set_font_size(False)
    table.set_fontsize(9.0)
    highlight = {'Adaptive': '#F2F2F2', 'RH': '#E6E6E6', 'CADR': '#E6E6E6'}
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor('#BDBDBD')
        cell.set_linewidth(0.6)
        if r == 0:
            cell.set_facecolor('#D9D9D9')
            cell.get_text().set_fontweight('bold')
            cell.get_text().set_fontsize(8.3)
        else:
            name = rows[r - 1][0]
            if name in highlight:
                cell.set_facecolor(highlight[name])
            if name in ('RH', 'CADR'):
                cell.get_text().set_fontweight('bold')

    axt.text(0.5, 0.00,
             'A4000, A4500 and A5000 are the three idle slots. Feasible counts J1 and J8; '
             'infeasible counts J2 and J7.\nAll ten policies see the same frozen state.',
             ha='center', va='bottom', fontsize=7.8, color='#333333',
             transform=axt.transAxes)

    _save(fig, 'snapshot_decision.pdf')


def fig_availability_history():
    """Stock-status record of the four fleet GPUs across all raw snapshots."""
    import glob
    from collections import Counter

    SNAP_DIR = HERE.parent.parent / 'data' / 'runpod_pricing'
    SUMMARY = HERE.parent.parent / 'data' / 'runpod_pricing_summary' / 'stock_status.csv'
    if not SNAP_DIR.exists() and not SUMMARY.exists():
        # No archive and no compact summary: never overwrite the committed figure
        # with empty axes.
        print(f'  Skipping availability_history: neither {SNAP_DIR} nor {SUMMARY} found')
        return
    FLEET = {
        'NVIDIA GeForce RTX 3090': 'RTX 3090',
        'NVIDIA RTX A5000':        'RTX A5000',
        'NVIDIA RTX A4500':        'RTX A4500',
        'NVIDIA RTX A4000':        'RTX A4000',
    }
    by_day  = {v: defaultdict(Counter) for v in FLEET.values()}
    by_hour = {v: defaultdict(Counter) for v in FLEET.values()}

    n_used = 0
    if SNAP_DIR.exists():
        files = sorted(glob.glob(str(SNAP_DIR / 'snapshot_*.json')))
        for fp in files:
            try:
                with open(fp) as f:
                    snap = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            ts = snap.get('timestamp', '')
            if len(ts) < 13:
                continue
            try:
                # timestamp is tagged UTC by the scraper's own code, but the
                # scraping machine's system clock was actually on GMT+7; shift by
                # +1h to Taiwan time (UTC+8), consistent with the other figures
                dt_tw = datetime.fromisoformat(ts) + timedelta(hours=1)
            except ValueError:
                continue
            day, hour = dt_tw.strftime('%Y-%m-%d'), dt_tw.hour
            n_used += 1
            for g in snap.get('gpus', []):
                name = FLEET.get(g.get('id'))
                if not name:
                    continue
                status = (g.get('lowestPrice') or {}).get('stockStatus') or 'Unavailable'
                by_day[name][day][status] += 1
                by_hour[name][hour][status] += 1
    else:
        seen = set()
        with open(SUMMARY, newline='') as f:
            for row in csv.DictReader(f):
                ts = row.get('timestamp_tw', '')
                if not ts:
                    continue
                try:
                    dt_tw = datetime.fromisoformat(ts)
                except ValueError:
                    continue
                if ts not in seen:
                    seen.add(ts)
                    n_used += 1
                name = row.get('gpu')
                if name not in by_day:
                    continue
                day, hour = dt_tw.strftime('%Y-%m-%d'), dt_tw.hour
                status = row.get('status') or 'Unavailable'
                by_day[name][day][status] += 1
                by_hour[name][hour][status] += 1

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.0, 6.2))
    gpu_colors = {'RTX 3090': '#E74C3C', 'RTX A5000': '#2E86C1',
                  'RTX A4500': '#28B463', 'RTX A4000': '#B7950B'}

    # Panel (a): daily share of High-stock snapshots per GPU
    all_days = sorted({d for g in by_day.values() for d in g})
    xs = np.arange(len(all_days))
    y_all = []
    for name, color in gpu_colors.items():
        ys = []
        for d in all_days:
            c = by_day[name][d]
            tot = sum(c.values())
            ys.append(100 * c.get('High', 0) / tot if tot else np.nan)
        y_all.extend(ys)
        ax1.plot(xs, ys, '-o', color=color, lw=1.3, ms=3, label=name, alpha=0.9)
    tick_idx = xs[::5]
    ax1.set_xticks(tick_idx)
    ax1.set_xticklabels([all_days[i][5:] for i in tick_idx], fontsize=8)
    ax1.set_ylabel('High-stock share of\ndaily snapshots (%)')
    ax1.set_title(f'(a) Daily high-stock availability per fleet GPU '
                  f'({n_used:,} snapshots)', fontsize=10)
    ax1.legend(fontsize=8, ncol=4, loc='upper right')
    # scale y to the data (essentially zero throughout): a 0-100% axis would
    # render the whole panel as dead space; keep enough headroom for the legend
    ymax = max((v for v in y_all if not np.isnan(v)), default=1.0)
    ax1.grid(True, alpha=0.3); ax1.set_ylim(-0.15, max(5.0, ymax * 1.8))

    # Panel (b): stock-status mix by hour of day (all fleet GPUs pooled)
    hours = np.arange(24)
    statuses = ['High', 'Medium', 'Low', 'Unavailable']
    scolors  = {'High': '#82E0AA', 'Medium': '#F8C471',
                'Low': '#EC7063', 'Unavailable': '#BDC3C7'}
    pooled = defaultdict(Counter)
    for name in gpu_colors:
        for h, c in by_hour[name].items():
            pooled[h].update(c)
    bottoms = np.zeros(24)
    for st in statuses:
        vals = []
        for h in hours:
            tot = sum(pooled[h].values())
            vals.append(100 * pooled[h].get(st, 0) / tot if tot else 0)
        ax2.bar(hours, vals, bottom=bottoms, color=scolors[st],
                label=st, width=0.82, edgecolor='white', lw=0.4)
        bottoms += np.array(vals)
    ax2.set_xticks(hours[::2]); ax2.set_xlim(-0.6, 23.6)
    ax2.set_xlabel('Hour of day (Taiwan time, UTC+8)')
    ax2.set_ylabel('Share of snapshots (%)')
    ax2.set_title('(b) Stock-status mix by hour of day, four fleet GPUs pooled',
                  fontsize=10)
    ax2.legend(fontsize=8.5, ncol=4, loc='upper center',
               bbox_to_anchor=(0.5, -0.22), frameon=False)
    ax2.set_ylim(0, 100)

    fig.suptitle('Observed RunPod pricing-API stock-status record (fleet GPU types)',
                 fontsize=11, fontweight='bold')
    fig.tight_layout()
    _save(fig, 'availability_history.pdf')


FIGURES = {
    'architecture':              fig_architecture,
    'phase2_comparison':         fig_phase2_comparison,
    'sensitivity_heatmap':       fig_sensitivity_heatmap,
    'deadline_sensitivity':      fig_deadline_sensitivity,
    'phase4_monthly':            fig_phase4_monthly,
    'tardiness_tradeoff':        fig_tardiness_tradeoff,
    'numerical_example':         fig_numerical_example,
    'data_overview':             fig_data_overview,
    'phase4_timeline':           fig_phase4_timeline,
    'diurnal_model':             fig_diurnal_model,
    'scheduling_decision':       fig_scheduling_decision,
    'availability_history':      fig_availability_history,
    'weight_sensitivity':        fig_weight_sensitivity,
    'cost_comparison':           fig_cost_comparison,
    'ttest_effects':             fig_ttest_matrix,
    'seed_distributions':        fig_seed_distributions,
    'problem_overview':          fig_problem_overview,
    'rescue_flow':               fig_rescue_flow,
    'rh_timeline':               fig_rh_timeline,
    'cadr_triage':               fig_cadr_triage,
    'cmax_sensitivity':          fig_cmax_sensitivity,
    'ablation_waterfall':        fig_ablation_waterfall,
    'phase3_ci':                 fig_phase3_ci,
    'scheduler_lineage':         fig_scheduler_lineage,
    'snapshot_decision':         fig_snapshot_decision,
    'job_lifecycle':             fig_job_lifecycle,
    'gpu_heatmap':               fig_gpu_heatmap,
    'threshold_sensitivity':     fig_threshold_sensitivity,
    'cr_sensitivity':            fig_cr_sensitivity,
    'reservation_sensitivity':   fig_reservation_sensitivity,
}

def main(targets=None):
    """Generate the requested figures and fail if any target cannot be written."""
    targets = list(FIGURES) if targets is None else targets
    failed = False
    print(f'Generating figures: {targets}')
    for name in targets:
        if name not in FIGURES:
            print(f'  Unknown figure "{name}". Available: {list(FIGURES.keys())}')
            failed = True
            continue
        try:
            FIGURES[name]()
        except Exception as e:
            failed = True
            print(f'  ERROR generating {name}: {e}')
            import traceback; traceback.print_exc()
    print('Failed.' if failed else 'Done.')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:] or None))

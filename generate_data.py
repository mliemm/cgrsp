"""
Generate LaTeX table fragments and stats macros from simulation JSON outputs.
Run from repo root:  python3 generate_data.py

Reads data from CGRSP/Outputs/ and writes .tex fragments to thesis/generated/.
"""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent   # repo root
CGRSP = ROOT / 'CGRSP'
OUT = ROOT / 'thesis' / 'generated'
OUT.mkdir(exist_ok=True)

SCHED_ORDER = ['RH', 'CADR', 'SPT+Rescue', 'Adaptive', 'SPT', 'EDF', 'LCF', 'Balanced', 'FIFO', 'Random']


def _fmt_ci(lo, hi, clip_lo=None):
    if clip_lo is not None:
        lo = max(lo, clip_lo)
    return f'[{lo:.2f},~{hi:.2f}]'


def _write(name, content):
    path = OUT / name
    path.write_text(content)
    print(f'  Wrote {path}')


# ── Table 1: Phase 2 benchmark (hectic day, seed 42) ─────────────────────────

def gen_table_phase2():
    with open(CGRSP / 'Outputs/runpod_scheduler/report.json') as f:
        d = json.load(f)

    rows = []
    _w = {s: d['results'][s]['timing_metrics']['avg_waiting_time'] for s in SCHED_ORDER if s in d['results']}
    _m = {s: d['results'][s]['job_counts']['deadline_miss_rate_pct'] for s in SCHED_ORDER if s in d['results']}
    best_wait_p2 = min(_w, key=_w.get)
    best_miss_p2 = min(_m, key=_m.get)
    for s in SCHED_ORDER:
        if s not in d['results']:
            continue
        r = d['results'][s]
        wait = r['timing_metrics']['avg_waiting_time'] / 60
        miss = r['job_counts']['deadline_miss_rate_pct']
        cost = r['cost_metrics']['total_cost']
        util = r['utilization_legacy']['average'] * 100
        completed = r['job_counts']['completed']
        submitted = r['job_counts']['submitted']

        wait_str = f'{wait:.2f}'
        miss_str = f'{miss:.2f}'
        s_label = s

        if s == best_wait_p2:
            wait_str = r'\textbf{' + wait_str + '}'
        if s == best_miss_p2:
            miss_str = r'\textbf{' + miss_str + '}'
        if s in (best_wait_p2, best_miss_p2):
            s_label = r'\textbf{' + s + '}'

        rows.append(
            f'{s_label} & {wait_str} & {miss_str} & {cost:.2f} & {util:.1f} & {completed}/{submitted}\\\\'
        )

    header = [
        r'\begin{tabular}{lrrrrr}',
        r'\toprule',
        r'\textbf{Scheduler} & \textbf{Avg Wait (min)} & \textbf{Miss \%} & \textbf{Cost (\$)} & \textbf{Util (\%)} & \textbf{Completed} \\',
        r'\midrule',
    ]
    footer = [r'\bottomrule', r'\end{tabular}']
    _write('table_phase2.tex', '\n'.join(header + rows + footer) + '\n')


# ── Table 2: Phase 3 statistical (30 seeds, hectic day) ──────────────────────

def gen_table_phase3_stats():
    with open(CGRSP / 'Outputs/phase3/statistical.json') as f:
        d = json.load(f)

    # Find best miss scheduler
    all_miss = {s: d['schedulers'][s]['miss_pct']['mean'] for s in SCHED_ORDER if s in d['schedulers']}
    best_miss_sched = min(all_miss, key=all_miss.get)
    all_wait = {s: d['schedulers'][s]['wait_min']['mean'] for s in SCHED_ORDER if s in d['schedulers']}
    best_wait_sched = min(all_wait, key=all_wait.get)  # data-driven, same rule as the miss column

    rows = []
    for s in SCHED_ORDER:
        if s not in d['schedulers']:
            continue
        r = d['schedulers'][s]
        w = r['wait_min']
        m = r['miss_pct']
        c = r['cost_usd']

        wait_mean = f'{w["mean"]:.2f}'
        miss_mean = f'{m["mean"]:.2f}'
        s_label = s

        if s == best_wait_sched:
            wait_mean = r'\textbf{' + wait_mean + '}'
        if s == best_miss_sched:
            miss_mean = r'\textbf{' + miss_mean + '}'
        if s in (best_wait_sched, best_miss_sched):
            s_label = r'\textbf{' + s + '}'

        ci_w = _fmt_ci(w["ci95_lo"], w["ci95_hi"])
        ci_m = _fmt_ci(m["ci95_lo"], m["ci95_hi"], clip_lo=0.0)
        tard_mean = r['avg_tardiness_min']['mean']
        rows.append(
            f'{s_label} & {wait_mean} & {ci_w} & {miss_mean} & {ci_m} & {tard_mean:.2f} & {c["mean"]:.2f}\\\\'
        )

    header = [
        r'\begin{tabular}{lrrrrrr}',
        r'\toprule',
        r'\textbf{Scheduler} & \textbf{Avg Wait (min)} & \textbf{95\% CI} & \textbf{Miss \%} & \textbf{95\% CI} & \textbf{Avg Tardiness (min)} & \textbf{Cost (\$)} \\',
        r'\midrule',
    ]
    footer = [r'\bottomrule', r'\end{tabular}']
    _write('table_phase3_stats.tex', '\n'.join(header + rows + footer) + '\n')


# ── Table: Ablation (FIFO -> SPT -> SPT+Rescue -> Adaptive) ──────────────────

def gen_table_ablation():
    with open(CGRSP / 'Outputs/phase3/statistical.json') as f:
        d = json.load(f)

    ablation_order = ['FIFO', 'SPT', 'EDF', 'SPT+Rescue', 'RH', 'CADR', 'Adaptive']
    tt = {(e['comparison'], e['metric']): e['test'] for e in d['t_tests']}
    extra = d['ablation_extra']['CADR-order-only']

    rows = []
    for s in ablation_order:
        r = d['schedulers'][s]
        w = r['wait_min']['mean']
        m = r['miss_pct']['mean']
        tard = r['avg_tardiness_min']['mean']
        s_label = s

        if s == 'RH':
            s_label = r'\textbf{' + s + '} (proposed)'
        elif s == 'CADR':
            s_label = r'\textbf{' + s + '} (proposed)'
        elif s == 'Adaptive':
            s_label = s + ' (EDF-SPT hybrid)'

        # Cohen's d vs FIFO for all except FIFO itself
        if s == 'FIFO':
            d_wait, d_miss = '--', '--'
        elif s == 'SPT':
            d_wait = f"{_exact_t_d(d['schedulers'], 'SPT vs FIFO', 'avg_wait_min')[1]:.3f}"
            d_miss  = f"{_exact_t_d(d['schedulers'], 'SPT vs FIFO', 'deadline_miss_pct')[1]:.3f}"
        elif s == 'EDF':
            d_wait = f"{_exact_t_d(d['schedulers'], 'EDF vs FIFO', 'avg_wait_min')[1]:.3f}"
            d_miss  = f"{_exact_t_d(d['schedulers'], 'EDF vs FIFO', 'deadline_miss_pct')[1]:.3f}"
        elif s == 'SPT+Rescue':
            # Rescue vs SPT: compute inline from raw data
            spt_wait = d['schedulers']['SPT']['wait_min']['mean']
            spt_miss = d['schedulers']['SPT']['miss_pct']['mean']
            rescue_wait = d['schedulers']['SPT+Rescue']['wait_min']['mean']
            rescue_miss = d['schedulers']['SPT+Rescue']['miss_pct']['mean']
            pool_w = ((d['schedulers']['SPT']['wait_min']['std']**2 +
                       d['schedulers']['SPT+Rescue']['wait_min']['std']**2) / 2) ** 0.5
            pool_m = ((d['schedulers']['SPT']['miss_pct']['std']**2 +
                       d['schedulers']['SPT+Rescue']['miss_pct']['std']**2) / 2) ** 0.5
            d_wait = f'{(rescue_wait - spt_wait) / pool_w:.3f}' if pool_w > 0 else '0.000'
            d_miss  = f'{(rescue_miss - spt_miss) / pool_m:.3f}' if pool_m > 0 else '0.000'
        elif s == 'RH':
            d_wait = f"{_exact_t_d(d['schedulers'], 'RH vs FIFO', 'avg_wait_min')[1]:.3f}"
            d_miss  = f"{_exact_t_d(d['schedulers'], 'RH vs FIFO', 'deadline_miss_pct')[1]:.3f}"
        elif s == 'CADR':
            d_wait = f"{_exact_t_d(d['schedulers'], 'CADR vs FIFO', 'avg_wait_min')[1]:.3f}"
            d_miss  = f"{_exact_t_d(d['schedulers'], 'CADR vs FIFO', 'deadline_miss_pct')[1]:.3f}"
        elif s == 'Adaptive':
            d_wait = f"{_exact_t_d(d['schedulers'], 'Adaptive vs FIFO', 'avg_wait_min')[1]:.3f}"
            d_miss  = f"{_exact_t_d(d['schedulers'], 'Adaptive vs FIFO', 'deadline_miss_pct')[1]:.3f}"

        rows.append(f'{s_label} & {w:.2f} & {m:.2f} & {tard:.2f} & {d_wait} & {d_miss} \\\\')

    # CADR-order-only ablation row
    rows.append(r'\midrule')
    oo_w = extra['wait_min']['mean']
    oo_m = extra['miss_pct']['mean']
    oo_tard = extra['avg_tardiness_min']['mean']
    rows.append(
        f'CADR-order-only & {oo_w:.2f} & {oo_m:.2f} & {oo_tard:.2f} & -- & -- \\\\'
    )

    header = [
        r'\begin{tabular}{lrrrrr}',
        r'\toprule',
        r'\textbf{Scheduler} & \textbf{Avg Wait (min)} & \textbf{Miss \%} & \textbf{Avg Tardiness (min)} & \textbf{Cohen $d$ (wait)} & \textbf{Cohen $d$ (miss)} \\',
        r'\midrule',
    ]
    footer = [r'\bottomrule', r'\end{tabular}']
    _write('table_ablation.tex', '\n'.join(header + rows + footer) + '\n')


# ── Table: C_max sensitivity ──────────────────────────────────────────────────

def gen_table_cmax_sensitivity():
    with open(CGRSP / 'Outputs/phase3/cmax_sensitivity.json') as f:
        d = json.load(f)

    cmax_values = d['metadata']['cmax_values']
    scheds = ['FIFO', 'SPT', 'EDF', 'SPT+Rescue', 'RH', 'CADR', 'Adaptive']

    heads = []
    for cmax in cmax_values:
        h = f'$C_{{\\max}}={cmax}$'
        if cmax == 5:
            h = r'\shortstack{' + h + r' \\ \textbf{(baseline)}}'
        heads.append(h)
    rows = []
    for s in scheds:
        vals = []
        for cmax in cmax_values:
            ss = d['results'][f'cmax{cmax}']['schedulers']
            vals.append(f"{ss[s]['avg_wait_min']:.1f}\\,/\\,{ss[s]['avg_miss_pct']:.1f}")
        rows.append(s + ' & ' + ' & '.join(vals) + r' \\')

    ncol = len(cmax_values)
    header = [
        r'\begin{tabular}{l' + 'c' * ncol + '}',
        r'\toprule',
        r'& \multicolumn{' + str(ncol) + r'}{c}{\textbf{Avg Wait (min) / Miss \%}} \\',
        r'\cmidrule(lr){2-' + str(ncol + 1) + '}',
        r'\textbf{Scheduler} & ' + ' & '.join(heads) + r' \\',
        r'\midrule',
    ]
    footer = [r'\bottomrule', r'\end{tabular}']
    _write('table_cmax_sensitivity.tex', '\n'.join(header + rows + footer) + '\n')


# ── Table: Threshold sensitivity ─────────────────────────────────────────────

def gen_table_threshold_sensitivity():
    with open(CGRSP / 'Outputs/phase3/threshold_sensitivity.json') as f:
        d = json.load(f)

    theta_values = [60, 180, 300, 600, 1200]
    rows = []
    # Find best miss for bolding
    best_miss = min(d['results'][f'theta{t}']['avg_miss_pct'] for t in theta_values)

    for theta in theta_values:
        key = f'theta{theta}'
        r = d['results'][key]
        wait_str = f'{r["avg_wait_min"]:.2f}'
        miss_str = f'{r["avg_miss_pct"]:.2f}'
        theta_label = f'$\\theta={theta}$\\,s'
        if theta == 600:
            theta_label = r'\textbf{' + theta_label + r'} (default)'
        if r['avg_miss_pct'] == best_miss:
            miss_str = r'\textbf{' + miss_str + '}'
        rows.append(f'{theta_label} & {wait_str} & {miss_str} \\\\')

    header = [
        r'\begin{tabular}{lrr}',
        r'\toprule',
        r'\textbf{Threshold $\theta$} & \textbf{Avg Wait (min)} & \textbf{Miss \%} \\',
        r'\midrule',
    ]
    footer = [r'\bottomrule', r'\end{tabular}']
    _write('table_threshold_sensitivity.tex', '\n'.join(header + rows + footer) + '\n')


def gen_table_weight_sensitivity():
    with open(CGRSP / 'Outputs/phase3/weight_sensitivity.json') as f:
        d = json.load(f)

    scheds = ['FIFO', 'SPT', 'EDF', 'SPT+Rescue', 'RH', 'CADR']

    def row(label, comp, winner):
        cells = []
        for s in scheds:
            v = f'{comp[s]:.2f}'
            if s == winner:
                v = r'\textbf{' + v + '}'
            cells.append(v)
        return f'{label} & ' + ' & '.join(cells) + r' \\'

    rows = []
    # Panel A: multi-objective weight w_cost sweep (QoS ratio fixed at baseline)
    for wc in ['0.1', '0.3', '0.5', '0.7', '0.9']:
        r = d['sweep_wcost'][wc]
        label = f'$w_{{\\text{{cost}}}}={wc}$'
        if wc == '0.5':
            label = label + r' \textbf{(baseline)}'
        rows.append(row(label, r['composite'], r['winner']))
    rows.append(r'\midrule')
    # Panel B: deadline-penalty ratio alpha2/alpha1 sweep (w_qos = w_cost = 0.5)
    for a2 in ['1', '5', '10', '20', '50']:
        r = d['sweep_alpha2'][a2]
        label = f'$\\alpha_2/\\alpha_1={a2}$'
        if a2 == '10':
            label = label + r' \textbf{(baseline)}'
        rows.append(row(label, r['composite'], r['winner']))

    header = [
        r'\begin{tabular}{lrrrrrr}',
        r'\toprule',
        r'& \multicolumn{6}{c}{\textbf{Composite objective (normalised, lower is better)}} \\',
        r'\cmidrule(lr){2-7}',
        r'\textbf{Weight setting} & \textbf{FIFO} & \textbf{SPT} & \textbf{EDF} & \textbf{SPT+Rescue} & \textbf{RH} & \textbf{CADR} \\',
        r'\midrule',
    ]
    footer = [r'\bottomrule', r'\end{tabular}']
    _write('table_weight_sensitivity.tex', '\n'.join(header + rows + footer) + '\n')


def gen_table_reservation_sensitivity():
    with open(CGRSP / 'Outputs/phase3/reservation_sensitivity.json') as f:
        d = json.load(f)['results']
    rho_values = [0, 1, 2, 3]
    rows = []
    for day, label in (('surge', 'Surge ($\\rho<1$)'), ('hectic', 'Hectic ($\\rho>1$)')):
        grp = d[day]
        best_miss = min(grp[f'r{r}']['avg_miss_pct'] for r in rho_values)
        for i, r in enumerate(rho_values):
            e = grp[f'r{r}']
            miss = f'{e["avg_miss_pct"]:.2f}'
            if e['avg_miss_pct'] == best_miss:
                miss = r'\textbf{' + miss + '}'
            day_cell = (r'\multirow{4}{*}{' + label + '}') if i == 0 else ''
            rho_lbl = f'{r}' + (' (none)' if r == 0 else '')
            rows.append(
                f'{day_cell} & {rho_lbl} & {e["avg_wait_min"]:.1f} & {miss} & '
                f'{e["tight_miss_pct"]:.1f} & {e["loose_miss_pct"]:.1f} \\\\')
        rows.append(r'\midrule' if day == 'surge' else '')
    rows = [r for r in rows if r != '']
    header = [r'\begin{tabular}{llrrrr}', r'\toprule',
              (r'\textbf{Day} & \textbf{$\rho_{\mathrm{res}}$} & \textbf{Wait (min)} & '
               r'\textbf{Miss \%} & \textbf{Tight \%} & \textbf{Loose \%} \\'),
              r'\midrule']
    footer = [r'\bottomrule', r'\end{tabular}']
    _write('table_reservation_sensitivity.tex', '\n'.join(header + rows + footer) + '\n')


# ── Table: CR* sensitivity ────────────────────────────────────────────────────

def gen_table_cr_sensitivity():
    with open(CGRSP / 'Outputs/phase3/cr_sensitivity.json') as f:
        d = json.load(f)

    cr_values = d['metadata']['cr_star_values']
    results = d['results']
    best_miss = min(results[str(float(cr))]['avg_miss_pct'] for cr in cr_values)

    rows = []
    for cr in cr_values:
        key = str(float(cr))
        r = results[key]
        wait_str = f'{r["avg_wait_min"]:.2f}'
        miss_str = f'{r["avg_miss_pct"]:.2f}'
        cr_label = f'$CR^{{*}}={cr}$'
        if r['avg_miss_pct'] == best_miss:
            miss_str = r'\textbf{' + miss_str + '}'
        rows.append(f'{cr_label} & {wait_str} & {miss_str} \\\\')

    header = [
        r'\begin{tabular}{lrr}',
        r'\toprule',
        r'\textbf{$CR^{*}$ threshold} & \textbf{Avg Wait (min)} & \textbf{Miss \%} \\',
        r'\midrule',
    ]
    footer = [r'\bottomrule', r'\end{tabular}']
    _write('table_cr_sensitivity.tex', '\n'.join(header + rows + footer) + '\n')


# ── Table: Tardiness-objective weight sweep ───────────────────────────────────

def gen_table_tardiness_objective():
    with open(CGRSP / 'Outputs/phase3/weight_sensitivity.json') as f:
        d = json.load(f)

    scheds = ['FIFO', 'SPT', 'EDF', 'SPT+Rescue', 'RH', 'CADR']
    panel = d['tardiness_objective']

    def row(label, comp, winner):
        cells = []
        for s in scheds:
            v = f'{comp[s]:.2f}'
            if s == winner:
                v = r'\textbf{' + v + '}'
            cells.append(v)
        return f'{label} & ' + ' & '.join(cells) + r' \\'

    rows = []
    for a2 in ['1', '5', '10', '20', '50']:
        r = panel[a2]
        label = f'$\\alpha_2/\\alpha_1={a2}$'
        rows.append(row(label, r['composite'], r['winner']))

    header = [
        r'\begin{tabular}{lrrrrrr}',
        r'\toprule',
        r'& \multicolumn{6}{c}{\textbf{Composite with tardiness term (normalised, lower is better)}} \\',
        r'\cmidrule(lr){2-7}',
        r'\textbf{$\alpha_2/\alpha_1$} & \textbf{FIFO} & \textbf{SPT} & \textbf{EDF} & \textbf{SPT+Rescue} & \textbf{RH} & \textbf{CADR} \\',
        r'\midrule',
    ]
    footer = [r'\bottomrule', r'\end{tabular}']
    _write('table_tardiness_objective.tex', '\n'.join(header + rows + footer) + '\n')


# ── Table 3: Phase 3 day-type sensitivity ────────────────────────────────────

def gen_table_sensitivity():
    with open(CGRSP / 'Outputs/phase3/sensitivity.json') as f:
        d = json.load(f)

    day_types = [('quiet', 6), ('normal', 100), ('hectic', 950), ('surge', 730)]
    col_scheds = ['SPT', 'LCF', 'Balanced', 'FIFO', 'EDF', 'Random', 'SPT+Rescue', 'RH', 'CADR', 'Adaptive']
    cells = {}
    for dt, n_jobs in day_types:
        ss = d['sensitivity'][f'fleet5_type{dt}']['schedulers']
        # In the discriminating regimes (hectic, surge) bold the lowest-miss
        # scheduler, chosen data-driven so the highlight tracks the actual winner.
        best_miss_s = (min(col_scheds, key=lambda s: ss[s]['avg_miss_pct'])
                       if dt in ('hectic', 'surge') else None)
        for s in col_scheds:
            v_str = f"{ss[s]['avg_wait_min']:.1f} / {ss[s]['avg_miss_pct']:.1f}"
            cells[(s, dt)] = r'\textbf{' + v_str + '}' if s == best_miss_s else v_str
    rows = [s + ' & ' + ' & '.join(cells[(s, dt)] for dt, _ in day_types) + ' \\\\'
            for s in col_scheds]

    header = [
        r'\begingroup\setlength{\tabcolsep}{9pt}',
        r'\begin{tabular}{lcccc}',
        r'\toprule',
        r'& \multicolumn{4}{c}{\textbf{Avg Wait (min) / Miss \%}} \\',
        r'\cmidrule(lr){2-5}',
        r'\textbf{Scheduler} & \textbf{Quiet} & \textbf{Normal} & \textbf{Hectic} & \textbf{Surge} \\',
        r'\midrule',
    ]
    footer = [r'\bottomrule', r'\end{tabular}', r'\endgroup']
    _write('table_sensitivity.tex', '\n'.join(header + rows + footer) + '\n')


# ── Table 4: Deadline tightness (surge day, C_max=5) ─────────────────────────

def gen_table_deadline():
    with open(CGRSP / 'Outputs/phase3/deadline.json') as f:
        d = json.load(f)

    tight_pcts = [10, 30, 50, 70, 90]
    scheds = ['FIFO', 'SPT', 'EDF', 'LCF', 'Balanced', 'Random', 'SPT+Rescue']
    fleet_key = 'fleet5'

    rows = []
    for tp in tight_pcts:
        frac_key = f'tight{tp}pct'
        r = d['results'][fleet_key][frac_key]['schedulers']
        # Same raw-mean rule as the DlXxx macros, so table and prose agree.
        miss_vals = {
            s: (sum(r[s]['raw_miss']) / len(r[s]['raw_miss'])
                if r[s].get('raw_miss')
                else r[s].get('avg_miss_pct', r[s].get('deadline_miss_pct', 0)))
            for s in scheds
        }
        best = min(miss_vals.values())
        vals = []
        for s in scheds:
            v = miss_vals[s]
            v_str = f'{v:.2f}'
            # Bold the lowest-miss point estimate in each row (data-driven, not
            # hardcoded): EDF at low tight fractions, SPT at high ones. The
            # 50% row is a statistical tie, so its bold marks only the lower
            # point estimate.
            if v == best:
                v_str = r'\textbf{' + v_str + '}'
            vals.append(v_str)
        rows.append(f'{tp}\\% & ' + ' & '.join(vals) + r'\\')

    header = [
        r'\begin{tabular}{cccccccc}',
        r'\toprule',
        r'\textbf{Tight \%} & \textbf{FIFO} & \textbf{SPT} & \textbf{EDF} & \textbf{LCF} & \textbf{Balanced} & \textbf{Random} & \textbf{SPT+Rescue} \\',
        r'\midrule',
    ]
    footer = [r'\bottomrule', r'\end{tabular}']
    _write('table_deadline.tex', '\n'.join(header + rows + footer) + '\n')


# ── Table 5: Phase 4 monthly summary ─────────────────────────────────────────

def gen_table_phase4():
    with open(CGRSP / 'Outputs/phase4/monthly_results.json') as f:
        d = json.load(f)

    all_wait = {s: d['summary'][s]['avg_wait_min']['mean'] for s in SCHED_ORDER}
    all_miss = {s: d['summary'][s]['deadline_miss_pct']['mean'] for s in SCHED_ORDER}
    best_wait_sched = min(all_wait, key=all_wait.get)
    best_miss_sched = min(all_miss, key=all_miss.get)

    rows = []
    for s in SCHED_ORDER:
        r = d['summary'][s]
        wait = r['avg_wait_min']['mean']
        miss = r['deadline_miss_pct']['mean']
        cost = r['total_cost']['mean']
        tard = r['avg_tardiness_min']['mean']

        wait_str = f'{wait:.2f}'
        miss_str = f'{miss:.2f}'
        s_label = s

        if s == best_wait_sched:
            wait_str = r'\textbf{' + wait_str + '}'
        if s == best_miss_sched:
            miss_str = r'\textbf{' + miss_str + '}'
        if s in (best_wait_sched, best_miss_sched):
            s_label = r'\textbf{' + s + '}'

        rows.append(
            f'{s_label} & {wait_str} & [{r["avg_wait_min"]["ci95_lo"]:.2f},~{r["avg_wait_min"]["ci95_hi"]:.2f}] & {miss_str} & {tard:.2f} & {cost:.2f} \\\\'
        )

    header = [
        r'\begin{tabular}{lrrrrr}',
        r'\toprule',
        r'\textbf{Scheduler} & \textbf{Avg Wait (min)} & \textbf{95\% CI} & \textbf{Miss\%} & \textbf{Avg Tardiness (min)} & \textbf{Cost (\$/mo)} \\',
        r'\midrule',
    ]
    footer = [r'\bottomrule', r'\end{tabular}']
    _write('table_phase4.tex', '\n'.join(header + rows + footer) + '\n')


# ── Macro file: \newcommand definitions for key statistics ───────────────────

def gen_stats_macros():
    with open(CGRSP / 'Outputs/phase3/statistical.json') as f:
        p3 = json.load(f)
    with open(CGRSP / 'Outputs/phase4/monthly_results.json') as f:
        p4 = json.load(f)
    with open(CGRSP / 'Outputs/runpod_scheduler/report.json') as f:
        p2_raw = json.load(f)
    with open(CGRSP / 'Outputs/phase3/sensitivity.json') as f:
        sens = json.load(f)
    with open(CGRSP / 'Outputs/phase3/deadline.json') as f:
        dl = json.load(f)
    with open(CGRSP / 'Outputs/phase3/threshold_sensitivity.json') as f:
        thr = json.load(f)['results']
    with open(CGRSP / 'Outputs/phase3/cr_sensitivity.json') as f:
        cr_sens = json.load(f)['results']
    with open(CGRSP / 'Outputs/phase3/reservation_sensitivity.json') as f:
        res_sens = json.load(f)['results']
    with open(CGRSP / 'Outputs/phase3/cmax_sensitivity.json') as f:
        cmax = json.load(f)['results']

    s3 = p3['schedulers']
    s4 = p4['summary']
    with open(CGRSP / 'Outputs/phase3/prior_robustness.json') as f:
        prior_all = json.load(f)
    for _metric in ('miss_pct_mean', 'wait_min_mean'):
        assert (prior_all['arm_c_arrival_prior']['hectic']['x0.5'][_metric]
                == prior_all['arm_c_arrival_prior']['hectic']['x0.6'][_metric])
    p2 = p2_raw['results']
    s_hec = sens['sensitivity']['fleet5_typehectic']['schedulers']
    s_srg = sens['sensitivity']['fleet5_typesurge']['schedulers']
    dl5 = dl['results']['fleet5']
    rs_srg = res_sens['surge']
    rs_hec = res_sens['hectic']

    def _d(name, val):
        return f'\\newcommand{{\\{name}}}{{{val}}}'

    def _fmt(v, dp=2):
        return f'{v:.{dp}f}'

    tt = {}
    for entry in p3['t_tests']:
        key = (entry['comparison'], entry['metric'])
        tt[key] = entry['test']

    _raw = {'avg_wait_min': 'raw_wait_min', 'deadline_miss_pct': 'raw_miss_pct'}

    def _md(stem, a, b, metric):
        md, lo, hi = _paired_ci(s3[a][_raw[metric]], s3[b][_raw[metric]])
        p = tt[(f'{a} vs {b}', metric)]['p']
        assert (lo < 0 < hi) == (p >= 0.05), (stem, lo, hi, p)
        return [_d(stem + 'MD', _fmt(md)), _d(stem + 'MDLo', _fmt(lo)), _d(stem + 'MDHi', _fmt(hi))]

    def _d_val(comp, metric):
        return _fmt(_exact_t_d(s3, comp, metric)[1], dp=3)

    def _t_val(comp, metric):
        return _fmt(_exact_t_d(s3, comp, metric)[0], dp=3)

    def _p_val(comp, metric):
        return _fmt(tt[(comp, metric)]['p'], dp=3)

    # Holm-Bonferroni step-down correction across the full pairwise t-test family.
    # Reports how many of the comparisons significant at the uncorrected 0.05
    # level remain significant after controlling the family-wise error rate.
    _holm_ps = sorted(e['test']['p'] for e in p3['t_tests'])
    _holm_m = len(_holm_ps)
    _holm_uncorr = sum(1 for _p in _holm_ps if _p < 0.05)  # significant before correction
    _holm_nsig = 0
    _holm_maxsig = 0.0
    _holm_ok = True
    _holm_cut = 0.0  # largest p that survives the step-down
    for _rank, _p in enumerate(_holm_ps):
        if _holm_ok and _p <= 0.05 / (_holm_m - _rank):
            _holm_nsig += 1
            _holm_maxsig = _p
            _holm_cut = _p
        else:
            _holm_ok = False
    # Uncorrected-significant comparisons the correction removes, and those of
    # them that Table tab:ttests prints (daggered); the rest are unprinted.
    _holm_removed = [(e['comparison'], e['metric']) for e in p3['t_tests']
                     if e['test']['p'] < 0.05 and e['test']['p'] > _holm_cut]
    _TABLE_ROWS = {
        ('SPT vs FIFO', 'avg_wait_min'), ('SPT vs FIFO', 'deadline_miss_pct'),
        ('SPT+Rescue vs FIFO', 'avg_wait_min'), ('SPT+Rescue vs FIFO', 'deadline_miss_pct'),
        ('EDF vs FIFO', 'avg_wait_min'), ('EDF vs FIFO', 'deadline_miss_pct'),
        ('Random vs FIFO', 'deadline_miss_pct'), ('SPT vs EDF', 'deadline_miss_pct'),
        ('RH vs FIFO', 'avg_wait_min'), ('RH vs FIFO', 'deadline_miss_pct'),
        ('RH vs SPT+Rescue', 'avg_wait_min'), ('RH vs SPT+Rescue', 'deadline_miss_pct'),
        ('RH vs CADR', 'deadline_miss_pct'),
    }
    _holm_printed = set(_holm_removed) & _TABLE_ROWS
    print('  Holm-removed comparisons:', sorted(_holm_removed), '| printed with dagger:', sorted(_holm_printed))
    assert len(_holm_removed) == _holm_uncorr - _holm_nsig

    # Same Holm step-down applied to the Wilcoxon p-values of the same family.
    _wil = sorted((e['test']['wilcoxon_p'], e['comparison'], e['metric']) for e in p3['t_tests'])
    _wil_nsig = 0
    for _rank, (_p, _c, _m) in enumerate(_wil):
        if _p <= 0.05 / (_holm_m - _rank):
            _wil_nsig += 1
        else:
            break
    _wil_spt_edf = next(e['test']['wilcoxon_p'] for e in p3['t_tests']
                        if e['comparison'] == 'SPT vs EDF' and e['metric'] == 'deadline_miss_pct')

    def _p2w(sched):
        return _fmt(p2[sched]['timing_metrics']['avg_waiting_time'] / 60)

    def _p2m(sched):
        return _fmt(p2[sched]['job_counts']['deadline_miss_rate_pct'])

    def _dlm(frac_key, sched):
        r = dl5[frac_key]['schedulers'][sched]
        raw = r.get('raw_miss')
        if raw:
            # Average the per-seed values directly: the stored 3 dp mean can land
            # on a .xx5 boundary and round differently from the true mean.
            return _fmt(sum(raw) / len(raw))
        return _fmt(r.get('avg_miss_pct', r.get('deadline_miss_pct', 0)))

    def _dlsr(tight_pct):
        return dl5[f'tight{tight_pct}pct']['spt_vs_edf_miss']

    def _dlt(tight_pct):
        return _fmt(_dlsr(tight_pct)['t'], dp=3)

    def _dlp(tight_pct):
        # Math fragment ("p=0.69" or "p<10^{-4}"), same convention as the beta
        # and prior-robustness p-value macros: an exact 0.000 would misread as
        # a p of exactly zero. The stored p is rounded to six decimals and the
        # tight ends of the sweep round to 0.000000, so recompute from the
        # per-seed arrays the sweep also stores.
        e = _dlsr(tight_pct)
        _, _, p = _paired_stats(e['raw_miss_spt'], e['raw_miss_edf'])
        return _fmt_p(p)

    def _thrm(theta):
        return _fmt(thr[f'theta{theta}']['avg_miss_pct'])

    def _thrw(theta):
        return _fmt(thr[f'theta{theta}']['avg_wait_min'])

    lines = [
        '% AUTO-GENERATED by generate_data.py -- DO NOT EDIT',
        '% Re-run: python3 generate_data.py',
        '% Phase 3 (30-seed hectic day, C_max=5)',
        _d('SPTwait',       _fmt(s3['SPT']['wait_min']['mean'])),
        _d('SPTmiss',       _fmt(s3['SPT']['miss_pct']['mean'])),
        _d('EDFmiss',       _fmt(s3['EDF']['miss_pct']['mean'])),
        _d('RescueWait',    _fmt(s3['SPT+Rescue']['wait_min']['mean'])),
        _d('RescueMiss',    _fmt(s3['SPT+Rescue']['miss_pct']['mean'])),
        _d('FIFOwait',      _fmt(s3['FIFO']['wait_min']['mean'])),
        _d('FIFOmiss',      _fmt(s3['FIFO']['miss_pct']['mean'])),
        _d('LCFmiss',       _fmt(s3['LCF']['miss_pct']['mean'])),
        _d('Balancedmiss',  _fmt(s3['Balanced']['miss_pct']['mean'])),
        '% Phase 3 t-test effect sizes (Cohen d)',
        _d('SPTFIFOwaitD',  _d_val('SPT vs FIFO', 'avg_wait_min')),
        _d('SPTFIFOmissD',  _d_val('SPT vs FIFO', 'deadline_miss_pct')),
        _d('EDFFIFOwaitD',  _d_val('EDF vs FIFO', 'avg_wait_min')),
        _d('EDFFIFOmissD',  _d_val('EDF vs FIFO', 'deadline_miss_pct')),
        _d('RescueFIFOwaitD', _d_val('SPT+Rescue vs FIFO', 'avg_wait_min')),
        _d('RescueFIFOmissD', _d_val('SPT+Rescue vs FIFO', 'deadline_miss_pct')),
        '% RH Phase 3 (30-seed hectic day)',
        _d('RHWait',        _fmt(s3['RH']['wait_min']['mean'])),
        _d('RHMiss',        _fmt(s3['RH']['miss_pct']['mean'])),
        _d('RHFIFOwaitD',   _d_val('RH vs FIFO', 'avg_wait_min')),
        _d('RHFIFOmissD',   _d_val('RH vs FIFO', 'deadline_miss_pct')),
        _d('RHRescueWaitD', _d_val('RH vs SPT+Rescue', 'avg_wait_min')),
        _d('RHRescueMissD', _d_val('RH vs SPT+Rescue', 'deadline_miss_pct')),
        _d('RHFIFOwaitT',   _t_val('RH vs FIFO', 'avg_wait_min')),
        _d('RHFIFOmissT',   _t_val('RH vs FIFO', 'deadline_miss_pct')),
        _d('RHRescueMissT', _t_val('RH vs SPT+Rescue', 'deadline_miss_pct')),
        _d('RHRescueMissP', _p_val('RH vs SPT+Rescue', 'deadline_miss_pct')),
        _d('RHRescueWaitT', _t_val('RH vs SPT+Rescue', 'avg_wait_min')),
        _d('RHRescueWaitP', _p_val('RH vs SPT+Rescue', 'avg_wait_min')),
        '% Wait ties asserted in the prose: RH vs SPT and CADR vs SPT+Rescue.',
        '% Both are non-significant, which is what licenses the word "tied".',
        _d('RHSPTwaitT',        _t_val('RH vs SPT', 'avg_wait_min')),
        _d('RHSPTwaitP',        _p_val('RH vs SPT', 'avg_wait_min')),
        _d('RHSPTwaitD',        _d_val('RH vs SPT', 'avg_wait_min')),
        _d('CADRRescueWaitT',   _t_val('CADR vs SPT+Rescue', 'avg_wait_min')),
        _d('CADRRescueWaitP',   _p_val('CADR vs SPT+Rescue', 'avg_wait_min')),
        _d('CADRRescueWaitD',   _d_val('CADR vs SPT+Rescue', 'avg_wait_min')),
        '% Random vs FIFO miss Cohen d and Random Phase 3 mean miss',
        _d('RandomFIFOmissD', _d_val('Random vs FIFO', 'deadline_miss_pct')),
        '% SPT vs EDF miss p-value and Cohen d (EDF significantly lower since the',
        '% 2026-07-19 complexity-mix fix; survives Holm)',
        _d('SPTEDFmissP',   _fmt(tt[('SPT vs EDF', 'deadline_miss_pct')]['p'], dp=3)),
        _d('SPTEDFmissD',   _d_val('SPT vs EDF', 'deadline_miss_pct')),
        '% Phase 3 t-statistics and non-trivial p-values (for results tables)',
        _d('SPTFIFOwaitT',    _t_val('SPT vs FIFO', 'avg_wait_min')),
        _d('SPTFIFOmissT',    _t_val('SPT vs FIFO', 'deadline_miss_pct')),
        _d('RescueFIFOwaitT', _t_val('SPT+Rescue vs FIFO', 'avg_wait_min')),
        _d('RescueFIFOmissT', _t_val('SPT+Rescue vs FIFO', 'deadline_miss_pct')),
        _d('EDFFIFOwaitT',    _t_val('EDF vs FIFO', 'avg_wait_min')),
        _d('EDFFIFOwaitP',    _p_val('EDF vs FIFO', 'avg_wait_min')),
        _d('EDFFIFOmissT',    _t_val('EDF vs FIFO', 'deadline_miss_pct')),
        _d('RandomFIFOmissT', _t_val('Random vs FIFO', 'deadline_miss_pct')),
        _d('RandomFIFOmissP', _p_val('Random vs FIFO', 'deadline_miss_pct')),
        _d('SPTEDFmissT',     _t_val('SPT vs EDF', 'deadline_miss_pct')),
        '% Holm-Bonferroni family-wise correction over the pairwise t-test family',
        _d('HolmFamily',  str(_holm_m)),
        _d('HolmUncorrSig', str(_holm_uncorr)),
        _d('HolmNSig',    str(_holm_nsig)),
        _d('WilcoxHolmNSig', str(_wil_nsig)),
        _d('SPTEDFmissWilcoxP', _fmt(_wil_spt_edf, dp=4)),
        _d('HolmMaxSigP', _fmt(_holm_maxsig, dp=3)),
        # Counts the prose spells out in words: how many uncorrected-significant
        # results the correction removes, and how many of those are outside the
        # two rows the results table prints.
        _d('HolmRemoved',          str(_holm_uncorr - _holm_nsig)),
        _d('HolmRemovedUnprinted', str(len(_holm_removed) - len(_holm_printed))),
        '% Phase 4 (30-day monthly, 10 replications)',
        _d('PhFourSPTwait',     _fmt(s4['SPT']['avg_wait_min']['mean'])),
        _d('PhFourSPTmiss',     _fmt(s4['SPT']['deadline_miss_pct']['mean'])),
        _d('PhFourRescueWait',  _fmt(s4['SPT+Rescue']['avg_wait_min']['mean'])),
        _d('PhFourRescueMiss',  _fmt(s4['SPT+Rescue']['deadline_miss_pct']['mean'])),
        _d('PhFourEDFwait',     _fmt(s4['EDF']['avg_wait_min']['mean'])),
        _d('PhFourEDFmiss',     _fmt(s4['EDF']['deadline_miss_pct']['mean'])),
        _d('PhFourFIFOmiss',    _fmt(s4['FIFO']['deadline_miss_pct']['mean'])),
        _d('PhFourBalancedMiss', _fmt(s4['Balanced']['deadline_miss_pct']['mean'])),
        _d('PhFourLCFmiss',     _fmt(s4['LCF']['deadline_miss_pct']['mean'])),
        _d('PhFourRandomMiss',  _fmt(s4['Random']['deadline_miss_pct']['mean'])),
        _d('PhFourRHWait', _fmt(s4['RH']['avg_wait_min']['mean'])),
        _d('PhFourRHMiss', _fmt(s4['RH']['deadline_miss_pct']['mean'])),
        '% Phase 4 hectic-day sub-rates (from monthly_results.json)',
        _d('PhFourHecSPTmiss',    _fmt(s4['SPT']['hectic_miss_pct']['mean'])),
        _d('PhFourHecRescueMiss', _fmt(s4['SPT+Rescue']['hectic_miss_pct']['mean'])),
        _d('PhFourHecFIFOmiss',   _fmt(s4['FIFO']['hectic_miss_pct']['mean'])),
        '% Phase 2 (seed 42, single hectic-day run)',
        _d('PhTwoSPTwait',    _p2w('SPT')),
        _d('PhTwoSPTmiss',    _p2m('SPT')),
        _d('PhTwoEDFmiss',    _p2m('EDF')),
        _d('PhTwoRescueWait', _p2w('SPT+Rescue')),
        _d('PhTwoRescueMiss', _p2m('SPT+Rescue')),
        _d('PhTwoFIFOmiss',   _p2m('FIFO')),
        _d('PhTwoUtilPct',    _fmt(p2['SPT']['utilization_legacy']['average'] * 100, dp=1)),
        '% Phase 3 derived differences (rounded to integer minutes)',
        _d('SPTFIFOwaitDiffPct', _fmt((s3['FIFO']['wait_min']['mean'] - s3['SPT']['wait_min']['mean'])
                                      / s3['FIFO']['wait_min']['mean'] * 100, dp=1)),
        '% Cmax sensitivity (n=10 seeds, hectic day)',
        _d('CmaxSevenFIFOmiss', _fmt(cmax['cmax7']['schedulers']['FIFO']['avg_miss_pct'], dp=1)),
        '% Sensitivity: hectic day (30-seed mean, ~950 jobs)',
        _d('SensHecSPTwait',    _fmt(s_hec['SPT']['avg_wait_min'],    dp=1)),
        _d('SensHecSPTmiss',    _fmt(s_hec['SPT']['avg_miss_pct'],    dp=1)),
        _d('SensHecFIFOmiss',   _fmt(s_hec['FIFO']['avg_miss_pct'],   dp=1)),
        _d('SensHecRHwait',     _fmt(s_hec['RH']['avg_wait_min'],     dp=1)),
        _d('SensHecRHmiss',     _fmt(s_hec['RH']['avg_miss_pct'],     dp=1)),
        '% Sensitivity: surge day (30-seed mean, ~730 jobs)',
        _d('SensSrgSPTmiss',    _fmt(s_srg['SPT']['avg_miss_pct'],        dp=1)),
        _d('SensSrgEDFmiss',    _fmt(s_srg['EDF']['avg_miss_pct'],        dp=1)),
        _d('SensSrgRescueMiss', _fmt(s_srg['SPT+Rescue']['avg_miss_pct'], dp=1)),
        '% Deadline tightness sweep (surge day, 30 seeds each)',
        _d('DlTenEDFmiss',     _dlm('tight10pct', 'EDF')),
        _d('DlTenSPTmiss',     _dlm('tight10pct', 'SPT')),
        _d('DlThirtyEDFmiss',  _dlm('tight30pct', 'EDF')),
        _d('DlThirtySPTmiss',  _dlm('tight30pct', 'SPT')),
        _d('DlFiftyEDFmiss',   _dlm('tight50pct', 'EDF')),
        _d('DlFiftySPTmiss',   _dlm('tight50pct', 'SPT')),
        _d('DlSeventyEDFmiss', _dlm('tight70pct', 'EDF')),
        _d('DlSeventySPTmiss', _dlm('tight70pct', 'SPT')),
        _d('DlNinetyEDFmiss',  _dlm('tight90pct', 'EDF')),
        _d('DlNinetySPTmiss',  _dlm('tight90pct', 'SPT')),
        '% Paired SPT vs EDF miss test at every sweep point (positive t = SPT',
        '% higher, i.e. EDF lower). The sweep is where the two policies swap',
        '% places, so the crossover claim needs a test at each point, not just',
        '% the point estimates: at 50% tight the two are indistinguishable.',
        _d('DlTenSPTEDFmissT',     _dlt(10)),
        _d('DlTenSPTEDFmissP',     _dlp(10)),
        _d('DlThirtySPTEDFmissT',  _dlt(30)),
        _d('DlThirtySPTEDFmissP',  _dlp(30)),
        _d('DlFiftySPTEDFmissT',   _dlt(50)),
        _d('DlFiftySPTEDFmissP',   _dlp(50)),
        _d('DlSeventySPTEDFmissT', _dlt(70)),
        _d('DlSeventySPTEDFmissP', _dlp(70)),
        _d('DlNinetySPTEDFmissT',  _dlt(90)),
        _d('DlNinetySPTEDFmissP',  _dlp(90)),
        '% Deadline sweep: baseline (FIFO/LCF/Balanced) range endpoints',
        _d('DlTenFIFOmiss',    _dlm('tight10pct', 'FIFO')),
        '% Rescue threshold sensitivity (hectic day, 10 seeds)',
        _d('ThrSixtyMiss',         _thrm(60)),
        _d('ThrOneEightyMiss',     _thrm(180)),
        _d('ThrThreeHundredMiss',  _thrm(300)),
        _d('ThrSixHundredMiss',    _thrm(600)),
        # How far the shipped default (600 s) sits above the sweep minimum (180 s).
        _d('ThrDefaultGap', _fmt(thr['theta600']['avg_miss_pct'] - thr['theta180']['avg_miss_pct'])),
        _d('ThrSixHundredWait',    _thrw(600)),
        _d('ThrTwelveHundredMiss', _thrm(1200)),
        # Full-sweep miss spread (max - min across all swept thresholds).
        _d('ThrMissSpread', _fmt(
            max(thr[f'theta{t}']['avg_miss_pct'] for t in (60, 180, 300, 600, 1200))
            - min(thr[f'theta{t}']['avg_miss_pct'] for t in (60, 180, 300, 600, 1200)))),
        '% Phase 3 operational cost means (USD per 24h run) and spread',
        _d('RHcost',          _fmt(s3['RH']['cost_usd']['mean'])),
        _d('RescueCost',      _fmt(s3['SPT+Rescue']['cost_usd']['mean'])),
        _d('SPTcost',         _fmt(s3['SPT']['cost_usd']['mean'])),
        _d('LCFcost',         _fmt(s3['LCF']['cost_usd']['mean'])),
        # Cheapest-to-dearest spread as a percentage of the cheapest policy.
        _d('CostSpreadPct', _fmt(
            100.0 * (max(s3[k]['cost_usd']['mean'] for k in s3)
                     - min(s3[k]['cost_usd']['mean'] for k in s3))
            / min(s3[k]['cost_usd']['mean'] for k in s3))),
        '% CADR Phase 3 (30-seed hectic day)',
        _d('CADRwait',        _fmt(s3['CADR']['wait_min']['mean'])),
        _d('CADRmiss',        _fmt(s3['CADR']['miss_pct']['mean'])),
        _d('CADRcost',        _fmt(s3['CADR']['cost_usd']['mean'])),
        _d('CADRtard',        _fmt(s3['CADR']['avg_tardiness_min']['mean'])),
        _d('CADRtardReduction', _fmt((1 - s3['CADR']['avg_tardiness_min']['mean'] / s3['RH']['avg_tardiness_min']['mean']) * 100, dp=0)),
        '% RH vs CADR on miss: paired test not significant; equivalence not established',
        _d('RHCADRmissT',     _t_val('RH vs CADR', 'deadline_miss_pct')),
        _d('RHCADRmissP',     _p_val('RH vs CADR', 'deadline_miss_pct')),
        _d('RHCADRmissD',     _d_val('RH vs CADR', 'deadline_miss_pct')),
        '% Phase 3 paired mean differences (first minus second scheduler), 95% t-CI, df=29',
        *_md('SPTFIFOwait', 'SPT', 'FIFO', 'avg_wait_min'),
        *_md('SPTFIFOmiss', 'SPT', 'FIFO', 'deadline_miss_pct'),
        *_md('RescueFIFOwait', 'SPT+Rescue', 'FIFO', 'avg_wait_min'),
        *_md('RescueFIFOmiss', 'SPT+Rescue', 'FIFO', 'deadline_miss_pct'),
        *_md('EDFFIFOwait', 'EDF', 'FIFO', 'avg_wait_min'),
        *_md('EDFFIFOmiss', 'EDF', 'FIFO', 'deadline_miss_pct'),
        *_md('RandomFIFOmiss', 'Random', 'FIFO', 'deadline_miss_pct'),
        *_md('SPTEDFmiss', 'SPT', 'EDF', 'deadline_miss_pct'),
        *_md('RHFIFOwait', 'RH', 'FIFO', 'avg_wait_min'),
        *_md('RHFIFOmiss', 'RH', 'FIFO', 'deadline_miss_pct'),
        *_md('RHRescueWait', 'RH', 'SPT+Rescue', 'avg_wait_min'),
        *_md('RHRescueMiss', 'RH', 'SPT+Rescue', 'deadline_miss_pct'),
        *_md('RHCADRmiss', 'RH', 'CADR', 'deadline_miss_pct'),
        '% Phase 3 tardiness for comparison set (mean, min)',
        _d('FIFOtard',        _fmt(s3['FIFO']['avg_tardiness_min']['mean'])),
        _d('SPTtard',         _fmt(s3['SPT']['avg_tardiness_min']['mean'])),
        _d('EDFtard',         _fmt(s3['EDF']['avg_tardiness_min']['mean'])),
        _d('RescueTard',      _fmt(s3['SPT+Rescue']['avg_tardiness_min']['mean'])),
        _d('RHtard',          _fmt(s3['RH']['avg_tardiness_min']['mean'])),
        '% Phase 4 CADR and tardiness macros',
        _d('PhFourCADRwait',     _fmt(s4['CADR']['avg_wait_min']['mean'])),
        _d('PhFourCADRMiss',     _fmt(s4['CADR']['deadline_miss_pct']['mean'])),
        _d('PhFourCADRTard',     _fmt(s4['CADR']['avg_tardiness_min']['mean'])),
        _d('PhFourRHtard',       _fmt(s4['RH']['avg_tardiness_min']['mean'])),
        _d('PhFourEDFtard',      _fmt(s4['EDF']['avg_tardiness_min']['mean'])),
        _d('PhFourFIFOtard',     _fmt(s4['FIFO']['avg_tardiness_min']['mean'])),
        _d('PhFourSPTtard',      _fmt(s4['SPT']['avg_tardiness_min']['mean'])),
        _d('PhFourRescueTard',   _fmt(s4['SPT+Rescue']['avg_tardiness_min']['mean'])),
        '% Sensitivity: hectic day CADR',
        _d('SensHecCADRwait',    _fmt(s_hec['CADR']['avg_wait_min'],    dp=1)),
        _d('SensHecCADRmiss',    _fmt(s_hec['CADR']['avg_miss_pct'],    dp=1)),
        '% CADR-order-only ablation (isolation of cost-aware placement)',
        _d('CADROOmiss',         _fmt(p3['ablation_extra']['CADR-order-only']['miss_pct']['mean'])),
        _d('CADROOwait',         _fmt(p3['ablation_extra']['CADR-order-only']['wait_min']['mean'])),
        _d('CADROOtard',         _fmt(p3['ablation_extra']['CADR-order-only']['avg_tardiness_min']['mean'])),
        '% CR* sensitivity prose macros. CRStarBestCR is the argmin, so the',
        '% prose never has to name the winning threshold by hand, and',
        '% CRStarDefaultGap is how far the shipped default sits above it.',
        _d('CRStarBestMiss',     _fmt(min(cr_sens[k]['avg_miss_pct'] for k in cr_sens))),
        _d('CRStarBestCR',       f"{float(min(cr_sens, key=lambda k: cr_sens[k]['avg_miss_pct'])):g}"),
        _d('CRStarDefaultMiss',  _fmt(cr_sens['3.0']['avg_miss_pct'])),
        _d('CRStarDefaultGap',   _fmt(cr_sens['3.0']['avg_miss_pct']
                                      - min(cr_sens[k]['avg_miss_pct'] for k in cr_sens))),
        _d('CRStarMissSpread',   _fmt(
            max(cr_sens[k]['avg_miss_pct'] for k in cr_sens)
            - min(cr_sens[k]['avg_miss_pct'] for k in cr_sens))),
        '% RH anticipation (reservation) payoff: surge below saturation',
        _d('ResSurgeOffMiss',    _fmt(rs_srg['r0']['avg_miss_pct'])),
        _d('ResSurgeOnMiss',     _fmt(rs_srg['r1']['avg_miss_pct'])),
        _d('ResSurgeOffTight',   _fmt(rs_srg['r0']['tight_miss_pct'])),
        _d('ResSurgeOnTight',    _fmt(rs_srg['r1']['tight_miss_pct'])),
        _d('ResSurgeOffWait',    _fmt(rs_srg['r0']['avg_wait_min'])),
        _d('ResSurgeOnWait',     _fmt(rs_srg['r1']['avg_wait_min'])),
        _d('ResSurgeOnLoose',    _fmt(rs_srg['r1']['loose_miss_pct'])),
        _d('ResHecticGateMiss',  _fmt(rs_hec['r0']['avg_miss_pct'])),
        '% Sensitivity: surge day RH (anticipation active below saturation)',
        _d('SensSrgRHmiss',      _fmt(s_srg['RH']['avg_miss_pct'],     dp=1)),
        _d('SensSrgRHwait',      _fmt(s_srg['RH']['avg_wait_min'],     dp=1)),
        '% Surge-day wait range over the seven policies and five fractions of the deadline-tightness sweep (ch5 prose)',
        _d('SensSrgWaitLo',      _fmt(min(v['avg_wait_min'] for f in dl['results']['fleet5'].values()
                                          for v in f['schedulers'].values()), dp=0)),
        _d('SensSrgWaitHi',      _fmt(max(v['avg_wait_min'] for f in dl['results']['fleet5'].values()
                                          for v in f['schedulers'].values()), dp=0)),
        '% Surge-day wait range over all ten schedulers in the day-type table (ch5 and ch6 prose)',
        _d('SensSrgDayWaitLo',   _fmt(min(v['avg_wait_min'] for v in s_srg.values()), dp=0)),
        _d('SensSrgDayWaitHi',   _fmt(max(v['avg_wait_min'] for v in s_srg.values()), dp=0)),
        _d('PriorGateLambda',    _fmt(prior_all['lambda_gate'], dp=4)),
        _d('PriorSurgeCrossX',   _fmt(prior_all['lambda_gate'] / prior_all['arm_c_arrival_prior']['surge']['x1.0']['true_arrival_rate'], dp=1)),
        _d('PriorGateShortfall', _fmt(100 * (1 - prior_all['lambda_gate'] / prior_all['arm_c_arrival_prior']['hectic']['x1.0']['true_arrival_rate']), dp=0)),
        _d('PriorHecUnderMiss',  _fmt(prior_all['arm_c_arrival_prior']['hectic']['x0.5']['miss_pct_mean'])),
        _d('PriorHecUnderWait',  _fmt(prior_all['arm_c_arrival_prior']['hectic']['x0.5']['wait_min_mean'])),
        _d('CmaxTenRHwait',      _fmt(cmax['cmax10']['schedulers']['RH']['avg_wait_min'], dp=1)),
        _d('CmaxTenOtherWaitLo', _fmt(min(v['avg_wait_min'] for k, v in cmax['cmax10']['schedulers'].items() if k != 'RH'), dp=1)),
        _d('CmaxTenOtherWaitHi', _fmt(max(v['avg_wait_min'] for k, v in cmax['cmax10']['schedulers'].items() if k != 'RH'), dp=1)),
        _d('CRStarWaitSpread',   _fmt(max(v['avg_wait_min'] for v in cr_sens.values()) - min(v['avg_wait_min'] for v in cr_sens.values()), dp=1)),
        '% Adaptive EDF-SPT hybrid (Phase 3 hectic 30-seed, Phase 4 monthly, sensitivity)',
        _d('AdaptiveWait',         _fmt(s3['Adaptive']['wait_min']['mean'])),
        _d('AdaptiveMiss',         _fmt(s3['Adaptive']['miss_pct']['mean'])),
        _d('AdaptiveTard',         _fmt(s3['Adaptive']['avg_tardiness_min']['mean'])),
        _d('PhFourAdaptiveWait',   _fmt(s4['Adaptive']['avg_wait_min']['mean'])),
        _d('PhFourAdaptiveMiss',   _fmt(s4['Adaptive']['deadline_miss_pct']['mean'])),
        _d('PhFourAdaptiveTard',   _fmt(s4['Adaptive']['avg_tardiness_min']['mean'])),
        _d('SensSrgAdaptiveMiss',  _fmt(s_srg['Adaptive']['avg_miss_pct'], dp=1)),
    ]
    _write('stats.tex', '\n'.join(lines) + '\n')




# ── Table: Availability-model (beta_g) sensitivity ───────────────────────────

def gen_table_beta_sensitivity():
    path = CGRSP / 'Outputs/phase3/beta_sensitivity.json'
    if not path.exists():
        print('  Skipping beta sensitivity table (no JSON yet)')
        return
    with open(path) as f:
        d = json.load(f)

    scheds = ['FIFO', 'SPT', 'EDF', 'SPT+Rescue', 'RH', 'CADR']
    rows = []
    for scale in d['scales']:
        key = f'{float(scale):.2f}'  # scales may be stored as floats or strings
        ss = d['results'][key]
        vals = []
        for name in scheds:
            eff = _beta_eff_miss(ss[name])
            vals.append(f"{ss[name]['wait_min_mean']:.1f}\\,/\\,{eff:.1f}")
        label = f'$\\times{key}$'
        if abs(float(scale) - 1.0) < 1e-9:
            label += r' \textbf{(baseline)}'
        rows.append(label + ' & ' + ' & '.join(vals) + r' \\')

    header = [
        r'\begin{tabular}{lrrrrrr}',
        r'\toprule',
        r'& \multicolumn{6}{c}{\textbf{Avg Wait (min) / Effective Miss \%}} \\',
        r'\cmidrule(lr){2-7}',
        r'\textbf{$\beta_g$ scale} & \textbf{FIFO} & \textbf{SPT} & \textbf{EDF} & \textbf{SPT+Rescue} & \textbf{RH} & \textbf{CADR} \\',
        r'\midrule',
    ]
    footer = [r'\bottomrule', r'\end{tabular}']
    _write('table_beta_sensitivity.tex', '\n'.join(header + rows + footer) + '\n')


def _beta_eff_miss(cell):
    """Effective miss %: jobs unfinished at the end of the 24 h window count
    as missed (every deadline in the hectic instance falls before the window
    end, so an unfinished job has certainly missed). Prefers the per-seed
    mean stored by the sweep script; falls back to composing cross-seed means
    (mean-of-products vs product-of-means differ when per-seed completion and
    miss covary, so the per-seed value is the correct estimator)."""
    if 'eff_miss_pct_mean' in cell:
        return cell['eff_miss_pct_mean']
    completed_pct = cell.get('completed_pct_mean', 100.0)
    on_time_pct = completed_pct * (1.0 - cell['miss_pct_mean'] / 100.0)
    return 100.0 - on_time_pct


def _betai(a, b, x):
    """Regularised incomplete beta function I_x(a, b) (Numerical Recipes 6.4);
    used for exact two-sided t-test p-values without a scipy dependency."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_front = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                + a * math.log(x) + b * math.log(1.0 - x))

    def betacf(a, b, x):
        MAXIT, EPS, FPMIN = 200, 3e-12, 1e-300
        qab, qap, qam = a + b, a + 1.0, a - 1.0
        c, d = 1.0, 1.0 - qab * x / qap
        if abs(d) < FPMIN:
            d = FPMIN
        d = 1.0 / d
        h = d
        for mm in range(1, MAXIT + 1):
            m2 = 2 * mm
            aa = mm * (b - mm) * x / ((qam + m2) * (a + m2))
            d = 1.0 + aa * d
            if abs(d) < FPMIN:
                d = FPMIN
            c = 1.0 + aa / c
            if abs(c) < FPMIN:
                c = FPMIN
            d = 1.0 / d
            h *= d * c
            aa = -(a + mm) * (qab + mm) * x / ((a + m2) * (qap + m2))
            d = 1.0 + aa * d
            if abs(d) < FPMIN:
                d = FPMIN
            c = 1.0 + aa / c
            if abs(c) < FPMIN:
                c = FPMIN
            d = 1.0 / d
            de = d * c
            h *= de
            if abs(de - 1.0) < EPS:
                break
        return h

    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(ln_front) * betacf(a, b, x) / a
    return 1.0 - math.exp(ln_front) * betacf(b, a, 1.0 - x) / b


def _paired_ci(a, b):
    """Paired mean difference a - b with its two-sided 95% t interval (df = 29)."""
    assert len(a) == len(b) == 30
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    md = sum(d) / n
    sd = (sum((x - md) ** 2 for x in d) / (n - 1)) ** 0.5
    half = 2.0452296421327034 * sd / n ** 0.5
    return md, md - half, md + half


_RAW_KEY = {'avg_wait_min': 'raw_wait_min', 'deadline_miss_pct': 'raw_miss_pct'}


def _exact_t_d(scheds, comp, metric):
    """Paired t and pooled-SD Cohen's d for 'A vs B', recomputed from the per-seed arrays.
    statistical.json stores both to 4 dp, so printing those at 3 dp would round twice."""
    a_name, b_name = comp.split(' vs ')
    a = scheds[a_name][_RAW_KEY[metric]]
    b = scheds[b_name][_RAW_KEY[metric]]
    _, t, _ = _paired_stats(a, b)
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    va = sum((x - ma) ** 2 for x in a) / (n - 1)
    vb = sum((x - mb) ** 2 for x in b) / (n - 1)
    pooled = (va + vb) / 2.0
    return t, ((ma - mb) / pooled ** 0.5 if pooled > 0 else 0.0)


def _paired_stats(a, b):
    """Paired t-test of matched arrays a vs b (difference a - b):
    returns (mean diff, t, two-sided p) with df = n - 1."""
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    md = sum(d) / n
    sd = (sum((x - md) ** 2 for x in d) / (n - 1)) ** 0.5
    if sd == 0.0:
        t = float('inf') if md > 0 else -float('inf') if md < 0 else 0.0
        return md, t, (0.0 if md else 1.0)
    t = md / (sd / n ** 0.5)
    df = n - 1
    p = _betai(df / 2.0, 0.5, df / (df + t * t))
    return md, t, p


def _fmt_p(p):
    """LaTeX math fragment for a two-sided p-value (use inside $...$):
    exact to 2-3 decimals when moderate, a power-of-ten bound when tiny."""
    if p >= 0.0995:
        return f"p={p:.2f}"
    if p >= 0.001:
        return f"p={p:.3f}"
    return f"p<10^{{{math.ceil(math.log10(p))}}}"


# ── Macros: capacity bound + beta-sweep key numbers ──────────────────────────

def gen_audit_macros():
    lines = []
    with open(CGRSP / 'Outputs/phase4/monthly_results.json') as f:
        monthly = json.load(f)
    reservation = monthly['reservation_ablation']['paired_vs_default']
    for name, metric in [('Wait', 'avg_wait_min'), ('Miss', 'deadline_miss_pct'),
                         ('Tard', 'avg_tardiness_min')]:
        row = reservation[metric]
        lines.append(f"\\newcommand{{\\MonthRes{name}Delta}}{{{abs(row['mean_delta']):.2f}}}")
        lines.append(f"\\newcommand{{\\MonthRes{name}P}}{{{_fmt_p(row['p_value'])}}}")
    lines.append(f"\\newcommand{{\\MonthResOffWait}}{{{reservation['avg_wait_min']['mean_ablated']:.2f}}}")
    lines.append(f"\\newcommand{{\\MonthResOffMiss}}{{{reservation['deadline_miss_pct']['mean_ablated']:.2f}}}")
    sparse_path = CGRSP / 'Outputs/phase3/sparse_sensitivity.json'
    with open(sparse_path) as f:
        sparse = json.load(f)['arms']
    exposure = [row['sparse_dispatch_pct'] for row in sparse['baseline']['schedulers'].values()]
    lines.append(f"\\newcommand{{\\SparseExposureLo}}{{{min(exposure):.2f}}}")
    lines.append(f"\\newcommand{{\\SparseExposureHi}}{{{max(exposure):.2f}}}")
    adjusted = {}
    running_p = 0.0
    for rank, (arm, row) in enumerate(sorted(sparse.items(), key=lambda item: item[1]['rh_vs_cadr_miss_p'])):
        running_p = max(running_p, min(1.0, (len(sparse) - rank) * row['rh_vs_cadr_miss_p']))
        adjusted[arm] = running_p
    lines.append(f"\\newcommand{{\\SparseSigmaHighP}}{{{sparse['sigma_200']['rh_vs_cadr_miss_p']:.3f}}}")
    lines.append(f"\\newcommand{{\\SparseSigmaHighAdjustedP}}{{{adjusted['sigma_200']:.3f}}}")
    for name, arm in [('Base', 'baseline'), ('MeanLow', 'mean_075'), ('MeanHigh', 'mean_125'),
                      ('SigmaLow', 'sigma_050'), ('SigmaHigh', 'sigma_200')]:
        for scheduler in ['RH', 'CADR']:
            value = sparse[arm]['schedulers'][scheduler]['miss_pct']
            lines.append(f"\\newcommand{{\\Sparse{name}{scheduler}Miss}}{{{value:.2f}}}")
    ks_path = CGRSP / 'Outputs/phase3/kappa_speed_sensitivity.json'
    with open(ks_path) as f:
        ks = json.load(f)['arms']
    for name, arm in [('Base', 'baseline'), ('AdjLow', 'adj_20'), ('AdjHigh', 'adj_40'),
                      ('Uninf', 'uninformative'), ('Slow', 'a4000_scene_matched')]:
        for scheduler in ['FIFO', 'SPT', 'EDF', 'SPTRescue', 'RH', 'CADR']:
            key = scheduler.replace('SPTRescue', 'SPT+Rescue')
            lines.append(f"\\newcommand{{\\Ks{name}{scheduler}Miss}}{{{ks[arm]['schedulers'][key]['miss_pct']:.2f}}}")
        lines.append(f"\\newcommand{{\\Ks{name}RHvCADRp}}{{{ks[arm]['rh_vs_cadr_miss_p']:.3f}}}")
    cb_path = CGRSP / 'Outputs/phase3/capacity_bound.json'
    if cb_path.exists():
        with open(cb_path) as f:
            cb = json.load(f)
        lines.append(f"\\newcommand{{\\CapBoundMeanPct}}{{{cb['capacity_bound_miss_pct_mean']:.2f}}}")
        lines.append(f"\\newcommand{{\\CapBoundMaxMisses}}{{{max(cb['capacity_bound_misses'])}}}")
        lines.append(f"\\newcommand{{\\RelaxEDFMeanPct}}{{{cb['relaxed_edf_miss_pct_mean']:.2f}}}")
    bs_path = CGRSP / 'Outputs/phase3/beta_sensitivity.json'
    if bs_path.exists():
        with open(bs_path) as f:
            bs = json.load(f)['results']
        def m(scale, sched):
            return _beta_eff_miss(bs[scale][sched])
        lines.append(f"\\newcommand{{\\BetaQuarterCompletedPct}}{{{bs['0.25']['RH']['completed_pct_mean']:.0f}}}")
        lines.append(f"\\newcommand{{\\BetaHalfCompletedPct}}{{{bs['0.50']['RH']['completed_pct_mean']:.0f}}}")
        lines.append(f"\\newcommand{{\\BetaThreeQRHmiss}}{{{m('0.75','RH'):.1f}}}")
        lines.append(f"\\newcommand{{\\BetaThreeQRescueMiss}}{{{m('0.75','SPT+Rescue'):.1f}}}")
        lines.append(f"\\newcommand{{\\BetaThreeQCADRmiss}}{{{m('0.75','CADR'):.1f}}}")
        lines.append(f"\\newcommand{{\\BetaThreeQFIFOmiss}}{{{m('0.75','FIFO'):.1f}}}")
        lines.append(f"\\newcommand{{\\BetaUpRHmiss}}{{{m('1.25','RH'):.2f}}}")
        lines.append(f"\\newcommand{{\\BetaUpSPTmiss}}{{{m('1.25','SPT'):.2f}}}")
        # Row extrema, so prose can quote a real bound instead of a round
        # number that some policy in the same row falls outside. The x0.50
        # bound covers exactly the six policies the table displays; the x0.75
        # range covers every policy simulated, which is wider than the six.
        table_scheds = ['FIFO', 'SPT', 'EDF', 'SPT+Rescue', 'RH', 'CADR']
        half_shown = [m('0.50', s) for s in table_scheds]
        three_q_all = [m('0.75', s) for s in bs['0.75']]
        lines.append(f"\\newcommand{{\\BetaHalfMinEffMiss}}{{{min(half_shown):.1f}}}")
        lines.append(f"\\newcommand{{\\BetaThreeQuarterMinEffMiss}}{{{min(three_q_all):.1f}}}")
        lines.append(f"\\newcommand{{\\BetaThreeQuarterMaxEffMiss}}{{{max(three_q_all):.1f}}}")
        # Paired t-tests on the per-seed effective-miss arrays at x0.75
        if 'eff_miss_pct_raw' in bs['0.75']['RH']:
            _, t_c, p_c = _paired_stats(bs['0.75']['RH']['eff_miss_pct_raw'],
                                        bs['0.75']['CADR']['eff_miss_pct_raw'])
            _, t_r, p_r = _paired_stats(bs['0.75']['RH']['eff_miss_pct_raw'],
                                        bs['0.75']['SPT+Rescue']['eff_miss_pct_raw'])
            lines.append(f"\\newcommand{{\\BetaRHvCADRt}}{{{t_c:.2f}}}")
            lines.append(f"\\newcommand{{\\BetaRHvCADRp}}{{{_fmt_p(p_c)}}}")
            lines.append(f"\\newcommand{{\\BetaRHvRescueT}}{{{t_r:.2f}}}")
            lines.append(f"\\newcommand{{\\BetaRHvRescueP}}{{{_fmt_p(p_r)}}}")
    pr_path = CGRSP / 'Outputs/phase3/prior_robustness.json'
    if pr_path.exists():
        with open(pr_path) as f:
            pr = json.load(f)
        a = pr['arm_a_reversed']
        for sched, macro in [('RH', 'RevRHmiss'), ('CADR', 'RevCADRmiss'),
                             ('SPT+Rescue', 'RevRescueMiss'),
                             ('SPT', 'RevSPTmiss'), ('FIFO', 'RevFIFOmiss'),
                             ('EDF', 'RevEDFmiss')]:
            lines.append(f"\\newcommand{{\\{macro}}}{{{a[sched]['miss_pct_mean']:.2f}}}")
        lines.append(f"\\newcommand{{\\NoPriorRHmiss}}{{{pr['arm_b_rh_noprior']['RH-noprior']['miss_pct_mean']:.2f}}}")
        lines.append(f"\\newcommand{{\\NoPriorRHwait}}{{{pr['arm_b_rh_noprior']['RH-noprior']['wait_min_mean']:.2f}}}")
        # Arm A: rank-reversed RH vs CADR paired test on the per-seed arrays
        if 'miss_pct_raw' in a['RH']:
            _, t_rev, p_rev = _paired_stats(a['RH']['miss_pct_raw'],
                                            a['CADR']['miss_pct_raw'])
            lines.append(f"\\newcommand{{\\RevRHvCADRt}}{{{t_rev:.2f}}}")
            lines.append(f"\\newcommand{{\\RevRHvCADRp}}{{{_fmt_p(p_rev)}}}")
        # Arm B: stock RH (Phase 3 seeds) vs prior-ablated RH, paired
        st_path = CGRSP / 'Outputs/phase3/statistical.json'
        nop = pr['arm_b_rh_noprior']['RH-noprior']
        if st_path.exists() and 'miss_pct_raw' in nop:
            with open(st_path) as f:
                st_rh = json.load(f)['schedulers']['RH']
            if 'raw_miss_pct' in st_rh:
                md_np, t_np, p_np = _paired_stats(st_rh['raw_miss_pct'],
                                                  nop['miss_pct_raw'])
                lines.append(f"\\newcommand{{\\NoPriorDiffPp}}{{{md_np:.2f}}}")
                lines.append(f"\\newcommand{{\\NoPriorT}}{{{t_np:.2f}}}")
                lines.append(f"\\newcommand{{\\NoPriorP}}{{{_fmt_p(p_np)}}}")
    ps_path = CGRSP / 'Outputs/phase3/pressure_sensitivity.json'
    if ps_path.exists():
        with open(ps_path) as f:
            ps = json.load(f)['results']
        miss = {p_: ps[p_]['miss_pct_mean'] for p_ in ps}
        best_p = min(miss, key=miss.get)
        lines.append(f"\\newcommand{{\\PswpBestP}}{{{best_p}}}")
        lines.append(f"\\newcommand{{\\PswpBestMiss}}{{{miss[best_p]:.2f}}}")
        lines.append(f"\\newcommand{{\\PswpBaseMiss}}{{{miss['10']:.2f}}}")
        lines.append(f"\\newcommand{{\\PswpSpread}}{{{max(miss.values()) - min(miss.values()):.2f}}}")
        lines.append(f"\\newcommand{{\\PswpBestTard}}{{{ps[best_p]['tard_min_mean']:.2f}}}")
        _bt = min(ps, key=lambda k: ps[k]['tard_min_mean'])
        lines.append(f"\\newcommand{{\\PswpBestTardVal}}{{{ps[_bt]['tard_min_mean']:.2f}}}")
        lines.append(f"\\newcommand{{\\PswpBestTardP}}{{{_bt}}}")
        # Best-of-sweep Adaptive vs RH on the same ten seeds (sweep seeds are
        # 0..9, the first ten of the 30-seed Phase 3 list, same generator)
        st_path = CGRSP / 'Outputs/phase3/statistical.json'
        if st_path.exists() and 'miss_pct_raw' in ps[best_p]:
            with open(st_path) as f:
                st_rh = json.load(f)['schedulers']['RH']
            if 'raw_miss_pct' in st_rh:
                n10 = len(ps[best_p]['miss_pct_raw'])
                md_ps, t_ps, p_ps = _paired_stats(ps[best_p]['miss_pct_raw'],
                                                  st_rh['raw_miss_pct'][:n10])
                lines.append(f"\\newcommand{{\\PswpVsRHdiff}}{{{md_ps:+.2f}}}")
                lines.append(f"\\newcommand{{\\PswpVsRHt}}{{{t_ps:.2f}}}")
                lines.append(f"\\newcommand{{\\PswpVsRHp}}{{{_fmt_p(p_ps)}}}")
    dg_path = CGRSP / 'Outputs/phase3/diagnostics.json'
    if dg_path.exists():
        with open(dg_path) as f:
            dg = json.load(f)
        tf = dg['part_b_rh_t_free']
        lines.append(f"\\newcommand{{\\TFreeCollapsedPct}}{{{tf['collapsed_pct']:.0f}}}")
        sd = dg['part_c_seed_detail']
        rh_tight = [sd[f'RH_seed{s}']['tight_misses'] for s in (0, 42)]
        rh_miss = [sd[f'RH_seed{s}']['misses'] for s in (0, 42)]
        lines.append(f"\\newcommand{{\\DiagRHTightLo}}{{{min(rh_tight)}}}")
        lines.append(f"\\newcommand{{\\DiagRHTightHi}}{{{max(rh_tight)}}}")
        lines.append(f"\\newcommand{{\\DiagRHMissLo}}{{{min(rh_miss)}}}")
        lines.append(f"\\newcommand{{\\DiagRHMissHi}}{{{max(rh_miss)}}}")
        late = [sd[f'RH_seed{s}']['tight_miss_mean_late_min'] for s in (0, 42)]
        lines.append(f"\\newcommand{{\\DiagRHTightLateLo}}{{{min(late):.0f}}}")
        lines.append(f"\\newcommand{{\\DiagRHTightLateHi}}{{{max(late):.0f}}}")
        cadr_tight = sum(sd[f'CADR_seed{s}']['tight_misses'] for s in (0, 42))
        lines.append(f"\\newcommand{{\\DiagCADRTightTotal}}{{{cadr_tight}}}")
        band = [sd[f'{n}_seed{s}']['dispatches_in_1200_1800_band'] for n in ('RH', 'CADR', 'SPT') for s in (0, 42)]
        lines.append(f"\\newcommand{{\\DiagBandLo}}{{{min(band)}}}")
        lines.append(f"\\newcommand{{\\DiagBandHi}}{{{max(band)}}}")
        ma = dg.get('part_a_mean_based_service')
        if ma:
            waits = [abs(v['wait_change_pct']) for k, v in ma.items() if 'wait_change_pct' in v]
            rels = [abs(v['miss_change_rel_pct']) for k, v in ma.items() if 'miss_change_rel_pct' in v]
            lines.append(f"\\newcommand{{\\MeanSvcWaitMaxPct}}{{{max(waits):.1f}}}")
            lines.append(f"\\newcommand{{\\MeanSvcMissMaxRelPct}}{{{max(rels):.0f}}}")
            lines.append(f"\\newcommand{{\\MeanSvcRHMiss}}{{{ma['RH']['mean_based_miss_pct']:.2f}}}")
            lines.append(f"\\newcommand{{\\MeanSvcCADRMiss}}{{{ma['CADR']['mean_based_miss_pct']:.2f}}}")
            lines.append(f"\\newcommand{{\\MeanSvcRHvCADRp}}{{{ma['RH_vs_CADR_miss']['p']:.2f}}}")
    if lines:
        _write('stats_audit.tex', '\n'.join(lines) + '\n')
    else:
        print('  Skipping audit macros (no JSONs yet)')


def gen_table_sparse_sensitivity():
    with open(CGRSP / 'Outputs/phase3/sparse_sensitivity.json') as f:
        data = json.load(f)
    schedulers = ['FIFO', 'SPT', 'EDF', 'SPT+Rescue', 'RH', 'CADR']
    rows = [r'\begin{tabular}{lrrrrrr}', r'\toprule',
            'Estimated-cell arm & ' + ' & '.join(schedulers) + r' \\', r'\midrule']
    labels = {'baseline': 'Baseline', 'mean_075': r'Mean $\times 0.75$',
              'mean_125': r'Mean $\times 1.25$', 'sigma_050': r'$\sigma_{\ln}\times 0.5$',
              'sigma_200': r'$\sigma_{\ln}\times 2$'}
    for arm, label in labels.items():
        results = data['arms'][arm]['schedulers']
        rows.append(label + ' & ' + ' & '.join(f"{results[name]['miss_pct']:.2f}" for name in schedulers) + r' \\')
    rows.extend([r'\bottomrule', r'\end{tabular}'])
    _write('table_sparse_sensitivity.tex', '\n'.join(rows) + '\n')


def gen_table_kappa_speed_sensitivity():
    with open(CGRSP / 'Outputs/phase3/kappa_speed_sensitivity.json') as f:
        data = json.load(f)
    schedulers = ['FIFO', 'SPT', 'EDF', 'SPT+Rescue', 'RH', 'CADR']
    rows = [r'\begin{tabular}{lrrrrrr}', r'\toprule',
            'Arm & ' + ' & '.join(schedulers) + r' \\', r'\midrule']
    labels = {'baseline': 'Baseline',
              'adj_20': r'Label error 20\%',
              'adj_40': r'Label error 40\%',
              'uninformative': 'Uninformative label',
              'a4000_scene_matched': r'A4000 $\times 1.165$'}
    for arm, label in labels.items():
        results = data['arms'][arm]['schedulers']
        rows.append(label + ' & ' + ' & '.join(f"{results[name]['miss_pct']:.2f}" for name in schedulers) + r' \\')
    rows.extend([r'\bottomrule', r'\end{tabular}'])
    _write('table_kappa_speed_sensitivity.tex', '\n'.join(rows) + '\n')


# ── Table: Classical and learned baselines (30 seeds, hectic day) ────────────

def gen_table_baselines():
    with open(CGRSP / 'Outputs/phase3/baselines.json') as f:
        d = json.load(f)
    s = d['schedulers']

    order = ['HEFT', 'MinMin', 'Bandit', 'Bandit-random', 'FIFO', 'RH', 'CADR']
    labels = {'HEFT': r'HEFT $=$ MaxMin'}

    wait = {p: s[p]['wait_min']['mean'] for p in order}
    miss = {p: s[p]['miss_pct']['mean'] for p in order}
    tard = {p: s[p]['avg_tardiness_min']['mean'] for p in order}
    best_wait = min(wait, key=wait.get)
    best_miss = min(miss, key=miss.get)
    best_tard = min(tard, key=tard.get)

    rows = []
    for p in order:
        wait_s = f'{wait[p]:.2f}'
        miss_s = f'{miss[p]:.2f}'
        tard_s = f'{tard[p]:.2f}'
        if p == best_wait:
            wait_s = r'\textbf{' + wait_s + '}'
        if p == best_miss:
            miss_s = r'\textbf{' + miss_s + '}'
        if p == best_tard:
            tard_s = r'\textbf{' + tard_s + '}'
        label = labels.get(p, p)
        rows.append(f'{label} & {wait_s} & {miss_s} & {tard_s}\\\\')

    lines = [
        r'\begin{tabular}{lrrr}',
        r'\toprule',
        r'\textbf{Scheduler} & \textbf{Avg Wait (min)} & \textbf{Miss \%} & \textbf{Mean Tardiness (min)} \\',
        r'\midrule',
        *rows,
        r'\bottomrule',
        r'\end{tabular}',
    ]
    _write('table_baselines.tex', '\n'.join(lines) + '\n')


# ── Table: Fleet-composition sensitivity (six mixes, 10 seeds) ───────────────

def gen_table_fleet_mix():
    with open(CGRSP / 'Outputs/phase3/fleet_mix.json') as f:
        d = json.load(f)

    policies = ['FIFO', 'SPT', 'EDF', 'SPT+Rescue', 'RH', 'CADR']
    order = ['all-types', '3090-heavy', 'A5000-heavy', 'A4000-heavy',
             '3090-only', 'A5000-only']
    display = {
        'all-types': 'All-types',
        '3090-heavy': '3090-heavy',
        'A5000-heavy': 'A5000-heavy',
        'A4000-heavy': 'A4000-heavy',
        '3090-only': '3090-only',
        'A5000-only': 'A5000-only',
    }
    slot_order = ['RTX_3090', 'RTX_A5000', 'RTX_A4500', 'RTX_A4000']

    rows = []
    for mix in order:
        entry = d['mixes'][mix]
        counts = entry['slot_counts']
        slots = '/'.join(str(counts.get(t, 0)) for t in slot_order)
        miss = {p: entry['policies'][p]['miss_pct']['mean'] for p in policies}
        best = min(miss, key=miss.get)
        cells = []
        for p in policies:
            value = f'{miss[p]:.2f}'
            if p == best:
                value = r'\textbf{' + value + '}'
            cells.append(value)
        rows.append(f"{display[mix]} & {slots} & " + ' & '.join(cells) + r' \\')

    lines = [
        r'\begin{tabular}{llrrrrrr}',
        r'\toprule',
        r'\textbf{Mix} & \textbf{Slots} & \textbf{FIFO} & \textbf{SPT} & \textbf{EDF} & \textbf{SPT+Rescue} & \textbf{RH} & \textbf{CADR} \\',
        r'\midrule',
        *rows,
        r'\bottomrule',
        r'\end{tabular}',
    ]
    _write('table_fleet_mix.tex', '\n'.join(lines) + '\n')


if __name__ == '__main__':
    print('Generating thesis table fragments and stats macros...')
    gen_table_phase2()
    gen_table_phase3_stats()
    gen_table_sensitivity()
    gen_table_deadline()
    gen_table_phase4()
    gen_table_ablation()
    gen_table_cmax_sensitivity()
    gen_table_threshold_sensitivity()
    gen_table_weight_sensitivity()
    gen_table_reservation_sensitivity()
    gen_table_cr_sensitivity()
    gen_table_tardiness_objective()
    gen_stats_macros()
    gen_table_beta_sensitivity()
    gen_audit_macros()
    gen_table_sparse_sensitivity()
    gen_table_kappa_speed_sensitivity()
    gen_table_baselines()
    gen_table_fleet_mix()
    print('Done.')

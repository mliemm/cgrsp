"""
validate_numbers.py — consistency checks between .tex prose and simulation JSON,
plus recurring-linguistic-error checks (delta_j inversions, SPT-lowest-miss claims)
and a cheat-sheet sync check pinning cheat_sheet.md's headline numbers and
flip-prone claims to generated/stats.tex.

Run from repo root:
    python3 validate_numbers.py

Exit 0 if all checks pass, 1 if any issue found.

Scope of the numeric check (be honest about what it can and cannot catch): it
scans for hardcoded literals matching the CURRENT canonical values, i.e. it
detects prose that duplicates a live number instead of using its macro. It
CANNOT detect a stale literal that no longer matches the JSON (there is no
list of historical values to search for); protection against staleness comes
from the macro-sync check plus the convention that prose numbers go through
generated/stats.tex macros. Linguistic checks catch convention inversions
that have recurred 5-8+ times in git history.

The empirical-claim provenance check exists because the other checks above
all verify INTERNAL consistency (prose vs. macro vs. JSON) and none of them
can catch an UPSTREAM input being wrong in the first place, which is exactly
what happened in job_generator.py's COMPLEXITY_DISTRIBUTION (DECISIONS.md
entry 14): every downstream check passed for months because the wrong value
was internally consistent with itself. This check forces a one-time human
sign-off, tracked by content hash, on every sentence that claims a number is
"empirical", "calibrated", "derived from", "measured from", or "fitted" to
the dataset. See the check's own docstring below for how sign-off works.
"""
import hashlib
import json
import math
import re
import sys
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT = Path(__file__).resolve().parent
CGRSP = ROOT / 'CGRSP'
THESIS = ROOT / 'thesis'
PAPER = ROOT / 'paper'

# Files checked for hardcoded numbers and linguistic/spelling errors
TEX_FILES = [
    THESIS / 'front_matter/abstract_en.tex',
    THESIS / 'front_matter/abstract_zh.tex',
    THESIS / 'chapters/ch1_intro.tex',
    THESIS / 'chapters/ch2_related.tex',
    THESIS / 'chapters/ch3_formulation.tex',
    THESIS / 'chapters/ch4_system.tex',
    THESIS / 'chapters/ch5_experiments.tex',
    THESIS / 'chapters/ch6_conclusion.tex',
]

LINGUISTIC_FILES = TEX_FILES

# ── Load simulation data ──────────────────────────────────────────────────────

def _mdci(a, b):
    """Independent recomputation of the paired mean difference a - b and its 95% t-CI (df=29)."""
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    m = sum(d) / n
    sd = (sum((x - m) ** 2 for x in d) / (n - 1)) ** 0.5
    h = 2.0452296421327034 * sd / n ** 0.5
    return round(m, 2), round(m - h, 2), round(m + h, 2)


_MD_ROWS = [
    ('SPTFIFOwait', 'SPT', 'FIFO', 'raw_wait_min'),
    ('SPTFIFOmiss', 'SPT', 'FIFO', 'raw_miss_pct'),
    ('RescueFIFOwait', 'SPT+Rescue', 'FIFO', 'raw_wait_min'),
    ('RescueFIFOmiss', 'SPT+Rescue', 'FIFO', 'raw_miss_pct'),
    ('EDFFIFOwait', 'EDF', 'FIFO', 'raw_wait_min'),
    ('EDFFIFOmiss', 'EDF', 'FIFO', 'raw_miss_pct'),
    ('RandomFIFOmiss', 'Random', 'FIFO', 'raw_miss_pct'),
    ('SPTEDFmiss', 'SPT', 'EDF', 'raw_miss_pct'),
    ('RHFIFOwait', 'RH', 'FIFO', 'raw_wait_min'),
    ('RHFIFOmiss', 'RH', 'FIFO', 'raw_miss_pct'),
    ('RHRescueWait', 'RH', 'SPT+Rescue', 'raw_wait_min'),
    ('RHRescueMiss', 'RH', 'SPT+Rescue', 'raw_miss_pct'),
    ('RHCADRmiss', 'RH', 'CADR', 'raw_miss_pct'),
]


def _load():
    with open(CGRSP / 'Outputs/phase3/statistical.json', encoding='utf-8') as f:
        p3 = json.load(f)
    with open(CGRSP / 'Outputs/phase4/monthly_results.json', encoding='utf-8') as f:
        p4 = json.load(f)
    with open(CGRSP / 'Outputs/runpod_scheduler/report.json', encoding='utf-8') as f:
        p2_raw = json.load(f)
    with open(CGRSP / 'Outputs/phase3/sensitivity.json', encoding='utf-8') as f:
        sens = json.load(f)
    with open(CGRSP / 'Outputs/phase3/deadline.json', encoding='utf-8') as f:
        dl = json.load(f)
    with open(CGRSP / 'Outputs/phase3/cr_sensitivity.json', encoding='utf-8') as f:
        cr_sens = json.load(f)
    try:
        with open(CGRSP / 'Outputs/phase3/beta_sensitivity.json', encoding='utf-8') as f:
            beta = json.load(f)['results']
    except FileNotFoundError:
        beta = None

    s3 = p3['schedulers']
    s4 = p4['summary']
    p2 = p2_raw['results']
    s_hec = sens['sensitivity']['fleet5_typehectic']['schedulers']
    s_srg = sens['sensitivity']['fleet5_typesurge']['schedulers']
    dl5 = dl['results']['fleet5']
    tt = {(e['comparison'], e['metric']): e['test'] for e in p3['t_tests']}

    def _dlm(frac_key, sched):
        r = dl5[frac_key]['schedulers'][sched]
        return round(r.get('avg_miss_pct', r.get('deadline_miss_pct', 0)), 2)

    # Effective miss (unfinished jobs counted as missed), same estimator as
    # generate_data._beta_eff_miss. The x0.50 bound covers the six policies the
    # table displays; the x0.75 range covers every policy simulated.
    _beta_table_scheds = ['FIFO', 'SPT', 'EDF', 'SPT+Rescue', 'RH', 'CADR']

    def _beta_eff(scale, sched):
        return beta[scale][sched]['eff_miss_pct_mean']

    beta_keys = {}
    if beta is not None:
        three_q = [_beta_eff('0.75', s) for s in beta['0.75']]
        beta_keys = {
            'Beta_half_min_eff_miss': round(
                min(_beta_eff('0.50', s) for s in _beta_table_scheds), 1),
            'Beta_threeq_min_eff_miss': round(min(three_q), 1),
            'Beta_threeq_max_eff_miss': round(max(three_q), 1),
        }

    return {
        **{f"{st}{suf}": v for st, a, b, k in _MD_ROWS
           for suf, v in zip(("MD", "MDLo", "MDHi"), _mdci(s3[a][k], s3[b][k]))},
        **beta_keys,
        # Phase 3 means
        'SPT_wait':           round(s3['SPT']['wait_min']['mean'], 2),
        'SPT_miss':           round(s3['SPT']['miss_pct']['mean'], 2),
        'EDF_wait':           round(s3['EDF']['wait_min']['mean'], 2),
        'EDF_miss':           round(s3['EDF']['miss_pct']['mean'], 2),
        'Rescue_wait':        round(s3['SPT+Rescue']['wait_min']['mean'], 2),
        'Rescue_miss':        round(s3['SPT+Rescue']['miss_pct']['mean'], 2),
        'FIFO_wait':          round(s3['FIFO']['wait_min']['mean'], 2),
        'FIFO_miss':          round(s3['FIFO']['miss_pct']['mean'], 2),
        # t-test effect sizes
        'SPT_FIFO_wait_d':    round(tt[('SPT vs FIFO', 'avg_wait_min')]['cohen_d'], 3),
        'SPT_FIFO_miss_d':    round(tt[('SPT vs FIFO', 'deadline_miss_pct')]['cohen_d'], 3),
        'EDF_FIFO_wait_d':    round(tt[('EDF vs FIFO', 'avg_wait_min')]['cohen_d'], 3),
        'EDF_FIFO_miss_d':    round(tt[('EDF vs FIFO', 'deadline_miss_pct')]['cohen_d'], 3),
        'Rescue_FIFO_wait_d': round(tt[('SPT+Rescue vs FIFO', 'avg_wait_min')]['cohen_d'], 3),
        'Rescue_FIFO_miss_d': round(tt[('SPT+Rescue vs FIFO', 'deadline_miss_pct')]['cohen_d'], 3),
        # RH Phase 3
        'RH_wait':        round(s3['RH']['wait_min']['mean'], 2),
        'RH_miss':        round(s3['RH']['miss_pct']['mean'], 2),
        'RH_FIFO_wait_d': round(tt[('RH vs FIFO', 'avg_wait_min')]['cohen_d'], 3),
        'RH_FIFO_miss_d': round(tt[('RH vs FIFO', 'deadline_miss_pct')]['cohen_d'], 3),
        'RH_Rescue_miss_d': round(tt[('RH vs SPT+Rescue', 'deadline_miss_pct')]['cohen_d'], 3),
        # RH Phase 4 monthly
        'P4_RH_wait':     round(s4['RH']['avg_wait_min']['mean'], 2),
        'P4_RH_miss':     round(s4['RH']['deadline_miss_pct']['mean'], 2),
        # SPT+Rescue vs SPT miss Cohen d (not in t_tests JSON — computed from pooled std)
        'Rescue_SPT_miss_d':  round(
            (s3['SPT+Rescue']['miss_pct']['mean'] - s3['SPT']['miss_pct']['mean']) /
            math.sqrt((s3['SPT+Rescue']['miss_pct']['std']**2 + s3['SPT']['miss_pct']['std']**2) / 2),
            3),
        # Random vs FIFO Cohen d and Phase 3 mean miss
        'Random_FIFO_miss_d': round(tt[('Random vs FIFO', 'deadline_miss_pct')]['cohen_d'], 3),
        'Random_miss':        round(s3['Random']['miss_pct']['mean'], 2),
        'SPT_EDF_miss_p':     round(tt[('SPT vs EDF', 'deadline_miss_pct')]['p'], 3),
        # Phase 4 monthly
        'P4_SPT_wait':        round(s4['SPT']['avg_wait_min']['mean'], 2),
        'P4_SPT_miss':        round(s4['SPT']['deadline_miss_pct']['mean'], 2),
        'P4_Rescue_wait':     round(s4['SPT+Rescue']['avg_wait_min']['mean'], 2),
        'P4_Rescue_miss':     round(s4['SPT+Rescue']['deadline_miss_pct']['mean'], 2),
        'P4_EDF_wait':        round(s4['EDF']['avg_wait_min']['mean'], 2),
        'P4_EDF_miss':        round(s4['EDF']['deadline_miss_pct']['mean'], 2),
        'P4_FIFO_miss':       round(s4['FIFO']['deadline_miss_pct']['mean'], 2),
        'P4_LCF_miss':        round(s4['LCF']['deadline_miss_pct']['mean'], 2),
        'P4_Balanced_miss':   round(s4['Balanced']['deadline_miss_pct']['mean'], 2),
        'P4_Random_miss':     round(s4['Random']['deadline_miss_pct']['mean'], 2),
        # Phase 3 LCF/Balanced wait (used in FIFO/LCF/Balanced cluster range)
        'LCF_wait':           round(s3['LCF']['wait_min']['mean'], 2),
        'Balanced_wait':      round(s3['Balanced']['wait_min']['mean'], 2),
        # Phase 4 hectic-day sub-rates
        'P4_hec_SPT_miss':    round(s4['SPT']['hectic_miss_pct']['mean'], 2),
        'P4_hec_Rescue_miss': round(s4['SPT+Rescue']['hectic_miss_pct']['mean'], 2),
        'P4_hec_FIFO_miss':   round(s4['FIFO']['hectic_miss_pct']['mean'], 2),
        # Phase 2 (seed 42 single run)
        'P2_SPT_wait':    round(p2['SPT']['timing_metrics']['avg_waiting_time'] / 60, 2),
        'P2_SPT_miss':    round(p2['SPT']['job_counts']['deadline_miss_rate_pct'], 2),
        'P2_EDF_wait':    round(p2['EDF']['timing_metrics']['avg_waiting_time'] / 60, 2),
        'P2_EDF_miss':    round(p2['EDF']['job_counts']['deadline_miss_rate_pct'], 2),
        'P2_Rescue_wait': round(p2['SPT+Rescue']['timing_metrics']['avg_waiting_time'] / 60, 2),
        'P2_Rescue_miss': round(p2['SPT+Rescue']['job_counts']['deadline_miss_rate_pct'], 2),
        'P2_FIFO_miss':   round(p2['FIFO']['job_counts']['deadline_miss_rate_pct'], 2),
        'P2_util_pct':    round(p2['SPT']['utilization_legacy']['average'] * 100, 1),
        # Sensitivity: hectic (30-seed mean, 1dp)
        'SensHec_SPT_wait':    round(s_hec['SPT']['avg_wait_min'], 1),
        'SensHec_SPT_miss':    round(s_hec['SPT']['avg_miss_pct'], 1),
        'SensHec_EDF_wait':    round(s_hec['EDF']['avg_wait_min'], 1),
        'SensHec_EDF_miss':    round(s_hec['EDF']['avg_miss_pct'], 1),
        'SensHec_Rescue_wait': round(s_hec['SPT+Rescue']['avg_wait_min'], 1),
        'SensHec_Rescue_miss': round(s_hec['SPT+Rescue']['avg_miss_pct'], 1),
        'SensHec_FIFO_miss':   round(s_hec['FIFO']['avg_miss_pct'], 1),
        # Sensitivity: surge (30-seed mean, 1dp)
        'SensSrg_SPT_miss':    round(s_srg['SPT']['avg_miss_pct'], 1),
        'SensSrg_EDF_miss':    round(s_srg['EDF']['avg_miss_pct'], 1),
        'SensSrg_Rescue_miss': round(s_srg['SPT+Rescue']['avg_miss_pct'], 1),
        # Deadline tightness (2dp)
        'Dl_10_EDF':  _dlm('tight10pct', 'EDF'),
        'Dl_30_EDF':  _dlm('tight30pct', 'EDF'),
        'Dl_50_EDF':  _dlm('tight50pct', 'EDF'),
        'Dl_70_EDF':  _dlm('tight70pct', 'EDF'),
        'Dl_70_SPT':  _dlm('tight70pct', 'SPT'),
        'Dl_90_EDF':  _dlm('tight90pct', 'EDF'),
        'Dl_90_SPT':  _dlm('tight90pct', 'SPT'),
        # Baseline (FIFO/LCF/Balanced) range endpoints for deadline sweep
        'Dl_10_FIFO': _dlm('tight10pct', 'FIFO'),
        'Dl_10_LCF':  _dlm('tight10pct', 'LCF'),
        'Dl_90_FIFO': _dlm('tight90pct', 'FIFO'),
        'Dl_90_Balanced': _dlm('tight90pct', 'Balanced'),
        # Phase 3 CADR
        'CADR_wait':  round(s3['CADR']['wait_min']['mean'], 2),
        'CADR_miss':  round(s3['CADR']['miss_pct']['mean'], 2),
        'CADR_cost':  round(s3['CADR']['cost_usd']['mean'], 2),
        'CADR_tard':  round(s3['CADR']['avg_tardiness_min']['mean'], 2),
        # Phase 3 tardiness by scheduler
        'FIFO_tard':   round(s3['FIFO']['avg_tardiness_min']['mean'], 2),
        'SPT_tard':    round(s3['SPT']['avg_tardiness_min']['mean'], 2),
        'EDF_tard':    round(s3['EDF']['avg_tardiness_min']['mean'], 2),
        'Rescue_tard': round(s3['SPT+Rescue']['avg_tardiness_min']['mean'], 2),
        'RH_tard':     round(s3['RH']['avg_tardiness_min']['mean'], 2),
        # Phase 4 CADR
        'P4_CADR_wait': round(s4['CADR']['avg_wait_min']['mean'], 2),
        'P4_CADR_miss': round(s4['CADR']['deadline_miss_pct']['mean'], 2),
        'P4_CADR_tard': round(s4['CADR']['avg_tardiness_min']['mean'], 2),
        # Phase 4 tardiness by scheduler
        'P4_RH_tard':     round(s4['RH']['avg_tardiness_min']['mean'], 2),
        'P4_SPT_tard':    round(s4['SPT']['avg_tardiness_min']['mean'], 2),
        'P4_EDF_tard':    round(s4['EDF']['avg_tardiness_min']['mean'], 2),
        'P4_Rescue_tard': round(s4['SPT+Rescue']['avg_tardiness_min']['mean'], 2),
        'P4_FIFO_tard':   round(s4['FIFO']['avg_tardiness_min']['mean'], 2),
        # Sensitivity hectic CADR (1dp, matching existing SensHec pattern)
        'SensHec_CADR_wait': round(s_hec['CADR']['avg_wait_min'], 1),
        'SensHec_CADR_miss': round(s_hec['CADR']['avg_miss_pct'], 1),
        # Ablation: CADR-order-only
        'CADROO_wait': round(p3['ablation_extra']['CADR-order-only']['wait_min']['mean'], 2),
        'CADROO_miss': round(p3['ablation_extra']['CADR-order-only']['miss_pct']['mean'], 2),
        'CADROO_tard': round(p3['ablation_extra']['CADR-order-only']['avg_tardiness_min']['mean'], 2),
        # CR sensitivity: best (min) miss and spread (max - min)
        'CRStar_best_miss':   round(min(v['avg_miss_pct'] for v in cr_sens['results'].values()), 2),
        'CRStar_miss_spread': round(
            max(v['avg_miss_pct'] for v in cr_sens['results'].values()) -
            min(v['avg_miss_pct'] for v in cr_sens['results'].values()), 2),
        # CR* default (the shipped threshold) and how far it sits above the
        # sweep's best point
        'CRStar_default_miss': round(cr_sens['results']['3.0']['avg_miss_pct'], 2),
        'CRStar_default_gap': round(
            cr_sens['results']['3.0']['avg_miss_pct'] -
            min(v['avg_miss_pct'] for v in cr_sens['results'].values()), 2),
        # p-values the prose leans on by name
        'RH_Rescue_miss_p':   round(tt[('RH vs SPT+Rescue', 'deadline_miss_pct')]['p'], 3),
        'RH_SPT_wait_p':      round(tt[('RH vs SPT', 'avg_wait_min')]['p'], 3),
        'CADR_Rescue_wait_p': round(tt[('CADR vs SPT+Rescue', 'avg_wait_min')]['p'], 3),
    }


# ── Numeric checks ────────────────────────────────────────────────────────────

def _build_checks(data):
    """Return list of (description, expected_str, [bad_literal, ...])."""
    checks = []

    def chk(desc, key, *extra_bad):
        val = str(data[key])
        bad = [val] + list(extra_bad)
        checks.append((desc, val, bad))

    # Phase 3
    chk('SPT Phase3 wait (min)', 'SPT_wait')
    chk('SPT Phase3 miss (%)',   'SPT_miss')
    chk('EDF Phase3 miss (%)',   'EDF_miss')
    chk('Rescue Phase3 wait',    'Rescue_wait')
    chk('Rescue Phase3 miss',    'Rescue_miss')
    chk('FIFO Phase3 wait',      'FIFO_wait')
    chk('FIFO Phase3 miss',      'FIFO_miss')
    chk('SPT vs FIFO wait d',    'SPT_FIFO_wait_d')
    chk('SPT vs FIFO miss d',    'SPT_FIFO_miss_d')
    chk('EDF vs FIFO wait d',    'EDF_FIFO_wait_d')
    chk('EDF vs FIFO miss d',    'EDF_FIFO_miss_d')
    chk('Rescue vs FIFO miss d', 'Rescue_FIFO_miss_d')
    chk('Rescue vs FIFO wait d', 'Rescue_FIFO_wait_d')
    chk('SPT vs EDF miss p',     'SPT_EDF_miss_p')
    # Phase 4
    chk('P4 SPT wait',           'P4_SPT_wait')
    chk('P4 SPT miss',           'P4_SPT_miss')
    chk('P4 Rescue wait',        'P4_Rescue_wait')
    chk('P4 Rescue miss',        'P4_Rescue_miss')
    chk('P4 EDF wait',           'P4_EDF_wait')
    chk('P4 EDF miss',           'P4_EDF_miss')
    chk('P4 FIFO miss',          'P4_FIFO_miss')
    chk('P4 LCF miss',           'P4_LCF_miss')
    chk('P4 Balanced miss',      'P4_Balanced_miss')
    chk('P4 Random miss',        'P4_Random_miss')
    chk('P3 LCF wait',           'LCF_wait')
    chk('P3 Balanced wait',      'Balanced_wait')
    # Phase 4 hectic sub-rates
    chk('P4 hectic SPT miss',    'P4_hec_SPT_miss')
    chk('P4 hectic Rescue miss', 'P4_hec_Rescue_miss')
    chk('P4 hectic FIFO miss',   'P4_hec_FIFO_miss')
    # Phase 2
    chk('P2 SPT wait',           'P2_SPT_wait')
    chk('P2 SPT miss',           'P2_SPT_miss')
    chk('P2 EDF wait',           'P2_EDF_wait')
    chk('P2 EDF miss',           'P2_EDF_miss')
    chk('P2 Rescue wait',        'P2_Rescue_wait')
    chk('P2 Rescue miss',        'P2_Rescue_miss')
    chk('P2 FIFO miss',          'P2_FIFO_miss')
    chk('P2 utilization %',      'P2_util_pct')
    # Sensitivity hectic (stored as float with 1dp — check as string)
    chk('SensHec SPT wait',      'SensHec_SPT_wait')
    chk('SensHec SPT miss',      'SensHec_SPT_miss')
    chk('SensHec EDF wait',      'SensHec_EDF_wait')
    chk('SensHec EDF miss',      'SensHec_EDF_miss')
    chk('SensHec Rescue wait',   'SensHec_Rescue_wait')
    chk('SensHec Rescue miss',   'SensHec_Rescue_miss')
    chk('SensHec FIFO miss',     'SensHec_FIFO_miss')
    # Sensitivity surge
    chk('SensSrg SPT miss',      'SensSrg_SPT_miss')
    chk('SensSrg EDF miss',      'SensSrg_EDF_miss')
    chk('SensSrg Rescue miss',   'SensSrg_Rescue_miss')
    # Deadline tightness
    chk('Dl 10% EDF miss',       'Dl_10_EDF')
    chk('Dl 30% EDF miss',       'Dl_30_EDF')
    chk('Dl 50% EDF miss',       'Dl_50_EDF')
    chk('Dl 70% EDF miss',       'Dl_70_EDF')
    chk('Dl 70% SPT miss',       'Dl_70_SPT')
    chk('Dl 90% EDF miss',       'Dl_90_EDF')
    chk('Dl 90% SPT miss',       'Dl_90_SPT')
    # SPT+Rescue vs SPT miss d (was hardcoded as -0.10 — now macro RescueSPTmissD)
    chk('Rescue vs SPT miss d',  'Rescue_SPT_miss_d')
    chk('RH Phase3 wait (min)', 'RH_wait')
    chk('RH Phase3 miss (%)',   'RH_miss')
    # Random Phase 3 stats
    chk('Random Phase3 miss',    'Random_miss')
    chk('Random vs FIFO miss d', 'Random_FIFO_miss_d')
    # Deadline sweep baseline range endpoints
    chk('Dl 10% FIFO miss',      'Dl_10_FIFO')
    chk('Dl 10% LCF miss',       'Dl_10_LCF')
    chk('Dl 90% FIFO miss',      'Dl_90_FIFO')
    chk('Dl 90% Balanced miss',  'Dl_90_Balanced')
    return checks


def _scan_file(path, checks, data):
    """Return list of (line_no, line, description, expected) for numeric mismatches."""
    issues = []
    try:
        lines = path.read_text(encoding='utf-8').splitlines()
    except FileNotFoundError:
        print(f'  SKIP (not found): {path}')
        return issues

    macro_def_re = re.compile(r'\\newcommand\{')
    # Strip whole macro tokens that expand to numeric values, then scan the
    # remainder. Stripping (rather than skipping the line) means hardcoded
    # numbers on a line that ALSO contains a macro are still caught.
    macro_strip_re = re.compile(
        r'\\(?:SPT|EDF|Rescue|FIFO|PhFourHec|PhFour|PhTwo|SensHec|SensSrg|'
        r'DlTen|DlThirty|DlFifty|DlSeventy|DlNinety|Balanced|LCF|Random|'
        r'CADR|CRStar)[A-Za-z]*'
    )
    # Strip LaTeX currency amounts (e.g. "\$0.46/hr") before scanning: a
    # dollar-prefixed number is always a price or cost, never one of the
    # wait/miss/effect-size/p-value statistics these checks fingerprint, and
    # small prices like 0.46 collide with miss-rate fact values.
    currency_strip_re = re.compile(r'\\?\$\s*\d[\d,]*(?:\.\d+)?')
    # Strip LaTeX graphics/sizing directives and braced parameter-sweep sets
    # before scanning. Image widths ("[width=0.9\textwidth]") and sweep
    # enumerations ("\{0.1, 0.3, 0.5, 0.7, 0.9\}") contain decimals that collide
    # (by substring) with single-decimal fact values such as a 0.9% surge miss,
    # but are never the statistic a check fingerprints.
    graphics_strip_re = re.compile(
        r'\\includegraphics\[[^\]]*\]'
        r'|\\resizebox\{[^}]*\}'
        r'|(?:width|height|scale)\s*=\s*[0-9.]+\\?[A-Za-z]*'
    )
    sweepset_strip_re = re.compile(r'\\\{.*?\\\}')
    # Strip Poisson arrival-rate lambda values ("$\lambda=0.020$ j/s") and
    # diurnal availability multipliers ("$\times$1.3") before scanning: both
    # are simulation-environment parameters, never one of the wait/miss/
    # effect-size/p-value statistics these checks fingerprint, but their
    # decimals can coincidentally equal a fact value (e.g. lambda=0.002 j/s
    # colliding with an unrelated p-value of 0.002).
    lambda_strip_re = re.compile(r'\\lambda\s*=\s*[0-9.]+')
    multiplier_strip_re = re.compile(r'\$?\\times\$?\s*[0-9.]+')

    for lineno, line in enumerate(lines, 1):
        # Table rows are scanned like any other line. They used to be skipped
        # on the presence of an ampersand, which silently exempted 189 of the
        # 1,388 prose lines, every row of the hand-written t-test table among
        # them: exactly the place a hardcoded statistic is most likely to sit
        # and least likely to be re-read.
        if 'AUTO-GENERATED' in line or macro_def_re.search(line):
            continue
        scan_line = macro_strip_re.sub(' ', line)
        scan_line = currency_strip_re.sub(' ', scan_line)
        scan_line = graphics_strip_re.sub(' ', scan_line)
        scan_line = sweepset_strip_re.sub(' ', scan_line)
        scan_line = lambda_strip_re.sub(' ', scan_line)
        scan_line = multiplier_strip_re.sub(' ', scan_line)

        for desc, expected, bad_lits in checks:
            for lit in bad_lits:
                # Whole-number match: 9.6 must not fire inside 69.6 or 9.65.
                if re.search(r'(?<![\d.])' + re.escape(lit) + r'(?![\d])', scan_line):
                    issues.append((lineno, line.strip(), desc, expected))
                    break
    return issues


# ── Linguistic checks ─────────────────────────────────────────────────────────
#
# Catches the four recurring errors from git history:
#   1. SPT claimed as lowest-miss outside deadline-tightness context
#      (EDF is significantly better than SPT on miss, p=0.002 as of the 2026-07-19
#      complexity-mix fix; never "tied" — live values in stats.tex)
#   2. delta_j=1 described as a deadline miss (convention: delta_j=1 = on time)
#   3. delta_j=0 described as on-time (convention: delta_j=0 = deadline missed)
#   4. "EDF dominated by SPT on both metrics" — false; EDF's miss point estimate is below SPT's

_LING = [
    (
        # SPT (not SPT+Rescue) as the DIRECT subject of "achieves/has lowest miss".
        # Uses [^;.]{0,60} to prevent matching "lowest miss" from a different clause
        # on the same line (e.g. "SPT achieves lowest wait; EDF achieves lowest miss").
        # The Chinese alternation is not decoration: abstract_zh.tex sits in
        # LINGUISTIC_FILES, and until it was added every guard here was
        # English-only, so the identical banned claim written in Chinese
        # ("SPT達成最低違反率") passed the file unread.
        re.compile(
            r'\bSPT\b(?!\s*\+)\s+(?:achieves?|has|is)\b[^;.]{0,60}\blowest\s+miss'
            r'|SPT(?!\s*\+)[^。；]{0,30}最低[^。；]{0,10}'
            r'(?:截止期限)?(?:違反率|違反比率|逾期率)',
            re.IGNORECASE,
        ),
        # Safe contexts: deadline-tightness sweep (>= 70% tight), or "near-lowest" language.
        # "statistically tied" is deliberately NOT a safe context: as of the 2026-07-19
        # complexity-mix fix, EDF is significantly better than SPT on miss (p=0.002), so
        # a "tied" caveat next to an SPT-lowest-miss claim is itself the banned phrasing,
        # not a safe hedge.
        re.compile(
            r'tight|\bgeq\b|\\geq|f_\{.*tight|70\s*\\?%|near.?lowest'
            r'|截止期限比例|緊急比例|接近最低',
            re.IGNORECASE,
        ),
        'SPT-lowest-miss claim outside deadline-tightness context '
        '(Phase 3: EDF is significantly better than SPT on miss, p=0.002 — never "tied"; live values in generated/stats.tex)',
    ),
    (
        # SPT (not SPT+Rescue) claimed as Pareto-dominant in isolation. Bare SPT is
        # dominant on neither metric (wait is tied with RH, miss is significantly
        # worse than EDF's). Note RH itself does NOT cleanly Pareto-dominate
        # SPT+Rescue either (its miss advantage survives Holm correction, its wait
        # advantage does not) — never write "Pareto-dominates" for that pair.
        # Safe context: "jointly" or "SPT+Rescue" nearby, then it's a joint claim, which is fine.
        re.compile(
            r'\bSPT\b(?!\s*\+)[^;.]{0,80}?\bPareto.?(?:dominant|optimal|superior)\b',
            re.IGNORECASE,
        ),
        re.compile(r'\bjointly\b|SPT\+Rescue|SPT\s+and\s+SPT', re.IGNORECASE),
        'bare SPT claimed as Pareto-dominant '
        '(SPT dominates on neither metric; see generated/stats.tex for live values)',
    ),
    (
        # SPT (not SPT+Rescue) claimed as "uniquely" best or having the best/fewest misses.
        re.compile(
            r'\bSPT\b(?!\s*\+)[^;.]{0,80}?\b(?:uniquely|best|fewest|lowest)\b[^;.]{0,40}?\b(?:miss|deadline)',
            re.IGNORECASE,
        ),
        # Safe contexts: "best wait", "best throughput", "fewest wait", or deadline-tightness sweep
        re.compile(
            r'\bwait\b|\bthroughput\b|\blatency\b|tight|\bgeq\b|\\geq|70\s*\\?%',
            re.IGNORECASE,
        ),
        'SPT claimed as having best/fewest/lowest miss or deadline performance '
        '(SPT advantage is wait, not miss; EDF is significantly better than SPT on miss — live values in generated/stats.tex)',
    ),
    (
        # EDF claimed as dominated by SPT on BOTH metrics.
        # SPT dominates EDF on wait only; EDF is significantly BETTER than SPT on
        # miss (p=0.002 as of the 2026-07-19 complexity-mix fix), so "dominated on
        # both" is false in the opposite direction it used to be (it's not even a
        # near-tie anymore).
        re.compile(
            r'\bEDF\b.{0,80}?\bdominated?\s+by\s+SPT\b.{0,80}?\bboth\b',
            re.IGNORECASE,
        ),
        re.compile(r'(?!x)^', re.IGNORECASE),  # never safe
        'EDF claimed as dominated by SPT on both metrics '
        '(SPT dominates EDF on wait only; EDF is significantly better than SPT on miss — live values in generated/stats.tex)',
    ),
    (
        # Pattern: delta_j = 1 co-located with "miss" or "violat" within 80 chars
        # (catches "delta_j = 1 means deadline missed" type errors)
        # Chinese alternation for the same reason as the SPT guard above: the
        # Chinese abstract is checked by this function, and the inversion is
        # just as easy to write in Chinese. The Chinese terms sit outside the
        # \b group on purpose, since a word boundary never falls between two
        # Han characters.
        re.compile(
            r'\\delta_[j\d]\s*=\s*1\b.{0,80}?'
            r'(?:\b(?:miss(?:ed|es)?|violat|exceeded)\b|違反|逾期|超時)',
            re.IGNORECASE,
        ),
        # Safe context: the formula "1 - delta_j" or correct definition "1 if ... on time"
        re.compile(r'1\s*-\s*\\delta|on.?time|meets?\s+deadline|deadline\s+met'
                   r'|準時|如期|按時|未違反|符合截止期限', re.IGNORECASE),
        'delta_j=1 co-occurring with "miss/violated/exceeded" '
        '(convention: delta_j=1 means on-time; delta_j=0 means deadline missed)',
    ),
    (
        # Pattern: delta_j = 0 co-located with "on time" or "meets deadline" within 80 chars
        re.compile(
            r'\\delta_[j\d]\s*=\s*0\b.{0,80}?\b(?:on.?time|meets?\s+deadline|deadline\s+met)\b',
            re.IGNORECASE,
        ),
        # Safe context: "miss" nearby (correct: delta=0 near "miss" is fine)
        re.compile(r'\bmiss\b', re.IGNORECASE),
        'delta_j=0 co-occurring with "on-time/meets deadline" '
        '(convention: delta_j=0 means deadline missed; delta_j=1 means on-time)',
    ),
]


# Chinese prose carries no ". " for _split_sentences_with_spans to break on,
# so a Chinese paragraph would come back as one scope. Narrow further on the
# full-width terminators. This narrowing is deliberately kept out of
# _split_sentences_with_spans itself: that helper's exact segmentation is what
# the empirical-claim hashes were signed off against, and re-cutting those
# segments would invalidate every registered Chinese claim.
_ZH_TERMINATOR_RE = re.compile(r'[。！？；]')


def _ling_scope(line, sentences, pos):
    """The sentence containing character `pos`, Chinese clauses included."""
    start, end = next(((s, e) for s, e, _ in sentences if s <= pos < e),
                      (0, len(line)))
    for m in _ZH_TERMINATOR_RE.finditer(line, start, end):
        if m.end() <= pos:
            start = m.end()
        else:
            end = m.start()
            break
    return line[start:end]


def _check_linguistic(files):
    """Return list of (path, line_no, line, message) for linguistic issues.

    The safe-context test is scoped to the SENTENCE the banned pattern matched
    in, never the whole physical line. This thesis's .tex source is one line
    per paragraph (ch6's results paragraph is over 1,600 characters), so a
    line-scoped safe context let a single incidental "tight" anywhere in a
    paragraph switch off the SPT-lowest-miss guard for every sentence in it,
    which is precisely the paragraph the guard exists to police. A fixed
    character window either side of the match is not enough either: the excuse
    word can sit any distance away and still be in the same paragraph, so the
    sentence boundary is the only scope that tracks the actual claim.
    """
    issues = []
    for path in files:
        try:
            lines = path.read_text(encoding='utf-8').splitlines()
        except FileNotFoundError:
            continue
        for lineno, line in enumerate(lines, 1):
            if 'AUTO-GENERATED' in line:
                continue
            sentences = _split_sentences_with_spans(line)
            for pattern, safe_re, message in _LING:
                for m in pattern.finditer(line):
                    scope = _ling_scope(line, sentences, m.start())
                    if safe_re and safe_re.search(scope):
                        continue  # safe context in the matched sentence
                    issues.append((path, lineno, line.strip(), message))
                    break
    return issues


# ── Spelling checks ───────────────────────────────────────────────────────────
# Thesis uses British English; paper uses American English (CLAUDE.md).

_AMERICAN = [  # wrong in thesis (British-English) files
    (re.compile(r'\bminimiz\w*', re.IGNORECASE), 'American "minimiz…" — use "minimis…"'),
    (re.compile(r'\boptimiz\w*', re.IGNORECASE), 'American "optimiz…" — use "optimis…"'),
    (re.compile(r'\borganiz\w*', re.IGNORECASE), 'American "organiz…" — use "organis…"'),
    (re.compile(r'\bmodeling\b',  re.IGNORECASE), 'American "modeling" — use "modelling"'),
    (re.compile(r'\bbehavior\w*', re.IGNORECASE), 'American "behavior…" — use "behaviour…"'),
]

_BRITISH = [  # wrong in paper (American-English) files
    (re.compile(r'\bminimis\w*', re.IGNORECASE), 'British "minimis…" — use "minimiz…"'),
    (re.compile(r'\boptimis\w*', re.IGNORECASE), 'British "optimis…" — use "optimiz…"'),
    (re.compile(r'\borganis\w*', re.IGNORECASE), 'British "organis…" — use "organiz…"'),
    (re.compile(r'\bmodelling\b', re.IGNORECASE), 'British "modelling" — use "modeling"'),
    (re.compile(r'\bbehaviour\w*', re.IGNORECASE), 'British "behaviour…" — use "behavior…"'),
]


def _check_spelling(tex_files):
    """Return list of (path, lineno, line, message) for wrong-variant spellings."""
    issues = []
    for path in tex_files:
        if THESIS in path.parents:
            patterns = _AMERICAN
        elif PAPER in path.parents:
            patterns = _BRITISH
        else:
            continue
        try:
            lines = path.read_text(encoding='utf-8').splitlines()
        except FileNotFoundError:
            continue
        for lineno, line in enumerate(lines, 1):
            stripped = line.lstrip()
            if stripped.startswith('%') or 'AUTO-GENERATED' in line:
                continue
            for pattern, message in patterns:
                if pattern.search(line):
                    issues.append((path, lineno, line.strip(), message))
                    break
    return issues


# ── Macro-sync check ─────────────────────────────────────────────────────────
# Catches the case where stats.tex macros are stale relative to the simulation
# JSONs (e.g. after run_all.sh regenerates JSONs but stats.tex was not updated,
# or vice versa).  Only the thesis copy is checked (both copies are identical).

_MACRO_SYNC = [
    # (macro_name, data_key)
    ('SPTwait',        'SPT_wait'),
    ('SPTmiss',        'SPT_miss'),
    ('EDFmiss',        'EDF_miss'),
    ('FIFOwait',       'FIFO_wait'),
    ('FIFOmiss',       'FIFO_miss'),
    ('SPTFIFOwaitD',   'SPT_FIFO_wait_d'),
    ('SPTFIFOmissD',   'SPT_FIFO_miss_d'),
    ('SPTEDFmissP',    'SPT_EDF_miss_p'),
    ('PhFourSPTwait',  'P4_SPT_wait'),
    ('PhFourSPTmiss',  'P4_SPT_miss'),
    ('PhFourEDFmiss',  'P4_EDF_miss'),
    ('PhFourFIFOmiss', 'P4_FIFO_miss'),
    ('RHWait',        'RH_wait'),
    ('RHMiss',        'RH_miss'),
    ('PhFourRHWait',  'P4_RH_wait'),
    ('PhFourRHMiss',  'P4_RH_miss'),
    # Phase 3 CADR
    ('CADRwait',  'CADR_wait'),
    ('CADRmiss',  'CADR_miss'),
    ('CADRcost',  'CADR_cost'),
    ('CADRtard',  'CADR_tard'),
    # Phase 3 tardiness by scheduler
    ('FIFOtard',   'FIFO_tard'),
    ('SPTtard',    'SPT_tard'),
    ('EDFtard',    'EDF_tard'),
    ('RescueTard', 'Rescue_tard'),
    ('RHtard',     'RH_tard'),
    # Phase 4 CADR
    ('PhFourCADRwait', 'P4_CADR_wait'),
    ('PhFourCADRMiss', 'P4_CADR_miss'),
    ('PhFourCADRTard', 'P4_CADR_tard'),
    # Phase 4 tardiness by scheduler
    ('PhFourRHtard',     'P4_RH_tard'),
    ('PhFourSPTtard',    'P4_SPT_tard'),
    ('PhFourEDFtard',    'P4_EDF_tard'),
    ('PhFourRescueTard', 'P4_Rescue_tard'),
    ('PhFourFIFOtard',   'P4_FIFO_tard'),
    # Sensitivity hectic CADR
    ('SensHecCADRwait', 'SensHec_CADR_wait'),
    ('SensHecCADRmiss', 'SensHec_CADR_miss'),
    # Ablation: CADR-order-only
    ('CADROOwait', 'CADROO_wait'),
    ('CADROOmiss', 'CADROO_miss'),
    ('CADROOtard', 'CADROO_tard'),
    # CR sensitivity
    ('CRStarBestMiss',   'CRStar_best_miss'),
    ('CRStarMissSpread', 'CRStar_miss_spread'),
    ('CRStarDefaultMiss', 'CRStar_default_miss'),
    ('CRStarDefaultGap',  'CRStar_default_gap'),
    # RH vs SPT+Rescue miss p-value: was typed as a bare 0.004 in four places
    # (results table and conclusion included) purely because no macro existed.
    ('RHRescueMissP', 'RH_Rescue_miss_p'),
    # The two wait ties the prose asserts in ten places. Both are now tested,
    # and a macro is the only thing that keeps the word "tied" answerable.
    ('RHSPTwaitP',      'RH_SPT_wait_p'),
    ('CADRRescueWaitP', 'CADR_Rescue_wait_p'),
    # all 39 paired mean-difference macros of tab:ttests, recomputed independently
    *[(f"{st}{suf}", f"{st}{suf}") for st, *_ in _MD_ROWS for suf in ("MD", "MDLo", "MDHi")],
    # Beta-sweep row extrema (these live in stats_audit.tex)
    ('BetaHalfMinEffMiss',        'Beta_half_min_eff_miss'),
    ('BetaThreeQuarterMinEffMiss', 'Beta_threeq_min_eff_miss'),
    ('BetaThreeQuarterMaxEffMiss', 'Beta_threeq_max_eff_miss'),
]

def _check_macro_sync(data):
    """Return list of (macro, expected, actual) where the generated macros are
    out of sync. Both generated macro files are read: stats_audit.tex is as
    load-bearing in the prose as stats.tex and was previously unchecked, so a
    macro defined there reported as "undefined" rather than being compared."""
    text = ''
    for fname in ('stats.tex', 'stats_audit.tex'):
        try:
            text += (THESIS / 'generated' / fname).read_text(encoding='utf-8') + '\n'
        except FileNotFoundError:
            pass
    if not text:
        return [('stats.tex', 'exists', 'missing')]

    import re as _re
    issues = []
    for macro, key in _MACRO_SYNC:
        expected_val = data[key]
        m = _re.search(r'\\newcommand\{\\' + macro + r'\}\{([^}]+)\}', text)
        if not m:
            issues.append((macro, str(expected_val), 'undefined'))
        else:
            try:
                actual_val = float(m.group(1))
                if abs(actual_val - expected_val) > 0.005:
                    issues.append((macro, str(expected_val), m.group(1)))
            except ValueError:
                issues.append((macro, str(expected_val), m.group(1)))
    return issues


# ── Deadline-crossover check ──────────────────────────────────────────────────
# Regression guard for the 2026-06-15 incident: the deadline-sweep crossover was
# stated as "EDF lowest at f_tight <= 50%, SPT lowest at >= 70%" in 10 places,
# while the sweep did not support it (root cause: hardcoded bolding in
# generate_data.py). This check derives the crossover from the JSON and flags
# any tight-fraction threshold in the prose that disagrees.
#
# The crossover is now read off the PAIRED SPT-vs-EDF test the sweep stores at
# every point, not off the point estimates. Ranking two means tells you which
# number is smaller and nothing about whether the difference is real: at
# f_tight = 50% EDF's mean is below SPT's, yet the paired test over the same 30
# seeds gives p = 0.69, so a point-estimate reading would have the prose
# announce a winner where the paired test does not resolve the difference.
# The claim the data supports has three parts: EDF significantly lower up to
# edf_sig_max, an unresolved midpoint, SPT significantly lower from spt_sig_min on.

_THRESH_RE = re.compile(r'\\(leq|geq)\s*\$?\s*(\d+)\s*\\?%')
_DEADLINE_CTX_RE = re.compile(
    r'tight[- ]deadline fraction|tight fraction|f_\{\\text\{tight|截止期限比例'
)
# Phrasings that count as acknowledging the unresolved middle. Any of
# these in the same sentence as a full crossover claim satisfies the third part.
# A "statistical tie" or "indistinguishable" overstates what the paired test proves.
_TIE_ACK_RE = re.compile(
    r'no significant|not significant|does not resolve (?:their|the) difference|'
    r'cannot be distinguish|cannot separate|無法區分|不顯著|無顯著|無差異',
    re.IGNORECASE,
)


# ── Baseline and fleet-mix table checks ──────────────────────────────────────
# The two journal tables are generated, but nothing else in this validator
# reads them. A hand edit or a stale regeneration would otherwise pass. These
# checks rebuild every cell from the JSON and pin the identity, duplicate-test
# and ordering-flag claims the new prose makes.

def _clean_table_cell(cell):
    cell = cell.strip()
    match = re.fullmatch(r'\\textbf\{(.*)\}', cell)
    return match.group(1).strip() if match else cell


def _table_body(path):
    """Return the data rows of a generated fragment, cells cleaned."""
    rows = []
    in_body = False
    for line in path.read_text(encoding='utf-8').splitlines():
        stripped = line.strip()
        if stripped.startswith(r'\midrule'):
            in_body = True
            continue
        if stripped.startswith(r'\bottomrule'):
            break
        if in_body and stripped:
            if stripped.endswith(r'\\'):
                stripped = stripped[:-2].rstrip()
            rows.append([_clean_table_cell(c) for c in stripped.split('&')])
    return rows


def _betai(a, b, x):
    """Regularised incomplete beta I_x(a, b); same implementation as generate_data.py."""
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


def _paired_pooled_d(a, b):
    """Two-sided paired t p-value and pooled-SD Cohen's d for a - b."""
    n = len(a)
    diff = [x - y for x, y in zip(a, b)]
    md = sum(diff) / n
    sd_diff = (sum((d - md) ** 2 for d in diff) / (n - 1)) ** 0.5
    t = md / (sd_diff / n ** 0.5)
    df = n - 1
    p = _betai(df / 2.0, 0.5, df / (df + t * t))
    mean_a = sum(a) / n
    mean_b = sum(b) / n
    var_a = sum((x - mean_a) ** 2 for x in a) / (n - 1)
    var_b = sum((x - mean_b) ** 2 for x in b) / (n - 1)
    pooled = ((var_a + var_b) / 2.0) ** 0.5
    return p, md / pooled


def _check_baseline_fleet_tables():
    issues = []
    b_path = CGRSP / 'Outputs/phase3/baselines.json'
    f_path = CGRSP / 'Outputs/phase3/fleet_mix.json'
    b_tex = THESIS / 'generated/table_baselines.tex'
    f_tex = THESIS / 'generated/table_fleet_mix.tex'
    for path in (b_path, f_path, b_tex, f_tex):
        if not path.exists():
            issues.append(f'missing {path.relative_to(ROOT)}')
    if issues:
        return issues

    b = json.load(open(b_path, encoding='utf-8'))
    s = b['schedulers']
    meta = b['metadata']
    if (meta['n_seeds'], meta['n_jobs'], meta['concurrency']) != (30, 950, 5):
        issues.append(f'baselines metadata changed: {meta["n_seeds"]} seeds, '
                      f'{meta["n_jobs"]} jobs, C_max={meta["concurrency"]}')
    if abs(meta['arrival_rate'] - 0.1) > 1e-12 or abs(meta['duration_hr'] - 24.0) > 1e-12:
        issues.append('baselines workload metadata changed from lambda=0.100, 24 h')

    metrics = ['raw_wait_min', 'raw_miss_pct', 'raw_cost_usd', 'raw_util_pct',
               'raw_avg_tardiness_min', 'raw_p95_tardiness_min']
    if any(s['HEFT'][m] != s['MaxMin'][m] for m in metrics):
        issues.append('HEFT and MaxMin are no longer identical on every per-seed array')

    unique = {(t['comparison'].replace('HEFT', 'MaxMin'), t['metric'])
              for t in b['t_tests']}
    if len(b['t_tests']) != 30 or len(unique) != 24:
        issues.append(f'Holm family is {len(b["t_tests"])} tests with '
                      f'{len(unique)} unique comparisons, expected 30/24')
    if any('p_holm' not in t['test'] for t in b['t_tests']):
        issues.append('a baseline t-test has no p_holm value')

    rows = _table_body(b_tex)
    expected_order = ['HEFT', 'MinMin', 'Bandit', 'Bandit-random', 'FIFO', 'RH', 'CADR']
    labels = {'HEFT': 'HEFT $=$ MaxMin'}
    if len(rows) != len(expected_order):
        issues.append(f'{b_tex.relative_to(ROOT)} has {len(rows)} data rows, '
                      f'expected {len(expected_order)}')
    for row, name in zip(rows, expected_order):
        if len(row) != 4:
            issues.append(f'{b_tex.relative_to(ROOT)} row {name} has {len(row)} cells')
            continue
        if row[0] != labels.get(name, name):
            issues.append(f'{b_tex.relative_to(ROOT)} row {name} label is {row[0]!r}')
        expected = [
            f'{s[name]["wait_min"]["mean"]:.2f}',
            f'{s[name]["miss_pct"]["mean"]:.2f}',
            f'{s[name]["avg_tardiness_min"]["mean"]:.2f}',
        ]
        if row[1:] != expected:
            issues.append(f'{b_tex.relative_to(ROOT)} row {name} is {row[1:]}, '
                          f'JSON expects {expected}')

    f = json.load(open(f_path, encoding='utf-8'))
    fmeta = f['metadata']
    if fmeta['n_seeds'] != 10 or fmeta['n_jobs'] != 950 or fmeta['concurrency'] != 5:
        issues.append(f'fleet-mix metadata changed: {fmeta["n_seeds"]} seeds, '
                      f'{fmeta["n_jobs"]} jobs, C_max={fmeta["concurrency"]}')
    if fmeta['policies'] != ['FIFO', 'SPT', 'EDF', 'SPT+Rescue', 'RH', 'CADR']:
        issues.append(f'fleet-mix policies changed: {fmeta["policies"]}')
    anchor = f['anchor_equivalence']
    if not anchor['passed'] or anchor['mismatch_count'] != 0:
        issues.append('fleet-mix anchor no longer reproduces statistical.json')
    if f['ordering_stable'] is not False:
        issues.append('fleet-mix ordering_stable is no longer false')

    two_lowest = {mix: v['rh_cadr_two_lowest'] for mix, v in f['ordering_checks'].items()}
    expected_two_lowest = {
        'all-types': True, '3090-heavy': True, 'A5000-heavy': False,
        'A4000-heavy': False, '3090-only': True, 'A5000-only': True,
    }
    if two_lowest != expected_two_lowest:
        issues.append(f'fleet-mix two-lowest map changed: {two_lowest}')
    for mix, check in f['ordering_checks'].items():
        if not all(check['deadline_aware_below_fifo'].values()):
            issues.append(f'{mix}: a deadline-aware policy is not below FIFO')

    a5 = f['mixes']['A5000-heavy']['policies']
    p, d = _paired_pooled_d(a5['RH']['raw_miss_pct'], a5['CADR']['raw_miss_pct'])
    if abs(p - 0.282) > 0.002 or abs(d - 0.287) > 0.002:
        issues.append(f'A5000-heavy RH vs CADR is p={p:.4f}, d={d:.3f}; '
                      'prose says p=0.282, d=0.287')
    p2, d2 = _paired_pooled_d(a5['RH']['raw_miss_pct'], a5['SPT+Rescue']['raw_miss_pct'])
    if abs(p2 - 0.916) > 0.002 or abs(d2 - 0.041) > 0.002:
        issues.append(f'A5000-heavy RH vs SPT+Rescue is p={p2:.4f}, d={d2:.3f}; '
                      'prose says p=0.916, d=0.041')
    a4 = f['mixes']['A4000-heavy']['policies']
    p3, d3 = _paired_pooled_d(a4['RH']['raw_miss_pct'], a4['CADR']['raw_miss_pct'])
    if p3 >= 0.001 or abs(d3 - 6.21) > 0.02:
        issues.append(f'A4000-heavy RH vs CADR is p={p3:.6f}, d={d3:.3f}; '
                      'prose says p<0.001, d=6.21')
    p4, _ = _paired_pooled_d(a4['RH']['raw_miss_pct'], a4['SPT+Rescue']['raw_miss_pct'])
    if abs(p4 - 0.0038) > 0.0002:
        issues.append(f'A4000-heavy RH vs SPT+Rescue is p={p4:.5f}; '
                      'prose says p=0.0038')

    mix_order = ['all-types', '3090-heavy', 'A5000-heavy', 'A4000-heavy',
                 '3090-only', 'A5000-only']
    argmin = {
        mix: min(fmeta['policies'],
                 key=lambda p: f['mixes'][mix]['policies'][p]['miss_pct']['mean'])
        for mix in mix_order
    }
    expected_argmin = {
        'all-types': 'RH', '3090-heavy': 'RH', 'A5000-heavy': 'CADR',
        'A4000-heavy': 'CADR', '3090-only': 'RH', 'A5000-only': 'RH',
    }
    if argmin != expected_argmin:
        issues.append(f'per-mix miss minima changed: {argmin}')
    for p in fmeta['policies']:
        best_mix = min(mix_order, key=lambda m: f['mixes'][m]['policies'][p]['miss_pct']['mean'])
        if best_mix != 'A4000-heavy':
            issues.append(f'{p} no longer has its lowest miss in A4000-heavy')
    cadr_gap = max(
        f['mixes'][m]['policies']['CADR']['miss_pct']['mean']
        - min(f['mixes'][m]['policies'][p]['miss_pct']['mean'] for p in fmeta['policies'])
        for m in mix_order
    )
    rh_a4000_gap = (f['mixes']['A4000-heavy']['policies']['RH']['miss_pct']['mean']
                    - f['mixes']['A4000-heavy']['policies']['CADR']['miss_pct']['mean'])
    if abs(cadr_gap - 1.01) > 0.005 or abs(rh_a4000_gap - 5.39) > 0.005:
        issues.append(f'gap claims changed: CADR max {cadr_gap:.2f}, '
                      f'RH A4000-heavy {rh_a4000_gap:.2f}')

    slot_order = ['RTX_3090', 'RTX_A5000', 'RTX_A4500', 'RTX_A4000']
    display = {
        'all-types': 'All-types', '3090-heavy': '3090-heavy',
        'A5000-heavy': 'A5000-heavy', 'A4000-heavy': 'A4000-heavy',
        '3090-only': '3090-only', 'A5000-only': 'A5000-only',
    }
    rows = _table_body(f_tex)
    mixes = list(display)
    if len(rows) != len(mixes):
        issues.append(f'{f_tex.relative_to(ROOT)} has {len(rows)} data rows, '
                      f'expected {len(mixes)}')
    for row, mix in zip(rows, mixes):
        if len(row) != 8:
            issues.append(f'{f_tex.relative_to(ROOT)} row {mix} has {len(row)} cells')
            continue
        counts = f['mixes'][mix]['slot_counts']
        slots = '/'.join(str(counts.get(t, 0)) for t in slot_order)
        if row[:2] != [display[mix], slots]:
            issues.append(f'{f_tex.relative_to(ROOT)} row {mix} label is {row[:2]!r}')
        expected = [f'{f["mixes"][mix]["policies"][p]["miss_pct"]["mean"]:.2f}'
                    for p in fmeta['policies']]
        if row[2:] != expected:
            issues.append(f'{f_tex.relative_to(ROOT)} row {mix} is {row[2:]}, '
                          f'JSON expects {expected}')

    for path in TEX_FILES:
        for lineno, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
            if re.search(r'p\s*=\s*0\.000\b', line):
                issues.append(f'{path.relative_to(ROOT)}:{lineno} writes p=0.000; '
                              'use p<0.001')
    return issues


def _deadline_crossover():
    """Return (edf_sig_max, spt_sig_min, tie_points) from deadline.json fleet5,
    using the stored paired SPT-vs-EDF miss test at each sweep point. The test
    is signed SPT minus EDF, so t > 0 means EDF is the lower of the two."""
    with open(CGRSP / 'Outputs/phase3/deadline.json', encoding='utf-8') as f:
        dl5 = json.load(f)['results']['fleet5']
    edf_sig, spt_sig, ties = [], [], []
    for p in (10, 30, 50, 70, 90):
        point = dl5[f'tight{p}pct']
        test = point.get('spt_vs_edf_miss')
        if test is None:
            # No paired test at this point: fall back to the row minimum, which
            # is weaker but better than dropping the point silently.
            miss = {s: r.get('avg_miss_pct', r.get('deadline_miss_pct', 0))
                    for s, r in point['schedulers'].items()}
            winner = min(miss, key=miss.get)
            (edf_sig if winner == 'EDF' else spt_sig if winner == 'SPT'
             else ties).append(p)
            continue
        if test['p'] >= 0.05:
            ties.append(p)
        elif test['t'] > 0:
            edf_sig.append(p)
        else:
            spt_sig.append(p)
    return (max(edf_sig) if edf_sig else None,
            min(spt_sig) if spt_sig else None,
            ties)


def _check_deadline_crossover(files):
    """On any prose line about the tight-deadline fraction: every '\\leq N%'
    must equal edf_sig_max, every '\\geq M%' must equal spt_sig_min, and a line
    that states both bounds must also acknowledge the tie points between them."""
    edf_sig_max, spt_sig_min, ties = _deadline_crossover()
    issues = []
    for path in files:
        try:
            lines = path.read_text(encoding='utf-8').splitlines()
        except FileNotFoundError:
            continue
        for lineno, line in enumerate(lines, 1):
            if 'AUTO-GENERATED' in line or not _DEADLINE_CTX_RE.search(line):
                continue
            sentences = _split_sentences_with_spans(line)
            ops = {}
            for m in _THRESH_RE.finditer(line):
                op, num = m.group(1), int(m.group(2))
                ops.setdefault(op, m.start())
                if op == 'leq' and edf_sig_max is not None and num != edf_sig_max:
                    issues.append((path, lineno, line.strip(),
                        f'tight-fraction "\\leq {num}%" disagrees with data: '
                        f'EDF is significantly lower-miss than SPT only at '
                        f'f_tight <= {edf_sig_max}%'))
                elif op == 'geq' and spt_sig_min is not None and num != spt_sig_min:
                    issues.append((path, lineno, line.strip(),
                        f'tight-fraction "\\geq {num}%" disagrees with data: '
                        f'SPT is significantly lower-miss than EDF from '
                        f'f_tight >= {spt_sig_min}%'))
            # A line carrying both bounds is making the whole crossover claim,
            # so it owes the reader the middle of the sweep as well. Omitting
            # it reads as an exhaustive partition of the range and hands the
            # tie point to whichever side the sentence happens to touch.
            middle = [p for p in ties
                      if (edf_sig_max is None or p > edf_sig_max)
                      and (spt_sig_min is None or p < spt_sig_min)]
            claim_scope = (_ling_scope(line, sentences, ops['geq'])
                           if 'geq' in ops else '')
            if (ops.keys() >= {'leq', 'geq'} and middle
                    and not _TIE_ACK_RE.search(claim_scope)):
                pts = ', '.join(f'{p}%' for p in middle)
                issues.append((path, lineno, line.strip(),
                    f'crossover claim omits the unresolved middle of the '
                    f'sweep (f_tight {pts}: paired SPT/EDF difference not '
                    f'significant). State that the paired test does not resolve '
                    f'their difference at the midpoint'))
    return issues


# ── Main ──────────────────────────────────────────────────────────────────────

# ── Cheat-sheet sync check ────────────────────────────────────────────────────
# The defense cheat sheet (cheat_sheet.md) is the one claim surface with no
# LaTeX macros: every number in it is hand-typed, which is exactly how the
# 2026-05/06 circular drift started in the chapters. The 2026-07-10 audit
# found all five of its discrepancies here and none in the macro-guarded
# chapters. This check (a) pins the sheet's headline numbers to the values in
# generated/stats.tex via anchor lines that must exist and match, and (b)
# applies the historically flip-prone claim guards to the sheet, with
# exceptions for the crossover scope and for Q&A text that quotes an attack.

CHEAT_SHEET = ROOT / 'cheat_sheet.md'


def _stats_macros():
    """Parse generated/stats.tex and stats_audit.tex into {name: float}."""
    vals = {}
    for fname in ('stats.tex', 'stats_audit.tex'):
        try:
            text = (THESIS / 'generated' / fname).read_text(encoding='utf-8')
        except FileNotFoundError:
            continue
        for m in re.finditer(r'\\newcommand\{\\(\w+)\}\{([^}]+)\}', text):
            try:
                vals[m.group(1)] = float(m.group(2))
            except ValueError:
                pass
    return vals


# Each anchor: (label, regex with numeric capture groups, macro name per group).
# The anchor must appear at least once (guards silent deletion of the line),
# and every captured number must equal its macro at the precision written
# (tolerance = half a unit in the last quoted decimal place).
_CHEAT_ANCHORS = [
    ('SPT/RH wait tie',
     re.compile(r'RH (\d+\.\d+) vs SPT (\d+\.\d+) min, t = -?[\d.]+, p = (\d+\.\d+)'),
     ['RHWait', 'SPTwait', 'RHSPTwaitP']),
    ('CADR/Rescue wait tie',
     re.compile(r'tested: ([\d.]+) vs ([\d.]+) min, t = -?[\d.]+, p = (\d+\.\d+)'),
     ['CADRwait', 'RescueWait', 'CADRRescueWaitP']),
    ('SPT/EDF miss-significance p-value',
     re.compile(r'SIGNIFICANTLY OUTPERFORMED by EDF on miss '
                r'\(EDF ([\d.]+)% vs SPT ([\d.]+)%, p = ([\d.]+)'),
     ['EDFmiss', 'SPTmiss', 'SPTEDFmissP']),
    # The crossover is the claim this thesis has restated wrongly more often
    # than any other, so all three parts are pinned, the tie point included.
    ('deadline crossover, EDF-significant end',
     re.compile(r'10%: EDF (\d+\.\d+) vs SPT (\d+\.\d+), t = (\d+\.\d+); '
                r'30%: EDF (\d+\.\d+) vs SPT (\d+\.\d+), t = (\d+\.\d+)'),
     ['DlTenEDFmiss', 'DlTenSPTmiss', 'DlTenSPTEDFmissT',
      'DlThirtyEDFmiss', 'DlThirtySPTmiss', 'DlThirtySPTEDFmissT']),
    ('deadline crossover, indistinguishable middle',
     re.compile(r'EDF (\d+\.\d+)% vs SPT (\d+\.\d+)%, t = (\d+\.\d+), p = (\d+\.\d+)'),
     ['DlFiftyEDFmiss', 'DlFiftySPTmiss', 'DlFiftySPTEDFmissT', None]),
    ('deadline crossover, SPT-significant end',
     re.compile(r'70%: SPT (\d+\.\d+) vs EDF (\d+\.\d+), t = (-\d+\.\d+); '
                r'90%: SPT (\d+\.\d+) vs EDF (\d+\.\d+), t = (-\d+\.\d+)'),
     ['DlSeventySPTmiss', 'DlSeventyEDFmiss', 'DlSeventySPTEDFmissT',
      'DlNinetySPTmiss', 'DlNinetyEDFmiss', 'DlNinetySPTEDFmissT']),
    ('deadline-blind trio (Phase 3)',
     re.compile(r'FIFO (\d+\.\d+), LCF (\d+\.\d+), Balanced (\d+\.\d+)'),
     ['FIFOmiss', 'LCFmiss', 'Balancedmiss']),
    ('Phase 4 miss ladder',
     re.compile(r'Miss: CADR ([\d.]+)% and RH ([\d.]+)% '
                r'\(no paired significance test is computed at Phase 4[^)]*\), '
                r'Adaptive ([\d.]+)%, Rescue ([\d.]+)%, EDF ([\d.]+)%, '
                r'SPT ([\d.]+)%, Random ([\d.]+)%, FIFO ([\d.]+)%'),
     ['PhFourCADRMiss', 'PhFourRHMiss', 'PhFourAdaptiveMiss', 'PhFourRescueMiss',
      'PhFourEDFmiss', 'PhFourSPTmiss', 'PhFourRandomMiss', 'PhFourFIFOmiss']),
    ('Phase 4 wait ladder',
     re.compile(r'Wait: SPT (\d+\.\d+) min \(lowest\); Rescue (\d+\.\d+), '
                r'RH (\d+\.\d+), CADR (\d+\.\d+)'),
     ['PhFourSPTwait', 'PhFourRescueWait', 'PhFourRHWait', 'PhFourCADRwait']),
    ('Phase 4 tardiness trade-off',
     re.compile(r'Tardiness: CADR ([\d.]+) min vs RH ([\d.]+) min'),
     ['PhFourCADRTard', 'PhFourRHtard']),
]

# Claim guards for the sheet: (pattern, safe-context pattern, message).
_CHEAT_CLAIMS = [
    (
        # "SPT ... lowest wait" recited without the RH tie. This is the claim
        # the 2026-07-10 audit found in three places; RH's point estimate is
        # lower and Table 5.2 bolds RH, so the tie must always be stated.
        re.compile(r'\bSPT\b(?!\s*\+)[^;.\n]{0,45}\blowest wait\b', re.IGNORECASE),
        # "tied" alone is not safe: the historical bug line said "lowest wait,
        # statistically tied with EDF on miss" (tied about the wrong metric).
        # The discriminator is an RH mention, which every correct phrasing has.
        re.compile(r'\bRH\b'),
        'SPT claimed lowest wait without the RH tie '
        '(SPT and RH are statistically tied; RH point estimate is lower)',
    ),
    (
        re.compile(r'\bSPT\b(?!\s*\+)[^;.\n]{0,60}\blowest miss\b', re.IGNORECASE),
        re.compile(r'tight fraction|f_tight|>=?\s*50|≥\s*50|never|do not', re.IGNORECASE),
        'SPT claimed lowest miss outside the deadline-tightness crossover context '
        '(RH is the miss leader; EDF is significantly better than SPT on miss)',
    ),
    (
        re.compile(r'\bSPT\b(?!\s*\+)[^;.\n]{0,80}?\bPareto.?(?:dominant|optimal|superior)\b',
                   re.IGNORECASE),
        re.compile(r'\bjointly\b|SPT\+Rescue', re.IGNORECASE),
        'bare SPT claimed Pareto-dominant '
        '(bare SPT dominates on neither metric; neither RH nor CADR cleanly '
        'Pareto-dominates SPT+Rescue either, do not claim that pair does)',
    ),
    (
        # delta_j written with "=1" next to miss language (Unicode or ASCII).
        re.compile(r'(?:δ|\\delta|delta)_?[j\d]?\s*=\s*1\b.{0,60}?'
                   r'\b(?:miss(?:ed|es)?|violat|late)\b', re.IGNORECASE),
        re.compile(r'1\s*-\s*(?:δ|\\delta|delta)|on.?time|deadline met|\bmet\b',
                   re.IGNORECASE),
        'delta_j=1 co-occurring with miss language (convention: delta_j=1 means on time)',
    ),
    (
        re.compile(r'(?:δ|\\delta|delta)_?[j\d]?\s*=\s*0\b.{0,60}?'
                   r'\b(?:on.?time|deadline met)\b', re.IGNORECASE),
        re.compile(r'\bmiss', re.IGNORECASE),
        'delta_j=0 co-occurring with on-time language (convention: delta_j=0 means missed)',
    ),
]


def _check_cheat_sheet():
    """Return list of (line_no, snippet, message) for cheat-sheet issues."""
    try:
        text = CHEAT_SHEET.read_text(encoding='utf-8')
    except FileNotFoundError:
        return [(0, '', 'cheat_sheet.md missing')]
    macros = _stats_macros()
    issues = []

    for label, pattern, macro_names in _CHEAT_ANCHORS:
        matches = list(pattern.finditer(text))
        if not matches:
            issues.append((0, label,
                           f'anchor line for "{label}" not found '
                           '(edited away or reworded; keep the anchor phrasing '
                           'or update _CHEAT_ANCHORS)'))
            continue
        for m in matches:
            lineno = text.count('\n', 0, m.start()) + 1
            for captured, name in zip(m.groups(), macro_names):
                if name is None:
                    # Captured for regex structure only: the corresponding
                    # macro is a LaTeX math fragment ("p=0.69"), not a float,
                    # so there is nothing numeric to compare against.
                    continue
                if name not in macros:
                    issues.append((lineno, m.group(0)[:100],
                                   f'{label}: macro \\{name} missing from generated stats'))
                    continue
                try:
                    cap = float(captured)
                except ValueError:
                    # A malformed anchor regex must be reported, never allowed
                    # to abort the whole run and take the remaining checks
                    # down with it.
                    issues.append((lineno, m.group(0)[:100],
                                   f'{label}: capture "{captured}" for \\{name} '
                                   'is not a number, fix the anchor regex'))
                    continue
                dp = len(captured.split('.')[1]) if '.' in captured else 0
                tol = 0.5 * 10 ** (-dp) + 1e-9
                if abs(cap - macros[name]) > tol:
                    issues.append((lineno, m.group(0)[:100],
                                   f'{label}: sheet says {captured}, '
                                   f'\\{name} = {macros[name]:g}'))

    for lineno, line in enumerate(text.splitlines(), 1):
        for pattern, safe_re, message in _CHEAT_CLAIMS:
            if pattern.search(line):
                if safe_re and safe_re.search(line):
                    continue
                issues.append((lineno, line.strip()[:100], message))
    return issues


# ── Empirical-claim provenance check ───────────────────────────────────────
#
# Root cause of the 2026-07-19 job_generator.py complexity-distribution bug
# (DECISIONS.md entry 14) and five further instances found the same day
# (entry 15): a sentence claims a specific number is "empirical", "derived
# from", "calibrated to", "measured from", or "fitted" to the dataset, and
# that CLAIM ITSELF was never independently re-derived from its named
# source, only whether the cited number happened to match a stats.tex
# macro (a different, insufficient check: internal consistency between
# prose/macros/JSON says nothing about whether the macro's own upstream
# input was correct in the first place). At least one sentence (the
# sigma_ln dispersion claim, ch3/ch5) was edited twice across a month by two
# different fix passes and was still wrong both times, because each edit
# only patched the number that was visibly stale (a leftover job count),
# never the substantive claim in the same sentence.
#
# This check does not try to verify such claims automatically, the real
# derivation can be checked against the tracked raw CSV and the simulator's
# stored calibration summary. It forces a one-time human sign-off: any sentence matching
# the trigger phrases below must have its exact normalised text's hash
# registered in _VERIFIED_EMPIRICAL_CLAIMS, or the check fails. Editing a
# signed-off sentence at all (even a word) changes its hash and re-triggers
# the check, so a silent, unreviewed drift back to a stale claim cannot
# pass this check again the way it did twice before. To sign off a new or
# edited claim: verify it against its actual source, then add the hash
# this check reports to _VERIFIED_EMPIRICAL_CLAIMS with a short note of
# what was checked and how.

_EMPIRICAL_CLAIM_TRIGGER = re.compile(
    r'\b(empirical(?:ly)?|calibrat\w*|derived\s+from|measured\s+from|'
    r'fitted\b|real\s+rendering\s+jobs?|'
    r'from\s+(?:the\s+)?[\d,]+[\s-]*job\s+dataset|from\s+real\s+data|'
    # Additional English provenance synonyms found slipping through a
    # sentence-scoped check when they share a .tex line with an already
    # -triggered sentence (2026-07-19 Track F audit): a quantitative claim
    # with no "empirical/calibrated/..." keyword of its own is otherwise
    # invisible to this check no matter how the line is split.
    r'range[sd]?\s+from|coefficient\s+of\s+variation|recorded\s+in\s+UTC|'
    r'spans?\s+(?:four|three|two)\s+GPU\s+types|'
    # Chinese equivalents (empirical/real/calibrated/derived/measured/fitted/
    # dataset), so abstract_zh.tex is structurally checkable at all: the
    # English-only trigger set could never match this file's Chinese prose,
    # a total blind spot, not a narrow miss (2026-07-19 Track F audit).
    r'真實|經驗|校準|校正|源自|衍生自|推導自|測量自|量測自|擬合|資料集)\b',
    re.IGNORECASE,
)

# Splits a line into sentence-like segments on ". "/"! "/"? " followed by a
# capital letter or a LaTeX command, so a hash covers just the one sentence
# making the claim, not an entire paragraph-per-line block (this thesis's
# .tex source is largely one physical line per paragraph). Imperfect around
# abbreviations, acceptable: worst case a claim's fingerprint is a slightly
# too-large window, not a missed claim.
_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z\\])')


def _split_sentences_with_spans(line):
    """Return [(start, end, text), ...] sentence-like segments of line."""
    starts = [0]
    ends = []
    for m in _SENTENCE_SPLIT_RE.finditer(line):
        ends.append(m.start())
        starts.append(m.end())
    ends.append(len(line))
    return [(s, e, line[s:e]) for s, e in zip(starts, ends)]


def _claim_hash(sentence):
    normalised = ' '.join(sentence.split())
    return hashlib.sha256(normalised.encode()).hexdigest()[:16]


# Hashes of claims that have been individually checked against their stated
# source (not just cross-checked against a macro) and confirmed correct, or
# confirmed to already be honestly framed as a modelling assumption rather
# than a measurement, or confirmed to carry no specific checkable number.
# Compute a new entry's hash from this check's own output, do not
# hand-compute it. Sign-off pass of 2026-07-19 (67 claims found, 66 signed
# off below after individual verification; the ch5 fig:data-overview median
# execution times were left unsigned, see the bottom of this dict).
_VERIFIED_EMPIRICAL_CLAIMS = {
    'b6e77c70cf8521b1': 'ch4:56 default worker quota five (docs.runpod.io, re-read 2026-11-02); calibration data exclude the 393 jobs that ran while more than five were in flight, recounted by scripts/filter_over_five_concurrency.py (2026-11-02)',
    'c66850c00375648d': 'ch4:56 peak in-flight 8 before the filter (2025-08-13, 08-27, 08-29, 09-01 and 2026-04-09) and 5 after, 393 jobs excluded, recounted by scripts/filter_over_five_concurrency.py; the quota-of-8 statement was removed and the sweep is {3,5,7,10} (2026-11-02)',
    'f05d48a9438a606f': 'ch5:298 sweep values {3,5,7,10} match cmax_sensitivity.json metadata (n=10 seeds, 950 jobs); budgets above five are simulated what-ifs because the filtered calibration data peak at five (2026-11-02)',
    '51827d0df29ccf55': 'ch6:89 C_max = 5 is RunPod default Serverless worker quota (docs.runpod.io, re-read 2026-11-02); 393 jobs above five in flight excluded (2026-11-02)',
    'ef89c10839bc6205': 'ch1:36 counts 6,234 = 6,030 PROD + 203 DEV/DEV_2 + 1 blank, recounted from data/rendering_performance_data.csv after scripts/filter_over_five_concurrency.py (2026-11-02)',
    '328e19b6463e75ce': 'ch1:52 service-time inputs calibrated to the 6,234 filtered jobs (execution_times.py recomputed from the CSV, 2026-11-02)',
    '33bfe882fd571c35': 'ch2:78 per-stratum means and dispersions refit from the filtered CSV match execution_times.py (2026-11-02)',
    '18406c11e117f232': 'ch2:89 stratum means derived from the filtered 6,234-job CSV (2026-11-02)',
    '4266cf21bbc0039f': 'ch4:56 peak in-flight jobs is 8 before the filter (days 2025-08-13, 08-27, 08-29, 09-01 and 2026-04-09, completion-timestamp basis) and 5 after; 393 jobs excluded (2026-11-02)',
    'e87354faa441cb56': 'ch4:56 August 2025 subset is 1,230 jobs (Taiwan time), 1,031 after the filter, recounted from the unfiltered and filtered CSVs (2026-11-02)',
    '6268c6d130614205': 'ch4:56 execution times derive from the filtered 6,234-job CSV (2026-11-02)',
    'b4606d4a8b89688c': 'ch4:60 Table 4.1 caption: filtered 6,234-job dataset, C_max = 5 (2026-11-02)',
    '8fb8a2232b7c3e84': 'ch4:75 per-GPU pooled means 72.59/72.90/85.97/98.35 s recomputed from the filtered CSV (2026-11-02)',
    '61d72bf8e807dc64': 'ch4:91 complexity counts 2,052/2,127/2,055 of 6,234 at the 65.2/73.9 s thresholds with strict less-than, shares 0.3292/0.3412/0.3296 as in job_generator.py (2026-11-02)',
    'e795356f3fbc35f6': 'ch4:91 Pearson corr(total_pixels, render_time_s) = -0.038 on the filtered CSV (2026-11-02)',
    'c265242a5d11cd89': 'ch5:11 counts 6,234 = 6,030 PROD + 203 DEV/DEV_2 + 1 blank, July 2025 to June 2026, recounted from the filtered CSV (2026-11-02)',
    '16973038bc49ec93': 'ch5:11 per-GPU counts 970/4,680/249/335 recounted from the filtered CSV; pixel range 3,102 to 48,320,224 (2026-11-02)',
    'c4b4466d76a87814': 'ch5:11 render_time_s range 29.79 to 352.29 s, aggregate CV 0.265 (rounds to 0.27), pooled n-weighted sigma_ln 0.105 on the filtered CSV (2026-11-02)',
    '2abe4484746dbb85': 'ch5:20 stratum means from the filtered 6,234-job CSV (2026-11-02)',
    'de2810f7d4bc70a9': 'ch5:20 sigma_ln range 0.029 to 0.234, n-weighted 0.105, variance-pooled 0.130, CV 0.265, recomputed on the filtered CSV (2026-11-02)',
    'c1975c11332d2b64': 'ch5:29 caption: 6,234 jobs; per-GPU medians A5000 67.94 s and A4000 87.40 s on the filtered CSV (2026-11-02)',
    'ca234c9b0dd60cbb': 'ch5:29 caption: only colour wording changed for the mono figure (red/grey -> dark/light grey); the 6,234-job count, 14:00-17:00 peak and medians A5000 67.94 s / A4000 87.40 s are unchanged and were previously verified against the filtered CSV (2026-11-11)',
    '1909bf8dca6e7977': 'ch6:12 calibrated against the 6,234 filtered jobs (2026-11-02)',
    'b61d7f4eb0b57b22': 'ch6:89 mean import time 10.78 s, about one seventh of the mean render time, recomputed on the filtered CSV (2026-11-02)',
    '7e5849041bf7db41': 'ch6:91 calibration set 204 non-production rows = 203 DEV/DEV_2 + 1 blank, 3.27% of 6,234, recounted from data/rendering_performance_data.csv; production-only means 3090 72.9, A5000 72.2, A4500 83.4, A4000 98.3 s preserve the GPU ordering (2026-11-22)',
    '0c9e5afd68bb8c9e': 'ch6:95 A5000 4,680 jobs; A4500 291 of 335 and A4000 247 of 249 at high complexity on the filtered CSV (2026-11-02)',
    'a787b6715832cb13': 'abstract_en: execution_times.py reads rendering_performance_data.csv and defines per-complexity GPU service-time means; runpod_availability.py BASE_HIGH_STOCK_PROB is an assumed stock-status model, so the abstract identifies the simulation boundary (2026-11-01)',
    '0fb7bd94dae86e8a': 'ch3 objective paragraph: alpha_1 = 1, alpha_2 = 10 per evaluation/metrics.py, so one miss equals ten seconds of wait and the C_QoS ordering follows mean wait (second audit)',
    '185d0a3d0688f096': 'ch5 service-model paragraph: strata with fewer than five jobs are (1,A4500)=0, (1,A4000)=0, (2,A4000)=3 in execution_times.py SAMPLE_SIZES; the one-off scratch mean-based rerun of the 30 hectic seeds changed FIFO/Adaptive waits by 2.57/2.41%, CADR miss 10.09 to 8.96 (-11.2%), EDF -5.3%, FIFO +6.0%, CADR below RH (8.96 vs 9.16); no committed arm, stated as scratch in the text (second audit)',
    '168dcbcfafa790c7': 'ch5 dataset caption: per-GPU render_time_s medians from rendering_performance_data.csv are A5000 68.21 s and A4000 87.30 s (second audit)',
    '3df5372400ce907e': 'ch5 availability bullet: 5,294 snapshot files and 77 distributions files in data/runpod_pricing spanning 2026-03-10 23:49 UTC to 2026-04-20; dash replaced by "to", Low-range share scoped to the Phase 3 day-type runs as measured by the instrumented delay draw (second audit)',
    'db5ce92945f7d468': 'ch5 availability bullet continuation: 77 calibration snapshots as above; wording change only (second audit)',
    'd05905b0aac9dbb8': 'ch6 simulator construction: nine strata with n >= 53 and three sparse strata (n = 0, 0, 3) per execution_times.py SAMPLE_SIZES (second audit)',
    '849881fa61b1d8ad': 'ch5 label-noise section opener: kappa is a render-time tercile (execution_times.py) and the A4000/A5000 scene-matched ratio 1.171 on 28 shared scenes was recomputed from the CSV on 2026-10-26; A4000 kappa=3 stratum mean 98.2 < A5000 99.6 from EXECUTION_TIME_LOOKUP (2026-10-26)',
    '60b89b59e82e4177': 'ch2:105 positioning claim: tab:related_comparison has exactly four columns (GPU hetero, stochastic availability, deadlines, statistical validation) and only the CGRSP row ticks all; SkyNomad, REACH and BOA are contrasted in ch2 line 15 and none observes coarse stock status (2026-10-26)',
    'da2cd30a6004c313': 'abstract_en: GPU types and sparse-stratum estimates match execution_times.py (low-complexity A4000/A4500 with 0 jobs, medium A4000 n=3) and DECISIONS.md entry 0 (2026-10-26)',
    '2fabf106e76b8b07': 'ch3:185 stock thresholds: runpod_availability.py lines 56-139 give RTX_3090 beta 0.50, OFF_PEAK x1.3, so p_high 0.65, p_medium min(0.90, 0.975) = 0.90, sum 1.55 and realised P(Medium) = 1 - p_high; the cumulative draw u<1 never reaches Low when the sum >= 1 (2026-10-26)',
    '98ca6abba11a538c': 'ch4:85 kappa ex-post label: thresholds 65.5/75.1 s are render-time terciles (execution_times.py COMPLEXITY_LEVELS); Pearson corr(total_pixels, render_time_s) = -0.054 over all 6,627 rows recomputed from the CSV (2026-10-26)',
    '4d654dded126cb4f': 'The five arms of phase3/sparse_sensitivity.json each have RH and CADR below FIFO on miss; the point-estimate order reverses in mean_075. Baseline raw arrays match statistical.json seeds 0 to 9. Recomputed from the generated arrays on 2026-10-24.',
    '5e54b9afa8c45760': 'ch5:378 no-prior RH ablation: RH 9.50 vs no-prior 9.72 miss, wait values and paired t=-0.51, p=0.62 recomputed from prior_robustness.json; difference now printed signed as stock minus no-prior (-0.22 pp) (2026-10-20)',
    '4d164049eb5494cb': 'ch6:12 simulator summary: execution times fitted to the CSV; hectic 950 and surge 730 jobs/day both exceed the busiest recorded day (415); beta_g stated as assumed at ch5:46 (2026-09-25)',
    'fa1ede760efd7346': 'abstract_en: 6,627 jobs from the RunPod Serverless rendering pipeline (6,408 PROD + 218 DEV/DEV_2 + 1 blank in the CSV environment field); 20/80 split is DECISIONS.md entry 2 design choice (re-verified 2026-09-25)',
    '4ffb601378b19351': 'ch2 Sargent V&V: data validity only partly checked (service-time means and sd(ln) refit from the CSV match execution_times.py); job counts and beta_g are assumptions, prices are Pods rates, per ch4/ch5 (2026-09-25)',
    'a93adeabacab7682': 'ch2 Krishnamachari simulator-validation item: service-time inputs checked, partial verification only, consistent with the ch2 Sargent paragraph (2026-09-25)',
    '1f3eb8bd59e327ef': 'ch5 dispersion: sd(ln) 0.033 to 0.231 recomputed from the CSV; import mean 11.6 s; per-type import share A4000 8.9%, A4500 11.1%, A5000 15.5%, 3090 16.9% of mean render time; adding import keeps the speed order (2026-09-25)',
    '1bb7191672d10afd': 'ch5:29 caption: 14:00-17:00 peak matches generate_figures.py and real arrival data; median 68 s (A5000) / 87 s (A4000) recomputed from the CSV; hectic/surge described as stress loads exceeding recorded daily volume (busiest day 415) (2026-09-25)',
    'd0662cab400c6234': 'ch6:80 availability limitation: beta_g stated as assumed fulfilment probabilities; snapshot record High almost never (5 of about 21k type-snapshots) re-verified; provisioning ranges not measured (2026-09-25)',
    'b1c4723d68e97316': 'abstract_en: GPU types measured on RunPod (production Serverless endpoint per ch4); no number beyond the fleet list (re-verified 2026-09-24)',
    'defae48ef26f2be5': 'ch1 objective 2: calibrated (service-time inputs) to 6,627 jobs and checked by the constraint verifier, not validated; consistent with ch2 Sargent paragraph (2026-09-24)',
    'ae23277ee937ac3d': 'ch1 contribution: simulator verified and calibrated against the 6,627-job dataset; no validation claimed (2026-09-24)',
    'b1f058e6121427ff': 'ch4 Serverless disclosure lead sentence: production responses in data/rendering_jobs come from the Serverless endpoint (2026-09-24)',
    '75c5d26138034a11': 'ch4 cost provenance: raw data/rendering_jobs responses carry no cost field; scripts/combine_json_data.py assigned Flex rates 0.684 (A5000, 3090) / 0.576 (A4000, A4500) from runpod_gpu_pricing.json to the 1,230 August 2025 rows of combined_rendering_data.jsonl; rendering_performance_data.csv lists securePrice 0.27/0.46/0.25/0.25; ratios 1.49, 2.30, 2.53 (checked 2026-09-24)',
    'b6351540a6f9db5f': 'ch4 Table 4.1 caption: 6,627 verified; C_max = 5 is RunPod default per-account Serverless worker quota (docs.runpod.io/serverless/workers/overview, re-read 2026-09-24)',
    '42f06763d1bcdcb5': 'ch4 execution-time paragraph: W is the render time only (execution_times.py fits render_time_s); log-normal form and E[W] = stratum mean unchanged (2026-09-24)',
    '87ccfc13b783b7fd': 'ch5 beta_g: stated as modelling assumptions; snapshot recount (secureCloud-filtered lowestPrice) shows High almost never (5 of about 21k type-snapshots, all A5000) and no-offer shares 6.2% to 36.7%, which the model excludes (2026-09-24)',
    # 2026-09-04 audit round (sentences rewritten against the regenerated data)
    'fb2bdf49a0896c9d': 'ch3 servicetime paragraph: mu = ln Wbar - sigma^2/2 is exactly ExecutionTimeModel.sample_execution_time; sigma fitted per stratum in ch5',
    '8b602028934c54e3': 'ch3 provisioning paragraph: p^H = min(0.95, beta phi), p^M = min(0.90, 1.5 beta phi) match runpod_availability.py; beta_g stated as modelling assumptions per ch5',
    'c6a527bbb497daab': 'ch3 Remark 1: log-normal with expectation equal to the stratum mean, dispersion fitted in ch5; matches execution_times.py',
    '589c80ba73d6ebf7': 'ch4 rescue threshold: high-complexity p99 render time 152.5 s and maximum 352.3 s recomputed from the CSV; 600 s clears both',
    '9e6491069db47a9d': 'ch5 log-normal: three sparse strata (A4500 low, A4000 low, A4000 medium n=3) use estimated means, flagged in tab:gpu_specs',

    # -- front_matter/abstract_en.tex --

    # -- ch1_intro.tex --
    '7c43ef96a3d79bd1': 'no specific number',
    'd93cc9a6786c6ec9': 'names GPU types only, specifics verified elsewhere (Table 4.1/gpu_specs)',
    '8cac04f1374074e5': 'July 2025-June 2026 range matches execution_times.py comments; 4 GPU types/3 complexity levels are structural facts',

    # -- ch2_related.tex --
    'd2f4df6b8014048a': '6,627 verified; per-stratum means verified via execution_times.py arithmetic',
    '466b0f8cdbe3843e': 'no specific number',
    '5ba59147d6de0cd5': 'no specific number',

    # -- ch3_formulation.tex --
    '5bad38239475bfa3': 'forward reference to ch5 calibration, no number here',
    '93c6dd762e21552e': '$0.46/$0.25 verified against execution_times.py PRICING_LOOKUP (3090 $0.46, A4500 and A4000 $0.25)',

    # -- ch4_system.tex --
    '2e52420e24219fcc': '6,627 verified',
    'b573d21ae5118853': '74.0/78.1/86.0/97.9s verified as SAMPLE_SIZES-weighted means of EXECUTION_TIME_LOOKUP (Track B, 2026-07-19)',
    '7272065fb1d7c803': 'pooled sigma_ln~=0.11 verified: POOLED_SIGMA_LN=0.108 reproduces exactly as n-weighted average (Track B, 2026-07-19)',
    'e627cff0829fe469': 'no specific number',
    '6e219a826daa4337': 'ch4 complexity split matches SAMPLE_SIZES in execution_times.py summed over GPUs: 2196/2243/2188 of 6627, used as 0.3314/0.3385/0.3301 in job_generator.py; these are calibration-pass counts, not a strict-threshold recount (2026-11-01)',
    '2efda70bac1ed2ff': 'ch4 threshold-boundary caveat: recounted tracked rendering_performance_data.csv render_time_s at <65.5 and <75.1: 2195/2242/2190; ties at 65.5 (1 row) and 75.1 (3 rows), compared with SAMPLE_SIZES (2026-11-01)',
    '9d2c788d5d790143': 'ch4 illustrative Gantt: generate_gantt.py JOBS gives J6 arrival 50/deadline 210 and FIFO/RESCUE lists use 65/75/120 s service with no provisioning; 120 is identified as an illustrative approximation (2026-11-01)',
    '2521f943053ff068': 'explicitly self-caveated as an illustrative approximation, not a claimed empirical value',
    'c67db310892cc0c7': 'explicitly says "approximated from empirical data", consistent illustrative framing',
    '94c529e716215e5c': 'reliability prior 0/0.05/0.15/0.40 verified against rolling_horizon_scheduler.py GPU_RELIABILITY_PENALTY (/100), ordering confirmed inverse of beta_g (2026-07-19 audit)',
    'a626808d59736a0c': 'forward reference to ch5, no number here',

    # -- ch5_experiments.tex --
    '491c6f41a098176c': 'terciles claim matches COMPLEXITY_THRESHOLDS at 33rd/67th percentile (Track B, 2026-07-19)',
    '4896660c2c5d4074': 'no specific number',
    '7ed5f4507762aa8c': 'standard queueing-theory assumption description, no specific number',
    '6d62cff25e108de9': '5,294/77 counts verified against data/runpod_pricing/ (77 distributions_*.json, usually two per collection run); window 11 March to 20 April 2026 in Taiwan time (first file tagged 20260310_234905 on a GMT+7 clock) matches data/README.md; rehashed 2026-09-24',
    'baa7bbf120fb8eb1': '"at most two minutes" matches WAIT_TIME_RANGES Medium=30-120s (120s=2min exactly), verified against runpod_availability.py',
    'cbc11c0808b12783': 'no specific number',
    '1b780f0695b09bbc': '1,200 = 4 day types x 30 seeds x 10 schedulers, arithmetic checked',
    'd03862991e40e63a': 'consistent with 1,200-run structure verified above',
    '257e9caadcbb2d4c': 'RH-wins-unconditionally framing verified against table_weight_sensitivity.tex during the entry-14 narrative rewrite',
    '0ee8db65cce9d0ea': 'no specific number',

    # -- ch6_conclusion.tex --
    '5c7b1dba8cc11384': '6,627 verified',
    '0e98bb2960a4f3dd': '4,916 A5000 jobs verified against JOB_DISTRIBUTION/SAMPLE_SIZES; A4000 254/257=99% and A4500 282/335=84% high-complexity, both consistent with "sampled almost entirely at high complexity" (2026-07-19)',
    '04c67143f9605614': 're-hash after 2026-07-19 Track 5 prose-quality split (one long sentence into three, no content change); describes methodology only (kappa proxied by render-time terciles), no specific derived number in this sentence',
    '723d0cc39b9a9183': 're-hash after 2026-07-19 Track 5 prose-quality fix (semicolon replaced with period, no content change); summary restatement of the execution-time model already verified in detail elsewhere in this registry (per-stratum means and pooled sigma_ln~=0.11 verified against execution_times.py, Track B 2026-07-19)',
    # -- ch5_experiments.tex:11, newly caught by the 2026-07-19 provenance-regex
    # widening (Track F blind-spot patch): these three sentences share a .tex
    # line with an already-signed-off sentence but make their own separate
    # checkable claims that the old sentence-scoped/English-only regex could
    # never surface.
    '3687deb73d5dc98b': 'timestamp/UTC-vs-CloudWatch provenance claim, not raw-CSV-checkable (an operational fact about the original AWS Lambda ingestion pipeline, outside this repo); corroborated identically and independently across 4 repo locations authored at different times (cheat_sheet.md changelog entry + Q&A A10, and matching code comments in generate_figures.py:646/770), consistent with a genuine investigation of a prior timezone-mislabelling bug rather than an invented claim (2026-07-19)',
    '27c783fa61729c0a': 'per-GPU job counts (3090=1,119, A5000=4,916, A4000=257, A4500=335) independently recomputed via df.gpu_used.value_counts() on data/rendering_performance_data.csv, exact match; resolution range not independently re-checked here (already covered by Track A/B prior verification of total_pixels min/max) (2026-07-19)',
    '7903cb4f319eea0d': 'execution-time range and CV independently recomputed from data/rendering_performance_data.csv: render_time_s.min()=29.79/max()=352.29 (rounds to stated 30s/352s), std/mean=0.2799 (matches stated aggregate CV approx 0.28); within-stratum pooled sigma_ln already verified elsewhere in this registry (2026-07-19)',
    '574dd2a1f2586443': 'sample sizes (384-1871 range) verified against SAMPLE_SIZES, all in the "hundreds to thousands" range claimed',
    'ed2b0006642c0cae': 'no specific number',
    # 2026-09-27 sixth audit round
    'caf96c1e0d574e1b': 'ch1:36: 6,627 = 6,408 PROD + 218 DEV/DEV_2 (151 + 67) + 1 blank environment (row 0, 2025-07-18T03-23-22.969Z) in data/rendering_performance_data.csv (re-verified 2026-09-27)',
    '93d6ec7ecba780d8': 'ch5:11: 6,627 = 6,408 PROD + 218 DEV/DEV_2 + 1 blank environment in the CSV; July 2025-June 2026 matches execution_times.py (re-verified 2026-09-27)',
    'f88df1601765f568': 'ch4 SPT: e_j = min over idle nodes of Wbar; tie order is the queue order; estimated_duration is never set by JobGenerator (None) and every runner substitutes 100.0, read by no scheduler (checked 2026-09-27)',
    '3f4705e9cf70aba8': 'ch4 RH node selection: prior 0/0.05/0.15/0.40 is GPU_RELIABILITY_PENALTY/100; higher-beta types are the cheaper ones (A4000/A4500 0.25, A5000 0.27, 3090 0.46); snapshot Secure Cloud offer shares 3090 93.8, A5000 85.7, A4500 72.2, A4000 63.3% recomputed from data/runpod_pricing/snapshot_*.json rank the types the other way (2026-09-27)',
    '254b55718876b0cd': 'ch5:46 tail: 77 calibration snapshots concentrated at 02:00-03:00 and 19:00-20:00 Taiwan time, content unchanged from the earlier sign-off (a45dec9d); rehashed because the preceding implementation note now ends in the same sentence span (2026-09-27)',
    '743eded11bf439df': 'ch5 beta sweep: x0.75 effective miss RH 43.4 < CADR 48.0 < SPT+Rescue 48.5, FIFO 56.5 and SPT 50.9 worst (table_beta_sensitivity.tex); claim now restricted to the scarcity side (2026-09-27)',
    '00823a30b3d94d8f': 'ch5 data-ordered permutation: offer share = lowestPrice.stockStatus non-null, a single-GPU listing field, recomputed from the 5,294 snapshots; REVERSED_BETA (phase3_prior_robustness.py) is 3090 .75 > A5000 .70 > A4500 .65 > A4000 .50, the same order (2026-09-27)',
    '610975f1d1692fe0': "ch5 no-prior ablation: macros NoPriorRHmiss 9.72, NoPriorRHwait 140.89, diff 0.22 pp, p=0.62 from stats_audit.tex; only 'calibrated' changed to 'baseline' (2026-09-27)",
    '1a381c5539fbaa68': 'ch6 availability limitation: scarcity side x0.75 to x1.00 order RH < CADR/SPT+Rescue < deadline-blind in table_beta_sensitivity.tex; claim restricted to that side (2026-09-27)',
}


def _check_empirical_claims(files):
    """Return list of (path, line_no, sentence, claim_hash) for empirical/
    calibrated/derived claims whose exact text is not yet signed off."""
    issues = []
    for path in files:
        try:
            lines = path.read_text(encoding='utf-8').splitlines()
        except FileNotFoundError:
            continue
        for lineno, line in enumerate(lines, 1):
            if 'AUTO-GENERATED' in line:
                continue
            seen_on_line = set()
            for trig in _EMPIRICAL_CLAIM_TRIGGER.finditer(line):
                for s, e, sentence in _split_sentences_with_spans(line):
                    if s <= trig.start() < e:
                        h = _claim_hash(sentence)
                        if h in seen_on_line:
                            break
                        seen_on_line.add(h)
                        if h not in _VERIFIED_EMPIRICAL_CLAIMS:
                            issues.append((path, lineno, ' '.join(sentence.split()), h))
                        break
    return issues


# ── Result-claim provenance hashes ──────────────────────────────────────────
# The sentences below were added with the journal baseline and fleet-mix
# integration. Each was checked against baselines.json or fleet_mix.json
# before its hash was registered: the table check above rebuilds every cell,
# and the p-values and pooled effect sizes in the A5000-heavy and A4000-heavy
# sentences are recomputed there. Editing one of these sentences changes its
# hash and fails the check, so the new text must be reverified against the
# JSON and re-registered rather than drifting silently.

_VERIFIED_RESULT_CLAIMS = {
    'eee15bf85aefbd0c': 'ch5 baselines: HEFT and max-min indistinguishable, identical on all six per-seed metrics, 30 comparisons with 6 duplicates (24 unique); checked against baselines.json',
    '26bff90a138d02fd': 'ch5 baselines: MinMin wait not separated from RH (p_holm=0.8904, d=-0.0283) or CADR (p=0.0831, p_holm=0.3326); checked against baselines.json t_tests',
    '5e1a78006f93c189': 'ch5 baselines: MinMin miss 19.06 vs RH 9.60 (d=+2.4006) and CADR 9.68 (d=+2.1770), both p_holm<0.001; checked against baselines.json',
    '9952dffdbf6527d1': 'ch5 baselines: bandit vs FIFO miss p=0.1213 and wait p=0.3383; bandit worse than RH and CADR on miss p<0.001; checked against baselines.json',
    '3e868d851e229107': 'ch5 baselines: random-arm control 13.57 miss against trained bandit 25.45; checked against baselines.json means',
    '261a12607b229abe': 'ch5 fleet mix: anchor reproduces statistical.json seeds 0-9, 360/360 values across six policies and six metrics; checked against fleet_mix.json anchor_equivalence',
    '2bed7a75117d6e5b': 'ch5 fleet mix: RH and CADR are the two lowest in four named mixes; checked against fleet_mix.json ordering_checks',
    'bb6d29a31391ca85': 'ch5 fleet mix: every deadline-aware policy below FIFO in all six mixes; checked against fleet_mix.json ordering_checks',
    '0239b95400a2b45e': 'ch5 fleet mix: ordering_stable false because A5000-heavy and A4000-heavy put CADR and SPT+Rescue first and second, RH third and fourth; checked against fleet_mix.json',
    'fe9bf39a640bbca9': 'ch5 fleet mix: A5000-heavy RH vs CADR 9.27 vs 8.15, p=0.282, pooled d=0.287; RH vs SPT+Rescue 9.27 vs 9.13, p=0.916, d=0.041; recomputed from raw arrays',
    'b81634f0f466f587': 'ch5 fleet mix: A4000-heavy CADR 0.57 vs RH 5.96, p<0.001, pooled d=6.21; SPT+Rescue 1.36 vs RH uncorrected p=0.0038; recomputed from raw arrays',
    'b0da922dc091066c': 'ch5 fleet mix: CADR largest gap to mix minimum 1.01 pp (3090-only), RH gap 5.39 pp (A4000-heavy); checked against fleet_mix.json means',
    '97e0600af49b2d0c': 'ch5 fleet mix: CADR at or near the minimum in every mix and leads in the two reversal mixes; checked against fleet_mix.json means',
    '8f2d6c86e90ddda1': 'ch6 concluding remarks: RH lowest in four of six mixes, CADR within about 1 pp of the minimum and leads in the other two; checked against fleet_mix.json means',
    '1d428f7b105d00ee': 'ch6 fleet-composition limitation: six-mix sweep, RH/CADR two lowest in four of six, A5000-heavy p=0.282 d=0.287, A4000-heavy 0.57 vs 5.96 p<0.001 d=6.21; checked against fleet_mix.json',
    'ea4f455820d63646': 'ch6 baseline coverage: HEFT/max-min identical, MinMin wait not separated but miss worse, shallow bandit not better than FIFO, random control lower miss; checked against baselines.json',
    '3beed9667db48b17': 'ch1 scope: six-mix sweep is a ten-seed sensitivity check, not a second studio trace; shallow bandit is not evidence about deep RL; checked against both JSONs and their metadata',
    '900d5f734de9733a': 'ch5 baselines: bandit arms FIFO/SPT/EDF/SPT+Rescue, 15 training seeds 500-514, evaluation seeds 0-29; checked against phase3_baselines.py and baselines.json metadata',
    '38df7c580cdb8a5d': 'ch5 baselines: HEFT and max-min highest wait 191.77 and miss 29.90, above FIFO 167.93 and 24.43; checked against baselines.json means',
    '7b97a4d5cd961bf7': 'ch5 baselines: MinMin lowest wait point estimate 141.42; checked against baselines.json means',
    '811b3e3e40e999a0': 'ch5 fleet mix: A4000-heavy FIFO 15.79 against anchor 23.67; checked against fleet_mix.json means',
}


def _check_result_claims(files):
    """Return registered result-claim hashes whose exact sentence is absent."""
    found = set()
    for path in files:
        try:
            lines = path.read_text(encoding='utf-8').splitlines()
        except FileNotFoundError:
            continue
        for line in lines:
            if 'AUTO-GENERATED' in line:
                continue
            for _, _, sentence in _split_sentences_with_spans(line):
                found.add(_claim_hash(sentence))
    return [(h, note) for h, note in _VERIFIED_RESULT_CLAIMS.items()
            if h not in found]


def main():
    data = _load()
    checks = _build_checks(data)
    total_issues = 0

    print('── Numeric checks ──────────────────────────────────────────────────')
    for path in TEX_FILES:
        rel = path.relative_to(ROOT)
        issues = _scan_file(path, checks, data)
        if issues:
            print(f'\n{rel}:')
            for lineno, line, desc, expected in issues:
                print(f'  line {lineno}: [{desc}] expected {expected}')
                print(f'    {line[:120]}')
            total_issues += len(issues)

    if total_issues == 0:
        print('  all numeric checks passed.')

    print('\n── Linguistic checks ───────────────────────────────────────────────')
    ling_issues = _check_linguistic(LINGUISTIC_FILES)
    if ling_issues:
        for path, lineno, line, message in ling_issues:
            rel = path.relative_to(ROOT)
            print(f'\n{rel} line {lineno}: {message}')
            print(f'    {line[:120]}')
        total_issues += len(ling_issues)
    else:
        print('  all linguistic checks passed.')

    print('\n── Spelling checks ─────────────────────────────────────────────────')
    spell_issues = _check_spelling(TEX_FILES)
    if spell_issues:
        for path, lineno, line, message in spell_issues:
            rel = path.relative_to(ROOT)
            print(f'\n{rel} line {lineno}: {message}')
            print(f'    {line[:120]}')
        total_issues += len(spell_issues)
    else:
        print('  all spelling checks passed.')

    print('\n── Macro-sync check ────────────────────────────────────────────────')
    sync_issues = _check_macro_sync(data)
    if sync_issues:
        for macro, expected, actual in sync_issues:
            print(f'  \\{macro}: stats.tex has {actual}, JSON expects {expected}')
        total_issues += len(sync_issues)
    else:
        print('  stats.tex macros match simulation JSON.')

    print('\n── Baseline and fleet-mix table checks ─────────────────────────────')
    bf_issues = _check_baseline_fleet_tables()
    if bf_issues:
        for issue in bf_issues:
            print('  ' + issue)
        total_issues += len(bf_issues)
    else:
        print('  baseline and fleet-mix tables match their JSON; identity, '
              'duplicate-test and ordering claims hold.')

    print('\n── Label/speed bracket check ───────────────────────────────────────')
    ks = json.load(open(ROOT / 'CGRSP/Outputs/phase3/kappa_speed_sensitivity.json', encoding='utf-8'))['arms']
    ks_issues = []
    for arm_name, arm in ks.items():
        miss = {k: v['miss_pct'] for k, v in arm['schedulers'].items()}
        ranked = sorted(miss, key=miss.get)
        if ranked[0] != 'RH' or miss['RH'] >= miss['FIFO']:
            ks_issues.append(f'{arm_name}: RH is not the lowest miss point estimate and below FIFO ({miss})')
        second_ok = ranked[1] == ('EDF' if arm_name == 'uninformative' else 'CADR')
        if not second_ok or miss['CADR'] >= miss['FIFO']:
            ks_issues.append(f'{arm_name}: second place is not CADR (EDF in the uninformative arm) ({miss})')
        if miss['RH'] >= miss['CADR']:
            ks_issues.append(f'{arm_name}: ch5 says RH has the lower point estimate than CADR in every arm ({miss})')
        separates = arm['rh_vs_cadr_miss_p'] < 0.05
        if separates != (arm_name == 'a4000_scene_matched'):
            ks_issues.append(f'{arm_name}: ch5 says only the scene-matched arm separates RH and CADR at raw 0.05')
    if ks['uninformative']['schedulers']['SPT']['miss_pct'] <= ks['uninformative']['schedulers']['FIFO']['miss_pct']:
        ks_issues.append('uninformative: SPT is no longer above FIFO; ch5 says it is')
    for issue in ks_issues:
        print('  ' + issue)
    total_issues += len(ks_issues)
    if not ks_issues:
        print('  ch5 label/speed claims match kappa_speed_sensitivity.json.')

    print('\n── Deadline-crossover check ────────────────────────────────────────')
    edf_sig_max, spt_sig_min, tie_pts = _deadline_crossover()
    cross_issues = _check_deadline_crossover(LINGUISTIC_FILES)
    if cross_issues:
        for path, lineno, line, message in cross_issues:
            rel = path.relative_to(ROOT)
            print(f'\n{rel} line {lineno}: {message}')
            print(f'    {line[:120]}')
        total_issues += len(cross_issues)
    else:
        ties = ', '.join(f'{p}%' for p in tie_pts) or 'none'
        print(f'  crossover prose matches data (EDF significantly lower '
              f'f_tight<={edf_sig_max}%, unresolved at {ties}, '
              f'SPT significantly lower f_tight>={spt_sig_min}%).')

    print('\n── Cheat-sheet sync check ──────────────────────────────────────────')
    # The defense cheat sheet is not part of the public package.
    cheat_issues = _check_cheat_sheet() if CHEAT_SHEET.exists() else None
    if cheat_issues is None:
        print('  skipped: cheat_sheet.md is not in this package.')
    elif cheat_issues:
        for lineno, snippet, message in cheat_issues:
            loc = f'cheat_sheet.md line {lineno}' if lineno else 'cheat_sheet.md'
            print(f'\n{loc}: {message}')
            if snippet:
                print(f'    {snippet}')
        total_issues += len(cheat_issues)
    else:
        print('  cheat-sheet anchors match generated stats; claim guards clean.')

    print('\n── Constraint-verifier check ───────────────────────────────────────')
    out = ROOT / 'CGRSP' / 'Outputs' / 'runpod_scheduler'
    ver = {**json.loads((out / 'report.json').read_text(encoding='utf-8'))['verification'],
           **json.loads((out / 'verification_proposed.json').read_text(encoding='utf-8'))['schedulers']}
    bad = [k for k, v in ver.items()
           if not v['all_passed'] or v['total_violations']
           or not all(c['checked'] and c['status'] == 'passed' for c in v['constraint_results'].values())
           or v['constraint_results']['assignment_uniqueness'].get('jobs_checked') != 950]
    if len(ver) != 10 or bad:
        print(f'  verifier coverage {len(ver)}/10, failing: {bad} '
              '(ch4 metrics collection and cheat sheet J6 claim zero violations for all ten, over all 950 jobs)')
        total_issues += 1
    else:
        print('  all ten schedulers pass the constraint verifier over all 950 jobs (dispatch log, [dispatch, completion) occupancy).')
    print('\n── Empirical-claim provenance check ────────────────────────────────')
    claim_issues = _check_empirical_claims(TEX_FILES)
    if claim_issues:
        for path, lineno, sentence, h in claim_issues:
            rel = path.relative_to(ROOT)
            print(f'\n{rel} line {lineno}: UNVERIFIED empirical/calibrated/derived claim '
                  f'(hash {h})')
            print(f'    {sentence[:200]}')
        print(f'\n  {len(claim_issues)} claim(s) need manual sign-off: verify each against its '
              'actual source (not just against a macro), then add its hash to '
              '_VERIFIED_EMPIRICAL_CLAIMS with a note of what was checked.')
        total_issues += len(claim_issues)
    else:
        print('  all empirical/calibrated/derived claims are signed off.')

    print('\n── Result-claim provenance check ───────────────────────────────────')
    result_issues = _check_result_claims(TEX_FILES)
    if result_issues:
        for h, note in result_issues:
            print(f'  registered result claim {h} is no longer present: {note}')
        total_issues += len(result_issues)
    else:
        print('  all registered baseline and fleet-mix result claims are present.')

    if total_issues == 0:
        print('\nvalidate_numbers: all checks passed.')
        sys.exit(0)
    else:
        print(f'\nvalidate_numbers: {total_issues} issue(s) found.')
        sys.exit(1)


if __name__ == '__main__':
    main()

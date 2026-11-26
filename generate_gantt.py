#!/usr/bin/env python3
"""
Gantt chart illustration for CGRSP: FIFO vs SPT+Rescue.

Scenario: C_max=3 nodes, 7 jobs. ONLY J6 has a tight deadline (d=210 s).
All other jobs have loose (8-hour) deadlines and are not shown with markers.

Schedule derivation:
  Nodes: N0=RTX 3090, N1=RTX A5000, N2=RTX A4000
  Execution times (illustrative per-complexity constants, not GPU-stratum means;
  provisioning delay is zero):
    κ=1 -> 65 s, κ=2 -> 75 s, κ=3 -> 120 s
  J6: arrive=50, κ=2, exec=75 s, tight deadline d=210 s (window = 160 s)

  FIFO (first-in-first-out, deadline-oblivious):
    t=0:   J0(κ3)->N0          (N0 idle)
    t=5:   J1(κ1)->N1          (N1 idle)
    t=10:  J2(κ1)->N2          (N2 idle)
    t=20:  J3 arrives; all busy → queue [J3]
    t=30:  J4 arrives → queue [J3,J4]
    t=40:  J5 arrives → queue [J3,J4,J5]
    t=50:  J6 arrives → queue [J3,J4,J5,J6]
    t=70:  J1 done; N1 free → J3→N1 (FIFO order)
    t=75:  J2 done; N2 free → J4→N2 (FIFO order)
    t=120: J0 done; N0 free → J5→N0 (FIFO order)
    t=140: J4 done; N2 free → J6→N2   ← J6 finishes 215 s, d=210 s → MISS

  SPT+Rescue (laxity threshold = 600 s):
    t=0..50: identical initial three assignments
    t=70: J1 done; N1 free. Laxity check on queue [J3,J4,J5,J6]:
           L_J6(70) = 210 - 70 - 75 = 65 s < 600 s → J6 URGENT
           J3,J4,J5 loose → normal
           Order: [J6(urgent), J4(SPT 65s), J3(SPT 75s), J5(SPT 120s)]
           → J6→N1 immediately                        ← J6 finishes 145 s < 210 s ✓
    t=75:  J2 done; N2 free. Queue [J4,J3,J5] all loose.
           SPT: J4(65s)→N2
    t=120: J0 done; N0 free. Queue [J3,J5]. SPT: J3(75s)→N0
    t=140: J4 done; N2 free. Queue [J5]. J5→N2

  Node utilisation:
    FIFO:    N0:[J0,J5]  N1:[J1,J3]         N2:[J2,J4,J6(MISS)]
    Rescue:  N0:[J0,J3]  N1:[J1,J6(MEETS)]  N2:[J2,J4,J5]

  J6 wait: FIFO = 140-50 = 90 s   SPT+Rescue = 70-50 = 20 s
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from pathlib import Path

# ── colour palette ────────────────────────────────────────────────────────────
KAPPA_COL = {1: '#E6E6E6', 2: '#9E9E9E', 3: '#333333'}
KAPPA_TXT = {1: '#1A1A1A', 2: '#1A1A1A', 3: 'white'}
DL_COL    = '#1A1A1A'
BH        = 0.44   # bar height

# ── scenario: only J6 has a tight deadline ────────────────────────────────────
JOBS = {
    0: {'arrive': 0,  'kappa': 3, 'deadline': None, 'tight': False},
    1: {'arrive': 5,  'kappa': 1, 'deadline': None, 'tight': False},
    2: {'arrive': 10, 'kappa': 1, 'deadline': None, 'tight': False},
    3: {'arrive': 20, 'kappa': 2, 'deadline': None, 'tight': False},
    4: {'arrive': 30, 'kappa': 1, 'deadline': None, 'tight': False},
    5: {'arrive': 40, 'kappa': 3, 'deadline': None, 'tight': False},
    6: {'arrive': 50, 'kappa': 2, 'deadline': 210,  'tight': True},
}

# Each entry: (job_id, node_idx, start, end)
FIFO = [
    (0, 0,   0, 120),
    (1, 1,   5,  70),
    (2, 2,  10,  75),
    (3, 1,  70, 145),  # J3 gets N1 (FIFO order)
    (4, 2,  75, 140),  # J4 gets N2 (FIFO order)
    (5, 0, 120, 240),  # J5 gets N0 (FIFO order)
    (6, 2, 140, 215),  # J6 last on N2 → MISS: 215 > 210
]

RESCUE = [
    (0, 0,   0, 120),
    (1, 1,   5,  70),
    (2, 2,  10,  75),
    (6, 1,  70, 145),  # J6 urgent (L=65s<600s) → dispatched immediately to N1
    (4, 2,  75, 140),  # J4 (SPT 65s, loose) gets N2
    (3, 0, 120, 195),  # J3 (SPT 75s, loose) gets N0
    (5, 2, 140, 260),  # J5 (120s, loose) gets N2 last
]

NODE_LBL = ['N0 (RTX 3090)', 'N1 (RTX A5000)', 'N2 (RTX A4000)']
XLIM = (-5, 290)


def render(ax, entries):
    for jid, node, t0, t1 in entries:
        job = JOBS[jid]
        y   = float(node)
        col = KAPPA_COL[job['kappa']]

        ax.barh(y, t1 - t0, left=t0, height=BH,
                color=col, edgecolor='white', linewidth=0.7, zorder=3)

        if t1 - t0 >= 25:
            ax.text((t0 + t1) / 2.0, y,
                    f'J{jid}\nκ={job["kappa"]}',
                    ha='center', va='center',
                    fontsize=7.2, color=KAPPA_TXT[job['kappa']], fontweight='bold', zorder=4)

        dl = job['deadline']
        if job['tight'] and dl is not None:
            ax.axvline(dl, color=DL_COL, lw=1.8, ls='--', zorder=5, alpha=0.85)

    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(NODE_LBL, fontsize=8.5)
    ax.set_xlim(*XLIM)
    ax.set_ylim(-0.58, 2.65)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))
    ax.grid(axis='x', which='major', ls=':', alpha=0.35, zorder=0)
    ax.set_xlabel('Time (seconds)', fontsize=9)
    ax.set_ylabel('GPU node', fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 5.8),
                                gridspec_kw={'hspace': 0.70})

render(ax1, FIFO)
ax1.set_title(
    r'$\mathbf{FIFO}$: J6 arrives $t\!=\!50\,$s; dispatched $t\!=\!140\,$s '
    r'(wait $= 90\,$s); misses $d_6\!=\!210\,$s',
    fontsize=9.5, loc='left', pad=4)
ax1.annotate(
    r'$\delta_6\!=\!0$: MISS ($+$5 s)',
    xy=(215, 2.22), xytext=(223, 2.52),
    fontsize=8.5, color=DL_COL, fontweight='bold', va='bottom', ha='left',
    arrowprops=dict(arrowstyle='->', color=DL_COL, lw=1.2,
                    connectionstyle='arc3,rad=-0.15'),
    zorder=7)

render(ax2, RESCUE)
ax2.set_title(
    r'$\mathbf{SPT{+}Rescue}$: J6 arrives $t\!=\!50\,$s; dispatched $t\!=\!70\,$s '
    r'(wait $= 20\,$s); meets $d_6\!=\!210\,$s',
    fontsize=9.5, loc='left', pad=4)
ax2.annotate(
    r'$L_{J_6}(70) = 210 - 70 - 75 = 65\,\mathrm{s} < 600\,\mathrm{s}$ threshold',
    xy=(72, 0.84), xytext=(95, 0.5),
    fontsize=8, color='#1A1A1A', fontweight='bold', va='center', ha='left',
    arrowprops=dict(arrowstyle='->', color='#1A1A1A', lw=1.0,
                    connectionstyle='arc3,rad=-0.15'),
    zorder=7)

legend_handles = [
    mpatches.Patch(color=KAPPA_COL[1], label=r'$\kappa=1$ Low complexity ($\approx\!65\,$s)'),
    mpatches.Patch(color=KAPPA_COL[2], label=r'$\kappa=2$ Medium ($\approx\!75\,$s)'),
    mpatches.Patch(color=KAPPA_COL[3], label=r'$\kappa=3$ High ($\approx\!120\,$s)'),
    plt.Line2D([0], [0], color=DL_COL, lw=2.0, ls='--',
               label=r'J6 tight deadline ($d_6\!=\!210\,$s)'),
]
fig.legend(handles=legend_handles, loc='lower center', ncol=4,
           fontsize=8.5, bbox_to_anchor=(0.5, -0.04),
           frameon=True, framealpha=0.95)

fig.suptitle(
    r'Illustrative CGRSP scheduling ($C_{\max}=3$ nodes, 7 jobs): '
    r'SPT+Rescue rescues J6 (only tight-deadline job) by promoting it over J4',
    fontsize=10, y=1.02)

out = Path(__file__).resolve().parent / 'thesis' / 'figures' / 'gantt_chart.pdf'
out.parent.mkdir(exist_ok=True)
fig.savefig(out, bbox_inches='tight', dpi=150)
print(f'Saved {out}')
plt.close(fig)

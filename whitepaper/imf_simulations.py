#!/usr/bin/env python3
"""
Inheritance Math Framework (IMF) -- Simulation Suite v1.2.1
Remnant Fieldworks Inc. | Author: Derek Hone

Canonical ODE (Postulate 4.1):
    dC/dt = -alpha * D_bar + beta * F + gamma * R * u_bar - delta * C

Equilibrium (constant coefficients):
    C* = (-alpha * D_bar + beta * F + gamma * R * u_bar) / delta

Blocking-issue fixes in this version:
    1. Fig 2: recurrence now includes epsilon term correctly
    2. Fig 3: F held constant across strategies -- only R varies
    3. Fig 5: categorical labels only -- no decimal scores
    4. All ODE figures: projected model with clipping as part of simulation
    5. Fig 6: prominent disclosure that AUC reflects model architecture

Usage:
    python imf_simulations.py --output <dir> --seed <int>
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    print("scikit-learn required for Figure 6.  pip install scikit-learn")
    sys.exit(1)

plt.style.use('seaborn-v0_8-whitegrid')
FIGSIZE = (10, 6)
DPI = 150

parser = argparse.ArgumentParser(description='IMF Simulation Suite v1.2.1')
parser.add_argument('--output', type=str,
                    default=os.path.dirname(os.path.abspath(__file__)),
                    help='Output directory for figures')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
args = parser.parse_args()

OUT = args.output
os.makedirs(OUT, exist_ok=True)
SEED = args.seed

manifest_lines = []
saved_files = []


def log_params(fig_num, title, params_dict):
    header = f"\nFIGURE {fig_num}: {title}"
    sep = '=' * 70
    print(f"\n{sep}\n{header}\n{sep}")
    manifest_lines.append(f"\n{sep}")
    manifest_lines.append(header)
    manifest_lines.append(sep)
    for k, v in params_dict.items():
        line = f"  {k}: {v}"
        print(line)
        manifest_lines.append(line)


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  -> Saved: {path}")
    saved_files.append(path)
    return path


def canonical_dCdt(C, D_bar, F, R, u_bar, alpha, beta, gamma, delta):
    """Postulate 4.1:  dC/dt = -alpha*D_bar + beta*F + gamma*R*u_bar - delta*C"""
    return -alpha * D_bar + beta * F + gamma * R * u_bar - delta * C


def equilibrium(D_bar, F, R, u_bar, alpha, beta, gamma, delta):
    """C* = numerator / delta.  Returns raw value (may be <0 or >1)."""
    return (-alpha * D_bar + beta * F + gamma * R * u_bar) / delta


def check_eq(label, D_bar, F, R, u_bar, alpha, beta, gamma, delta,
             allow_negative=False):
    """Compute and print C*.  Raise if > 1 or < 0 (unless allow_negative)."""
    cstar = equilibrium(D_bar, F, R, u_bar, alpha, beta, gamma, delta)
    tag = 'OK' if 0 <= cstar <= 1 else ('->0' if cstar < 0 else 'VIOLATION')
    print(f"    C*({label}) = {cstar:.4f}  [{tag}]")
    if cstar > 1.0:
        raise ValueError(f"C* > 1 for {label}: {cstar:.4f}")
    if cstar < 0 and not allow_negative:
        raise ValueError(f"C* < 0 for {label}: {cstar:.4f}")
    return cstar


def add_param_table(ax, col_labels, table_data, loc='lower left',
                    bbox=None):
    if bbox is None:
        bbox = [0.02, 0.02, 0.55, 0.22]
    tbl = ax.table(cellText=table_data, colLabels=col_labels,
                   loc=loc, bbox=bbox, cellLoc='center', edges='closed')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    for key, cell in tbl.get_celld().items():
        cell.set_edgecolor('#cccccc')
        cell.set_linewidth(0.5)
        if key[0] == 0:
            cell.set_facecolor('#e6e6e6')
            cell.set_text_props(fontweight='bold')
    return tbl


# ===================================================================
# FIGURE 1 -- Core Balance Law: Three Scenarios
# ===================================================================
def fig1_coherence_dynamics():
    # ---- Parameters (common) ----
    ALPHA = 0.10
    BETA  = 0.08
    GAMMA = 0.11
    DELTA = 0.12
    C0    = 0.65
    D0_A  = 0.10  # initial D_bar for unmanaged
    T     = 60
    dt    = 0.05
    C_MIN = 0.20
    T_SWITCH_B = 15  # time stewardship kicks in for scenario B

    t = np.arange(0, T + dt / 2, dt)
    N = len(t)

    # ---- Scenario definitions ----
    # A: Unmanaged -- D_bar grows, no stewardship
    #    F=0.15, R=0, u_bar=0
    #    C* starts ~0.07 and goes negative as D_bar grows
    #    -> system degrades toward 0 (expected; clip is part of the simulation model for scenarios with negative forcing)
    A = {'F': 0.15, 'R': 0.0, 'u_bar': 0.0}

    # B: Delayed Stewardship at t=15
    #    Pre-switch: same as A
    #    Post-switch: F=0.55, R=0.55, u_bar=0.60, D_bar decays
    B_pre  = {'F': 0.15, 'R': 0.0,  'u_bar': 0.0}
    B_post = {'F': 0.55, 'R': 0.55, 'u_bar': 0.60}

    # C: Resonance-Matched (constant parameters)
    #    D_bar = 0.25 (constant, well-managed)
    C_sc = {'D_bar': 0.25, 'F': 0.70, 'R': 0.85, 'u_bar': 0.75}

    # ---- Verify equilibria ----
    print("  Figure 1 equilibrium checks:")
    check_eq('A at D=0.10', 0.10, A['F'], A['R'], A['u_bar'],
             ALPHA, BETA, GAMMA, DELTA, allow_negative=False)
    check_eq('A at D=0.50', 0.50, A['F'], A['R'], A['u_bar'],
             ALPHA, BETA, GAMMA, DELTA, allow_negative=True)
    cstar_B_post = check_eq('B post-switch D=0.30',
             0.30, B_post['F'], B_post['R'], B_post['u_bar'],
             ALPHA, BETA, GAMMA, DELTA)
    cstar_B_post_low = check_eq('B post-switch D=0.15',
             0.15, B_post['F'], B_post['R'], B_post['u_bar'],
             ALPHA, BETA, GAMMA, DELTA)
    cstar_C = check_eq('C resonance', C_sc['D_bar'], C_sc['F'],
             C_sc['R'], C_sc['u_bar'], ALPHA, BETA, GAMMA, DELTA)

    # ---- Simulate ----
    results = {}
    eq_traces = {}  # for plotting C*(t)

    for label in ['A', 'B', 'C']:
        C_arr = np.zeros(N)
        D_arr = np.zeros(N)
        Cstar_arr = np.zeros(N)
        C_arr[0] = C0

        if label == 'A':
            D_arr[0] = D0_A
        elif label == 'B':
            D_arr[0] = D0_A
        else:
            D_arr[0] = C_sc['D_bar']

        for i in range(N):
            if label == 'A':
                D_arr[i] = D0_A + 0.012 * t[i]  # grows linearly
                F_i, R_i, u_i = A['F'], A['R'], A['u_bar']
            elif label == 'B':
                if t[i] < T_SWITCH_B:
                    D_arr[i] = D0_A + 0.012 * t[i]
                    F_i, R_i, u_i = B_pre['F'], B_pre['R'], B_pre['u_bar']
                else:
                    D_at_switch = D0_A + 0.012 * T_SWITCH_B
                    D_arr[i] = max(D_at_switch - 0.005 * (t[i] - T_SWITCH_B),
                                   0.12)
                    F_i = B_post['F']
                    R_i = B_post['R']
                    u_i = B_post['u_bar']
            else:  # C
                D_arr[i] = C_sc['D_bar']
                F_i, R_i, u_i = C_sc['F'], C_sc['R'], C_sc['u_bar']

            Cstar_arr[i] = equilibrium(
                D_arr[i], F_i, R_i, u_i, ALPHA, BETA, GAMMA, DELTA)

            if i > 0:
                dCdt = canonical_dCdt(
                    C_arr[i - 1], D_arr[i], F_i, R_i, u_i,
                    ALPHA, BETA, GAMMA, DELTA)
                # Projected model boundary -- clipping is part of the
                # simulation model for scenarios with negative forcing
                C_arr[i] = np.clip(C_arr[i - 1] + dt * dCdt, 0.0, 1.0)

        results[label] = C_arr
        eq_traces[label] = Cstar_arr

    # ---- Plot ----
    colors = {'A': '#d62728', 'B': '#2ca02c', 'C': '#1f77b4'}
    labels = {
        'A': 'A: Unmanaged Drift',
        'B': f'B: Delayed Stewardship (t={T_SWITCH_B})',
        'C': 'C: Resonance-Matched',
    }
    ls_map = {'A': '-', 'B': '-', 'C': '-.'}

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.fill_between(t, 0, C_MIN, color='red', alpha=0.08)
    ax.axhline(C_MIN, color='red', ls='--', lw=1.2, alpha=0.7)
    ax.text(1, C_MIN + 0.01, '$C_{min}$', color='red', fontsize=10,
            fontweight='bold')

    for sc in ['A', 'B', 'C']:
        ax.plot(t, results[sc], color=colors[sc], lw=2.2,
                ls=ls_map[sc], label=labels[sc])
        # Equilibrium as thin dotted
        eq_clipped = np.clip(eq_traces[sc], 0, 1)
        ax.plot(t, eq_clipped, color=colors[sc], lw=1, ls=':', alpha=0.5)

    # Stewardship annotation
    ax.axvline(T_SWITCH_B, color='#2ca02c', ls=':', lw=1, alpha=0.5)
    idx_sw = int(T_SWITCH_B / dt)
    ax.annotate('Stewardship begins', xy=(T_SWITCH_B, results['B'][idx_sw]),
                xytext=(T_SWITCH_B + 4, 0.50), fontsize=9,
                arrowprops=dict(arrowstyle='->', color='#2ca02c'),
                color='#2ca02c')

    # Equilibrium annotation for C
    ax.annotate(f'$C^* = {cstar_C:.2f}$',
                xy=(T * 0.7, cstar_C), xytext=(T * 0.7 + 3, cstar_C + 0.06),
                fontsize=9, color='#1f77b4',
                arrowprops=dict(arrowstyle='->', color='#1f77b4', lw=0.8))

    # Parameter table
    col_labels = ['Scenario', r'$\alpha$', r'$\beta$', r'$\gamma$',
                  r'$\delta$', 'F', 'R', r'$\bar{u}$', r'$\bar{D}$']
    table_data = [
        ['A: Unmanaged', f'{ALPHA}', f'{BETA}', f'{GAMMA}', f'{DELTA}',
         '0.15', '0', '0', 'grows'],
        ['B: Delayed',   f'{ALPHA}', f'{BETA}', f'{GAMMA}', f'{DELTA}',
         '0.15/0.55', '0/0.55', '0/0.60', 'grows/falls'],
        ['C: Resonance', f'{ALPHA}', f'{BETA}', f'{GAMMA}', f'{DELTA}',
         '0.70', '0.85', '0.75', '0.25'],
    ]
    add_param_table(ax, col_labels, table_data,
                    bbox=[0.02, 0.02, 0.70, 0.22])

    ax.set_xlabel('Time $t$ (arbitrary units)', fontsize=12)
    ax.set_ylabel('Coherence $C(t)$', fontsize=12)
    ax.set_title(
        'Core Balance Law (Postulate 4.1): Three Scenarios',
        fontsize=13, fontweight='bold')
    ax.set_xlim(0, T)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)

    fig.text(
        0.5, -0.04,
        r'$dC/dt = -\alpha\cdot\bar{D} + \beta\cdot F'
        r' + \gamma\cdot R\cdot\bar{u} - \delta\cdot C$'
        '    |    Dotted lines show instantaneous $C^*(t)$\n'
        'Projected discrete model: $C_{n+1} = \\Pi_{[0,1]}[C_n + \\Delta t \\cdot f(C_n)]$.\n'
        'In the Unmanaged scenario, raw forcing becomes negative (pressure against the lower boundary).\n'
        'Clipping to [0,1] is part of the simulation model, not a post-hoc safety guard.',
        ha='center', fontsize=7, fontstyle='italic', color='#555')

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    save(fig, 'imf_fig1_coherence_dynamics.png')

    log_params(1, 'Core Balance Law: Three Scenarios', {
        'ODE': 'dC/dt = -alpha*D_bar + beta*F + gamma*R*u_bar - delta*C',
        'dt': dt, 'T': T, 'C(0)': C0, 'C_min': C_MIN,
        'alpha': ALPHA, 'beta': BETA, 'gamma': GAMMA, 'delta': DELTA,
        'A_Unmanaged':
            f'F={A["F"]}, R={A["R"]}, u={A["u_bar"]}, '
            f'D_bar={D0_A}+0.012t',
        'B_Delayed':
            f'Pre-t={T_SWITCH_B}: same as A; Post: '
            f'F={B_post["F"]}, R={B_post["R"]}, u={B_post["u_bar"]}, '
            f'D decays from switch value at -0.005/t',
        'C_Resonance':
            f'D_bar={C_sc["D_bar"]}, F={C_sc["F"]}, '
            f'R={C_sc["R"]}, u={C_sc["u_bar"]}, '
            f'C*={cstar_C:.3f}',
    })


# ===================================================================
# FIGURE 2 -- Actual-vs-Ideal State-at-Landmark Error E_k (Proposition 4.3)
# ===================================================================
def fig2_drift_accumulation():
    K  = 10     # generations
    E0 = 0.10   # initial deviation
    k  = np.arange(K + 1)

    regimes = {
        'Near-unity contractivity': {'L': 0.95, 'eps': 0.03,
                            'color': '#1f77b4', 'marker': 'o'},
        'Moderate contractivity':   {'L': 0.80, 'eps': 0.08,
                            'color': '#ff7f0e', 'marker': 's'},
        'Strong contractivity':     {'L': 0.60, 'eps': 0.15,
                            'color': '#d62728', 'marker': '^'},
    }

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 8),
        gridspec_kw={'height_ratios': [1.3, 1]}, sharex=True)

    for name, p in regimes.items():
        L, eps = p['L'], p['eps']
        E_inf = eps / (1 - L)  # stable floor

        # ---- Recurrence: E_{k+1} = L * E_k + eps ----
        E_sim = np.zeros(K + 1)
        E_sim[0] = E0
        for i in range(K):
            E_sim[i + 1] = L * E_sim[i] + eps  # MUST include eps

        # ---- Closed-form bound (constant L, eps) ----
        # E_K = E_0 * L^K + eps * (1 - L^K) / (1 - L)
        E_bound = np.array([
            E0 * L**n + eps * (1 - L**n) / (1 - L) for n in k])

        # Verify match (they should be identical for constant L, eps)
        max_err = np.max(np.abs(E_sim - E_bound))
        print(f"  {name}: max |sim - bound| = {max_err:.2e}, "
              f"E_inf = {E_inf:.3f}")

        # Top panel: E_k trajectory + bound
        ax1.plot(k, E_sim, color=p['color'], lw=2.2,
                 marker=p['marker'], ms=7, zorder=3,
                 label=f"{name} (L={L}, $\\varepsilon$={eps})")
        ax1.plot(k, E_bound, color=p['color'], lw=1, ls='--',
                 alpha=0.6, zorder=2)

        # Stable floor
        ax1.axhline(E_inf, color=p['color'], ls=':', lw=1.2, alpha=0.5)
        ax1.text(K + 0.2, E_inf, f'$E_\\infty$={E_inf:.2f}',
                 fontsize=8, color=p['color'], va='center')

        # Bottom panel: generation-by-generation error increment
        increments = np.diff(E_sim)
        ax2.bar(k[1:] - 0.25 + 0.25 * list(regimes.keys()).index(name),
                increments, width=0.25, color=p['color'], alpha=0.7,
                edgecolor=p['color'], lw=0.8, label=name)

    ax1.set_ylabel('Inheritance Error $E_k$', fontsize=12)
    ax1.set_title(
        'Actual-vs-Ideal State-at-Landmark Error $E_k$ (Proposition 4.3)',
        fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9, loc='upper left', framealpha=0.9)
    ax1.set_ylim(bottom=-0.02)

    # Model text box
    props = dict(boxstyle='round,pad=0.4', facecolor='#f0f0f0',
                 alpha=0.9, edgecolor='#999')
    ax1.text(0.98, 0.45,
             'Recurrence: $E_{k+1} = L_k \\cdot E_k + \\varepsilon_k$\n'
             'Bound: $E_K = E_0 L^K + \\varepsilon\\frac{1 - L^K}{1 - L}$\n'
             'Floor: $E_\\infty = \\varepsilon / (1 - L)$\n'
             '(Dashed = closed-form bound)',
             transform=ax1.transAxes, fontsize=8, va='center', ha='right',
             bbox=props)

    ax2.set_xlabel('Generation index $k$', fontsize=12)
    ax2.set_ylabel('$\\Delta E_k = E_{k} - E_{k-1}$', fontsize=10)
    ax2.legend(fontsize=8, loc='upper right', framealpha=0.9)
    ax2.set_xticks(k)

    fig.text(
        0.5, -0.03,
        'Near-unity contractivity ($L \\approx 1$) retains prior error strongly, yielding the highest '
        'stable floor despite low per-step perturbation.\n'
        'Strong contractivity ($L = 0.60$) reduces the floor even under larger perturbations. '
        '$E_k$ measures actual-vs-ideal trajectory deviation.',
        ha='center', fontsize=8, fontstyle='italic', color='#555')

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    save(fig, 'imf_fig2_drift_accumulation.png')

    log_params(2, 'Actual-vs-Ideal State-at-Landmark Error (Proposition 4.3)', {
        'Recurrence': 'E_{k+1} = L * E_k + eps (includes eps term)',
        'Bound': 'E_K = E_0 * L^K + eps * (1 - L^K) / (1 - L)',
        'K': K, 'E_0': E0,
        'Near-unity_contractivity': f'L=0.95, eps=0.03, E_inf={0.03/0.05:.2f}',
        'Moderate_contractivity': f'L=0.80, eps=0.08, E_inf={0.08/0.20:.2f}',
        'Strong_contractivity': f'L=0.60, eps=0.15, E_inf={0.15/0.40:.2f}',
    })


# ===================================================================
# FIGURE 3 -- Recovery Under Resonance: F Held Constant
# ===================================================================
def fig3_resonance_correction():
    # ---- Parameters (common across all strategies) ----
    ALPHA = 0.10
    BETA  = 0.08
    GAMMA = 0.11
    DELTA = 0.12
    F_CONST = 0.60  # HELD CONSTANT -- blocking issue 3
    U_BAR = 0.65    # same u_bar for all
    C0 = 0.75       # pre-disruption level
    T  = 55
    dt = 0.05
    T_DISRUPT = 5   # disruption at t=5
    C_RECOVERY = 0.65

    # D_bar trajectory (shared): stable, then disruption, then recovery
    def D_bar_fn(ti):
        if ti < T_DISRUPT:
            return 0.15  # stable pre-disruption
        else:
            # jumps to 0.70, then exponentially recovers
            return 0.15 + 0.55 * np.exp(-0.04 * (ti - T_DISRUPT))

    strategies = {
        'Resonance-Matched': {'R': 0.90, 'color': '#2ca02c', 'ls': '-'},
        'Generic':           {'R': 0.35, 'color': '#7f7f7f', 'ls': '-'},
        'Misaligned':        {'R': 0.10, 'color': '#d62728', 'ls': '--'},
    }

    t = np.arange(0, T + dt / 2, dt)
    N = len(t)

    # ---- Verify equilibria ----
    print("  Figure 3 equilibrium checks (F=0.60 held constant):")
    for name, s in strategies.items():
        # At peak disruption (D=0.70)
        check_eq(f'{name} D=0.70', 0.70, F_CONST, s['R'], U_BAR,
                 ALPHA, BETA, GAMMA, DELTA, allow_negative=True)
        # At recovery (D~0.20)
        check_eq(f'{name} D=0.20', 0.20, F_CONST, s['R'], U_BAR,
                 ALPHA, BETA, GAMMA, DELTA)

    # ---- Simulate ----
    results = {}
    recovery_times = {}

    for name, s in strategies.items():
        C_arr = np.zeros(N)
        C_arr[0] = C0
        rt = None
        for i in range(1, N):
            D_i = D_bar_fn(t[i])
            dCdt = canonical_dCdt(
                C_arr[i - 1], D_i, F_CONST, s['R'], U_BAR,
                ALPHA, BETA, GAMMA, DELTA)
            # Projected model boundary -- clipping is part of the
            # simulation model for scenarios with negative forcing
            C_arr[i] = np.clip(C_arr[i - 1] + dt * dCdt, 0.0, 1.0)
            if rt is None and t[i] > T_DISRUPT + 2 and C_arr[i] >= C_RECOVERY:
                rt = t[i]
        results[name] = C_arr
        recovery_times[name] = rt
        print(f"    {name}: recovery at t={rt:.1f}" if rt else
              f"    {name}: no recovery by t={T}")

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.axvspan(T_DISRUPT, T, color='#ffffcc', alpha=0.25,
               label='Post-disruption period')
    ax.axvline(T_DISRUPT, color='grey', ls=':', lw=1, alpha=0.6)
    ax.text(T_DISRUPT + 0.5, 0.95, 'Disruption', fontsize=9, color='grey',
            fontweight='bold', rotation=90, va='top')
    ax.axhline(C_RECOVERY, color='grey', ls=':', lw=0.8, alpha=0.4)
    ax.text(0.3, C_RECOVERY + 0.01,
            f'Recovery target $C={C_RECOVERY}$', fontsize=8, color='grey')

    y_offsets = iter([0.04, -0.06, -0.04])
    for name, s in strategies.items():
        ax.plot(t, results[name], color=s['color'], lw=2.2, ls=s['ls'],
                label=f"{name} (R={s['R']})")
        rt = recovery_times[name]
        yoff = next(y_offsets)
        if rt is not None:
            ax.axvline(rt, color=s['color'], ls=':', lw=0.7, alpha=0.4)
            ax.annotate(
                f't={rt:.1f}', xy=(rt, C_RECOVERY),
                xytext=(rt + 1.5, C_RECOVERY + yoff),
                fontsize=9, color=s['color'], fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=s['color'], lw=1))
        else:
            ax.text(T - 8, results[name][-1] + 0.02,
                    'No recovery', fontsize=9,
                    color=s['color'], fontweight='bold')

    # Parameter table
    col_labels = ['Strategy', 'R', 'F (const)', r'$\bar{u}$',
                  r'$\alpha$', r'$\beta$', r'$\gamma$', r'$\delta$']
    table_data = [
        ['Resonance', '0.90', '0.60', '0.65',
         f'{ALPHA}', f'{BETA}', f'{GAMMA}', f'{DELTA}'],
        ['Generic',   '0.35', '0.60', '0.65',
         f'{ALPHA}', f'{BETA}', f'{GAMMA}', f'{DELTA}'],
        ['Misaligned','0.10', '0.60', '0.65',
         f'{ALPHA}', f'{BETA}', f'{GAMMA}', f'{DELTA}'],
    ]
    add_param_table(ax, col_labels, table_data, loc='lower right',
                    bbox=[0.40, 0.02, 0.58, 0.18])

    ax.set_xlabel('Time $t$ (arbitrary units)', fontsize=12)
    ax.set_ylabel('Coherence $C(t)$', fontsize=12)
    ax.set_title(
        'Recovery Under Resonance Alignment (Proposition 4.4)',
        fontsize=13, fontweight='bold')

    ax.text(
        0.5, 0.97,
        'F held constant at 0.60 across all strategies; only R varies. '
        'This isolates the resonance contribution per Proposition 4.4.',
        transform=ax.transAxes, fontsize=8, ha='center', va='top',
        fontstyle='italic', color='#555')

    ax.set_xlim(0, T)
    ax.set_ylim(0.0, 1.0)
    ax.legend(fontsize=9, loc='center right', framealpha=0.9)
    fig.tight_layout()
    save(fig, 'imf_fig3_resonance_correction.png')

    log_params(3, 'Recovery Under Resonance Alignment (F Constant)', {
        'ODE': 'dC/dt = -alpha*D_bar + beta*F + gamma*R*u_bar - delta*C',
        'F_CONSTANT': F_CONST,
        'u_bar': U_BAR,
        'alpha': ALPHA, 'beta': BETA, 'gamma': GAMMA, 'delta': DELTA,
        'C(0)': C0, 'T_disrupt': T_DISRUPT, 'C_recovery': C_RECOVERY,
        'D_bar(t)': '0.15 + 0.55*exp(-0.04*(t-5)) for t>=5, else 0.15',
        'Resonance_R': 0.90, 'Generic_R': 0.35, 'Misaligned_R': 0.10,
        'Recovery_times': {n: f'{v:.1f}' if v else 'None'
                          for n, v in recovery_times.items()},
    })


# ===================================================================
# FIGURE 4 -- Qualitative Coherence Across Four IMF Lanes
# ===================================================================
def fig4_lane_comparison():
    np.random.seed(SEED + 4)  # deterministic noise
    ALPHA_COMMON = 0.10
    BETA_COMMON  = 0.08
    GAMMA_COMMON = 0.11
    T  = 60
    dt = 0.05
    C0 = 0.50  # all lanes start from same initial state

    # Lane-specific parameters designed for target C*
    lanes = {
        'Quantum Information': {
            'delta': 0.20,  # fast convergence (tau=5)
            'D_bar': 0.08, 'F': 0.80, 'R': 0.70, 'u_bar': 0.65,
            'D_growth': 0.004,  # fast decoherence
            'noise_sigma': 0.0,
            'color': '#9467bd', 'ls': '-',
        },
        'Software Systems': {
            'delta': 0.12,  # moderate (tau=8.3)
            'D_bar': 0.10, 'F': 0.68, 'R': 0.62, 'u_bar': 0.60,
            'D_growth': 0.002,
            'noise_sigma': 0.0,
            'color': '#ff7f0e', 'ls': '--',
        },
        'Enterprise Governance': {
            'delta': 0.10,  # slow (tau=10)
            'D_bar': 0.10, 'F': 0.58, 'R': 0.55, 'u_bar': 0.52,
            'D_growth': 0.001,
            'noise_sigma': 0.0,
            'color': '#1f77b4', 'ls': '-.',
        },
        'Ecological Inheritance': {
            'delta': 0.08,  # slowest (tau=12.5)
            'D_bar': 0.10, 'F': 0.35, 'R': 0.50, 'u_bar': 0.50,
            'D_growth': 0.003,  # notable drift
            'noise_sigma': 0.012,  # highest variance
            'color': '#2ca02c', 'ls': ':',
        },
    }

    t = np.arange(0, T + dt / 2, dt)
    N = len(t)

    # ---- Verify equilibria ----
    print("  Figure 4 equilibrium checks:")
    cstar_values = {}
    for name, p in lanes.items():
        # Initial equilibrium
        cs0 = check_eq(f'{name} t=0',
                       p['D_bar'], p['F'], p['R'], p['u_bar'],
                       ALPHA_COMMON, BETA_COMMON, GAMMA_COMMON,
                       p['delta'])
        # Final equilibrium (D_bar increased)
        D_final = p['D_bar'] + p['D_growth'] * T
        cs_f = check_eq(f'{name} t={T}',
                        D_final, p['F'], p['R'], p['u_bar'],
                        ALPHA_COMMON, BETA_COMMON, GAMMA_COMMON,
                        p['delta'])
        cstar_values[name] = (cs0, cs_f)

    # ---- Simulate ----
    results = {}
    for name, p in lanes.items():
        C_arr = np.zeros(N)
        C_arr[0] = C0
        for i in range(1, N):
            D_i = p['D_bar'] + p['D_growth'] * t[i]
            dCdt = canonical_dCdt(
                C_arr[i - 1], D_i, p['F'], p['R'], p['u_bar'],
                ALPHA_COMMON, BETA_COMMON, GAMMA_COMMON, p['delta'])
            noise = p['noise_sigma'] * np.random.randn() if p['noise_sigma'] > 0 else 0.0
            # Projected model boundary -- clipping is part of the
            # simulation model for scenarios with negative forcing
            C_arr[i] = np.clip(C_arr[i - 1] + dt * dCdt + noise * np.sqrt(dt),
                               0.0, 1.0)
        results[name] = C_arr

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=FIGSIZE)

    for name, p in lanes.items():
        ax.plot(t, results[name], color=p['color'], lw=2.5,
                ls=p['ls'], label=name)
        C_final = results[name][-1]
        # Mark actual simulated value at final time step
        ax.plot(T, C_final, marker='o', ms=5,
                color=p['color'], zorder=5)
        ax.text(T + 0.5, C_final,
                f'C({T})={C_final:.2f}', fontsize=7,
                color=p['color'], va='center')

    # Parameter table
    col_labels = ['Lane', r'$\delta$', r'$\bar{D}_0$',
                  'F', 'R', r'$\bar{u}$', r'$D_{gr}$', r'$\sigma$']
    table_data = []
    for name, p in lanes.items():
        short = name.split()[0]
        table_data.append([
            short, f"{p['delta']}", f"{p['D_bar']}",
            f"{p['F']}", f"{p['R']}", f"{p['u_bar']}",
            f"{p['D_growth']}", f"{p['noise_sigma']}"])
    add_param_table(ax, col_labels, table_data, loc='lower left',
                    bbox=[0.02, 0.02, 0.58, 0.22])

    ax.set_xlabel('Time $t$ (arbitrary units)', fontsize=12)
    ax.set_ylabel('Coherence $C(t)$ (arbitrary units)', fontsize=12)
    ax.set_title(
        'Qualitative Coherence Dynamics Across Four IMF Lanes',
        fontsize=13, fontweight='bold')

    ax.text(
        0.5, 0.97,
        'Qualitative illustration only. Parameter values are '
        'domain-inspired but not empirically calibrated.\n'
        'All four systems use the canonical ODE with '
        'domain-appropriate coefficient choices. '
        r'Common: $\alpha$='
        f'{ALPHA_COMMON}, '
        r'$\beta$='
        f'{BETA_COMMON}, '
        r'$\gamma$='
        f'{GAMMA_COMMON}',
        transform=ax.transAxes, fontsize=7, ha='center', va='top',
        fontstyle='italic', color='#555')

    ax.set_xlim(0, T + 8)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=9, loc='upper right', framealpha=0.9)
    fig.tight_layout()
    save(fig, 'imf_fig4_lane_comparison.png')

    log_params(4, 'Qualitative Coherence Across Four IMF Lanes', {
        'ODE': 'dC/dt = -alpha*D_bar + beta*F + gamma*R*u_bar - delta*C',
        'alpha_common': ALPHA_COMMON,
        'beta_common': BETA_COMMON,
        'gamma_common': GAMMA_COMMON,
        'dt': dt, 'T': T, 'C(0)': C0,
        **{f'{n}_Cstar': f'{cstar_values[n][0]:.3f} -> {cstar_values[n][1]:.3f}'
           for n in lanes},
        **{n: (f"delta={p['delta']}, D0={p['D_bar']}, F={p['F']}, "
              f"R={p['R']}, u={p['u_bar']}, D_growth={p['D_growth']}, "
              f"sigma={p['noise_sigma']}")
           for n, p in lanes.items()},
    })


# ===================================================================
# FIGURE 5 -- ExecutionProof Engine Alignment Matrix (Categorical)
# ===================================================================
def fig5_executionproof_mapping():
    imf_rows = [
        'Drift $\\bar{D}$',
        'Fidelity $F$',
        'Resonance $R$',
        'Stewardship $\\bar{u}$',
        'Coherence $C$',
        'Error Floor $E_\\infty$',
    ]
    engines = [
        'Authority', 'Evidence', 'Constraint',
        'Control/\nRecord', 'Coherence\nEnvelope',
    ]

    # 0=None, 1=Indirect, 2=Secondary, 3=Primary
    LEVEL_NAMES = ['None', 'Indirect', 'Secondary', 'Primary']
    data = np.array([
        # Auth  Evid  Constr  Ctrl/Rec  Coh.Env
        [0,    2,    1,      2,        3],   # Drift D_bar
        [1,    3,    1,      2,        2],   # Fidelity F
        [1,    1,    0,      2,        2],   # Resonance R
        [2,    1,    2,      3,        1],   # Stewardship u_bar
        [1,    2,    2,      3,        3],   # Coherence C
        [3,    3,    3,      2,        3],   # Error Floor E_inf
    ])

    cmap_colors = ['#f7f7f7', '#c6dbef', '#4292c6', '#08306b']
    cmap = mcolors.ListedColormap(cmap_colors)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(10, 6.5))
    im = ax.imshow(data, cmap=cmap, norm=norm, aspect='auto')

    ax.set_xticks(range(len(engines)))
    ax.set_xticklabels(engines, fontsize=11, fontweight='bold')
    ax.set_yticks(range(len(imf_rows)))
    ax.set_yticklabels(imf_rows, fontsize=11)
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')

    # Text annotations -- CATEGORICAL LABELS ONLY (no decimals)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            txt_color = 'white' if val >= 2 else '#333'
            ax.text(j, i, LEVEL_NAMES[val], ha='center', va='center',
                    fontsize=9, color=txt_color,
                    fontweight='bold' if val == 3 else 'normal')

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3], shrink=0.8,
                        pad=0.02)
    cbar.ax.set_yticklabels(LEVEL_NAMES, fontsize=9)
    cbar.ax.set_ylabel('Alignment Level', fontsize=10, rotation=270,
                       labelpad=15)

    # Grid lines
    for i in range(data.shape[0] + 1):
        ax.axhline(i - 0.5, color='white', lw=2)
    for j in range(data.shape[1] + 1):
        ax.axvline(j - 0.5, color='white', lw=2)

    ax.set_title(
        'ExecutionProof Engine Alignment with IMF Quantities '
        '-- Qualitative Design Matrix',
        fontsize=13, fontweight='bold', pad=15)

    fig.text(
        0.5, -0.04,
        'Qualitative design mapping. Labels are conceptual '
        'alignment categories, not empirically measured scores.\n'
        'Categorical alignment only -- not derived from '
        'empirical measurement.',
        ha='center', fontsize=9, fontstyle='italic', color='#555')

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    save(fig, 'imf_fig5_executionproof_mapping.png')

    log_params(5, 'ExecutionProof Engine Alignment Matrix (Categorical)', {
        'IMF_rows': [r.replace('$', '').replace('\\', '')
                     for r in imf_rows],
        'Engines': ['Authority', 'Evidence', 'Constraint',
                    'Control/Record', 'Coherence Envelope'],
        'Levels': '0=None, 1=Indirect, 2=Secondary, 3=Primary',
        'Format': 'Categorical labels only -- NO decimal scores',
    })


# ===================================================================
# FIGURE 6 -- Drift as Early Warning (Landmark AUC)
# ===================================================================
def fig6_early_warning():
    np.random.seed(SEED)
    dt = 0.1
    T  = 120
    N_TRIALS = 200
    C_MIN = 0.25
    CONSEC = 3       # consecutive steps below C_min for collapse
    SIGMA_D = 0.02
    SIGMA_C = 0.03
    D0 = 0.10

    t = np.arange(0, T + dt / 2, dt)
    N = len(t)
    TAU_TIME = np.array([5, 10, 15, 20, 25])
    TAU_STEPS = (TAU_TIME / dt).astype(int)

    all_C = np.zeros((N_TRIALS, N))
    all_D = np.zeros((N_TRIALS, N))
    collapse_step = np.full(N_TRIALS, N, dtype=int)

    for trial in range(N_TRIALS):
        D = np.zeros(N)
        C = np.zeros(N)
        D[0] = D0
        C[0] = np.clip(1.0 - D[0] ** 1.5 + SIGMA_C * np.random.randn(),
                        0.0, 1.0)
        consec = 0
        collapsed = False
        for i in range(1, N):
            lam = 0.005 + 0.003 * max(0.0, t[i] - 40) / 60.0
            D[i] = np.clip(
                D[i - 1] + lam * dt
                + SIGMA_D * np.sqrt(dt) * np.random.randn(),
                0.0, 1.0)
            # Radial coupling: C = phi(D_bar) + noise
            C[i] = np.clip(
                1.0 - D[i] ** 1.5 + SIGMA_C * np.random.randn(),
                0.0, 1.0)
            if not collapsed:
                if C[i] < C_MIN:
                    consec += 1
                    if consec >= CONSEC:
                        collapse_step[trial] = i
                        collapsed = True
                else:
                    consec = 0
        all_C[trial] = C
        all_D[trial] = D

    collapse_times = collapse_step * dt
    collapsed_mask = collapse_step < N
    n_collapsed = collapsed_mask.sum()
    if n_collapsed > 0:
        mean_ct = collapse_times[collapsed_mask].mean()
        std_ct = collapse_times[collapsed_mask].std()
    else:
        mean_ct, std_ct = T, 0.0
    print(f"  Collapsed: {n_collapsed}/{N_TRIALS}, "
          f"mean t={mean_ct:.1f} +/- {std_ct:.1f}")

    # ---- Landmark-time AUC (post-collapse windows excluded) ----
    auc_values = []
    for tau_s, tau_t in zip(TAU_STEPS, TAU_TIME):
        labels = []
        scores = []
        for trial in range(N_TRIALS):
            cs = collapse_step[trial]
            # Find first time C < C_MIN for this trial (any crossing)
            first_below = N  # default: never below
            for fb_idx in range(N):
                if all_C[trial, fb_idx] < C_MIN:
                    first_below = fb_idx
                    break
            for t_idx in range(0, N - tau_s, 10):
                # Exclude post-collapse time points
                if t_idx >= first_below:
                    continue
                label = 1 if (cs >= t_idx and cs < t_idx + tau_s) else 0
                labels.append(label)
                scores.append(all_D[trial, t_idx])
        labels = np.array(labels)
        scores = np.array(scores)
        if len(np.unique(labels)) < 2:
            auc_values.append(0.5)
        else:
            auc_values.append(roc_auc_score(labels, scores))
        print(f"    tau={tau_t}: AUC={auc_values[-1]:.3f} "
              f"(pos={labels.sum()}, neg={(1-labels).sum()})")

    # ---- Statistics ----
    C_mean = all_C.mean(axis=0)
    C_std  = all_C.std(axis=0)
    D_mean = all_D.mean(axis=0)
    D_std  = all_D.std(axis=0)

    # ---- Plot ----
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(10, 11),
        gridspec_kw={'height_ratios': [1, 1, 0.8]})

    # -- Top: C(t) --
    ax1.fill_between(t, C_mean - C_std, C_mean + C_std,
                     color='#1f77b4', alpha=0.2)
    ax1.plot(t, C_mean, color='#1f77b4', lw=2,
             label=r'$C(t)$ mean $\pm$ 1$\sigma$')
    ax1.axhline(C_MIN, color='red', ls='--', lw=1.2, alpha=0.7)
    ax1.text(1, C_MIN + 0.01, f'$C_{{min}} = {C_MIN}$',
             fontsize=9, color='red')
    if n_collapsed > 0:
        ax1.axvspan(mean_ct - std_ct, mean_ct + std_ct,
                    color='red', alpha=0.08)
        ax1.axvline(mean_ct, color='#d62728', ls=':', lw=1.2, alpha=0.7)
        ax1.text(mean_ct + 1, 0.88,
                 f'Mean collapse\n$t={mean_ct:.0f} \\pm {std_ct:.0f}$',
                 fontsize=9, color='#d62728', fontweight='bold')
    ax1.set_ylabel('Coherence $C(t)$', fontsize=11)
    ax1.set_title(
        'Drift as Early Warning -- Internal Model Recovery Test',
        fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9, loc='upper right')
    ax1.set_ylim(-0.05, 1.05)

    # -- Middle: D(t) --
    tau_warn = 15
    if n_collapsed > 0:
        warn_start = mean_ct - tau_warn
        ax2.axvspan(warn_start, mean_ct, color='#fff3cd', alpha=0.6,
                    label=r'Early warning window ($\tau$=15)')
    ax2.fill_between(t, D_mean - D_std, D_mean + D_std,
                     color='#d62728', alpha=0.15)
    ax2.plot(t, D_mean, color='#d62728', lw=2,
             label=r'$\bar{D}(t)$ mean $\pm$ 1$\sigma$')
    ax2.set_ylabel(r'Drift $\bar{D}(t)$', fontsize=11)
    ax2.legend(fontsize=9, loc='upper left')
    ax2.set_ylim(-0.05, 1.05)

    # -- Bottom: AUC bars --
    bar_colors = ['#2ca02c' if a >= 0.75 else '#ff7f0e' if a >= 0.5
                  else '#d62728' for a in auc_values]
    x_labels = [f'$\\tau$={tt}' for tt in TAU_TIME]
    bars = ax3.bar(x_labels, auc_values, color=bar_colors,
                   edgecolor='#333', lw=0.8)
    ax3.axhline(0.5, color='grey', ls='--', lw=1, alpha=0.6)
    ax3.text(len(TAU_TIME) - 0.7, 0.52, 'Chance', fontsize=8, color='grey')
    ax3.set_ylabel('AUC (landmark-time labels)', fontsize=11)
    ax3.set_xlabel(r'Prediction horizon $\tau$ (time units)', fontsize=11)
    ax3.set_ylim(0, 1.05)
    for bar, auc in zip(bars, auc_values):
        ax3.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.02,
                 f'{auc:.2f}', ha='center', fontsize=9, fontweight='bold')

    # ---- PROMINENT DISCLOSURE (blocking issue 5) ----
    disclosure = (
        'INTERNAL MODEL RECOVERY TEST ONLY\n'
        r'$C$ is directly constructed from $\bar{D}$ via '
        r'$C = 1 - \bar{D}^{1.5} + \mathrm{noise}$.'
        '\nThe high AUC reflects model architecture '
        '(radial coupling),\nnot empirical predictive validity. '
        'This figure does NOT use Postulate 4.1.\n'
        'Post-collapse time points excluded from landmark labeling.'
    )
    props = dict(boxstyle='round,pad=0.5', facecolor='#fff3cd',
                 edgecolor='#e0a800', alpha=0.95, lw=1.5)
    ax1.text(0.02, 0.30, disclosure, transform=ax1.transAxes,
             fontsize=8, va='top', ha='left', bbox=props,
             fontstyle='italic', color='#664d00')

    for ax in (ax1, ax2):
        ax.set_xlim(0, T)

    fig.tight_layout()
    save(fig, 'imf_fig6_early_warning.png')

    log_params(6, 'Drift as Early Warning (Landmark AUC)', {
        'Coupling': 'C(t) = clip(1 - D_bar^1.5 + sigma_C*noise, 0, 1)',
        'NOT_Postulate_4.1': 'This figure uses explicit radial coupling',
        'D_dynamics': 'dD/dt = lambda(t) + sigma_D * xi',
        'lambda(t)': '0.005 + 0.003 * max(0, t-40)/60',
        'dt': dt, 'T': T, 'N_trials': N_TRIALS, 'seed': SEED,
        'C_min': C_MIN, 'consec_steps': CONSEC,
        'sigma_D': SIGMA_D, 'sigma_C': SIGMA_C, 'D_bar(0)': D0,
        'tau_values': TAU_TIME.tolist(),
        'AUC_values': [f'{a:.3f}' for a in auc_values],
        'AUC_method': 'landmark-time labels (collapse within next tau, post-collapse excluded)',
        'collapsed': f'{n_collapsed}/{N_TRIALS}',
        'mean_collapse': f'{mean_ct:.1f} +/- {std_ct:.1f}',
    })


# ===================================================================
# MAIN
# ===================================================================
if __name__ == '__main__':
    banner = (
        f"{'=' * 70}\n"
        f"IMF Simulation Suite v1.2.1 -- Patch Build\n"
        f"Remnant Fieldworks Inc. | Inheritance Math Framework\n"
        f"Canonical ODE: dC/dt = -alpha*D_bar + beta*F "
        f"+ gamma*R*u_bar - delta*C\n"
        f"Output: {OUT}   Seed: {SEED}\n"
        f"{'=' * 70}"
    )
    print(f"\n{banner}\n")

    fig1_coherence_dynamics()
    fig2_drift_accumulation()
    fig3_resonance_correction()
    fig4_lane_comparison()
    fig5_executionproof_mapping()
    fig6_early_warning()

    # ---- Save manifest ----
    manifest_path = os.path.join(OUT, 'parameters_manifest.txt')
    with open(manifest_path, 'w') as f:
        f.write('IMF Simulation Suite v1.2.1 -- Parameters Manifest\n')
        f.write(f'Remnant Fieldworks Inc. | seed={SEED}\n')
        f.write('Canonical ODE: dC/dt = -alpha*D_bar + beta*F '
                '+ gamma*R*u_bar - delta*C\n')
        f.write('\n'.join(manifest_lines))
        f.write('\n')
    saved_files.append(manifest_path)
    print(f'\n  -> Saved: {manifest_path}')

    print(f"\n{'=' * 70}")
    print('All output files:')
    for fp in sorted(saved_files):
        print(f'  {fp}')
    print(f'  {os.path.abspath(__file__)}')
    print(f"{'=' * 70}\n")

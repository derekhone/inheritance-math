#!/usr/bin/env python3
"""
Solver Convergence Check for IMF v1.2.1
Compares Euler integrations (dt=0.10, 0.05, 0.01) against
RK45 (scipy.integrate.solve_ivp) reference for the
resonance-matched scenario from Figure 1.

Usage: python solver_convergence_check.py
"""

import csv
import os

import numpy as np
from scipy.integrate import solve_ivp

# ---- Exact v1.2 resonance-matched parameters (from imf_simulations.py) ----
ALPHA = 0.10
BETA  = 0.08
GAMMA = 0.11
DELTA = 0.12
D_BAR = 0.25   # constant for resonance-matched scenario
F     = 0.70
R     = 0.85
U_BAR = 0.75
C0    = 0.65   # initial coherence (same as Fig 1)
T     = 60.0   # integration horizon (same as Fig 1)

C_THRESHOLD = 0.75  # threshold-crossing target

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(OUT_DIR, 'solver_convergence_results.csv')


def dCdt(t, C):
    """Canonical ODE: dC/dt = -alpha*D_bar + beta*F + gamma*R*u_bar - delta*C"""
    return -ALPHA * D_BAR + BETA * F + GAMMA * R * U_BAR - DELTA * C


def euler_integrate(dt):
    """Forward Euler with projected clipping to [0,1]."""
    t_arr = np.arange(0, T + dt / 2, dt)
    N = len(t_arr)
    C = np.zeros(N)
    C[0] = C0
    for i in range(1, N):
        C[i] = np.clip(C[i - 1] + dt * dCdt(t_arr[i - 1], C[i - 1]), 0.0, 1.0)
    return t_arr, C


def find_crossing(t_arr, C_arr, threshold):
    """First time C exceeds threshold. Returns None if never crossed."""
    for i in range(len(C_arr)):
        if C_arr[i] >= threshold:
            return t_arr[i]
    return None


def main():
    # ---- RK45 reference solution ----
    sol = solve_ivp(
        dCdt, [0, T], [C0],
        method='RK45', rtol=1e-8, atol=1e-10,
        dense_output=True,
        max_step=0.01,
    )
    assert sol.success, f"RK45 failed: {sol.message}"

    # ---- Euler runs ----
    dt_values = [0.10, 0.05, 0.01]
    euler_results = []
    for dt in dt_values:
        t_e, C_e = euler_integrate(dt)
        euler_results.append((dt, t_e, C_e))

    # ---- Compute deviations ----
    # Evaluate RK45 on a fine common grid for comparison
    t_fine = np.arange(0, T + 0.001, 0.001)
    C_rk45_fine = sol.sol(t_fine)[0]
    rk45_crossing = find_crossing(t_fine, C_rk45_fine, C_THRESHOLD)

    rows = []
    for dt, t_e, C_e in euler_results:
        # Interpolate RK45 onto Euler grid
        C_rk45_at_euler = sol.sol(t_e)[0]
        max_dev = np.max(np.abs(C_e - C_rk45_at_euler))
        euler_crossing = find_crossing(t_e, C_e, C_THRESHOLD)
        if euler_crossing is not None and rk45_crossing is not None:
            crossing_diff = euler_crossing - rk45_crossing
        else:
            crossing_diff = None
        rows.append({
            'Solver': 'Euler',
            'dt': f'{dt}',
            'MaxAbsDev': f'{max_dev:.6f}',
            'ThresholdCrossingTime': f'{euler_crossing:.2f}' if euler_crossing else 'N/A',
            'ThresholdCrossingDiff': f'{crossing_diff:+.4f}' if crossing_diff is not None else 'N/A',
        })

    rows.append({
        'Solver': 'RK45 (ref)',
        'dt': '--',
        'MaxAbsDev': '0.000000',
        'ThresholdCrossingTime': f'{rk45_crossing:.2f}' if rk45_crossing else 'N/A',
        'ThresholdCrossingDiff': '(reference)',
    })

    # ---- Print report ----
    print()
    print('Solver Convergence Report -- IMF v1.2.1')
    print(f'Equation: dC/dt = -alpha*Dbar + beta*F + gamma*R*ubar - delta*C')
    print(f'Scenario: Resonance-Matched (v1.2 parameters)')
    print(f'  alpha={ALPHA}, beta={BETA}, gamma={GAMMA}, delta={DELTA}')
    print(f'  D_bar={D_BAR}, F={F}, R={R}, u_bar={U_BAR}, C(0)={C0}, T={T}')
    print(f'  Threshold: C > {C_THRESHOLD}')
    print('=' * 80)
    header = f'{"Solver":<16} {"dt":>6}   {"Max |dev vs RK45|":>18}  {"Threshold-crossing time":>24}'
    print(header)
    print('-' * 80)
    for r in rows:
        tcross = r['ThresholdCrossingTime']
        tdiff = r['ThresholdCrossingDiff']
        tcross_str = f'{tcross} (diff: {tdiff})' if tdiff != '(reference)' else f'{tcross} (reference)'
        print(f'{r["Solver"]:<16} {r["dt"]:>6}   {r["MaxAbsDev"]:>18}  {tcross_str:>24}')
    print('=' * 80)

    # ---- Write CSV ----
    fieldnames = ['Solver', 'dt', 'MaxAbsDev', 'ThresholdCrossingTime', 'ThresholdCrossingDiff']
    with open(CSV_PATH, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f'\nConvergence check complete. Results saved to {CSV_PATH}')


if __name__ == '__main__':
    main()

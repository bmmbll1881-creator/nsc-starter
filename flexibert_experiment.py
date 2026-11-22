#!/usr/bin/env python3
"""Evaluate the frozen orthogonalized tilt on FlexiBERT's 500 BERT architectures.

Data: the official `training-free-nas` repository's `data/BERT_benchmark.json`.
Each entry contains per-layer operation type, heads and FFN width plus a GLUE
score.  Capacities follow the NSC FlexiBERT convention:
  SA  : psi(3H,H) + psi(H,H)
  LT  : 2 psi(H,H)
  DSC : psi(H,H)
  FFN : for each of nff blocks, psi(f,H) + psi(H,f)

Two capacity conventions are reported:
  * Xavier: sqrt(2/(fan_in+fan_out)), matching the Nsc FlexiBERT protocol;
  * fixed: sigma=0.02, matching student_score.py's GPT-2 convention.

The frozen method uses the coefficients and BETA in student_score.py.  A
FlexiBERT-only train/test fit is added as an adaptation diagnostic.
"""
import json
import math
import random
import statistics
from pathlib import Path

from evaluate import correlation, pair_summary, ranks
from nsc import psi_mp
from student_score import (
    BETA, COEF_INTERCEPT, COEF_LOG_CAPACITY, COEF_LOG_DEPTH,
)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data' / 'BERT_benchmark.json'


def rho(a, b):
    """Compute Spearman correlation using average ranks for ties."""
    return correlation(ranks(a), ranks(b))


def tau_b(a, b):
    """Compute Kendall tau-b over all unordered architecture pairs."""
    return pair_summary(a, b, [(i, j) for i in range(len(a))
                               for j in range(i)])['tau_b']


def xavier(m, n):
    """Compute Xavier initialization scale for a matrix shape."""
    return math.sqrt(2.0 / (m + n))


def layer_capacity(layer, hidden, sigma_mode):
    """Compute a FlexiBERT block capacity for its operation and projections."""
    op = layer['operation_type']
    nff = layer.get('num_feed_forward', 1)
    ff_total = layer.get('feed_forward_dimension', 1024)
    ff = ff_total // nff

    def sigma(m, n):
        """Choose the fixed or Xavier scale for the projection."""
        return xavier(m, n) if sigma_mode == 'xavier' else 0.02

    if op == 'SA':
        value = (psi_mp(3 * hidden, hidden, sigma(3 * hidden, hidden))
                 + psi_mp(hidden, hidden, sigma(hidden, hidden)))
    elif op == 'LT':
        value = 2 * psi_mp(hidden, hidden, sigma(hidden, hidden))
    elif op == 'DSC':
        value = psi_mp(hidden, hidden, sigma(hidden, hidden))
    else:
        raise ValueError(f'unknown operation type {op}')
    value += 2 * nff * psi_mp(ff, hidden, sigma(ff, hidden))
    return value


def features(arch, sigma_mode):
    """Summarize layer capacities as total, depth and positional moment."""
    h = arch['hparams']['model_hparam_overrides']['hidden_size']
    layers = arch['hparams']['model_hparam_overrides']['nas_config']['encoder_layers']
    caps = [layer_capacity(layer, h, sigma_mode) for layer in layers]
    length = len(caps)
    total = math.fsum(caps)
    tilt = math.fsum(((i + 0.5) / length - 0.5) * c
                     for i, c in enumerate(caps))
    return {'S0': total, 'D': tilt, 'L': length,
            'glue': arch['scores']['glue']}


def fit_ols(rows, values):
    """Fit FlexiBERT tilt residualization coefficients."""
    X = [[1.0, math.log(r['S0']), math.log(r['L'])] for r in rows]
    p = 3
    A = [[sum(row[a] * row[b] for row in X) for b in range(p)]
         for a in range(p)]
    b = [sum(row[a] * value for row, value in zip(X, values))
         for a in range(p)]
    # Gaussian elimination
    M = [A[i][:] + [b[i]] for i in range(p)]
    for col in range(p):
        pivot = max(range(col, p), key=lambda r: abs(M[r][col]))
        M[col], M[pivot] = M[pivot], M[col]
        scale = M[col][col] or 1e-15
        for r in range(col + 1, p):
            factor = M[r][col] / scale
            for c in range(col, p + 1):
                M[r][c] -= factor * M[col][c]
    out = [0.0] * p
    for r in range(p - 1, -1, -1):
        out[r] = (M[r][p] - sum(M[r][c] * out[c]
                                for c in range(r + 1, p))) / M[r][r]
    return out


def residual(row, coef):
    """Subtract the predicted tilt from the observed positional tilt."""
    predicted = (coef[0] + coef[1] * math.log(row['S0'])
                 + coef[2] * math.log(row['L']))
    return row['D'] - predicted


def main():
    """Evaluate NSC variants against the FlexiBERT GLUE labels."""
    archs = json.loads(DATA.read_text(encoding='utf-8'))
    print(f'{len(archs)} FlexiBERT architectures')
    print(f'Frozen BETA={BETA:+.2f}, coefficients from student_score.py')
    out = {'n': len(archs), 'beta': BETA, 'conventions': {}}
    frozen_coef = [COEF_INTERCEPT, COEF_LOG_CAPACITY, COEF_LOG_DEPTH]

    for mode in ('xavier', 'fixed'):
        rows = [features(a, mode) for a in archs]
        y = [r['glue'] for r in rows]
        base = [r['S0'] for r in rows]
        final = [r['S0'] + BETA * residual(r, frozen_coef) for r in rows]
        r0, r1 = rho(base, y), rho(final, y)
        t0, t1 = tau_b(base, y), tau_b(final, y)
        print(f'\n=== {mode} ===')
        print(f'  NSC   rho={r0:.5f} tau={t0:.5f}')
        print(f'  tilt  rho={r1:.5f} tau={t1:.5f} '
              f'delta_rho={r1-r0:+.5f}')

        # FlexiBERT-only adaptation diagnostic: 250/250 random split.
        rng = random.Random(42)
        order = list(range(len(rows)))
        rng.shuffle(order)
        train, test = order[:250], order[250:]
        coef = fit_ols([rows[i] for i in train],
                       [rows[i]['D'] for i in train])
        grid = [i * 0.02 for i in range(-300, 301)]
        beta = max(grid, key=lambda b: rho(
            [rows[i]['S0'] + b * residual(rows[i], coef) for i in train],
            [y[i] for i in train]))
        pred = [rows[i]['S0'] + beta * residual(rows[i], coef) for i in test]
        test_base = [rows[i]['S0'] for i in test]
        test_y = [y[i] for i in test]
        print(f'  adapted beta={beta:+.2f} test_rho={rho(pred, test_y):.5f} '
              f'(NSC test={rho(test_base, test_y):.5f})')
        out['conventions'][mode] = {
            'nsc_rho': r0, 'nsc_tau': t0,
            'tilt_rho': r1, 'tilt_tau': t1, 'delta_rho': r1 - r0,
            'adapted_beta': beta,
            'adapted_test_rho': rho(pred, test_y),
            'adapted_nsc_test_rho': rho(test_base, test_y),
        }

    out['paper_reference'] = {'nsc_rho': 0.8843, 'resnsc_rho': 0.8833,
                              'nsc_tau': 0.6946, 'resnsc_tau': 0.6921}
    path = ROOT / 'flexibert_results.json'
    path.write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(f'\nWrote {path}')


if __name__ == '__main__':
    main()

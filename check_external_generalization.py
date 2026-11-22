#!/usr/bin/env python3
"""Held-out generalisation check on public LiteTransformerSearch logs.

The local 200-architecture panel is itself a re-split of the public
`random_GPT2_wt103` logs, so it cannot serve as an external test.  This script
uses independently trained architecture sets from the same upstream release:
GPT-2 on LM1B, Transformer-XL on WikiText-103 / LM1B, a homogeneous-depth
stress set, and a selected Pareto set on LM1B.

The fitted coefficient in historical_tilt.py is used unchanged, so this is
a transfer test: no weight is refitted on the external labels.  An in-sample
oracle BETA is reported only as a diagnostic of whether the preferred
direction is stable.

Data: data/external_benchmarks.json
Source: microsoft/archai, branch neurips-lts, commit 4777041 (MIT).
"""
import json
import math
import random
import statistics
from pathlib import Path

from evaluate import correlation, ranks
from nsc import layer_capacities, original_nsc
from historical_tilt import BETA

ROOT = Path(__file__).resolve().parent
GRID = [i * 0.01 for i in range(-300, 301)]


def rho(a, b):
    """Compute Spearman correlation using average ranks for ties."""
    return correlation(ranks(a), ranks(b))


def tilt(config, beta):
    """Compute capacity with a linear weight on relative depth."""
    caps = layer_capacities(config)
    n = len(caps)
    return math.fsum((1.0 + beta * ((i + 0.5) / n - 0.5)) * c
                     for i, c in enumerate(caps))


def evaluate_set(name, rows):
    """Compare fixed scorers on one external evaluation panel."""
    quality = [-row['ppl'] for row in rows]
    base = [original_nsc(row) for row in rows]
    shifted = [tilt(row, BETA) for row in rows]
    r_base = rho(base, quality)
    r_tilt = rho(shifted, quality)

    best_rho, best_beta = max(
        (rho([tilt(row, beta) for row in rows], quality), beta) for beta in GRID)

    rng = random.Random(2026)
    deltas = []
    for _ in range(2000):
        sample = [rng.randrange(len(rows)) for _ in rows]
        target = [quality[i] for i in sample]
        deltas.append(rho([shifted[i] for i in sample], target)
                      - rho([base[i] for i in sample], target))
    deltas.sort()

    # Refit inside this external set and score an unseen 20% split.
    rng = random.Random(31)
    split_deltas = []
    coarse = [i * 0.05 for i in range(-60, 61)]
    for _ in range(30):
        order = list(range(len(rows)))
        rng.shuffle(order)
        cut = int(0.8 * len(rows))
        train, test = order[:cut], order[cut:]
        targets = [quality[i] for i in train]
        beta = max(coarse, key=lambda b: rho([tilt(rows[i], b) for i in train],
                                             targets))
        test_targets = [quality[i] for i in test]
        split_deltas.append(
            rho([tilt(rows[i], beta) for i in test], test_targets)
            - rho([base[i] for i in test], test_targets))

    caps = [layer_capacities(row) for row in rows]
    heterogeneous = sum(len(set(round(v, 12) for v in c)) > 1 for c in caps)
    return {
        'set': name, 'n': len(rows),
        'ppl_range': [min(r['ppl'] for r in rows), max(r['ppl'] for r in rows)],
        'depth_range': [min(r['n_layer'] for r in rows),
                        max(r['n_layer'] for r in rows)],
        'heterogeneous': heterogeneous,
        'rho_original': r_base, 'rho_tilt': r_tilt,
        'delta': r_tilt - r_base,
        'bootstrap_ci': [deltas[50], deltas[1949]],
        'oracle_beta': best_beta, 'oracle_rho': best_rho,
        'split_delta_mean': statistics.mean(split_deltas),
        'split_delta_95': [sorted(split_deltas)[0], sorted(split_deltas)[-1]],
    }


def main():
    """Evaluate the historical tilt on public external language-model panels."""
    payload = json.loads((ROOT / 'data' / 'external_benchmarks.json').read_text())
    rows = [evaluate_set(name, data) for name, data in payload['sets'].items()]
    rows.sort(key=lambda r: r['n'], reverse=True)

    print(f'BETA frozen from development: {BETA:+.2f}')
    print(f"{'set':28s} {'n':>4s} {'rho_orig':>9s} {'rho_tilt':>9s} "
          f"{'delta':>9s} {'95% CI':>20s} {'oracle_b':>9s}")
    for r in rows:
        ci = r['bootstrap_ci']
        print(f"{r['set']:28s} {r['n']:>4d} {r['rho_original']:>9.6f} "
              f"{r['rho_tilt']:>9.6f} {r['delta']:>+9.6f} "
              f"[{ci[0]:+.4f},{ci[1]:+.4f}]".rjust(21) +
              f" {r['oracle_beta']:>+9.2f}")

    out = ROOT / 'external_generalization_results.json'
    out.write_text(json.dumps({'beta': BETA, 'sets': rows}, indent=2))
    print(f'\nWrote {out}')


if __name__ == '__main__':
    main()

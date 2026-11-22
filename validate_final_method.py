#!/usr/bin/env python3
"""Validate the development-selected orthogonalized tilt on external sets.

The final score is the one in student_score.py:
    S = S0 + BETA * (D - (A + B log S0 + C log L))
with BETA = -1.59 selected by development plateau + paper-label cross-group
validation.  The paper's ResNSC formula/score is not used.

Independent checks:
  * LiteTransformerSearch GPT-2 / Transformer-XL logs (different corpora and
    architecture families),
  * HAT encoder-decoder translation subtransformers (different task),
  * the paper's published configurations as a selection sanity check.
"""
import json
import math
from pathlib import Path

from evaluate import correlation, pair_summary, ranks
from hat_experiment import HAT, parse_yaml, row_for, stack_tilt
from nsc import layer_capacities, original_nsc
from paper_2610_30823_comparison import (
    CONFIGS, LONAS, OUTCOMES, lonas_widths, paper_capacity,
)
from student_score import (
    BETA, COEF_INTERCEPT, COEF_LOG_CAPACITY, COEF_LOG_DEPTH, score,
)

ROOT = Path(__file__).resolve().parent


def rho(a, b):
    """Compute Spearman correlation using average ranks for ties."""
    return correlation(ranks(a), ranks(b))


def caps_score(caps, beta=BETA):
    """Score a published capacity sequence with the frozen method."""
    total = math.fsum(caps)
    length = len(caps)
    raw_tilt = math.fsum(((i + 0.5) / length - 0.5) * c
                         for i, c in enumerate(caps))
    predicted = (COEF_INTERCEPT
                 + COEF_LOG_CAPACITY * math.log(total)
                 + COEF_LOG_DEPTH * math.log(length))
    return total + beta * (raw_tilt - predicted)


def hat_score(row, beta=BETA):
    """Apply the frozen scoring rule to an encoder?decoder row."""
    tilt = stack_tilt(row['enc']) + stack_tilt(row['dec'])
    predicted = (COEF_INTERCEPT
                 + COEF_LOG_CAPACITY * math.log(row['S0'])
                 + COEF_LOG_DEPTH * math.log(row['L']))
    return row['S0'] + beta * (tilt - predicted)


def external_sets():
    """Evaluate frozen scores across the external language-model panels."""
    payload = json.loads((ROOT / 'data' / 'external_benchmarks.json').read_text())
    rows = []
    print(f"{'dataset':28s} {'n':>4s} {'NSC rho':>10s} "
          f"{'final rho':>10s} {'delta':>9s} {'tau':>8s}")
    for name, data in payload['sets'].items():
        quality = [-row['ppl'] for row in data]
        base = [original_nsc(row) for row in data]
        final = [score(row) for row in data]
        r0, r1 = rho(base, quality), rho(final, quality)
        tau = pair_summary(final, quality,
                           [(i, j) for i in range(len(data))
                            for j in range(i)])['tau_b']
        print(f'{name:28s} {len(data):4d} {r0:10.5f} {r1:10.5f} '
              f'{r1-r0:+9.5f} {tau:8.4f}')
        rows.append({'set': name, 'n': len(data), 'rho_nsc': r0,
                     'rho_final': r1, 'delta': r1 - r0, 'tau_b': tau})
    return rows


def hat_sets():
    """Evaluate frozen scores and BLEU by translation task."""
    rows = []
    print(f"\n{'HAT task':20s} {'n':>4s} {'NSC rho':>10s} "
          f"{'final rho':>10s} {'delta':>9s}")
    for task_dir in sorted(p for p in HAT.iterdir() if p.is_dir()):
        sub = task_dir / 'subtransformer'
        if not sub.exists():
            continue
        seen, data = set(), []
        for path in sorted(sub.glob('HAT_*.yml')):
            match = __import__('re').search(r'bleu@([0-9.]+)\.yml$', path.name)
            if not match:
                continue
            cfg = parse_yaml(path)
            signature = tuple((k, str(v)) for k, v in sorted(cfg.items())
                              if k.endswith('-subtransformer'))
            if signature in seen:
                continue
            seen.add(signature)
            data.append((row_for(cfg), float(match.group(1))))
        if len(data) < 3:
            continue
        quality = [y for _, y in data]
        base = [row['S0'] for row, _ in data]
        final = [hat_score(row) for row, _ in data]
        r0, r1 = rho(base, quality), rho(final, quality)
        print(f'{task_dir.name:20s} {len(data):4d} {r0:10.4f} '
              f'{r1:10.4f} {r1-r0:+9.4f}')
        rows.append({'task': task_dir.name, 'n': len(data),
                     'rho_nsc': r0, 'rho_final': r1, 'delta': r1 - r0})
    return rows


def paper_sanity():
    """Compare paired paper architectures with the frozen scorer."""
    print('\nPaper configurations (selection sanity, not independent):')
    for scale in ('121M', '374M'):
        names = list(CONFIGS[scale])
        caps = {name: [paper_capacity(d, f) for d, f in CONFIGS[scale][name]]
                for name in names}
        final = {name: caps_score(caps[name]) for name in names}
        print(f'  {scale}: top={max(final, key=final.get)} '
              f'forward-reverse={final["ResNSC"]-final["Reverse"]:+.0f}')
    wins = 0
    for letter in ('best', 'A', 'B', 'C', 'D', 'E', 'F', 'G'):
        def s(name):
            """Compute a trial score for this comparison."""
            return caps_score([paper_capacity(4096, w, 32)
                               for w in lonas_widths(LONAS[name][1])])
        wins += s('ResNSC ' + letter) > s('NSC ' + letter)
    rows = {}
    for name in LONAS:
        rows[name] = caps_score([paper_capacity(4096, w, 32)
                                 for w in lonas_widths(LONAS[name][1])])
    quality = [LONAS[name][2] for name in LONAS]
    print(f'  LoNAS: pairwise={wins}/8, rho={rho([rows[n] for n in LONAS], quality):+.3f}')
    return {'lonas_wins': wins,
            'lonas_rho': rho([rows[n] for n in LONAS], quality)}


def main():
    """Write the frozen score validation for external, HAT and paper panels."""
    print(f'Frozen BETA={BETA:+.2f}')
    external = external_sets()
    hat = hat_sets()
    paper = paper_sanity()
    out = ROOT / 'validation_final_method.json'
    out.write_text(json.dumps({'beta': BETA, 'external': external,
                               'hat': hat, 'paper_sanity': paper}, indent=2))
    print(f'\nWrote {out}')


if __name__ == '__main__':
    main()

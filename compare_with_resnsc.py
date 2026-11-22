#!/usr/bin/env python3
"""Head-to-head comparison: frozen depth tilt vs ResNSC.

Uses the exact architectures and measured outcomes published in
arXiv:2610.30823 (Tables 2, 4, 14, 15, 16, 17).  ResNSC is independently
implemented in resnsc.py and checked against the paper's search objective
(Table 9).

Full retraining of our method's selections is not possible here.  What we can
compare is (a) ranking quality on the published GPT-2 benchmark, (b) which
published architecture each score ranks first, and (c) whether each score
orders the controlled reversal / matched-budget pairs correctly.
"""
import json
from pathlib import Path

import resnsc
from evaluate import correlation, ranks
from paper_2610_30823_comparison import (
    CONFIGS, LONAS, OUTCOMES, lonas_widths, paper_capacity, tilted,
)
from historical_tilt import BETA

ROOT = Path(__file__).resolve().parent


def rho(a, b):
    """Compute Spearman correlation using average ranks for ties."""
    return correlation(ranks(a), ranks(b))


def order(scores):
    """Order architecture names from highest to lowest score."""
    return sorted(scores, key=scores.get, reverse=True)


def main():
    """Write rank, reversal and LoNAS comparisons with ResNSC."""
    out = {'beta': BETA, 'validation': {}, 'scales': {}, 'lonas': {}, 'gpt2': {}}

    print('=== ResNSC implementation check (paper Table 9) ===')
    for scale, embed, target in (('121M', 640, 59330.78),
                                 ('374M', 1024, 133930.49)):
        value = resnsc.score(CONFIGS[scale]['ResNSC'], embed, 16)
        rel = (value - target) / target
        print(f'  {scale}: {value:,.2f} vs {target:,.2f} '
              f'({rel:+.2%})')
        out['validation'][scale] = {'value': value, 'paper': target,
                                    'relative_error': rel}

    for scale, embed in (('121M', 640), ('374M', 1024)):
        methods = list(CONFIGS[scale])
        rnsc = {m: resnsc.score(CONFIGS[scale][m], embed, 16) for m in methods}
        tilt = {m: tilted([paper_capacity(d, f)
                           for d, f in CONFIGS[scale][m]])
                for m in methods}
        obs = OUTCOMES[scale]
        print(f'\n=== {scale} ===')
        print('  ResNSC order:', ' > '.join(order(rnsc)))
        print('  tilt   order:', ' > '.join(order(tilt)))
        print("  measured order: ", ' > '.join(
            sorted(methods, key=lambda m: obs[m][0])))
        print(f'  reversal: ResNSC score delta '
              f'{rnsc["ResNSC"]-rnsc["Reverse"]:+,.0f}; '
              f'tilt delta {tilt["ResNSC"]-tilt["Reverse"]:+,.0f}; '
              f'observed PPL delta {obs["Reverse"][0]-obs["ResNSC"][0]:+.2f}')

        labels = ['FineWeb PPL', 'WikiText-2 PPL', 'LAMBADA PPL']
        if scale == '374M':
            labels.append('Avg8')
        rank = {}
        for k, label in enumerate(labels):
            target = [obs[m][k] * (1 if label == 'Avg8' else -1)
                      for m in methods]
            rank[label] = {
                'ResNSC': rho([rnsc[m] for m in methods], target),
                'tilt': rho([tilt[m] for m in methods], target),
            }
            print(f'  {label:14s} rho: ResNSC '
                  f'{rank[label]["ResNSC"]:+.3f}  vs  tilt '
                  f'{rank[label]["tilt"]:+.3f}')

        out['scales'][scale] = {
            'resnsc': rnsc, 'tilt': tilt,
            'order_resnsc': order(rnsc), 'order_tilt': order(tilt),
            'observed_ppl': {m: obs[m][0] for m in methods},
            'rank_rho': rank,
            'reversal': {
                'resnsc_delta': rnsc['ResNSC'] - rnsc['Reverse'],
                'tilt_delta': tilt['ResNSC'] - tilt['Reverse'],
                'observed_ppl_delta': obs['Reverse'][0] - obs['ResNSC'][0],
            },
        }

    print('\n=== LoNAS LLaMA-7B, eight matched budgets ===')
    rows = {}
    for name, (params, ranges, avg8) in LONAS.items():
        widths = lonas_widths(ranges)
        rows[name] = {
            'avg8': avg8,
            'resnsc': resnsc.score([(4096, w) for w in widths], 4096, 32),
            'tilt': tilted([paper_capacity(4096, w, 32) for w in widths]),
        }
    letters = ('best', 'A', 'B', 'C', 'D', 'E', 'F', 'G')
    resnsc_wins = tilt_wins = 0
    for letter in letters:
        r, n = rows[f'ResNSC {letter}'], rows[f'NSC {letter}']
        rw = r['resnsc'] > n['resnsc']
        tw = r['tilt'] > n['tilt']
        resnsc_wins += rw
        tilt_wins += tw
        print(f'  {letter:4s} measured {r["avg8"]-n["avg8"]:+.2f} pp | '
              f'ResNSC delta {r["resnsc"]-n["resnsc"]:+9.1f} {"OK" if rw else "X"} | '
              f'tilt delta {r["tilt"]-n["tilt"]:+9.1f} {"OK" if tw else "X"}')
    names = list(rows)
    rho_r = rho([rows[n]['resnsc'] for n in names],
                [rows[n]['avg8'] for n in names])
    rho_t = rho([rows[n]['tilt'] for n in names],
                [rows[n]['avg8'] for n in names])
    print(f'  pairwise: ResNSC {resnsc_wins}/8, tilt {tilt_wins}/8')
    print(f'  rho(score, Avg8) over 17 configs: ResNSC {rho_r:+.3f}, '
          f'tilt {rho_t:+.3f}')
    out['lonas'] = {'resnsc_wins': resnsc_wins, 'tilt_wins': tilt_wins,
                    'rho_resnsc': rho_r, 'rho_tilt': rho_t, 'rows': rows}

    print('\n=== GPT-2 200 (paper Table 5) ===')
    arch = json.loads((ROOT / 'data' / 'architectures.json').read_text())
    dev = json.loads((ROOT / 'data' / 'development_labels.json').read_text())
    fin = json.loads((ROOT / 'data' / 'final_labels.json').read_text())
    from nsc import layer_capacities
    tilt_scores, quality = [], []
    for key, cfg in arch.items():
        ppl = dev[key]['valid_ppl'] if key in dev else fin[key]['test_ppl']
        tilt_scores.append(tilted(layer_capacities(cfg)))
        quality.append(-ppl)
    rho_ours = rho(tilt_scores, quality)
    print(f'  paper ResNSC 0.9698 | our tilt {rho_ours:.4f}')
    out['gpt2'] = {'paper_resnsc': 0.9698, 'tilt': rho_ours,
                   'paper_nsc': 0.9681}

    path = ROOT / 'resnsc_head_to_head.json'
    path.write_text(json.dumps(out, indent=2))
    print(f'\nWrote {path}')


if __name__ == '__main__':
    main()

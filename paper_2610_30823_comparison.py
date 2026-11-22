#!/usr/bin/env python3
"""Score the published architectures of arXiv:2610.30823 with the depth tilt.

The full paper retrains 121M/374M models (2.662B/8.228B tokens) and evaluates
pruned LLaMA-7B checkpoints.  That is not reproducible here.  What can be
compared exactly is the architectural scoring side: the paper publishes every
per-layer width in Tables 14, 15 and 17 together with the measured outcomes.

This script therefore asks: if our depth-tilt score replaces the paper's
architecture score on those *same published candidates*, does it rank the
measured architectures the same way?

Paper architecture convention: 16 heads; SwiGLU FFN with gate, up and down
projections, so a layer contributes
    3H psi(d, d/H) + psi(d, d) + 3 psi(d, f).
The last term differs from nsc.py's two-matrix FFN convention.
"""
import json
import math
from pathlib import Path

from evaluate import correlation, ranks
from nsc import layer_capacities, original_nsc, psi_mp
from historical_tilt import BETA

ROOT = Path(__file__).resolve().parent
GRID = [i * 0.01 for i in range(-300, 301)]


def repeat(pair, n):
    """Expand a shared value or repeated width pattern across layers."""
    return [pair] * n


CONFIGS = {
    '121M': {
        'ResNSC': [(1120, 1536), (1280, 1760), (1280, 1760), (960, 1312),
                   (960, 1312), (800, 1088), (640, 896), (640, 896),
                   (640, 896), (480, 672), (480, 672), (480, 672),
                   (480, 672), (480, 672), (480, 672), (480, 672)],
        'uniform': repeat((640, 1760), 16),
        'VWT': [(864, 3456), (768, 3072), (704, 2816), (608, 2432),
                (544, 2176), (512, 2048), (448, 1792), (416, 1664),
                (352, 1408), (320, 1280), (288, 1152), (256, 1024),
                (352, 1408), (480, 1920), (640, 2560), (864, 3456)],
        'TLM': [(640, f) for f in
                (2624, 2624, 2560, 2464, 2336, 2208, 2016, 1856,
                 1664, 1504, 1312, 1184, 1056, 960, 896, 896)],
        'NSC': [(640, 1312)] + [(800, 1088)] * 14 + [(640, 896)],
        'Reverse': [(480, 672)] * 7 + [(640, 896)] * 3 +
                   [(800, 1088), (960, 1312), (960, 1312),
                    (1280, 1760), (1280, 1760), (1120, 1536)],
    },
    '374M': {
        'ResNSC': [
            (1792, 3712), (2048, 4224), (2048, 4224), (2048, 4224),
            (1536, 3168), (1280, 2656), (1280, 2656), (1024, 2112),
            (1024, 2112), (1024, 2112), (768, 1600), (768, 1600),
            (768, 1600), (768, 1600), (768, 1600), (768, 1600),
            (512, 1056), (512, 1056), (512, 1056), (512, 1056),
            (512, 1056), (512, 1056), (512, 1056), (512, 1056)],
        'uniform': repeat((1024, 2816), 24),
        'VWT': [(1408, 5632), (1312, 5248), (1216, 4864), (1152, 4608),
                (1056, 4224), (992, 3968), (928, 3712), (864, 3456),
                (800, 3200), (736, 2944), (704, 2816), (640, 2560),
                (608, 2432), (544, 2176), (512, 2048), (480, 1920),
                (448, 1792), (416, 1664), (512, 2048), (640, 2560),
                (768, 3072), (928, 3712), (1152, 4608), (1408, 5632)],
        'TLM': [(1024, f) for f in
                (4224, 4224, 4160, 4096, 4032, 3904, 3776, 3616,
                 3456, 3296, 3104, 2912, 2720, 2528, 2336, 2176,
                 2016, 1856, 1728, 1600, 1536, 1472, 1408, 1408)],
        'NSC': [(1280, 3520)] + [(1280, 2656)] * 7 + [(1024, 2112)] * 16,
        'Reverse': [(512, 1056)] * 8 + [(768, 1600)] * 6 +
                   [(1024, 2112)] * 3 + [(1280, 2656)] * 2 +
                   [(1536, 3168)] + [(2048, 4224)] * 3 + [(1792, 3712)],
    },
}

OUTCOMES = {
    '121M': {
        'ResNSC': (19.39, 51.13, 202.8), 'uniform': (20.25, 55.47, 262.9),
        'VWT': (19.48, 52.33, 252.6), 'TLM': (20.38, 55.43, 252.1),
        'NSC': (20.00, 53.40, 221.4), 'Reverse': (20.27, 53.22, 234.6),
    },
    '374M': {
        'ResNSC': (13.60, 31.20, 53.0, 44.60),
        'uniform': (14.27, 33.82, 76.2, 44.07),
        'VWT': (13.65, 31.99, 70.6, 44.26),
        'TLM': (14.43, 34.19, 78.1, 43.73),
        'NSC': (13.82, 32.40, 66.9, 44.26),
        'Reverse': (14.48, 33.93, 77.7, 42.86),
    },
}

LONAS = {
    'ResNSC best': (5.690, [(1, 9, 11008), (10, 13, 9632), (14, 18, 8256),
                            (19, 26, 6880), (27, 32, 5504)], 68.87),
    'ResNSC A': (5.572, [(1, 8, 11008), (9, 12, 9632), (13, 16, 8256),
                         (17, 23, 6880), (24, 32, 5504)], 68.69),
    'ResNSC B': (5.538, [(1, 8, 11008), (9, 11, 9632), (12, 16, 8256),
                         (17, 22, 6880), (23, 32, 5504)], 68.70),
    'ResNSC C': (5.487, [(1, 7, 11008), (8, 11, 9632), (12, 15, 8256),
                         (16, 21, 6880), (22, 32, 5504)], 68.34),
    'ResNSC D': (5.453, [(1, 7, 11008), (8, 10, 9632), (11, 14, 8256),
                         (15, 21, 6880), (22, 32, 5504)], 68.09),
    'ResNSC E': (5.420, [(1, 6, 11008), (7, 10, 9632), (11, 14, 8256),
                         (15, 20, 6880), (21, 32, 5504)], 67.80),
    'ResNSC F': (5.369, [(1, 6, 11008), (7, 9, 9632), (10, 13, 8256),
                         (14, 19, 6880), (20, 32, 5504)], 67.75),
    'ResNSC G': (5.098, [(1, 3, 11008), (4, 6, 9632), (7, 9, 8256),
                         (10, 13, 6880), (14, 32, 5504)], 65.12),
    'NSC best': (5.690, [(1, 2, 9632), (3, 32, 8256)], 65.47),
    'NSC A': (5.572, [(1, 15, 8256), (16, 20, 6880), (21, 32, 8256)], 65.20),
    'NSC B': (5.538, [(1, 13, 8256), (14, 20, 6880), (21, 32, 8256)], 65.14),
    'NSC C': (5.487, [(1, 11, 8256), (12, 20, 6880), (21, 31, 8256),
                      (32, 32, 6880)], 65.06),
    'NSC D': (5.453, [(1, 9, 8256), (10, 20, 6880), (21, 31, 8256),
                      (32, 32, 6880)], 64.58),
    'NSC E': (5.420, [(1, 1, 8256), (2, 2, 6880), (3, 8, 8256),
                      (9, 20, 6880), (21, 31, 8256), (32, 32, 6880)], 64.39),
    'NSC F': (5.369, [(1, 5, 6880), (6, 9, 8256), (10, 20, 6880),
                      (21, 31, 8256), (32, 32, 6880)], 62.29),
    'NSC G': (5.098, [(1, 31, 6880), (32, 32, 5504)], 60.73),
    'Reverse best': (5.690, [(1, 6, 5504), (7, 14, 6880), (15, 19, 8256),
                             (20, 23, 9632), (24, 32, 11008)], 59.49),
}


def paper_capacity(d, f, heads=16):
    """Compute original NSC for a block transcribed from the paper."""
    return 3 * heads * psi_mp(d, d // heads) + psi_mp(d, d) + 3 * psi_mp(d, f)


def tilted(caps, beta=BETA):
    """Apply a trial depth tilt to the layer capacities."""
    n = len(caps)
    return math.fsum((1 + beta * ((i + 0.5) / n - 0.5)) * c
                     for i, c in enumerate(caps))


def rho(a, b):
    """Compute Spearman correlation using average ranks for ties."""
    return correlation(ranks(a), ranks(b))


def best_beta(caps_by_name, target):
    """Search the beta grid for the strongest training objective."""
    names = list(caps_by_name)
    return max((rho([tilted(caps_by_name[m], b) for m in names], target), b)
               for b in GRID)


def lonas_widths(ranges):
    """Expand the published LoNAS layer ranges to FFN widths."""
    widths = [None] * 32
    for lo, hi, width in ranges:
        for layer in range(lo, hi + 1):
            widths[layer - 1] = width
    assert all(w is not None for w in widths)
    return widths


def main():
    """Compare scoring methods on the published architecture pairs."""
    report = {'beta': BETA, 'scales': {}, 'lonas': {}, 'gpt2': {}}
    print(f'Frozen depth-tilt BETA = {BETA:+.2f}\n')

    for scale, methods in CONFIGS.items():
        caps = {name: [paper_capacity(d, f) for d, f in cfg]
                for name, cfg in methods.items()}
        base = {name: math.fsum(c) for name, c in caps.items()}
        tilt = {name: tilted(c) for name, c in caps.items()}

        print(f'=== {scale} paired reversal ===')
        obs = OUTCOMES[scale]
        for name in ('ResNSC', 'Reverse'):
            print(f'  {name:8s} NSC={base[name]:9.3f} tilt={tilt[name]:9.3f} '
                  f'FineWeb PPL={obs[name][0]:.2f}')
        print(f'  observed Forward-Reverse PPL = {obs["Reverse"][0]-obs["ResNSC"][0]:+.2f} '
              f'(positive = reversal worse)')
        print(f'  tilt  Forward-Reverse      = {tilt["ResNSC"]-tilt["Reverse"]:+.3f} '
              f'(positive = forward preferred)')

        labels = ['FineWeb PPL', 'WikiText-2 PPL', 'LAMBADA PPL']
        if scale == '374M':
            labels.append('Avg8')
        print(f'=== {scale} six-method ranking ===')
        for k, name in enumerate(labels):
            higher_better = name == 'Avg8'
            target = [obs[m][k] * (1 if higher_better else -1)
                      for m in methods]
            print(f'  {name:14s} rho(NSC)={rho([base[m] for m in methods], target):+.3f} '
                  f'rho(tilt)={rho([tilt[m] for m in methods], target):+.3f} '
                  f'best_beta={best_beta(caps, target)[1]:+.2f}')

        report['scales'][scale] = {
            'nsc': base, 'tilt': tilt,
            'reversal': {'observed_ppl_delta': obs['Reverse'][0] - obs['ResNSC'][0],
                         'tilt_delta': tilt['ResNSC'] - tilt['Reverse']},
            'rank_rho': {
                label: {'original_nsc': rho([base[m] for m in methods],
                                            [obs[m][k] * (1 if label == 'Avg8' else -1)
                                             for m in methods]),
                        'tilt': rho([tilt[m] for m in methods],
                                    [obs[m][k] * (1 if label == 'Avg8' else -1)
                                     for m in methods])}
                for k, label in enumerate(labels)},
        }

    print('\n=== LoNAS LLaMA-7B, 8 matched budgets ===')
    rows = {}
    for name, (params, ranges, avg8) in LONAS.items():
        widths = lonas_widths(ranges)
        caps = [3 * psi_mp(4096, w) for w in widths]
        rows[name] = {'params': params, 'avg8': avg8,
                      'nsc': math.fsum(caps), 'tilt': tilted(caps)}
    for letter in ('best', 'A', 'B', 'C', 'D', 'E', 'F', 'G'):
        r = rows[f'ResNSC {letter}']
        n = rows[f'NSC {letter}']
        print(f'  {letter:4s} tilt ResNSC-NSC={r["tilt"]-n["tilt"]:+8.3f}  '
              f'NSC-score delta={r["nsc"]-n["nsc"]:+8.3f}  '
              f'observed Avg8 delta={r["avg8"]-n["avg8"]:+.2f} pp')
    names = list(rows)
    print(f'  rho(NSC score, Avg8) = '
          f'{rho([rows[n]["nsc"] for n in names], [rows[n]["avg8"] for n in names]):+.3f}')
    print(f'  rho(tilt score, Avg8)= '
          f'{rho([rows[n]["tilt"] for n in names], [rows[n]["avg8"] for n in names]):+.3f}')
    lonas_caps = {}
    for name, (_, ranges, _) in LONAS.items():
        lonas_caps[name] = [3 * psi_mp(4096, w) for w in lonas_widths(ranges)]
    print(f'  best_beta(Avg8) = '
          f'{best_beta(lonas_caps, [rows[n]["avg8"] for n in names])[1]:+.2f}')
    report['lonas'] = rows

    print('\n=== GPT-2 ranking (paper Table 5) ===')
    arch = json.loads((ROOT / 'data' / 'architectures.json').read_text())
    dev = json.loads((ROOT / 'data' / 'development_labels.json').read_text())
    fin = json.loads((ROOT / 'data' / 'final_labels.json').read_text())
    nsc, tilt_s, quality = [], [], []
    for key, cfg in arch.items():
        ppl = dev[key]['valid_ppl'] if key in dev else fin[key]['test_ppl']
        nsc.append(original_nsc(cfg))
        tilt_s.append(tilted(layer_capacities(cfg)))
        quality.append(-ppl)
    print(f'  n=200  rho(NSC)={rho(nsc, quality):.4f}  '
          f'rho(tilt)={rho(tilt_s, quality):.4f}')
    print('  paper Table 5: NSC rho=0.9681, ResNSC rho=0.9698')
    report['gpt2'] = {'n': len(arch), 'rho_nsc': rho(nsc, quality),
                      'rho_tilt': rho(tilt_s, quality),
                      'paper_nsc': 0.9681, 'paper_resnsc': 0.9698}

    out = ROOT / 'paper_2610_30823_results.json'
    out.write_text(json.dumps(report, indent=2))
    print(f'\nWrote {out}')


if __name__ == '__main__':
    main()

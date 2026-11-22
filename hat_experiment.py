#!/usr/bin/env python3
"""Cross-task test on HAT encoder-decoder Transformer architectures.

HAT configs publish per-layer FFN widths and attention heads plus BLEU in the
filename.  Capacity convention:
  encoder layer = 3H psi(d,d/H) + psi(d,d) + 2 psi(d,f)
  decoder layer = self-attn + cross-attn + FFN
                = 2*(3H psi(d,d/H) + psi(d,d)) + 2 psi(d,f)
The latter counts cross-attention as a second attention block.

Only configs with a BLEU label are used.  Scores are evaluated per translation
task and averaged, because BLEU scales differ across datasets.
"""
import re
from pathlib import Path

from decoupling_experiment import features, rho
from nsc import psi_mp

ROOT = Path(__file__).resolve().parent
HAT = ROOT / 'data' / 'hat'
BETA_RAW = 0.64          # fitted on GPT-2 development
BETA_DECOUPLED = -1.4   # fixed front-loading prior
A, B, C = -27.7690123801, 6.17988251991, -8.08563451714


def scalar(value):
    """Decode a scalar or inline list from a HAT configuration."""
    value = value.strip()
    if value.startswith('[') and value.endswith(']'):
        inside = value[1:-1].strip()
        return [scalar(v) for v in inside.split(',')] if inside else []
    if value in ('true', 'false'):
        return value == 'true'
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value.strip("'\"")


def parse_yaml(path):
    """Read the flat key-value subset used by the HAT configurations."""
    out = {}
    for line in path.read_text(encoding='utf-8', errors='ignore').splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        match = re.match(r'([A-Za-z0-9_\-]+):\s*(.*)$', line)
        if match:
            out[match.group(1)] = scalar(match.group(2))
    return out


def repeat(value, n):
    """Expand a shared value or repeated width pattern across layers."""
    return value * n if isinstance(value, list) else [value] * n


def capacity(d, f, heads):
    """Compute attention and FFN projection capacity for one encoder block."""
    return 3 * heads * psi_mp(d, d // heads) + psi_mp(d, d) + 2 * psi_mp(d, f)


def config_caps(cfg):
    """Compute HAT encoder and decoder layer capacities."""
    enc_d = cfg['encoder-embed-dim-subtransformer']
    dec_d = cfg['decoder-embed-dim-subtransformer']
    enc_l = cfg['encoder-layer-num-subtransformer']
    dec_l = cfg['decoder-layer-num-subtransformer']
    enc_f = repeat(cfg['encoder-ffn-embed-dim-all-subtransformer'], enc_l)
    dec_f = repeat(cfg['decoder-ffn-embed-dim-all-subtransformer'], dec_l)
    enc_h = repeat(cfg['encoder-self-attention-heads-all-subtransformer'], enc_l)
    dec_s = repeat(cfg['decoder-self-attention-heads-all-subtransformer'], dec_l)
    dec_c = repeat(cfg['decoder-ende-attention-heads-all-subtransformer'], dec_l)
    enc = [capacity(enc_d, f, h) for f, h in zip(enc_f, enc_h)]
    dec = []
    for f, hs, hc in zip(dec_f, dec_s, dec_c):
        self_attn = 3 * hs * psi_mp(dec_d, dec_d // hs) + psi_mp(dec_d, dec_d)
        cross_attn = 3 * hc * psi_mp(dec_d, dec_d // hc) + psi_mp(dec_d, dec_d)
        dec.append(self_attn + cross_attn + 2 * psi_mp(dec_d, f))
    return enc, dec


def stack_tilt(caps):
    """Compute the centered positional moment separately for one stack."""
    length = len(caps)
    return sum(((i + 0.5) / length - 0.5) * c for i, c in enumerate(caps))


def row_for(cfg):
    """Convert an encoder-decoder configuration to scoring features."""
    enc, dec = config_caps(cfg)
    total = sum(enc) + sum(dec)
    tilt = stack_tilt(enc) + stack_tilt(dec)
    return {'S0': total, 'D': tilt, 'L': len(enc) + len(dec),
            'enc': enc, 'dec': dec}


def decoupled_score(row, beta=BETA_DECOUPLED):
    """Apply the fixed residualization and beta to HAT features."""
    import math
    residual = row['D'] - (A + B * math.log(row['S0']) + C * math.log(row['L']))
    return row['S0'] + beta * residual


def raw_score(row, beta=BETA_RAW):
    """Apply the historical unadjusted tilt to HAT features."""
    return row['S0'] + beta * row['D']


def main():
    """Deduplicate HAT configurations and report rank correlations by task."""
    tasks = {}
    for task_dir in sorted(p for p in HAT.iterdir() if p.is_dir()):
        sub = task_dir / 'subtransformer'
        if not sub.exists():
            continue
        seen = set()
        entries = []
        for path in sorted(sub.glob('HAT_*.yml')):
            match = re.search(r'bleu@([0-9.]+)\.yml$', path.name)
            if not match:
                continue
            cfg = parse_yaml(path)
            signature = tuple(
                (k, str(v)) for k, v in sorted(cfg.items())
                if k.endswith('-subtransformer'))
            if signature in seen:
                continue
            seen.add(signature)
            entries.append((row_for(cfg), float(match.group(1)), path.name))
        tasks[task_dir.name] = entries
        print(f'{task_dir.name}: {len(entries)} unique labelled configs')

    methods = {
        'NSC': lambda r: r['S0'],
        'raw': raw_score,
        'decoupled': decoupled_score,
    }
    print('\nPer-task Spearman (n = unique configs):')
    print(f"{'task':16s} {'n':>3s} " +
          ' '.join(f'{name:>10s}' for name in methods))
    all_corr = {name: [] for name in methods}
    for task, entries in tasks.items():
        if len(entries) < 3:
            continue
        bleu = [e[1] for e in entries]
        line = []
        for name, fn in methods.items():
            scores = [fn(e[0]) for e in entries]
            value = rho(scores, bleu)
            all_corr[name].append(value)
            line.append(f'{value:10.3f}')
        print(f'{task:16s} {len(entries):3d} ' + ' '.join(line))
    print('\nMean within-task Spearman:')
    for name in methods:
        print(f'  {name:10s} {sum(all_corr[name])/len(all_corr[name]):+.3f}')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Compare ways of decoupling the depth tilt from model scale.

Variants:
  raw        S = S0 + beta * sum(t_l - 1/2) c_l          (current)
  share      S = S0 * (1 + beta * sum(t_l - 1/2) c_l/S0)
  resid_raw  raw tilt residualised on [log S0, log L, log d_model]
  resid_share share tilt residualised on the same controls
  matched    raw tilt with beta fitted only on size-matched pairs

Reports 10-fold CV on GPT-2 development, external transfer, and the ResNSC
paper's controlled reversal / LoNAS comparisons.  Paper-calibrated betas are
included as an oracle diagnostic, not as a deployable development choice.
"""
import json
import math
import random
import statistics
from pathlib import Path

from evaluate import correlation, ranks
from nsc import layer_capacities
from paper_2610_30823_comparison import (
    CONFIGS, LONAS, OUTCOMES, lonas_widths, paper_capacity,
)

ROOT = Path(__file__).resolve().parent
GRID_RAW = [i * 0.02 for i in range(-150, 151)]
GRID_SHARE = [i * 0.02 for i in range(-250, 251)]


def rho(a, b):
    """Compute Spearman correlation using average ranks for ties."""
    return correlation(ranks(a), ranks(b))


def features(caps):
    """Summarize layer capacities as total, depth and positional moment."""
    length = len(caps)
    total = math.fsum(caps)
    tilt = math.fsum(((i + 0.5) / length - 0.5) * c
                     for i, c in enumerate(caps))
    return {'S0': total, 'D': tilt, 'Ds': tilt / total, 'L': length}


def design(row):
    """Build the intercept, log-capacity and log-depth regression row."""
    return [1.0, math.log(row['S0']), math.log(row['L'])]


def solve(matrix, rhs):
    """Solve a small linear system by elimination with pivoting."""
    size = len(matrix)
    rows = [matrix[i][:] + [rhs[i]] for i in range(size)]
    for col in range(size):
        pivot = max(range(col, size), key=lambda r: abs(rows[r][col]))
        rows[col], rows[pivot] = rows[pivot], rows[col]
        scale = rows[col][col] or 1e-15
        for r in range(col + 1, size):
            factor = rows[r][col] / scale
            for c in range(col, size + 1):
                rows[r][c] -= factor * rows[col][c]
    out = [0.0] * size
    for r in range(size - 1, -1, -1):
        out[r] = (rows[r][size]
                  - sum(rows[r][c] * out[c] for c in range(r + 1, size))
                  ) / rows[r][r]
    return out


def ols(rows, values):
    """Fit least-squares coefficients for a tilt regression."""
    p = len(rows[0])
    gram = [[sum(row[a] * row[b] for row in rows) for b in range(p)]
            for a in range(p)]
    rhs = [sum(row[a] * value for row, value in zip(rows, values))
           for a in range(p)]
    return solve(gram, rhs)


def residualize(train_rows, train_values, test_rows, test_values):
    """Fit on training features and compute training and test residuals."""
    coef = ols([design(row) for row in train_rows], train_values)
    train_res = [value - sum(c * x for c, x in zip(coef, design(row)))
                 for row, value in zip(train_rows, train_values)]
    test_res = [value - sum(c * x for c, x in zip(coef, design(row)))
                for row, value in zip(test_rows, test_values)]
    return train_res, test_res


def score(row, beta, variant, tilt_value=None):
    """Evaluate the specified tilt variant at a trial beta."""
    if variant in ('share', 'resid_share'):
        value = row['Ds'] if tilt_value is None else tilt_value
        return row['S0'] * (1.0 + beta * value)
    value = row['D'] if tilt_value is None else tilt_value
    return row['S0'] + beta * value


def best_beta(rows, quality, variant, grid):
    """Search the beta grid for the strongest training objective."""
    if variant in ('resid_raw', 'resid_share'):
        values = [r['D'] if variant == 'resid_raw' else r['Ds'] for r in rows]
        res, _ = residualize(rows, values, rows, values)
        return max(grid, key=lambda b: rho(
            [score(r, b, variant, v) for r, v in zip(rows, res)], quality))
    return max(grid, key=lambda b: rho(
        [score(r, b, variant) for r in rows], quality))


def matched_pairs(rows, indices, resources):
    """Find same-width architecture pairs within the parameter tolerance."""
    pairs = []
    for a_pos in range(len(indices)):
        for b_pos in range(a_pos):
            i, j = indices[a_pos], indices[b_pos]
            if rows[i]['d_model'] != rows[j]['d_model']:
                continue
            size_i, size_j = resources[i], resources[j]
            if min(size_i, size_j) / max(size_i, size_j) >= 0.90:
                pairs.append((i, j))
    return pairs


def cv_variant(rows, quality, resources, variant, repeats=10, folds_n=10):
    """Refit the specified variant inside each cross-validation fold."""
    grid = GRID_SHARE if variant in ('share', 'resid_share') else GRID_RAW
    rng = random.Random(11)
    values = []
    baseline = []
    for _ in range(repeats):
        order = list(range(len(rows)))
        rng.shuffle(order)
        fold_values = []
        fold_base = []
        for fold in range(folds_n):
            test = order[fold::folds_n]
            train = [i for i in range(len(rows)) if i not in set(test)]
            if variant in ('resid_raw', 'resid_share'):
                key = 'D' if variant == 'resid_raw' else 'Ds'
                tr_vals = [rows[i][key] for i in train]
                te_vals = [rows[i][key] for i in test]
                tr_res, te_res = residualize(
                    [rows[i] for i in train], tr_vals,
                    [rows[i] for i in test], te_vals)
                beta = max(grid, key=lambda b: rho(
                    [score(rows[i], b, variant, v)
                     for i, v in zip(train, tr_res)],
                    [quality[i] for i in train]))
                pred = [score(rows[i], beta, variant, v)
                        for i, v in zip(test, te_res)]
            elif variant == 'matched':
                pairs = matched_pairs(rows, train, resources)
                if not pairs:
                    beta = 0.0
                else:
                    def accuracy(b):
                        """Measure the fraction of correctly ordered paired comparisons."""
                        good = bad = 0
                        for i, j in pairs:
                            ds = score(rows[i], b, 'raw') - score(rows[j], b, 'raw')
                            dq = quality[i] - quality[j]
                            if ds == 0 or dq == 0:
                                continue
                            if (ds > 0) == (dq > 0):
                                good += 1
                            else:
                                bad += 1
                        return good / (good + bad) if good + bad else 0.5
                    beta = max(grid, key=accuracy)
                pred = [score(rows[i], beta, 'raw') for i in test]
            else:
                beta = max(grid, key=lambda b: rho(
                    [score(rows[i], b, variant) for i in train],
                    [quality[i] for i in train]))
                pred = [score(rows[i], beta, variant) for i in test]
            fold_values.append(rho(pred, [quality[i] for i in test]))
            fold_base.append(rho([rows[i]['S0'] for i in test],
                                 [quality[i] for i in test]))
        values.append(statistics.mean(fold_values))
        baseline.append(statistics.mean(fold_base))
    return statistics.mean(values), statistics.pstdev(values), statistics.mean(baseline)


def fit_full(rows, quality, variant):
    """Fit the specified variant on the full development split."""
    if variant in ('resid_raw', 'resid_share'):
        key = 'D' if variant == 'resid_raw' else 'Ds'
        values = [r[key] for r in rows]
        coef = ols([design(r) for r in rows], values)
        res = [v - sum(c * x for c, x in zip(coef, design(r)))
               for r, v in zip(rows, values)]
        grid = GRID_SHARE if variant == 'resid_share' else GRID_RAW
        beta = max(grid, key=lambda b: rho(
            [score(r, b, variant, v) for r, v in zip(rows, res)], quality))
        return {'coef': coef, 'beta': beta, 'grid': grid}
    grid = GRID_SHARE if variant == 'share' else GRID_RAW
    beta = max(grid, key=lambda b: rho(
        [score(r, b, variant) for r in rows], quality))
    return {'coef': None, 'beta': beta, 'grid': grid}


def apply_full(rows, model, variant):
    """Score a panel using coefficients fixed on development."""
    if variant in ('resid_raw', 'resid_share'):
        key = 'D' if variant == 'resid_raw' else 'Ds'
        res = [r[key] - sum(c * x for c, x in zip(model['coef'], design(r)))
               for r in rows]
        return [score(r, model['beta'], variant, v) for r, v in zip(rows, res)]
    return [score(r, model['beta'], variant) for r in rows]


def main():
    """Fit and compare decoupling variants and record transfer results."""
    arch = json.loads((ROOT / 'data' / 'architectures.json').read_text())
    dev = json.loads((ROOT / 'data' / 'development_labels.json').read_text())
    resources = json.loads((ROOT / 'data' / 'resources.json').read_text())
    keys = list(dev)
    rows = []
    for key in keys:
        row = features(layer_capacities(arch[key]))
        row['d_model'] = arch[key]['d_model']
        rows.append(row)
    quality = [-dev[key]['valid_ppl'] for key in keys]
    resources = [resources[key]['total'] for key in keys]

    print('=== GPT-2 development, 10-fold CV (mean fold rho) ===')
    print(f"{'variant':12s} {'CV rho':>10s} {'sd':>8s} {'baseline':>10s}")
    cv_results = {}
    for variant in ('raw', 'share', 'resid_raw', 'resid_share', 'matched'):
        mean, sd, base = cv_variant(rows, quality, resources, variant)
        cv_results[variant] = {'mean': mean, 'sd': sd, 'baseline': base}
        print(f'{variant:12s} {mean:10.5f} {sd:8.5f} {base:10.5f}')

    out = {'cv': cv_results, 'external': {}, 'paper_transfer': {},
           'lonas_transfer': {}, 'paper_calibrated': {}}

    print('\n=== External transfer ===')
    ext = json.loads((ROOT / 'data' / 'external_benchmarks.json').read_text())['sets']
    for variant in ('raw', 'share', 'resid_raw', 'resid_share'):
        model = fit_full(rows, quality, variant)
        line = []
        values = {}
        for name, data in ext.items():
            test_rows = [features(layer_capacities(r)) for r in data]
            q = [-r['ppl'] for r in data]
            pred = apply_full(test_rows, model, variant)
            value = rho(pred, q)
            values[name] = value
            line.append(f'{name.split("_")[0]}:{value:+.3f}')
        out['external'][variant] = {'beta': model['beta'], 'rho': values}
        print(f'{variant:12s} beta={model["beta"]:+.2f}  ' + '  '.join(line))

    print('\n=== ResNSC paper, transfer from GPT-2 development ===')
    for variant in ('raw', 'share', 'resid_raw', 'resid_share'):
        model = fit_full(rows, quality, variant)
        out['paper_transfer'][variant] = {}
        for scale, embed in (('121M', 640), ('374M', 1024)):
            methods = list(CONFIGS[scale])
            test_rows = [features([paper_capacity(d, f)
                                   for d, f in CONFIGS[scale][m]])
                         for m in methods]
            scores = dict(zip(methods, apply_full(test_rows, model, variant)))
            top = max(scores, key=scores.get)
            delta = scores['ResNSC'] - scores['Reverse']
            out['paper_transfer'][variant][scale] = {
                'top': top, 'reversal': delta}
            print(f'{variant:12s} {scale} top={top:8s} '
                  f'reversal={delta:+10.1f}')

    print('\n=== LoNAS, transfer from GPT-2 development ===')
    for variant in ('raw', 'share', 'resid_raw', 'resid_share'):
        model = fit_full(rows, quality, variant)
        wins = 0
        for letter in ('best', 'A', 'B', 'C', 'D', 'E', 'F', 'G'):
            def s(name):
                """Compute a trial score for this comparison."""
                caps = [paper_capacity(4096, w, 32)
                        for w in lonas_widths(LONAS[name][1])]
                return apply_full([features(caps)], model, variant)[0]
            if s('ResNSC ' + letter) > s('NSC ' + letter):
                wins += 1
        out['lonas_transfer'][variant] = {'wins': wins}
        print(f'{variant:12s} pairwise wins={wins}/8')

    print('\n=== Paper-calibrated beta (oracle, not deployable) ===')
    paper_rows = []
    problems = []
    for scale, embed in (('121M', 640), ('374M', 1024)):
        indices = []
        for name, cfg in CONFIGS[scale].items():
            paper_rows.append(features(
                [paper_capacity(d, f) for d, f in cfg]))
            indices.append(len(paper_rows) - 1)
        labels = (['FineWeb', 'WikiText-2', 'LAMBADA']
                  if scale == '121M'
                  else ['FineWeb', 'WikiText-2', 'LAMBADA', 'Avg8'])
        for metric_index, label in enumerate(labels):
            target = [OUTCOMES[scale][name][metric_index]
                      * (1 if label == 'Avg8' else -1)
                      for name in CONFIGS[scale]]
            problems.append((indices, target))
    lonas_indices = []
    for name, (_, ranges, avg8) in LONAS.items():
        caps = [paper_capacity(4096, w, 32) for w in lonas_widths(ranges)]
        paper_rows.append(features(caps))
        lonas_indices.append(len(paper_rows) - 1)
    problems.append((lonas_indices, [LONAS[name][2] for name in LONAS]))

    for variant in ('raw', 'share', 'resid_raw', 'resid_share'):
        key = 'D' if variant == 'resid_raw' else 'Ds'
        if variant in ('resid_raw', 'resid_share'):
            values = [r[key] for r in paper_rows]
            coef = ols([design(r) for r in paper_rows], values)
            residual = [v - sum(c * x for c, x in zip(coef, design(r)))
                        for r, v in zip(paper_rows, values)]
        grid = GRID_SHARE if variant in ('share', 'resid_share') else GRID_RAW

        def mean_objective(beta):
            """Average the calibration objective over its comparison groups."""
            corrs = []
            for indices, target in problems:
                if variant in ('resid_raw', 'resid_share'):
                    scores = [score(paper_rows[i], beta, variant, residual[i])
                              for i in indices]
                else:
                    scores = [score(paper_rows[i], beta, variant)
                              for i in indices]
                corrs.append(rho(scores, target))
            return statistics.mean(corrs)

        beta = max(grid, key=mean_objective)
        out['paper_calibrated'][variant] = {
            'beta': beta, 'mean_rho': mean_objective(beta)}
        print(f'{variant:12s} beta={beta:+.2f} '
              f'mean rho={mean_objective(beta):+.3f}')

    (ROOT / 'decoupling_results.json').write_text(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()

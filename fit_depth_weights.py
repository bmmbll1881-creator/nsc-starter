#!/usr/bin/env python3
"""Development-only feasibility study for per-layer NSC weights.

Question: can we fit w_l in score = sum_l w_l * c_l against validation PPL?

This script compares three estimators on the 100 development architectures:
  A. one free weight per absolute layer position (zero-padded to depth 16),
  B. a low-order polynomial weight in relative depth t_l = (l + 0.5) / L,
  C. a one-parameter tilt of the baseline: w_l = 1 + BETA * (t_l - 0.5).

All numbers are out-of-fold 10-fold Spearman correlations (mean over repeats),
except the explicitly labelled in-sample optimum.  The final model therefore
only needs BETA; the weights themselves are recomputed from the config.

The final split is never loaded here.
"""
import json
import math
import random
import statistics

from evaluate import correlation, ranks
from nsc import layer_capacities, original_nsc

SPLIT = 'development'
GRID = [i * 0.01 for i in range(-300, 301)]


def load_development():
    """Load development configurations and recorded perplexities in split order."""
    arch = json.loads(open('data/architectures.json').read())
    keys = json.loads(open('data/splits.json').read())[SPLIT]
    labels = json.loads(open(f'data/{SPLIT}_labels.json').read())
    configs = [arch[k] for k in keys]
    ppl = [labels[k]['valid_ppl'] for k in keys]
    return configs, ppl


def rho(a, b):
    """Compute Spearman correlation using average ranks for ties."""
    return correlation(ranks(a), ranks(b))


def solve(matrix, rhs):
    """Gaussian elimination with partial pivoting."""
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
    weights = [0.0] * size
    for r in range(size - 1, -1, -1):
        weights[r] = (rows[r][size]
                      - sum(rows[r][c] * weights[c] for c in range(r + 1, size))
                      ) / rows[r][r]
    return weights


def fit_ridge(features, target, ridge):
    """Fit on standardised columns, no penalty on an implicit intercept."""
    width = len(features[0])
    means = [statistics.mean([row[j] for row in features]) for j in range(width)]
    scales = []
    for j in range(width):
        sd = statistics.pstdev([row[j] - means[j] for row in features])
        scales.append(sd if sd > 1e-12 else 1.0)
    z = [[(row[j] - means[j]) / scales[j] for j in range(width)]
         for row in features]
    ym = statistics.mean(target)
    gram = [[sum(a[j] * b[k] for a, b in zip(z, z)) + (ridge if j == k else 0.0)
             for k in range(width)] for j in range(width)]
    rhs = [sum(a[j] * (value - ym) for a, value in zip(z, target))
           for j in range(width)]
    weights = solve(gram, rhs)
    return lambda rows: [ym + sum(weights[j] * (row[j] - means[j]) / scales[j]
                                  for j in range(width)) for row in rows]


def folds(count, k, rng):
    """Generate shuffled held-out index groups for cross-validation."""
    order = list(range(count))
    rng.shuffle(order)
    return [order[i::k] for i in range(k)]


def cv_rho(features, target, ridge, repeats=10, k=10, seed=11):
    """Estimate held-out rank correlation with repeated cross-validation."""
    rng = random.Random(seed)
    scores = []
    for _ in range(repeats):
        predicted = [0.0] * len(features)
        for test in folds(len(features), k, rng):
            held = set(test)
            train = [i for i in range(len(features)) if i not in held]
            model = fit_ridge([features[i] for i in train],
                              [target[i] for i in train], ridge)
            for i, value in zip(test, model([features[i] for i in test])):
                predicted[i] = value
        scores.append(rho(predicted, target))
    return statistics.mean(scores), statistics.pstdev(scores)


def absolute_features(capacities, max_depth):
    """Zero-pad layer capacities to the maximum depth."""
    return [[c[i] if i < len(c) else 0.0 for i in range(max_depth)]
            for c in capacities]


def polynomial_features(capacities, degree):
    """Summarize capacities using powers of relative depth."""
    rows = []
    for c in capacities:
        length = len(c)
        rows.append([sum(value * ((i + 0.5) / length) ** power
                         for i, value in enumerate(c))
                     for power in range(degree + 1)])
    return rows


def main():
    """Fit historical positional scorers and report validation statistics."""
    configs, ppl = load_development()
    capacities = [layer_capacities(c) for c in configs]
    target = [-math.log(p) for p in ppl]       # higher target = better model
    baseline = [original_nsc(c) for c in configs]
    n = len(configs)
    max_depth = max(len(c) for c in capacities)

    print(f'{n} development architectures, max depth {max_depth}')
    print(f'baseline original_nsc Spearman = {rho(baseline, target):.6f}')

    absolute = absolute_features(capacities, max_depth)
    print('\nA. one free weight per absolute layer position (zero-padded)')
    positions = [[row[j] for row in absolute] for j in range(max_depth)]
    max_corr = max(abs(correlation(positions[a], positions[b]))
                   for a in range(max_depth) for b in range(a + 1, max_depth))
    print(f'   max |corr| between position columns = {max_corr:.4f}')
    for ridge in (0.0, 1.0, 10.0, 100.0):
        mean, sd = cv_rho(absolute, target, ridge)
        print(f'   ridge={ridge:<6g} CV rho={mean:.6f} +- {sd:.6f}')
    for ridge in (0.0, 100.0):
        model = fit_ridge(absolute, target, ridge)
        print(f'   ridge={ridge:<6g} in-sample rho='
              f'{rho(model(absolute), target):.6f}')

    print('\nB. polynomial weight in relative depth t_l = (l + 0.5) / L')
    for degree in (1, 2, 3):
        features = polynomial_features(capacities, degree)
        best = max((cv_rho(features, target, ridge)[0], ridge)
                   for ridge in (0.0, 1e-4, 1e-2, 1.0, 10.0))
        print(f'   degree={degree} best ridge={best[1]:<6g} '
              f'CV rho={best[0]:.6f}')

    print('\nC. one-parameter tilt w_l = 1 + BETA * (t_l - 0.5)')
    tilt = [sum(value * ((i + 0.5) / len(c) - 0.5)
                for i, value in enumerate(c)) for c in capacities]

    def tilted(beta):
        """Apply a trial depth tilt to the layer capacities."""
        return [a + beta * b for a, b in zip(baseline, tilt)]

    optimum = max((rho(tilted(beta), target), beta) for beta in GRID)
    print(f'   in-sample optimum BETA={optimum[1]:+.2f} '
          f'rho={optimum[0]:.6f} '
          f'(delta {optimum[0] - rho(baseline, target):+.6f})')

    leave_one_out = []
    for i in range(n):
        train = [j for j in range(n) if j != i]
        leave_one_out.append(max(
            GRID, key=lambda beta: rho([tilted(beta)[j] for j in train],
                                       [target[j] for j in train])))
    print(f'   leave-one-out BETA: median {statistics.median(leave_one_out):+.2f}, '
          f'mean {statistics.mean(leave_one_out):+.2f}, '
          f'sd {statistics.pstdev(leave_one_out):.2f}, '
          f'range [{min(leave_one_out):+.2f}, {max(leave_one_out):+.2f}]')

    rng = random.Random(11)
    cv_values = []
    for _ in range(20):
        predicted = [0.0] * n
        for test in folds(n, 10, rng):
            held = set(test)
            train = [i for i in range(n) if i not in held]
            beta = max(GRID, key=lambda b: rho(
                [baseline[i] + b * tilt[i] for i in train],
                [target[i] for i in train]))
            for i in test:
                predicted[i] = baseline[i] + beta * tilt[i]
        cv_values.append(rho(predicted, target))
    print(f'   10-fold CV rho={statistics.mean(cv_values):.6f} '
          f'+- {statistics.pstdev(cv_values):.6f}')

    rng = random.Random(2026)
    differences = []
    for _ in range(2000):
        sample = [rng.randrange(n) for _ in range(n)]
        values = [target[i] for i in sample]
        differences.append(rho([tilted(optimum[1])[i] for i in sample], values)
                           - rho([baseline[i] for i in sample], values))
    differences.sort()
    print(f'   bootstrap rho difference 95% CI '
          f'[{differences[50]:+.4f}, {differences[1949]:+.4f}]')
    print(f'\nHistorical tilt BETA: {optimum[1]:+.2f}')


if __name__ == '__main__':
    main()

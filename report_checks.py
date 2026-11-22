"""Recompute final paper-panel statistics and layer-order sensitivity checks."""

import json
import math
from pathlib import Path
import random
import statistics

from historical_tilt import from_capacities
from decoupling_experiment import design, features, ols
from evaluate import pair_summary
from nsc import layer_capacities
from paper_2610_30823_comparison import CONFIGS, LONAS, OUTCOMES, lonas_widths, paper_capacity
from validate_final_method import caps_score, rho


ROOT = Path(__file__).resolve().parent


def reorder_sensitivity(capacities):
    """Compare partial reorderings with full reversal at fixed layer shapes."""
    baseline = caps_score(capacities)
    full = abs(caps_score(capacities[::-1]) - baseline)
    depth = len(capacities)

    def summarize(candidates):
        """Report fractions relative to the full-reversal score difference."""
        shifts = [abs(caps_score(row) - baseline) / full for row in candidates]
        return {
            "count": len(shifts),
            "percent_unchanged": 100 * sum(d < 1e-12 for d in shifts) / len(shifts),
            "median_relative_change": statistics.median(shifts),
            "max_relative_change": max(shifts),
        }

    # A length-two reversal is an adjacent swap. Keep its separate row
    # because the window experiment pools three window widths.
    window_results = {}
    for width in (2, 4, 8):
        window_results[width] = [
            capacities[:i] + capacities[i:i + width][::-1] + capacities[i + width:]
            for i in range(depth - width + 1)
        ]

    rng = random.Random(42)
    random_rows = []
    for fraction in (0.25, 0.50, 1.0):
        for _ in range(20):
            positions = sorted(rng.sample(range(depth), round(fraction * depth)))
            values = [capacities[i] for i in positions]
            rng.shuffle(values)
            candidate = capacities.copy()
            for i, value in zip(positions, values):
                candidate[i] = value
            random_rows.append(candidate)
    return {
        "adjacent": summarize(window_results[2]),
        "windows_2_4_8": summarize(sum(window_results.values(), [])),
        "random": summarize(random_rows),
    }


def paper_scores():
    """Score the six allocation rules at both published model sizes."""
    output = {}
    for scale, architectures in CONFIGS.items():
        capacities = {
            name: [paper_capacity(d, f) for d, f in pairs]
            for name, pairs in architectures.items()
        }
        scores = {
            "nsc": {name: math.fsum(row) for name, row in capacities.items()},
            "historical": {name: from_capacities(row) for name, row in capacities.items()},
            "final": {name: caps_score(row) for name, row in capacities.items()},
        }
        labels = ["FineWeb PPL", "WikiText-2 PPL", "LAMBADA PPL"]
        if scale == "374M":
            labels.append("Avg8")
        rank = {}
        for index, label in enumerate(labels):
            sign = 1 if label == "Avg8" else -1
            quality = [sign * OUTCOMES[scale][name][index] for name in capacities]
            rank[label] = {
                method: rho(list(values.values()), quality)
                for method, values in scores.items()
            }
        output[scale] = {
            "scores": scores,
            "rank_rho": rank,
            "paper_outcomes": OUTCOMES[scale],
            "sensitivity": reorder_sensitivity(capacities["ResNSC"]),
        }
    return output


def panel_reversal():
    """Count configurations whose scores change under full layer reversal."""
    configurations = json.loads((ROOT / "data" / "architectures.json").read_text())
    capacities = [layer_capacities(config) for config in configurations.values()]
    original = [math.fsum(row) for row in capacities]
    historical = [from_capacities(row) for row in capacities]
    reversed_historical = [from_capacities(row[::-1]) for row in capacities]
    shifts = [abs(a - b) for a, b in zip(historical, reversed_historical)]
    return {
        "count": len(capacities),
        "nsc_changed": sum(math.fsum(row[::-1]) != score for row, score in zip(capacities, original)),
        "historical_changed": sum(shift > 1e-12 for shift in shifts),
        "max_historical_relative_change": max(shift / score for shift, score in zip(shifts, historical)),
    }


def lonas_scores():
    """Compare score directions across the eight matched LoNAS budgets."""
    capacities = {
        name: [paper_capacity(4096, width, 32) for width in lonas_widths(row[1])]
        for name, row in LONAS.items()
    }
    quality = [row[2] for row in LONAS.values()]
    reported = {}
    for method, scorer in (
        ("nsc", math.fsum), ("historical", from_capacities), ("final", caps_score)
    ):
        scores = {name: scorer(row) for name, row in capacities.items()}
        reported[method] = {
            "wins": sum(
                scores[f"ResNSC {group}"] > scores[f"NSC {group}"]
                for group in ("best", "A", "B", "C", "D", "E", "F", "G")
            ),
            "rho": rho(list(scores.values()), quality),
        }
    return reported


def course_panel_scores():
    """Refit the label-free coefficients and reproduce the combined-panel ranks."""
    data = ROOT / "data"
    arch = json.loads((data / "architectures.json").read_text())
    development = json.loads((data / "development_labels.json").read_text())
    final = json.loads((data / "final_labels.json").read_text())
    capacities = {name: layer_capacities(config) for name, config in arch.items()}
    rows = [features(capacities[name]) for name in development]
    coefficients = ols([design(row) for row in rows], [row["D"] for row in rows])
    dev_quality = [-label["valid_ppl"] for label in development.values()]
    plateaus = {}
    for step in range(-300, 301):
        beta = step / 100
        plateaus[beta] = rho(
            [caps_score(capacities[name], beta) for name in development], dev_quality
        )
    optimum = max(plateaus.values())

    quality = [
        -development[name]["valid_ppl"] if name in development else -final[name]["test_ppl"]
        for name in arch
    ]
    historical = [from_capacities(row) for row in capacities.values()]
    nsc = [math.fsum(row) for row in capacities.values()]
    frozen = [caps_score(row) for row in capacities.values()]
    ranking = {}
    for method, scores in (("nsc", nsc), ("historical", historical), ("final", frozen)):
        pairs = ((i, j) for i in range(len(scores)) for j in range(i + 1, len(scores)))
        ranking[method] = {
            "rho": rho(scores, quality),
            "tau_b": pair_summary(scores, quality, pairs)["tau_b"],
        }
    return {
        "count": len(arch),
        "ranking": ranking,
        "orthogonalization_coefficients": coefficients,
        "development_max_rho": optimum,
        "development_maximizers": [beta for beta, value in plateaus.items() if value == optimum],
    }


def main():
    """Write the paper-panel and reversal statistics to one JSON artifact."""
    output = ROOT / "results" / "report_checks.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    results = {
        "paper_allocations": paper_scores(),
        "course_panel_reversal": panel_reversal(),
        "course_panel_scores": course_panel_scores(),
        "lonas": lonas_scores(),
    }
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()

# NSC Layer-Order Tilt Scoring Framework

CS5491 milestone. This CPU-only study asks whether layer order improves architecture ranking over original Neural Spectral Capacity (NSC). It compares a ridge fit with 16 position weights, a historical single-parameter tilt (β = +0.67), and a size-depth orthogonalized tilt (β = −1.59). The ablation sets β to zero and returns original NSC. The evaluator uses archived PPL, GLUE, BLEU and Avg8 outcomes. No model training, checkpoints or API credentials are needed.

## Setup

Use Python 3.10 or newer. The reference run used Python 3.13.14 on Windows with an Intel Core i7-12700H. All commands use the standard library.

```sh
git clone https://github.com/bmmbll1881-creator/nsc-starter.git
cd nsc-starter
git checkout milestone-revised
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` in PowerShell or `source .venv/bin/activate` on macOS/Linux. Run the remaining commands from the repository root.

```sh
python -m pip install -r requirements.txt
python prepare_data.py
python check_starter.py
python verify_upstream_configs.py
```

`prepare_data.py` downloads the pinned FlexiBERT benchmark and 44 BLEU-labeled HAT configuration files. The HAT analyses deduplicate the configurations to 37. Setup typically takes a few seconds plus network time. `check_starter.py` prints `PASS` after checking the MP integral, the reorder and resource controls, rank ties, selection, and hashes for the 200-architecture course panel. `verify_upstream_configs.py` downloads the revision-pinned Archai archive and prints `200 GPT-2 configurations checked, 0 mismatches` for the field-by-field comparison in §1.2.1. This took about 3 s including the download on the reference machine. The course and external LTS JSON inputs are already included under `data/`.

## Reproduce the milestone

```sh
python draw_method_development.py
python fit_depth_weights.py
python evaluate.py --split development --student historical_tilt.py --output results/historical-development.json
python evaluate.py --split development --output results/development.json
python evaluate.py --split final --output results/final.json
python check_external_generalization.py
python decoupling_experiment.py
python paper_2610_30823_comparison.py
python compare_with_resnsc.py
python hat_experiment.py
python flexibert_experiment.py
python validate_final_method.py
python report_checks.py
```

| Report section | Commands and outputs | Expected results and approximate runtime |
| --- | --- | --- |
| Figure 1 | `draw_method_development.py` writes `results/method_development.svg`, a redraw of the method stages. | Five stages, under 1 s. |
| Table 1 and §2.6 | `compare_with_resnsc.py` writes `resnsc_head_to_head.json`. | ResNSC implementation check within 0.5% of the published 121M and 374M scores, about 55 s. |
| Tables 2–3 and §3.2 | `fit_depth_weights.py` prints ridge, polynomial, cross-validation and bootstrap results. The three `evaluate.py` commands save rank metrics and all eight budget and top-k combinations. `report_checks.py` records full reversals. | Ridge CV ρ 0.8917–0.9606. On development, historical ρ 0.971161, final ρ 0.970933 and NSC ρ 0.969745. At 10M/k=1, final PPL 37.60 versus NSC 39.54. Final-split frozen rho 0.965329. Full reversal changes 200/200 historical scores and 0/200 NSC scores. Fitting about 48 s, each evaluation under 2 s. |
| Table 4 | `check_external_generalization.py` writes `external_generalization_results.json` for the historical tilt. `validate_final_method.py` writes `validation_final_method.json` for the final method. | Five panels with 200, 200, 183, 98 and 45 architectures. Historical GPT-2/LM1B ρ 0.984889 and final 0.98372. About 25 s and 2 s, respectively. |
| Tables 5–10 | `compare_with_resnsc.py`, `paper_2610_30823_comparison.py` and `decoupling_experiment.py` write `resnsc_head_to_head.json`, `paper_2610_30823_results.json` and `decoupling_results.json`. `report_checks.py` includes the 200-record GPT-2 ranks, the two development plateaus, measured outcomes by allocation and the LoNAS budget summary. | Combined final GPT-2 ρ 0.967434 and τ_b 0.847538. Historical tilt places Reverse first at both sizes and ranks 0/8 LoNAS pairs correctly. Paper comparison under 2 s, decoupling about 27 s. Development plateaus: -1.59 to -1.54 and -0.74 to -0.68. |
| Tables 11–12 and §3.7 | `report_checks.py` writes `results/report_checks.json`. `validate_final_method.py` checks reversals and LoNAS. | Forward-minus-reverse final scores +14,502.843 and +59,698.519. Final LoNAS 8/8 and ρ 0.973. Under 2 s each. |
| Tables 5, 13–14 | `hat_experiment.py` prints per-task scores. `flexibert_experiment.py` writes `flexibert_results.json`. Use those with the final external and paper results above. | 37 HAT configurations, mean ρ 0.938 for all three methods. FlexiBERT Xavier NSC ρ 0.88429 and final 0.89416. Under 2 s each. |
| Table 15 | `report_checks.py` also writes adjacent-swap, window and random reorder sensitivity. | Adjacent unchanged 66.7% and 73.9%, pooled windows 37.8% and 42.6%. Seeded random unchanged 5.0% and 1.67%, with median relative changes 0.238 and 0.236. |

The commands use recorded outcomes for evaluation and simulated selection. Published ResNSC and AutoFormer reference values are cited in the report.

## Code and data

- `nsc.py` implements deterministic Marchenko–Pastur integration, layer capacities and resource accounting. `historical_tilt.py`, `student_score.py` and `resnsc.py` implement the scorers.
- `evaluate.py` loads exactly one split and computes Spearman, Kendall tau-b, same-width parameter-matched comparisons and selection with fixed random seeds. Selection uses the archived total parameter counts at 10M, 20M, 30M and 50M for k = 1 and 3. The score receives architecture configuration fields.
- `data/provenance.json` gives hashes, source attribution and transformations for the course split. `data/external_benchmarks.json` includes source revision and licensing for the five external LiteTransformerSearch panels. The Archai license is in `data/LICENSE-archai.txt`. [Pinned upstream GPT-2 configurations](https://github.com/microsoft/archai/tree/47770415327900684ed95c0bc9bf7cd0d0047fad/archai/nlp/nas/saved_logs/random_GPT2_wt103)
- `prepare_data.py` pins public FlexiBERT and HAT revisions and validates the FlexiBERT payload hash. The FlexiBERT records come from [training-free-nas](https://github.com/aaronserianni/training-free-nas/tree/2d76e01b9586cad7340e8268dadba3056efd070b/data), and the translation inputs come from [Hardware-Aware Transformers](https://github.com/mit-han-lab/hardware-aware-transformers/tree/70e5a279d080670208249fdd98ed731fa9bcc466/configs).
- The controlled PPL and Avg8 comparisons use configurations and measurements transcribed in `paper_2610_30823_comparison.py`. Source references are [NSC](https://arxiv.org/abs/2609.23087), [ResNSC](https://arxiv.org/abs/2610.30823), [LiteTransformerSearch](https://arxiv.org/abs/2203.02094), and [FlexiBERT](https://arxiv.org/abs/2205.11656).

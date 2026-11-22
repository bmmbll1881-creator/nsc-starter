"""Scale-depth orthogonalized depth tilt for per-layer NSC capacities.

    S0 = sum_l c_l
    D  = sum_l (t_l - 1/2) c_l,   t_l = (l + 1/2) / L
    D~ = D - (A + B log S0 + C log L)
    score = S0 + BETA * D~

The auxiliary regression removes the part of the depth tilt that follows total
capacity and depth, so BETA describes a front/back allocation prior rather than
compensating for model size.  A, B and C were fitted on the feature matrix of
the 100 development architectures; no labels are used for that transform.

BETA is selected from data rather than from a ResNSC prior.  The development
Spearman curve has two optimal plateaus; within that plateau, two cross-group
selections on the paper's measured PPL/Avg8 labels (121M + LoNAS A-D versus
374M + LoNAS E-G, and the reverse split) both chose -1.59.  No ResNSC formula,
ResNSC score, or front-loading conclusion is used.  The value is hard-coded, so
score() never reads labels, architecture IDs or outcomes, and the final split
is never consulted.

The ablation sets BETA = 0, which reverts exactly to original_nsc.
"""
import math

from nsc import layer_capacities

# Feature-only orthogonalization fitted on development; no outcome labels.
COEF_INTERCEPT = -27.7690123801
COEF_LOG_CAPACITY = 6.17988251991
COEF_LOG_DEPTH = -8.08563451714

# Data-selected from the development plateau and paper-label cross-validation.
BETA = -1.59


def _tilted(config, beta):
    """Add the weighted size-depth residual to original NSC."""
    capacities = layer_capacities(config)
    length = len(capacities)
    total = math.fsum(capacities)
    raw_tilt = math.fsum(
        ((i + 0.5) / length - 0.5) * c
        for i, c in enumerate(capacities)
    )
    predicted = (COEF_INTERCEPT
                 + COEF_LOG_CAPACITY * math.log(total)
                 + COEF_LOG_DEPTH * math.log(length))
    residual = raw_tilt - predicted
    return total + beta * residual


def score(config):
    """Return a finite scalar, with larger values predicting better quality."""
    return _tilted(config, BETA)


def ablation(config):
    """Disable only the tilt; BETA=0 reproduces original NSC."""
    return _tilted(config, 0.0)

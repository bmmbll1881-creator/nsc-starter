"""Historical positive depth tilt used in the method-development comparison."""

import math

from nsc import layer_capacities, original_nsc


BETA = 0.67


def from_capacities(capacities, beta=BETA):
    """Compute the unadjusted tilt for a sequence of layer capacities."""
    depth = len(capacities)
    return math.fsum(
        (1.0 + beta * ((i + 0.5) / depth - 0.5)) * capacity
        for i, capacity in enumerate(capacities)
    )


def score(config):
    """Score a GPT-2/Transformer-XL specification with the historical tilt."""
    return from_capacities(layer_capacities(config))


def ablation(config):
    """Remove layer ordering by returning original NSC."""
    return original_nsc(config)

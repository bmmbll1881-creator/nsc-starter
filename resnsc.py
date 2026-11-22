#!/usr/bin/env python3
"""Independent implementation of ResNSC scoring (arXiv:2610.30823).

Follows Appendix B-D: residual ledger, read/write deterministic equivalents,
32-point Gauss-Hermite SiLU gate law.  Standard library only; the quadrature
nodes and weights are hard-coded.

This is a scoring implementation, not ResNSC-DP.  Input is an ordered list of
(residual_width d_l, FFN_width f_l) pairs.
"""
import math

# 32-point Gauss-Hermite rule for the standard normal (weights sum to one).
GH_Z = [
    -7.125813909830728, -6.4094981492696608, -5.8122259495159136, -5.2755509865158805,
    -4.777164503502596, -4.3055479533511987, -3.8537554854714449, -3.4171674928185709,
    -2.9924908250023741, -2.5772495377323175, -2.1694991836061122, -1.7676541094632015,
    -1.3703764109528718, -0.97650046358968279, -0.58497876543593241, -0.19484074156939934,
    0.19484074156939934, 0.58497876543593241, 0.97650046358968279, 1.3703764109528718,
    1.7676541094632015, 2.1694991836061122, 2.5772495377323175, 2.9924908250023741,
    3.4171674928185709, 3.8537554854714449, 4.3055479533511987, 4.777164503502596,
    5.2755509865158805, 5.8122259495159136, 6.4094981492696608, 7.125813909830728,
]
GH_W = [
    4.1246074890182323e-23, 5.2084495919608421e-19, 6.7552902236701287e-16, 2.3780648557777909e-13,
    3.3475012398012285e-11, 2.312518412074231e-09, 8.8812907131058625e-08, 2.0596221039534304e-06,
    3.0559803060896349e-05, 0.00030255702581706253, 0.0020620510513078838, 0.0099034617023205963,
    0.034109847726092095, 0.085344808272080672, 0.1565389937575985, 0.21170556988047937,
    0.21170556988047937, 0.1565389937575985, 0.085344808272080672, 0.034109847726092095,
    0.0099034617023205963, 0.0020620510513078838, 0.00030255702581706253, 3.0559803060896349e-05,
    2.0596221039534304e-06, 8.8812907131058625e-08, 2.312518412074231e-09, 3.3475012398012285e-11,
    2.3780648557777909e-13, 6.7552902236701287e-16, 5.2084495919608421e-19, 4.1246074890182323e-23,
]


def ledger(depths, embed_width):
    """Return the residual-power ledger before each of the 2L branches."""
    max_width = max([embed_width] + list(depths))
    p = [1.0 if i < embed_width else 0.0 for i in range(max_width)]
    states = []
    for d in depths:
        states.append(p[:d])
        p[:d] = [v + 1.0 for v in p[:d]]
        states.append(p[:d])
        p[:d] = [v + 1.0 for v in p[:d]]
    return states, p


def z_depth(L):
    """Compute the residual calibration term at the given depth."""
    return 2 * L / math.fsum(1.0 / (b + 2) for b in range(2 * L))


def read_score(tau, m, v):
    """Deterministic read evaluation, Equations (D.1)-(D.2)."""
    if not any(tau > 0 for tau in tau):
        return 0.0
    lo, hi = 0.0, 1.0
    for _ in range(48):
        xi = 0.5 * (lo + hi)
        rhs = math.fsum(v * xi * t / (1.0 + m * v * xi * t)
                        for t in tau if t > 0)
        if 1.0 - xi - rhs > 0:
            lo = xi
        else:
            hi = xi
    xi = 0.5 * (lo + hi)
    return (math.fsum(math.log1p(m * v * xi * t) for t in tau if t > 0)
            - m * math.log(xi) - m * (1.0 - xi))


def gate_law(d, f):
    """Normalised squared-SiLU law, Equation (D.5)."""
    q = 2.0 * d / (d + f)
    raw = []
    for z, w in zip(GH_Z, GH_W):
        silu = math.sqrt(q) * z / (1.0 + math.exp(-math.sqrt(q) * z))
        raw.append((silu * silu, w))
    norm = math.fsum(u * w for u, w in raw)
    return [(u / norm, w) for u, w in raw]


def write_score(chi, rho, k, v_in, v_out, gate):
    """Deterministic write evaluation, Equations (D.3)-(D.4)."""
    if not any(c > 0 for c in chi):
        return 0.0
    alpha = v_in * math.fsum(chi)
    beta = v_out * math.fsum(rho)

    def step(alpha, beta):
        """Compute the residual-context term for this projection step."""
        gamma = math.fsum(w * u / (1.0 + alpha * beta * u) for u, w in gate)
        a = v_in * math.fsum(c / (1.0 + k * v_in * beta * gamma * c)
                             for c in chi if c > 0)
        b = v_out * math.fsum(r / (1.0 + k * v_out * alpha * gamma * r)
                              for r in rho if r > 0)
        return a, b, gamma

    for _ in range(1000):
        a, b, gamma = step(alpha, beta)
        na = 0.5 * (alpha + a)
        nb = 0.5 * (beta + b)
        change = max(abs(na - alpha) / max(1.0, abs(na)),
                     abs(nb - beta) / max(1.0, abs(nb)))
        alpha, beta = na, nb
        if change < 2e-12:
            break
    gamma = math.fsum(w * u / (1.0 + alpha * beta * u) for u, w in gate)
    return (math.fsum(math.log1p(k * v_in * beta * gamma * c)
                      for c in chi if c > 0)
            + math.fsum(math.log1p(k * v_out * alpha * gamma * r)
                        for r in rho if r > 0)
            + k * math.fsum(w * math.log1p(alpha * beta * u) for u, w in gate)
            - 2.0 * k * alpha * beta * gamma)


def block_score(p, d, f, L, heads=16):
    """One block, Equation (C.6).  p is the ledger before attention."""
    z = z_depth(L)
    p = list(p)

    def metrics(pv):
        """Collect residual read and write statistics for this block."""
        pbar = math.fsum(pv) / len(pv)
        if pbar <= 0:
            return None
        sin = [v / pbar for v in pv]
        sread = [z * v / (pbar * (1.0 + v)) for v in pv]
        rho = [z / (1.0 + v) for v in pv]
        return sin, sread, rho

    m0 = metrics(p)
    if m0 is None:
        return 0.0
    sin0, sread0, rho0 = m0
    # 3 complete Q/K/V reads: d x d stacked heads, per-head variance.
    v_a = 2.0 / (d + d / heads)
    score = 3.0 * read_score(sread0, d, v_a)
    # O write: k=d, v_in=1/d, v_out=1/d, delta gate.
    score += write_score(sin0, rho0, d, 1.0 / d, 1.0 / d, [(1.0, 1.0)])

    # Attention write increments the ledger.
    p1 = [v + 1.0 for v in p]
    m1 = metrics(p1)
    if m1 is None:
        return 0.0
    sin1, sread1, rho1 = m1
    v_f = 2.0 / (d + f)
    score += 2.0 * read_score(sread1, f, v_f)
    score += write_score(sin1, rho1, f, 1.0 / d, v_f, gate_law(d, f))
    return score


def score(pairs, embed_width, heads=16):
    """ResNSC score for an ordered list of (d_l, f_l) pairs."""
    depths = [d for d, _ in pairs]
    L = len(pairs)
    states, _ = ledger(depths, embed_width)
    return math.fsum(block_score(states[2 * i], d, f, L, heads)
                     for i, (d, f) in enumerate(pairs))

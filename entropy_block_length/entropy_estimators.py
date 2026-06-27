"""Binary entropy-rate estimators and synthetic source generators."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from statistics import mean
from typing import Callable, Iterable

BitSeq = list[int]


def binary_entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)


def logsumexp2(values: Iterable[float]) -> float:
    values = list(values)
    m = max(values)
    return m + math.log2(sum(2.0 ** (v - m) for v in values))


def kt_log_probability(n0: int, n1: int) -> float:
    total = 0.0
    for i in range(n0):
        total += math.log2((i + 0.5) / (i + 1.0))
    for i in range(n1):
        total += math.log2((i + 0.5) / (n0 + i + 1.0))
    return total


class ContextNode:
    def __init__(self) -> None:
        self.counts = [0, 0]
        self.children: dict[int, ContextNode] = {}


def build_context_tree(bits: BitSeq, depth: int) -> tuple[ContextNode, int]:
    root = ContextNode()
    usable = 0
    for i in range(depth, len(bits)):
        node = root
        symbol = bits[i]
        node.counts[symbol] += 1
        for d in range(1, depth + 1):
            context_bit = bits[i - d]
            node = node.children.setdefault(context_bit, ContextNode())
            node.counts[symbol] += 1
        usable += 1
    return root, usable


def ctw_node_log_probability(node: ContextNode, current_depth: int, max_depth: int) -> float:
    leaf_log = kt_log_probability(node.counts[0], node.counts[1])
    if current_depth == max_depth or not node.children:
        return leaf_log
    children_log = sum(
        ctw_node_log_probability(child, current_depth + 1, max_depth)
        for child in node.children.values()
    )
    return logsumexp2([math.log2(0.5) + leaf_log, math.log2(0.5) + children_log])


def ctw_entropy(bits: BitSeq, depth: int = 6) -> float:
    root, usable = build_context_tree(bits, depth)
    if usable <= 0:
        return float("nan")
    return -ctw_node_log_probability(root, 0, depth) / usable


def lz_gkb_entropy(bits: BitSeq, window: int = 1024, max_match: int = 128) -> float:
    terms: list[float] = []
    n = len(bits)
    text = "".join(str(bit) for bit in bits)
    if n <= 16:
        return float("nan")
    window = min(window, max(8, n // 2))

    for i in range(window, n):
        history = text[i - window : i]
        longest = 0
        cap = min(max_match, n - i)
        for length in range(1, cap + 1):
            pattern = text[i : i + length]
            if pattern in history:
                longest = length
            else:
                break
        if longest > 0:
            terms.append(math.log2(window) / (longest + 0.5))
    return mean(terms) if terms else float("nan")


def fixed_block_actw_entropy(bits: BitSeq, block_length: int, depth: int = 5) -> float:
    """ACTW-style estimate using non-overlapping fixed blocks (no change-point labels)."""
    if len(bits) <= depth + 1:
        return float("nan")
    block_length = min(block_length, len(bits))
    estimates: list[float] = []
    for start in range(0, len(bits) - block_length + 1, block_length):
        estimate = ctw_entropy(bits[start : start + block_length], depth=depth)
        if not math.isnan(estimate):
            estimates.append(estimate)
    if not estimates:
        return ctw_entropy(bits, depth=depth)
    return mean(estimates)


def random_window_actw_entropy(
    bits: BitSeq,
    min_window: int,
    max_window: int,
    samples: int = 32,
    depth: int = 5,
    rng: random.Random | None = None,
) -> float:
    """Random local-window ACTW; window range is a hyperparameter, not segment labels."""
    if len(bits) <= depth + 1:
        return float("nan")
    rng = rng or random.Random(0)
    max_window = min(max_window, len(bits))
    min_window = min(min_window, max_window)
    estimates: list[float] = []
    for _ in range(samples):
        window = rng.randint(min_window, max_window)
        end = rng.randint(window, len(bits))
        estimate = ctw_entropy(bits[end - window : end], depth=depth)
        if not math.isnan(estimate):
            estimates.append(estimate)
    return mean(estimates) if estimates else ctw_entropy(bits, depth=depth)


def sample_bernoulli(n: int, p: float, rng: random.Random) -> BitSeq:
    return [1 if rng.random() < p else 0 for _ in range(n)]


def sample_markov(n: int, p1_given_0: float, p1_given_1: float, rng: random.Random) -> BitSeq:
    x = 1 if rng.random() < 0.5 else 0
    bits = [x]
    for _ in range(1, n):
        p = p1_given_1 if bits[-1] == 1 else p1_given_0
        bits.append(1 if rng.random() < p else 0)
    return bits


def sample_order2_markov(n: int, transitions: dict[tuple[int, int], float], rng: random.Random) -> BitSeq:
    bits = [1 if rng.random() < 0.5 else 0, 1 if rng.random() < 0.5 else 0]
    for _ in range(2, n):
        p = transitions[(bits[-2], bits[-1])]
        bits.append(1 if rng.random() < p else 0)
    return bits


def sample_piecewise_bernoulli(n: int, ps: list[float], rng: random.Random) -> BitSeq:
    bits: BitSeq = []
    segment = n // len(ps)
    for idx, p in enumerate(ps):
        length = segment if idx < len(ps) - 1 else n - len(bits)
        bits.extend(sample_bernoulli(length, p, rng))
    return bits


def sample_drifting_bernoulli(n: int, p_start: float, p_end: float, rng: random.Random) -> BitSeq:
    return [
        1 if rng.random() < (p_start + (p_end - p_start) * t / max(1, n - 1)) else 0
        for t in range(n)
    ]


def sample_piecewise_markov(
    n: int, segments: list[tuple[float, float]], rng: random.Random
) -> BitSeq:
    bits: BitSeq = []
    segment_length = n // len(segments)
    last = 1 if rng.random() < 0.5 else 0
    for idx, (p01, p11) in enumerate(segments):
        length = segment_length if idx < len(segments) - 1 else n - len(bits)
        segment_bits = [last]
        for _ in range(1, length):
            p = p11 if segment_bits[-1] == 1 else p01
            segment_bits.append(1 if rng.random() < p else 0)
        bits.extend(segment_bits)
        last = bits[-1]
    return bits[:n]


def stationary_markov_entropy(p1_given_0: float, p1_given_1: float) -> float:
    pi1 = p1_given_0 / (p1_given_0 + 1.0 - p1_given_1)
    pi0 = 1.0 - pi1
    return pi0 * binary_entropy(p1_given_0) + pi1 * binary_entropy(p1_given_1)


def stationary_order2_entropy(transitions: dict[tuple[int, int], float]) -> float:
    states = [(0, 0), (0, 1), (1, 0), (1, 1)]
    distribution = {state: 0.25 for state in states}
    for _ in range(10_000):
        next_distribution = {state: 0.0 for state in states}
        for state, mass in distribution.items():
            p1 = transitions[state]
            next_distribution[(state[1], 0)] += mass * (1.0 - p1)
            next_distribution[(state[1], 1)] += mass * p1
        distribution = next_distribution
    return sum(distribution[state] * binary_entropy(transitions[state]) for state in states)


def clipped(value: float, low: float = 0.01, high: float = 0.99) -> float:
    return min(high, max(low, value))


@dataclass(frozen=True)
class Source:
    name: str
    family: str
    generator: Callable[[int, random.Random], tuple[BitSeq, float]]


def make_sources() -> list[Source]:
    base_order2 = {(0, 0): 0.05, (0, 1): 0.80, (1, 0): 0.35, (1, 1): 0.60}

    def bernoulli_source(n: int, rng: random.Random) -> tuple[BitSeq, float]:
        p = rng.uniform(0.20, 0.40)
        return sample_bernoulli(n, p, rng), binary_entropy(p)

    def markov1_source(n: int, rng: random.Random) -> tuple[BitSeq, float]:
        p01, p11 = rng.uniform(0.05, 0.14), rng.uniform(0.78, 0.92)
        return sample_markov(n, p01, p11, rng), stationary_markov_entropy(p01, p11)

    def markov2_source(n: int, rng: random.Random) -> tuple[BitSeq, float]:
        transitions = {s: clipped(p + rng.uniform(-0.08, 0.08)) for s, p in base_order2.items()}
        return sample_order2_markov(n, transitions, rng), stationary_order2_entropy(transitions)

    def piecewise_bernoulli_source(n: int, rng: random.Random) -> tuple[BitSeq, float]:
        center, spread = rng.uniform(0.12, 0.88), rng.uniform(0.04, 0.50)
        ps = [clipped(center - spread), center, clipped(center + spread)]
        return sample_piecewise_bernoulli(n, ps, rng), mean(binary_entropy(p) for p in ps)

    def drifting_bernoulli_source(n: int, rng: random.Random) -> tuple[BitSeq, float]:
        center, span = rng.uniform(0.12, 0.88), rng.uniform(0.06, 0.90)
        p_start, p_end = clipped(center - span / 2.0), clipped(center + span / 2.0)
        if rng.random() < 0.5:
            p_start, p_end = p_end, p_start
        return (
            sample_drifting_bernoulli(n, p_start, p_end, rng),
            mean(binary_entropy(p_start + (p_end - p_start) * t / 999.0) for t in range(1000)),
        )

    def piecewise_markov_source(n: int, rng: random.Random) -> tuple[BitSeq, float]:
        base_p01, base_p11 = rng.uniform(0.01, 0.50), rng.uniform(0.38, 0.98)
        shift = rng.uniform(0.01, 0.44)
        segments = [
            (clipped(base_p01 - shift), clipped(base_p11 + shift)),
            (base_p01, base_p11),
            (clipped(base_p01 + shift), clipped(base_p11 - shift)),
        ]
        return sample_piecewise_markov(n, segments, rng), mean(
            stationary_markov_entropy(a, b) for a, b in segments
        )

    return [
        Source("bernoulli", "stationary", bernoulli_source),
        Source("markov1", "stationary", markov1_source),
        Source("markov2", "stationary", markov2_source),
        Source("piecewise_bernoulli", "nonstationary", piecewise_bernoulli_source),
        Source("drifting_bernoulli", "nonstationary", drifting_bernoulli_source),
        Source("piecewise_markov1", "nonstationary", piecewise_markov_source),
    ]

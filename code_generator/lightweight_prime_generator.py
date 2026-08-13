#!/usr/bin/env python3
"""Lightweight (sparse) NTT-prime recognizer and bounded search.

CONSERVATIVE, internal/experimental, DEFAULT-OFF. This module recognizes whether a
modulus q is usable as an NTT prime for a given transform length N and, if so, whether
the EXISTING shift-add reduce emitters could serve it; otherwise it routes to Barrett.
It also offers a bounded, deterministic search for lightweight-prime candidates.

It changes NO datapath / template / emission behavior on its own. It reuses, never
duplicates, the existing arithmetic:
  * ``twiddle_generator.is_prime64``           — primality.
  * ``q-1`` divisibility by ``2N``             — 2N-th-root (NTT-friendliness) precondition.
  * ``reduce_recipe.build_reduce_recipe``      — the ground-truth shift-add eligibility oracle.
  * the {51,61} anchor convention              — from ``shiftadd_generator._shiftadd_q_anchor``.

Supported anchor families are derived from ``reduce_generator.ANCHOR_FAMILIES``.
The recipe builder decides whether a recognized modulus can use shift-add or must
use Barrett.
"""
import itertools
from typing import List, Optional, Sequence, Tuple

from .twiddle_generator import is_prime64
from .reduce_recipe import build_reduce_recipe
from .reduce_generator import anchor_family_for, _anchor_b_for, ANCHOR_FAMILIES
import dataclasses

POLICIES: Tuple[str, ...] = ("off", "recognize", "generate")
_POLICY_ENV = "STREAMNTT_LIGHTWEIGHT_PRIME_POLICY"
# The supported target-bit set follows from anchors {31,51,61,63}.
_MAX_ENVELOPE_BITS = 64
_SUPPORTED_TARGET_BITS: Tuple[int, ...] = (32, 52, 62, 64)
# anchor-family default K_groups (derived from reduce_generator.ANCHOR_FAMILIES, the single source of truth).
_KGROUP_TO_ANCHOR = {f.default_K_group: f.anchor_B for f in ANCHOR_FAMILIES}   # {32:31,52:51,62:61,64:63}
_LWP_EXP_LO = 23                              # signed-sparse lower exponent bound
_LWP_MAX_SPARSE_WEIGHT = 5                    # supported signed-sparse weight cap


class LightweightPrimeError(RuntimeError):
    """Raised when a bounded search finds no safe candidate (clear failure)."""


@dataclasses.dataclass(frozen=True)
class LightweightPrimeCandidate:
    """A recognized prime annotated with its supported route (immutable)."""
    q: int
    bit_length: int
    k_group: int                  # 52 | 62
    anchor_B: int                 # 51 | 61
    sparse_score: int             # NAF (signed-binary) weight; lower = sparser
    ntt_friendly: bool            # 2N | (q-1) for the requested N
    shiftadd_eligible: bool       # build_reduce_recipe(...).can_emit_shiftadd_reduce
    route: str                    # "shiftadd_reduce" | "barrett"
    source: str                   # "table" | "generated" | "user"
    notes: Tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class LightweightPrimeBatch:
    """Per-batch recognition result."""
    candidates: Tuple[Optional[LightweightPrimeCandidate], ...]
    class_map: Tuple[int, ...]    # sorted unique k_groups of recognized primes
    all_recognized: bool
    all_shiftadd_eligible: bool


# ---------------------------------------------------------------------------
# Policy (env-gated; default off)
# ---------------------------------------------------------------------------
def policy_from_env(environ) -> str:
    """Resolve the lightweight-prime policy from an environ-like mapping.

    Unset / empty / whitespace -> "off" (default; byte-identical legacy path).
    Case-insensitive. Raises ValueError on any other value.
    """
    raw = environ.get(_POLICY_ENV)
    value = (raw or "").strip().lower()
    if value == "":
        return "off"
    if value not in POLICIES:
        raise ValueError(
            "%s=%r is not valid. Use one of %s (unset/empty = off)."
            % (_POLICY_ENV, raw, POLICIES))
    return value


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _naf_weight(n: int) -> int:
    """Number of nonzero digits in the non-adjacent form of |n| (sparseness metric)."""
    n = abs(int(n))
    weight = 0
    while n:
        if n & 1:
            n -= 2 - (n & 3)      # subtract +1 or -1 so the next bit clears
            weight += 1
        n >>= 1
    return weight


def _k_group_and_anchor(q: int) -> Optional[Tuple[int, int]]:
    """Discover (K_group, anchor_B) from the prime's sparse magnitude.

    Uses reduce_generator.anchor_family_for as the single source of truth, then the family's
    default K_group. None when the prime's anchor pushes K_group outside the <=64-bit envelope. Known
    anchors {31,51,61,63} -> K_group {32,52,62,64}; existing 51/61 primes keep 52/62."""
    if q.bit_length() > _MAX_ENVELOPE_BITS:
        return None
    fam = anchor_family_for(q)
    if fam.default_K_group > _MAX_ENVELOPE_BITS:
        return None                          # e.g. anchor 64 -> K_group 65: out of the 64-bit envelope
    return (fam.default_K_group, fam.anchor_B)


def _ntt_friendly(q: int, N: int) -> bool:
    """q admits a 2N-th root: 2N | (q-1). Primality is checked separately."""
    if N <= 0:
        raise ValueError("N must be a positive int, got %r" % (N,))
    return (q - 1) % (2 * N) == 0


def _shiftadd_eligible(q: int) -> bool:
    """True if the EXISTING emitters could serve q on the shift-add reduce route."""
    try:
        recipe = build_reduce_recipe([("INT_%d" % q, q)])
    except (ValueError, RuntimeError):
        return False
    return bool(recipe.can_emit_shiftadd_reduce)


# ---------------------------------------------------------------------------
# Recognizer
# ---------------------------------------------------------------------------
def recognize_lightweight_prime(q, *, N: int, target_k_group: Optional[int] = None,
                                source: str = "user") -> Optional[LightweightPrimeCandidate]:
    """Recognize q for transform length N.

    Returns None when q is not prime, is outside the
    <=64-bit envelope, fails the 2N-th-root precondition, or mismatches an
    explicit target_k_group. Otherwise returns a candidate annotated with
    route="shiftadd_reduce" (eligible) or route="barrett" (valid prime, shift-add
    ineligible -> fall back to Barrett).
    """
    if isinstance(q, bool) or not isinstance(q, int):
        raise ValueError("q must be an int, got %r" % (q,))
    if not is_prime64(q):
        return None
    ka = _k_group_and_anchor(q)
    if ka is None:
        return None                          # Outside the supported envelope.
    k_group, anchor_B = ka
    if target_k_group is not None and k_group != target_k_group:
        return None
    if not _ntt_friendly(q, N):
        return None
    eligible = _shiftadd_eligible(q)
    route = "shiftadd_reduce" if eligible else "barrett"
    note = ("recipe can_emit_shiftadd_reduce"
            if eligible else "shift-add ineligible -> Barrett fallback")
    return LightweightPrimeCandidate(
        q=q, bit_length=q.bit_length(), k_group=k_group, anchor_B=anchor_B,
        sparse_score=_naf_weight(q), ntt_friendly=True, shiftadd_eligible=eligible,
        route=route, source=source, notes=(note,))


def recognize_batch(qs: Sequence[int], *, N: int) -> LightweightPrimeBatch:
    """Recognize each prime in qs; report the class map and batch eligibility."""
    cands = tuple(recognize_lightweight_prime(q, N=N) for q in qs)
    recognized = [c for c in cands if c is not None]
    class_map = tuple(sorted({c.k_group for c in recognized}))
    return LightweightPrimeBatch(
        candidates=cands,
        class_map=class_map,
        all_recognized=all(c is not None for c in cands),
        all_shiftadd_eligible=bool(recognized) and all(c.shiftadd_eligible for c in recognized),
    )


# ---------------------------------------------------------------------------
# Bounded, deterministic search
# ---------------------------------------------------------------------------
def _signed_sparse_candidates(anchor_B: int, *, exp_lo: int = _LWP_EXP_LO,
                              max_weight: int = _LWP_MAX_SPARSE_WEIGHT, max_steps: int = 20000):
    """Yield SIGNED-sparse candidates q = 2^anchor_B +/- 2^a +/- 2^b ... + 1 (Proth-style +1 unit so
    q-1 keeps >= exp_lo trailing zeros = 2N-friendly), deterministically. ANCHOR-FIRST: the leading
    term +2^anchor_B is fixed by the anchor (NOT bit_length); up to (max_weight-2) signed extra terms
    with exponents in [exp_lo, anchor_B-1]. Fewer-term (sparser) candidates first. Bounded by max_steps."""
    base = 1 << anchor_B
    hi = anchor_B - 1
    steps = 0
    for n_extra in range(0, max(0, max_weight - 2) + 1):     # weight = leading(1) + unit(1) + n_extra
        for exps in itertools.combinations(range(exp_lo, hi + 1), n_extra):
            for ssigns in itertools.product((1, -1), repeat=n_extra):
                if steps >= max_steps:
                    return
                steps += 1
                q = base + sum(s * (1 << e) for s, e in zip(ssigns, exps)) + 1
                if q > 1:
                    yield q


def _generative_candidates(target_bits: int, max_steps: int):
    """Yield signed-sparse candidates for the target family, anchored ANCHOR-FIRST at the family's
    anchor_B (32->31, 52->51, 62->61, 64->63). Signed forms q = 2^anchor +/- 2^a ... + 1, weight <= 5.

    Lower exponent bound is _LWP_EXP_LO (23) WHERE APPLICABLE, but is scaled down for SMALL anchors:
    the shift-add fold needs alpha (= the lower term) << 2^(anchor_B/2), so a 31-bit anchor cannot use
    a >= 23 AND stay shift-add-eligible. exp_lo = max(11, min(23, anchor_B-12)) keeps 2N-friendliness
    (>=11) while letting the 31-anchor family reach shift-add-eligible small-alpha primes (a ~ 17-19)."""
    anchor_B = _KGROUP_TO_ANCHOR.get(target_bits, target_bits - 1)
    exp_lo = max(11, min(_LWP_EXP_LO, anchor_B - 12))
    yield from _signed_sparse_candidates(anchor_B, exp_lo=exp_lo, max_steps=max_steps)


def search_lightweight_primes(*, target_bits: int, N: int, count: int,
                              k_group: Optional[int] = None,
                              pool: Optional[Sequence[int]] = None,
                              max_steps: int = 20000,
                              sparse_weight_limit: int = _LWP_MAX_SPARSE_WEIGHT,
                              require_shiftadd: bool = True
                              ) -> List[LightweightPrimeCandidate]:
    """Find up to ``count`` lightweight primes near ``target_bits`` for length ``N``.

    Deterministic: an optional ``pool`` (e.g. the known table) is scanned ascending
    first, then a bounded generative sweep tops up. Raises ValueError for an
    unsupported ``target_bits`` (only 52 / 62; true-64 is out of envelope) and
    LightweightPrimeError when no safe candidate is found (clear failure).
    """
    if target_bits not in _SUPPORTED_TARGET_BITS:
        raise ValueError(
            "target_bits=%r unsupported; only %s (true 64-bit is out of envelope)."
            % (target_bits, _SUPPORTED_TARGET_BITS))
    want_k = k_group if k_group is not None else target_bits   # target_bits in {32,52,62,64} == K_group
    found: List[LightweightPrimeCandidate] = []
    seen = set()

    def consider(q: int, src: str) -> None:
        if q in seen:
            return
        seen.add(q)
        cand = recognize_lightweight_prime(q, N=N, target_k_group=want_k, source=src)
        if cand is None:
            return
        if require_shiftadd and not cand.shiftadd_eligible:
            return
        if cand.sparse_score > sparse_weight_limit:
            return
        found.append(cand)

    for q in sorted(pool or []):
        if len(found) >= count:
            break
        consider(q, "table")
    if len(found) < count:
        for q in _generative_candidates(target_bits, max_steps):
            if len(found) >= count:
                break
            consider(q, "generated")

    if not found:
        raise LightweightPrimeError(
            "no lightweight prime found (target_bits=%d, N=%d, k_group=%s, pool=%d, "
            "max_steps=%d, require_shiftadd=%s)."
            % (target_bits, N, want_k, len(pool or []), max_steps, require_shiftadd))
    return found[:count]


# ---------------------------------------------------------------------------
# Generate-mode parameters and same-K prime list
# ---------------------------------------------------------------------------
_GEN_TARGET_BITS_ENV = "STREAMNTT_LWP_TARGET_BITS"
_GEN_KGROUP_ENV = "STREAMNTT_LWP_KGROUP"
_GEN_COUNT_ENV = "STREAMNTT_LWP_COUNT"
_GEN_MAX_STEPS_ENV = "STREAMNTT_LWP_MAX_STEPS"
_DEFAULT_TARGET_BITS = 52
_DEFAULT_MAX_STEPS = 20000
_GEN_MIXED_ENV = "STREAMNTT_LWP_MIXED"
_GEN_COUNT_52_ENV = "STREAMNTT_LWP_COUNT_52"
_GEN_COUNT_62_ENV = "STREAMNTT_LWP_COUNT_62"


@dataclasses.dataclass(frozen=True)
class GenerateParams:
    """Resolved generate-mode parameters (same-K, conservative)."""
    target_bits: int     # 32 | 52 | 62 | 64
    k_group: int         # Must equal the target_bits class in same-K mode.
    count: int
    max_steps: int


def _int_env(environ, key: str, default: int) -> int:
    raw = environ.get(key)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return int(str(raw).strip())
    except ValueError:
        raise ValueError("%s=%r must be an integer." % (key, raw))


def _bool_env(environ, key: str) -> bool:
    return (environ.get(key) or "").strip().lower() in ("1", "true", "yes", "on")


def generate_params_from_env(environ, *, default_count: int) -> GenerateParams:
    """Resolve generate-mode params from env (same-K; conservative defaults).

    Defaults: target_bits=52, k_group=class(target_bits), count=default_count, max_steps=20000.
    Raises ValueError when target_bits or k_group is not a supported anchor-family
    width, when k_group differs from class(target_bits) in same-K mode, for non-positive count/max_steps,
    or non-integer values.
    """
    target_bits = _int_env(environ, _GEN_TARGET_BITS_ENV, _DEFAULT_TARGET_BITS)
    if target_bits not in _SUPPORTED_TARGET_BITS:
        raise ValueError("%s=%d unsupported; only %s (true 64-bit is out of envelope)."
                         % (_GEN_TARGET_BITS_ENV, target_bits, _SUPPORTED_TARGET_BITS))
    expected_kg = target_bits   # anchor-first: target_bits in {32,52,62,64} == K_group
    k_group = _int_env(environ, _GEN_KGROUP_ENV, expected_kg)
    if k_group not in _SUPPORTED_TARGET_BITS:
        raise ValueError("%s=%d unsupported; only %s."
                         % (_GEN_KGROUP_ENV, k_group, _SUPPORTED_TARGET_BITS))
    if k_group != expected_kg:
        raise ValueError(
            "same-K generate: %s=%d must match the target_bits class %d. "
            "Use mixed generation mode for a mixed-K batch." % (_GEN_KGROUP_ENV, k_group, expected_kg))
    count = _int_env(environ, _GEN_COUNT_ENV, default_count)
    if count <= 0:
        raise ValueError("%s must be a positive int, got %d." % (_GEN_COUNT_ENV, count))
    max_steps = _int_env(environ, _GEN_MAX_STEPS_ENV, _DEFAULT_MAX_STEPS)
    if max_steps <= 0:
        raise ValueError("%s must be a positive int, got %d." % (_GEN_MAX_STEPS_ENV, max_steps))
    return GenerateParams(target_bits=target_bits, k_group=k_group, count=count, max_steps=max_steps)


def _generate_mixed_prime_list(*, N: int, environ, default_count: int) -> List[int]:
    """Mixed-K generated prime list: a 52-class block followed by a 62-class block.

    Per-class counts from STREAMNTT_LWP_COUNT_52 / _62 (default: default_count split evenly, >=1 each).
    Both classes use require_shiftadd=True. The concatenated INT_<q> list classifies as mixed_bit
    {52,62} downstream. Fails fast (LightweightPrimeError) if either class is short; ValueError on
    invalid params.
    """
    default_each = max(1, default_count // 2)
    count_52 = _int_env(environ, _GEN_COUNT_52_ENV, default_each)
    count_62 = _int_env(environ, _GEN_COUNT_62_ENV, default_each)
    if count_52 <= 0 or count_62 <= 0:
        raise ValueError("mixed generate: per-class counts must be positive (got 52=%d, 62=%d)."
                         % (count_52, count_62))
    max_steps = _int_env(environ, _GEN_MAX_STEPS_ENV, _DEFAULT_MAX_STEPS)
    if max_steps <= 0:
        raise ValueError("%s must be a positive int, got %d." % (_GEN_MAX_STEPS_ENV, max_steps))
    c52 = search_lightweight_primes(target_bits=52, N=N, count=count_52, k_group=52,
                                    max_steps=max_steps, require_shiftadd=True)
    c62 = search_lightweight_primes(target_bits=62, N=N, count=count_62, k_group=62,
                                    max_steps=max_steps, require_shiftadd=True)
    qs = [c.q for c in c52] + [c.q for c in c62]
    for q in qs:
        if (q - 1) % (2 * N) != 0:
            raise LightweightPrimeError(
                "generated prime %d is not 2N-friendly for N=%d (internal inconsistency)." % (q, N))
    return qs


def generate_prime_list(*, N: int, environ, default_count: int) -> List[int]:
    """Generated prime list (integer q's) for transform length N.

    Dispatches on STREAMNTT_LWP_MIXED: truthy -> mixed-K {52,62} batch; otherwise same-K. Same-K
    resolves env params and runs the shift-add-eligible search (require_shiftadd=True). Every returned
    q is prime, ≤62-bit, and 2N-friendly for THIS N. Fails fast (LightweightPrimeError) when fewer than
    the requested count are found; ValueError on invalid params.
    """
    if _bool_env(environ, _GEN_MIXED_ENV):
        return _generate_mixed_prime_list(N=N, environ=environ, default_count=default_count)
    params = generate_params_from_env(environ, default_count=default_count)
    # shiftadd recipe builds for anchors {31,51,61,63} (small-alpha primes; exp_lo scaled per family).
    _req_sa = params.k_group in (32, 52, 62, 64)
    cands = search_lightweight_primes(
        target_bits=params.target_bits, N=N, count=params.count, k_group=params.k_group,
        max_steps=params.max_steps, require_shiftadd=_req_sa)
    qs = [c.q for c in cands]
    if len(qs) < params.count:
        raise LightweightPrimeError(
            "requested %d lightweight prime(s) but found only %d (N=%d, target_bits=%d, "
            "k_group=%d, max_steps=%d)."
            % (params.count, len(qs), N, params.target_bits, params.k_group, params.max_steps))
    # Defensive re-check (plan 6.0-C §7): every generated prime must admit a 2N-th root for THIS N.
    for q in qs:
        if (q - 1) % (2 * N) != 0:
            raise LightweightPrimeError(
                "generated prime %d is not 2N-friendly for N=%d (internal inconsistency)." % (q, N))
    return qs


# ---------------------------------------------------------------------------
# Integration helper (stdout-only; consumed by generate_code recognize mode)
# ---------------------------------------------------------------------------
def recognition_report_lines(aliases_q: Sequence[Tuple[str, int]], *, N: int) -> List[str]:
    """One human-readable recognition line per (alias, q) pair. No emission effect."""
    lines: List[str] = []
    for alias, q in aliases_q:
        cand = recognize_lightweight_prime(q, N=N, source="user")
        if cand is None:
            lines.append(
                "lightweight-prime: %s bitlen=%d -> UNRECOGNIZED "
                "(non-prime / not 2N-friendly @N=%d / >62-bit)"
                % (alias, q.bit_length(), N))
        else:
            lines.append(
                "lightweight-prime: %s bitlen=%d k_group=%d anchor_B=%d sparse_score=%d route=%s"
                % (alias, cand.bit_length, cand.k_group, cand.anchor_B, cand.sparse_score, cand.route))
    return lines

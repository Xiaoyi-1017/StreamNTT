"""reduce_recipe.py -- ReduceRecipe composition layer (Phase 3.5 S3b).

PURE DATA ONLY in this slice: two frozen dataclasses (PrimeReduceFacts per
alias, ReduceRecipe per prime set) plus a TRANSITION known-table constructor
that derives every fact from q via the EXISTING pure helpers of the two leaf
generators. NOTHING here is consumed by the generation flow yet (consumption
by shiftadd_generator = S3c, by duck-typing); no emission changes; no
fail-fast behavior changes (the INT/custom-on-shiftadd fail-fast = S3d).

Layering (S3b plan Q1, Option B): this module sits ABOVE the two leaf
generators and imports DOWN into both:
    reduce_generator    -> _k_arith_for, classify_reduce_set (S1b class metadata)
    shiftadd_generator  -> anchor / NAF / envelope / resolver / self-check helpers
It must NEVER import the top-level generator module (which will construct
recipes in S3c) and shiftadd_generator must NEVER import this module (it
stays a leaf and will consume a recipe INSTANCE by duck-typing only).

REDUCE-ROUTE STATUS = TWO DISTINCT, NEVER-CONFLATED LAYERS (S3b user
decision, 2026-06-07, refining S3a F1):
  FIXED POINT  reduce_default_self_check_ok/_params -- the math self-check
      verdict at the FIXED default point (fold_depth=2, positive_only; the
      emit_shiftadd_header signature defaults). Truthful per prime: the
      large B=61 family passes 28/28; the small B=51 family is MIXED
      (21/41 pass incl. solo TII_33; 20/41 fail hard). A fixed-point FAIL
      does NOT mean the route is unsupported.
  LIVE RESOLVER  reduce_resolver_emit_ok/_params/_reason and
      can_emit_shiftadd_reduce -- what the CURRENT emit path would actually
      do: the reduce emitter hard-codes auto fold/correction resolution
      internally (fold in {2, 3} x {positive_only, signed}) and then
      self-checks at the RESOLVED point. This is the practical route
      capability. Probed live: every known-family solo emits (41/41 small,
      28/28 large); small-family reduce is a parameterization/resolver
      matter, not unsupported.
F1 REFINEMENT (recorded in the S3b records + handoff docs): the S3a
reading "small-family shiftadd=reduce fails / would raise at emit" was too
broad -- it held only for the fixed evaluation point at family granularity.

TRANSITION constructor (binding corrections C1-C5, S3a checkpoint 20260607_2054):
  C1  mixed {52,62} shiftadd unsupported = CURRENT single-anchor-family batch
      limitation (k_spread>=3), NOT permanent, NOT an S1 regression. The
      recipe RECORDS it (batch_legal_shiftadd / fail_fast_reasons).
  C2  route capability / self-check status is carried SEPARATELY for the
      reduce and mul routes, each with the params it was evaluated under.
  C3  anchor_exp == k_arith - 1 is an OBSERVED property of the current known
      families ONLY, NEVER a derivation rule: anchor_exp is STORED as
      independent data (from _shiftadd_ell, the current heuristic).
  C4  derivation-from-q is a TRANSITION observation, not the final sparse
      recognizer; KNOWN_PRIMES source expression text is NOT preserved.
  C5  INT/custom primes are RECORDED (provenance "unsupported_custom"),
      never raised on, in S3b. T_user != T_floor with a None
      mul_unsupported_reason is SUPPORTED-WITH-CORRECTION (delta = T_floor -
      T_user, delta_naf), NOT unsupported.

Terminology guard: ReduceRecipe.anchor_k_group is the shiftadd ANCHOR-side
batch width group max(q.bit_length()) -- the _shiftadd_q_anchor K_group --
and is pinned to it by the transition equivalence assert. It is deliberately
NOT named bare "k_group": it is NOT the S1b reduce-class K_group (max
per-prime K_arith in {52, 62}) and NOT final_K; that classification data
lives in class_map / rel_class / k_arith.
"""
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Optional, Sequence, Tuple

from code_generator.reduce_generator import ReduceClassInfo, _k_arith_for, classify_reduce_set
from code_generator.shiftadd_generator import (
    _decompose_signed_pow2,
    _shiftadd_class_sub_batches,
    _shiftadd_ell,
    _shiftadd_envelope_widths,
    _shiftadd_known_anchor_families,
    _shiftadd_math_self_check,
    _shiftadd_mul_self_check,
    _shiftadd_naf,
    _shiftadd_q_anchor,
    _shiftadd_resolve_fold_and_correction,
)

# The FIXED default reduce-route evaluation point (S3a F1 / plan R2): the
# emit_shiftadd_header signature defaults. The live emit path does NOT run
# at this point (it auto-resolves); statuses recorded against it carry it
# explicitly so they are never read as a param-independent or capability verdict.
REDUCE_SELF_CHECK_DEFAULT_PARAMS: Tuple[int, str] = (2, "positive_only")

# Bit lengths the current {51, 61} anchor heuristic accepts (_shiftadd_ell).
SHIFTADD_ANCHOR_BIT_LENGTHS: Tuple[int, ...] = (51, 52, 61, 62)

# Validity frame the recorded facts hold under (S3a section E; descriptive).
DOMAIN_TAGS: Tuple[str, ...] = (
    "bu_full_data2_wide_accumulator",
    "t_exactness: T_user + delta == T_floor (delta-corrected exact reciprocal)",
    "envelope: per-prime X in [0, (q-1)^2]; K_env is the Data2 container width only",
    "mul_hard_rules: r_pre in [0, 3q) with <= 2 corrections; r_pre < 0 or >= 3q unsupported",
)

# EMPTY curated-override hook (plan Q5 non-goals): reserved for future
# explicit metadata ("future_recognized" provenance); S3b must keep it empty.
# Populating it without implementing consumption fail-fasts at construction.
_RECIPE_OVERRIDES: dict = {}

_MIXED_BATCH_REASON = (
    "k_spread>=3 (current single-anchor-family limitation; mixed {52,62} "
    "shiftadd currently unsupported, NOT a permanent impossibility): "
    "K_spread=%d")


@dataclass(frozen=True)
class PrimeReduceFacts:
    """Per-alias reduce/mul route facts (PURE DATA; S3b plan Q2 as amended).

    Optional fields are None when the current helpers cannot evaluate them
    (bit_length outside SHIFTADD_ANCHOR_BIT_LENGTHS -> no anchor -> no route
    facts; bit_length > 62 -> no k_arith). Reduce-route status is recorded at
    TWO never-conflated layers (module docstring): the FIXED default point
    and the LIVE resolver. reduce_unsupported_reason is set ONLY when the
    current emit path could not serve the prime (resolver layer); a
    fixed-point FAIL alone does NOT set it. A None mul_unsupported_reason
    with t_match False means SUPPORTED-WITH-CORRECTION, never unsupported (C5).
    Per-prime resolver facts are the SOLO verdict (a single-prime case);
    batch verdicts live on ReduceRecipe.
    """
    # identity / class
    alias: str
    q: int
    k_bits: int                              # q.bit_length()
    k_arith: Optional[int]                   # 52 | 62 (_k_arith_for); None if > 62 bits
    rel_class: Optional[int]                 # set-relative index into the recipe class_map (S1b)
    anchor_exp: Optional[int]                # 51 | 61, STORED INDEPENDENTLY (C3); from _shiftadd_ell
    provenance: str                          # "known_table" | "unsupported_custom"
    #                                          ("future_recognized" RESERVED, never built in S3b)
    # reduce route -- working forms (q-derived envelope facts)
    alpha: Optional[int]                     # (1 << anchor_exp) - q, signed
    alpha_naf: Optional[Tuple[Tuple[int, int], ...]]   # _shiftadd_naf(alpha): ((sign, exp), ...)
    alpha_naf_weight: Optional[int]          # len(alpha_naf)
    wfold1: Optional[int]                    # signed fold-1 width (q-derived envelope)
    wfold2: Optional[int]                    # signed fold-2 width
    wfold3: Optional[int]                    # signed fold-3 width (fold_depth=3 envelope)
    f2_ok: Optional[bool]                    # fold-2 envelope gate: f2 in [-q, 2q)
    f3_ok: Optional[bool]                    # fold-3 envelope gate
    # reduce route -- FIXED-POINT status (recorded data, NOT capability)
    reduce_default_self_check_ok: Optional[bool]
    reduce_default_self_check_params: Optional[Tuple[int, str]]  # (fold_depth, correction)
    # reduce route -- LIVE-RESOLVER status (solo; the practical capability)
    reduce_resolver_emit_ok: Optional[bool]  # would the current emit path serve a solo case?
    reduce_resolver_params: Optional[Tuple[int, str]]  # (resolved_fold_depth, resolved_correction)
    reduce_resolver_reason: Optional[str]    # why not emittable / not evaluable; None on emit-ok
    reduce_unsupported_reason: Optional[str] # None iff reduce_resolver_emit_ok is True
    # mul route (Barrett-preserving reduce_shiftadd_mul; base = anchor_exp;
    # single evaluation point -- the mul emitter has no resolver)
    s: Optional[int]                         # q - 2^base - 1
    s_naf: Optional[Tuple[Tuple[int, int], ...]]       # _shiftadd_naf(s)
    t_user: Optional[int]                    # 2^base - s - 1
    t_floor: Optional[int]                   # floor(2^(2*base) / q) (exact reciprocal)
    t_match: Optional[bool]                  # t_user == t_floor (49/69 True; 20 False = corrected)
    delta: Optional[int]                     # t_floor - t_user (additive exactness correction)
    delta_naf: Optional[Tuple[Tuple[int, int], ...]]   # _decompose_signed_pow2(delta)
    mul_corrections: Optional[int]           # required correction steps (1 | 2); None if unsupported
    mul_unsupported_reason: Optional[str]


@dataclass(frozen=True)
class ClassShiftaddRecipe:
    """Per-K_group-class shiftadd sub-recipe for a known TWO-FAMILY batch.

    User K_group semantics: a known {52, 62} mixed set is shiftadd-LEGAL with
    per-class recipe selection (cls / PRIME_CLASS); each class sub-batch is a
    proven single-family batch with its own anchor and resolver verdict. The
    reduce resolver params here are the class sub-batch verdict (the values
    the mixed emitter consumes, assert-shimmed against its live resolution);
    mul corrections stay self-check-governed at emit (informational here,
    matching the single-family S3c choice).
    """
    k_class: int                              # 52 | 62 (S1b reduce class)
    b_anchor: int                             # 51 | 61 (class fold/mul base)
    aliases: Tuple[str, ...]
    reduce_resolver_emit_ok: bool
    reduce_resolver_params: Optional[Tuple[int, str]]
    reduce_resolver_reason: Optional[str]
    mul_ok: bool
    mul_corrections: Optional[int]


@dataclass(frozen=True)
class ReduceRecipe:
    """Per-set reduce-family recipe (PURE DESCRIPTIVE DATA; S3b plan Q3 as
    amended by the 2026-06-07 field-semantics decision).

    can_emit_shiftadd_reduce / can_emit_shiftadd_mul describe what the
    CURRENT emitters would do for THIS batch (batch rule included):
    can_emit_shiftadd_reduce is definitionally the batch
    reduce_resolver_emit_ok (live auto-resolver capability -- NEVER the
    fixed-point status); can_emit_shiftadd_mul is the mul emitter's
    self-check capability. A False value does NOT raise here -- it records
    that the existing emit path would refuse. S3b changes no control flow.
    """
    aliases: Tuple[str, ...]
    primes: Tuple[PrimeReduceFacts, ...]
    b_anchor: Optional[int]                  # min anchor_exp over the batch; None if any anchor unknown
    anchor_family: Optional[int]             # 51 | 61 when single-family; None otherwise
    anchor_k_group: int                      # shiftadd anchor-family group from _shiftadd_q_anchor:
    #                                          max q.bit_length(). NOT the S1b reduce-class K_group
    #                                          (max K_arith) and NOT final_K.
    k_spread: int                            # max - min q.bit_length()
    class_map: Tuple[int, ...]               # S1b classes PRESENT ascending; () if unavailable
    # reduce route, FIXED-POINT layer (recorded data, NOT capability):
    # batch verdict = AND over members' solo fixed-point verdicts (composable,
    # deterministic); None when the batch is shiftadd-illegal or a member is
    # not evaluable.
    reduce_default_self_check_ok: Optional[bool]
    reduce_default_self_check_params: Tuple[int, str]   # the fixed point, always recorded
    # reduce route, LIVE-RESOLVER layer (the practical batch capability):
    # the actual auto-resolution + self-check the current emit path would run
    # on THIS batch. Illegal batches record emit_ok=False with the reason.
    reduce_resolver_emit_ok: bool
    reduce_resolver_params: Optional[Tuple[int, str]]   # (resolved_fold_depth, resolved_correction)
    reduce_resolver_reason: Optional[str]    # None on emit-ok; refusal/illegality reason otherwise
    # route capabilities (C2; descriptive of the CURRENT emitters)
    can_emit_shiftadd_reduce: bool           # == reduce_resolver_emit_ok (live capability)
    can_emit_shiftadd_mul: bool              # batch rule + no mul_unsupported_reason member
    barrett_always: bool                     # Barrett serves every prime (bit_length <= 62); recipe-free
    # emit-derived batch params
    max_slots: int                           # max(2, max nonzero-exp alpha_naf terms) -- emitter rule
    slot_heuristic: str                      # "desc_exp" if max_slots == 3 else "asc_exp" -- emitter rule
    mul_corrections: Optional[int]           # batch max over supported primes (>= 1); None if none supported
    domain_tags: Tuple[str, ...]
    batch_legal_shiftadd: bool               # current batch rule + anchor availability
    fail_fast_reasons: Tuple[str, ...]       # why shiftadd batch construction would fail-fast today
    provenance: str                          # worst over primes: unsupported_custom dominates known_table
    # K_group-mixed two-family batches ONLY: per-class sub-recipes in cls
    # order (52-class first). None for every single-family set (existing
    # consumers unaffected).
    cls_recipes: Optional[Tuple[ClassShiftaddRecipe, ...]] = None


def _provenance_for_alias(alias: str) -> str:
    """Synthetic 'INT_<value>' aliases (resolve_primes_with_aliases upstream
    convention) are custom primes: recorded, never raised on in S3b (C5)."""
    return "unsupported_custom" if alias.startswith("INT_") else "known_table"


def _resolver_verdict(pairs: Sequence[Tuple[str, int]], k_env: int,
                      default_ok: Optional[bool]
                      ) -> Tuple[bool, Optional[Tuple[int, str]], Optional[str]]:
    """LIVE emit-path verdict for a shiftadd=reduce batch: auto fold/correction
    resolution exactly as the emitter runs it, then the math self-check at the
    RESOLVED point. Returns (emit_ok, resolved_params, reason). For a SOLO
    batch whose resolution lands on the fixed default point, default_ok is
    reused -- that call is LITERALLY identical (same primes, same fixed RNG
    seed/stream). Multi-prime batches always run the real batch call (the
    per-prime sample streams depend on batch composition)."""
    try:
        rfd, rcorr, _report = _shiftadd_resolve_fold_and_correction(
            list(pairs), k_env, "auto", "auto")
    except RuntimeError as exc:
        return False, None, "resolver refused: %s" % exc
    if (len(pairs) == 1 and default_ok is not None
            and (rfd, rcorr) == REDUCE_SELF_CHECK_DEFAULT_PARAMS):
        ok = default_ok
    else:
        ok, _log = _shiftadd_math_self_check(
            list(pairs), k_env, fold_depth=rfd, correction=rcorr)
    if ok:
        return True, (rfd, rcorr), None
    return False, (rfd, rcorr), (
        "math self-check FAILED at the resolver-chosen params "
        "(fold_depth=%d, correction=%s); the emit path would raise" % (rfd, rcorr))


@lru_cache(maxsize=None)
def _alias_facts(alias: str, q: int, k_env: int) -> PrimeReduceFacts:
    """Set-INDEPENDENT per-alias facts via the live helpers (rel_class=None).

    Cached: every fact is a pure function of (alias, q) -- k_env is the Data2
    container width only and gates nothing (kept in the key out of caution).
    Solo evaluation is exact for every legal batch's WORKING FORMS: known bit
    lengths map to family anchors {51, 61}, so a same-family batch's shared
    anchor equals each member's solo anchor (the batch equivalence assert
    re-proves this per construction). Solo resolver facts describe a
    single-prime case; the batch resolver verdict is computed per recipe.
    rel_class is SET-relative and injected by build_reduce_recipe via
    dataclasses.replace.
    """
    k_bits = q.bit_length()
    provenance = _provenance_for_alias(alias)
    try:
        k_arith = _k_arith_for(q)
    except RuntimeError:
        k_arith = None
    try:
        anchor_exp = _shiftadd_ell(q)
    except ValueError:
        anchor_exp = None
    if anchor_exp is None:
        no_anchor = ("bit_length %d outside {51, 52, 61, 62}: no shift-add "
                     "anchor (_shiftadd_ell rejects; Barrett route "
                     "unaffected)" % k_bits)
        return PrimeReduceFacts(
            alias=alias, q=q, k_bits=k_bits, k_arith=k_arith, rel_class=None,
            anchor_exp=None, provenance=provenance,
            alpha=None, alpha_naf=None, alpha_naf_weight=None,
            wfold1=None, wfold2=None, wfold3=None, f2_ok=None, f3_ok=None,
            reduce_default_self_check_ok=None,
            reduce_default_self_check_params=None,
            reduce_resolver_emit_ok=None, reduce_resolver_params=None,
            reduce_resolver_reason=no_anchor,
            reduce_unsupported_reason=no_anchor,
            s=None, s_naf=None, t_user=None, t_floor=None, t_match=None,
            delta=None, delta_naf=None, mul_corrections=None,
            mul_unsupported_reason=no_anchor)

    solo = [(alias, q)]
    # reduce route: q-derived envelopes at fold depth 2 (default gate) and 3.
    _w1, _w2, _w3f, _wf, ell2, _ok2, info2 = _shiftadd_envelope_widths(
        solo, k_env, fold_depth=2)
    if ell2 != anchor_exp:
        raise RuntimeError(
            "reduce_recipe transition guard: solo envelope anchor %r != "
            "_shiftadd_ell %r for %s" % (ell2, anchor_exp, alias))
    p2 = info2[0]
    info3 = _shiftadd_envelope_widths(solo, k_env, fold_depth=3)[6]
    p3 = info3[0]
    # FIXED-POINT layer (recorded data; never the capability verdict).
    fold_depth, correction = REDUCE_SELF_CHECK_DEFAULT_PARAMS
    default_ok, _sc_log = _shiftadd_math_self_check(
        solo, k_env, fold_depth=fold_depth, correction=correction)
    # LIVE-RESOLVER layer (solo practical capability).
    emit_ok, resolver_params, resolver_reason = _resolver_verdict(
        solo, k_env, default_ok)
    alpha = p2["alpha"]
    alpha_naf = tuple(p2["alpha_naf"])

    # mul route: base = the prime's anchor (the batch base for any legal batch).
    m_info, _bc, m_unsup, _m_log = _shiftadd_mul_self_check(solo, anchor_exp)
    m = m_info[0]
    delta = m["T_floor"] - m["T_user"]
    mul_reason = None
    if any(exp == 0 for _sign, exp in m["s_naf"]):
        # emit-time hard rule checked BEFORE the self-check in the mul emitter.
        mul_reason = ("s_naf contains a 2^0 term (collides with the constant "
                      "tail; the mul emitter raises)")
    elif m_unsup:
        _a, _q, reasons = m_unsup[0]
        mul_reason = "mul self-check unsupported: %s" % ", ".join(sorted(reasons))

    return PrimeReduceFacts(
        alias=alias, q=q, k_bits=k_bits, k_arith=k_arith, rel_class=None,
        anchor_exp=anchor_exp, provenance=provenance,
        alpha=alpha, alpha_naf=alpha_naf, alpha_naf_weight=len(alpha_naf),
        wfold1=p2["WFold1"], wfold2=p2["WFold2"], wfold3=p3["WFold3"],
        f2_ok=p2["F2_OK"], f3_ok=p3["F3_OK"],
        reduce_default_self_check_ok=default_ok,
        reduce_default_self_check_params=(fold_depth, correction),
        reduce_resolver_emit_ok=emit_ok,
        reduce_resolver_params=resolver_params,
        reduce_resolver_reason=resolver_reason,
        reduce_unsupported_reason=None if emit_ok else resolver_reason,
        s=m["s"], s_naf=tuple(m["s_naf"]),
        t_user=m["T_user"], t_floor=m["T_floor"], t_match=m["T_match"],
        delta=delta, delta_naf=tuple(_decompose_signed_pow2(delta)),
        mul_corrections=m["required_corrections"],
        mul_unsupported_reason=mul_reason)


def _validate_aliases_q(aliases_q: Sequence[Tuple[str, int]]) -> Tuple[Tuple[str, int], ...]:
    """Structural input validation (caller-bug guard, NOT a prime-value gate)."""
    pairs = tuple(aliases_q)
    if not pairs:
        raise ValueError("build_reduce_recipe: empty prime set.")
    for entry in pairs:
        if (not isinstance(entry, tuple)) or len(entry) != 2:
            raise ValueError(
                "build_reduce_recipe: expected (alias, q) pairs, got %r." % (entry,))
        alias, q = entry
        if not isinstance(alias, str):
            raise ValueError("build_reduce_recipe: alias must be str, got %r." % (alias,))
        if isinstance(q, bool) or not isinstance(q, int) or q <= 0:
            raise ValueError("build_reduce_recipe: q must be a positive int, got %r." % (q,))
    return pairs


def build_reduce_recipe(aliases_q: Sequence[Tuple[str, int]], *,
                        k_env: Optional[int] = None,
                        classify_info: Optional[ReduceClassInfo] = None) -> ReduceRecipe:
    """TRANSITION known-table constructor (S3b plan Q5). Pure; no file writes.

    Derives every fact from q via the EXISTING helpers (C4), stores
    anchor_exp independently (C3), and asserts batch equivalence to the
    current _shiftadd_q_anchor heuristic: a legal batch must reproduce
    (B, K_group, K_spread) exactly; a k_spread>=3 batch must match the
    heuristic's raise (reason text compared), recorded as
    batch_legal_shiftadd=False (C1). When some anchor is unknown (custom
    bit lengths outside {51, 52, 61, 62}) the heuristic comparison is
    SKIPPED and the recipe records the conservative _shiftadd_ell view
    (the recipe subsumes both helpers -- S3a F2).

    Reduce-route status is recorded at BOTH layers (module docstring):
    the fixed default point (data) and the live batch resolver verdict
    (capability = can_emit_shiftadd_reduce). INT/custom primes:
    provenance "unsupported_custom" RECORDED, no raise (C5; the S3d
    shiftadd fail-fast is future work). Raises only on structurally
    invalid input, a populated override hook, or a violated transition
    guard (internal-consistency bug, not prime input).

    k_env is the Data2 container width only (inert for every stored fact);
    None derives it like the live flow (classification final_k; max
    bit_length when classification is unavailable). classify_info, when
    provided, must match aliases_q pairwise (OD-s3b-4 single source).
    """
    pairs = _validate_aliases_q(aliases_q)
    if _RECIPE_OVERRIDES:
        raise NotImplementedError(
            "reduce_recipe: the curated override hook is RESERVED and must "
            "stay empty in S3b (no consumption is implemented).")

    classify_reason = None
    if classify_info is None:
        try:
            classify_info = classify_reduce_set(pairs, None)
        except RuntimeError as exc:   # > 62-bit prime: record, never raise (C5)
            classify_reason = str(exc)
    else:
        got = tuple((a, q) for a, q, _k, _r in classify_info.per_prime)
        if got != pairs:
            raise ValueError(
                "build_reduce_recipe: classify_info does not match aliases_q "
                "(%r vs %r)." % (got, pairs))
    class_map = classify_info.class_map if classify_info is not None else ()
    rel_by_index = ([r for _a, _q, _k, r in classify_info.per_prime]
                    if classify_info is not None else [None] * len(pairs))
    if k_env is None:
        k_env = (classify_info.final_k if classify_info is not None
                 else max(q.bit_length() for _a, q in pairs))

    primes = tuple(replace(_alias_facts(alias, q, k_env), rel_class=rel)
                   for (alias, q), rel in zip(pairs, rel_by_index))

    # batch facts (ANCHOR-side semantics; see the module terminology guard).
    k_bits_list = [f.k_bits for f in primes]
    anchor_k_group = max(k_bits_list)
    k_spread = anchor_k_group - min(k_bits_list)
    anchors = [f.anchor_exp for f in primes]
    all_anchors_known = all(a is not None for a in anchors)
    b_anchor = min(anchors) if all_anchors_known else None
    anchor_family = (anchors[0] if all_anchors_known and len(set(anchors)) == 1
                     else None)

    # K_group semantics: a batch spanning BOTH known anchor families
    # ({51, 61} == K_groups {52, 62}) is shiftadd-LEGAL with per-class recipe
    # selection (cls / PRIME_CLASS) -- the per-class mixed emitters serve it.
    # The k_spread>=3 illegality remains ONLY for wide sets that are NOT a
    # known two-family batch (custom bit lengths).
    known_two_family = (_shiftadd_known_anchor_families(pairs) == [51, 61])

    fail_fast_reasons = []
    if k_spread >= 3 and not known_two_family:
        fail_fast_reasons.append(_MIXED_BATCH_REASON % k_spread)
    for f in primes:
        if f.anchor_exp is None:
            fail_fast_reasons.append(
                "%s: bit_length %d outside {51, 52, 61, 62}" % (f.alias, f.k_bits))
    if classify_reason is not None:
        fail_fast_reasons.append("classification unavailable: %s" % classify_reason)
    batch_legal_shiftadd = not fail_fast_reasons

    # transition equivalence guard vs the current batch heuristic (plan Q5.4).
    if known_two_family:
        # per-class equivalence: each class sub-batch is single-family and
        # must reproduce the legacy heuristic anchor exactly.
        for _k_cls, _b_cls, _sub in _shiftadd_class_sub_batches(pairs):
            heur = _shiftadd_q_anchor(_sub)
            if heur[0] != _b_cls:
                raise RuntimeError(
                    "reduce_recipe transition guard: class %d sub-batch "
                    "anchor %r != %d for %r"
                    % (_k_cls, heur, _b_cls, [a for a, _q in _sub]))
    elif k_spread >= 3:
        try:
            _shiftadd_q_anchor(pairs)
        except ValueError as exc:
            if ("K_spread=%d >= 3" % k_spread) not in str(exc):
                raise RuntimeError(
                    "reduce_recipe transition guard: unexpected "
                    "_shiftadd_q_anchor reason for k_spread=%d: %s"
                    % (k_spread, exc))
        else:
            raise RuntimeError(
                "reduce_recipe transition guard: _shiftadd_q_anchor accepted "
                "a K_spread=%d >= 3 batch it must reject." % k_spread)
    elif batch_legal_shiftadd:
        heur = _shiftadd_q_anchor(pairs)
        if heur != (b_anchor, anchor_k_group, k_spread):
            raise RuntimeError(
                "reduce_recipe transition guard: (b_anchor, anchor_k_group, "
                "k_spread)=%r != _shiftadd_q_anchor %r for %r"
                % ((b_anchor, anchor_k_group, k_spread), heur,
                   [a for a, _q in pairs]))
    # else: k_spread < 3 with unknown anchors -- heuristic comparison skipped;
    # the recipe records the conservative _shiftadd_ell view (S3a section D gap).

    # reduce route, FIXED-POINT batch layer: AND over solo verdicts
    # (composable, deterministic); None when illegal or not fully evaluable.
    if batch_legal_shiftadd and all(
            f.reduce_default_self_check_ok is not None for f in primes):
        batch_default_ok = all(f.reduce_default_self_check_ok for f in primes)
    else:
        batch_default_ok = None

    # K_group-mixed two-family batches: per-class sub-recipes (cls order;
    # 52-class first). A 1-member class reuses the cached solo facts (the
    # solo resolver IS the class batch); multi-member classes run the real
    # class batch call.
    cls_recipes = None
    if known_two_family:
        cls_entries = []
        for _k_cls, _b_cls, _sub in _shiftadd_class_sub_batches(pairs):
            sub_set = set(_sub)
            sub_facts = [f for (a, q), f in zip(pairs, primes)
                         if (a, q) in sub_set]
            if len(_sub) == 1:
                f0 = sub_facts[0]
                c_ok = bool(f0.reduce_resolver_emit_ok)
                c_params = f0.reduce_resolver_params
                c_reason = f0.reduce_resolver_reason
            else:
                if all(f.reduce_default_self_check_ok is not None
                       for f in sub_facts):
                    sub_default_ok = all(
                        f.reduce_default_self_check_ok for f in sub_facts)
                else:
                    sub_default_ok = None
                c_ok, c_params, c_reason = _resolver_verdict(
                    _sub, k_env, sub_default_ok)
            sub_corr = [f.mul_corrections for f in sub_facts
                        if f.mul_corrections is not None]
            cls_entries.append(ClassShiftaddRecipe(
                k_class=_k_cls, b_anchor=_b_cls,
                aliases=tuple(a for a, _q in _sub),
                reduce_resolver_emit_ok=c_ok,
                reduce_resolver_params=c_params,
                reduce_resolver_reason=c_reason,
                mul_ok=all(f.mul_unsupported_reason is None
                           for f in sub_facts),
                mul_corrections=max(sub_corr) if sub_corr else None))
        cls_recipes = tuple(cls_entries)

    # reduce route, LIVE-RESOLVER batch layer: the verdict the current emit
    # path would produce for THIS batch (the practical capability). For a
    # known two-family batch the emit path is the per-class mixed emitter, so
    # the batch verdict is the AND over class verdicts and the resolved
    # params live per class (reduce_resolver_params stays None).
    if known_two_family:
        resolver_emit_ok = all(ce.reduce_resolver_emit_ok for ce in cls_recipes)
        resolver_params = None
        resolver_reason = (None if resolver_emit_ok else "; ".join(
            "cls %d (%d-class): %s" % (i, ce.k_class, ce.reduce_resolver_reason)
            for i, ce in enumerate(cls_recipes)
            if not ce.reduce_resolver_emit_ok))
    elif batch_legal_shiftadd:
        resolver_emit_ok, resolver_params, resolver_reason = _resolver_verdict(
            pairs, k_env, batch_default_ok)
    else:
        resolver_emit_ok, resolver_params = False, None
        resolver_reason = "; ".join(fail_fast_reasons)

    can_emit_shiftadd_reduce = resolver_emit_ok
    can_emit_shiftadd_mul = batch_legal_shiftadd and all(
        f.mul_unsupported_reason is None for f in primes)
    barrett_always = all(f.k_bits <= 62 for f in primes)

    # emit-derived batch params (exact emitter rules, recorded not enforced):
    # required slots = max nonzero-exponent alpha_naf terms; >= 2.
    nz_counts = [sum(1 for _s, e in f.alpha_naf if e != 0)
                 for f in primes if f.alpha_naf is not None]
    max_slots = max(2, max(nz_counts) if nz_counts else 2)
    slot_heuristic = "desc_exp" if max_slots == 3 else "asc_exp"
    supported_corr = [f.mul_corrections for f in primes
                      if f.mul_corrections is not None]
    mul_corrections = max(supported_corr) if supported_corr else None

    provenance = ("unsupported_custom"
                  if any(f.provenance == "unsupported_custom" for f in primes)
                  else "known_table")
    return ReduceRecipe(
        aliases=tuple(a for a, _q in pairs), primes=primes,
        b_anchor=b_anchor, anchor_family=anchor_family,
        anchor_k_group=anchor_k_group, k_spread=k_spread, class_map=class_map,
        reduce_default_self_check_ok=batch_default_ok,
        reduce_default_self_check_params=REDUCE_SELF_CHECK_DEFAULT_PARAMS,
        reduce_resolver_emit_ok=resolver_emit_ok,
        reduce_resolver_params=resolver_params,
        reduce_resolver_reason=resolver_reason,
        can_emit_shiftadd_reduce=can_emit_shiftadd_reduce,
        can_emit_shiftadd_mul=can_emit_shiftadd_mul,
        barrett_always=barrett_always,
        max_slots=max_slots, slot_heuristic=slot_heuristic,
        mul_corrections=mul_corrections,
        domain_tags=DOMAIN_TAGS,
        batch_legal_shiftadd=batch_legal_shiftadd,
        fail_fast_reasons=tuple(fail_fast_reasons),
        provenance=provenance,
        cls_recipes=cls_recipes)


__all__ = [
    "PrimeReduceFacts",
    "ClassShiftaddRecipe",
    "ReduceRecipe",
    "build_reduce_recipe",
    "REDUCE_SELF_CHECK_DEFAULT_PARAMS",
    "SHIFTADD_ANCHOR_BIT_LENGTHS",
    "DOMAIN_TAGS",
]

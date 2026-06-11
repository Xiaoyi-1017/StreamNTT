"""ICBU (iterative butterfly unit) generation & control logic for the segmented StreamNTT generator.

Owns Layer-1 (atomic BankSpec) + Layer-2 (_alias_icbu_recipe_schedule) recipe-schedule parsing /
normalization / validation / resolution feeding stage_generator.gen_bf_unit_storage (Phase 3.3.4).

First slice (SP-A..SP-D): parse + validate the schedule grammar and partition tokens; resolve a
per-stage-group (storage, partition) schedule into per-bank impl tuples; R-UNIFORM only (one recipe
on the single shared bf_unit child). Partition tokens are RECOGNIZED + legality-checked but NOT
emitted as #pragma HLS array_partition (only `none` is realizable; non-none -> upstream fail-fast).

Deferred (NOT here): array_partition emission, R-hetero realization, Layer-3 grouping planner,
`complete` partition, `ge` range syntax, explicit per-bank layouts.

Stdlib-only; MUST NOT import generate_code/stage_generator (avoids circular imports)."""
from __future__ import annotations

import re
from typing import List, NamedTuple, Optional, Tuple

__all__ = [
    "BankSpec",
    "IMPL_CODE_TO_NAME",
    "PARTITION_TOKENS",
    "parse_storage_count_form",
    "parse_partition_token",
    "check_bank_legality",
    "ResolvedSchedule",
    "parse_icbu_recipe_schedule",
]

# ---------------------------------------------------------------------------
# Layer 1: atomic BankSpec (per mem bank)
# ---------------------------------------------------------------------------
# impl single-letter code -> canonical lowercase impl string (matches bind_storage impl= token and
# stage_generator._BF_UNIT_STORAGE_RECIPES value vocabulary).
IMPL_CODE_TO_NAME = {"L": "lutram", "B": "bram", "U": "uram"}

# Canonical first-slice fill order for the count form L<l>B<b>U<u> -> banks mem0..3: all U(ram)
# first, then B(ram), then L(utram). DP1: first-slice deterministic default, NOT a permanent
# physical-optimal rule; explicit per-bank layouts (UBUB/...) are deferred.
_FILL_ORDER = ("U", "B", "L")
_COUNT_FORM_RE = re.compile(r"^L(\d+)B(\d+)U(\d+)$")


def parse_storage_count_form(token: str) -> Tuple[str, str, str, str]:
    """Parse a count-form storage token 'L<l>B<b>U<u>' into a per-bank impl tuple (mem0..3).

    Banks fill in canonical order U -> B -> L (DP1). Counts must sum to exactly 4 (C4). Returns a
    4-tuple of impl strings ('uram'|'bram'|'lutram'). Raises ValueError on malformed token / sum!=4.
    """
    m = _COUNT_FORM_RE.match(token)
    if not m:
        raise ValueError(
            "malformed storage count form %r (expected 'L<l>B<b>U<u>', e.g. 'L0B2U2')" % token
        )
    l, b, u = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if l + b + u != 4:
        raise ValueError(
            "storage counts in %r must sum to 4 (got L%d B%d U%d = %d) [C4]" % (token, l, b, u, l + b + u)
        )
    counts = {"U": u, "B": b, "L": l}
    banks: List[str] = []
    for code in _FILL_ORDER:
        banks.extend([IMPL_CODE_TO_NAME[code]] * counts[code])
    return tuple(banks)  # len == 4


# Partition token -> (part_type, factor). First-slice whitelist (DP3). `complete` (DP4) NOT included.
PARTITION_TOKENS = {
    "none":    ("none", 1),
    "block2":  ("block", 2),
    "block4":  ("block", 4),
    "cyclic2": ("cyclic", 2),
    "cyclic4": ("cyclic", 4),
}


def parse_partition_token(token: str) -> Tuple[str, int]:
    """Parse a partition token -> (part_type, factor). Whitelist = none/block2/block4/cyclic2/cyclic4
    (DP3). `complete` (DP4) and abbreviations (p2/c2) are rejected (deferred / not readable)."""
    if token not in PARTITION_TOKENS:
        raise ValueError(
            "unknown or deferred partition token %r (allowed: %s; 'complete' is deferred)"
            % (token, ", ".join(PARTITION_TOKENS))
        )
    return PARTITION_TOKENS[token]


class BankSpec(NamedTuple):
    impl: str         # 'lutram' | 'bram' | 'uram'
    part_type: str    # 'none' | 'block' | 'cyclic'  ('complete' deferred)
    part_factor: int  # 1 for none; 2 or 4 for block/cyclic


def check_bank_legality(spec: BankSpec, depth_half: int) -> None:
    """Validate one BankSpec against the bank array length depth_half (= DEPTH/2). Fail-fast (C5).
    First slice: factor in {2,4} and must divide depth_half. Exact per-impl geometry bounds
    (URAM 4Kx72 / BRAM / LUTRAM) confirmed later via csynth (T4) + pre-merge vertex (T5)."""
    if spec.part_type == "none":
        return
    if spec.part_factor not in (2, 4):
        raise ValueError("partition factor must be 2 or 4 (got %d) [C5]" % spec.part_factor)
    if depth_half % spec.part_factor != 0:
        raise ValueError("partition factor %d must divide DEPTH/2=%d [C5]" % (spec.part_factor, depth_half))


# ---------------------------------------------------------------------------
# Layer 3 (FIXED today): generated stage-order groups
# ---------------------------------------------------------------------------
def _lstage_groups(num_l_stage: int, precompute_boundary: int = 4) -> Tuple[Tuple[int, int], ...]:
    """Precompute/OTF grouping as inclusive (lo,hi) ranges: singletons s0..s(B-1) + one s_geB =
    [B, num_l_stage) when num_l_stage>B. Phase 3.4 D1: DERIVED from the stage_plan.StagePlan SINGLE
    SOURCE OF TRUTH (PRECOMP-boundary constructor); the Layer-2 schedule attaches recipes to THESE
    groups and may not split them (C6). The full configurable Layer-3 planner remains DEFERRED."""
    from code_generator.stage_plan import plan_from_precomp_boundary
    return plan_from_precomp_boundary(num_l_stage, precompute_boundary).lstage_groups()


# ---------------------------------------------------------------------------
# Layer 2: _alias_icbu_recipe_schedule (parse / validate / normalize / resolve)
# ---------------------------------------------------------------------------
class ResolvedSchedule(NamedTuple):
    normalized: str                                        # canonical normalized schedule string
    groups: Tuple[Tuple[int, int], ...]                    # Layer-3 group ranges (lo,hi)
    per_group_impls: Tuple[Tuple[str, str, str, str], ...] # 4 impl strings per group
    per_group_partition: Tuple[str, ...]                   # partition TOKEN per group (e.g. 'none')
    uniform: bool
    uniform_impls: Optional[Tuple[str, str, str, str]]
    uniform_partition: Optional[str]


_ENTRY_RE = re.compile(r"^\s*(\d+)(?:-(\d+))?\s*:\s*([A-Za-z0-9]+)\s*:\s*([A-Za-z0-9]+)\s*$")
_DEFAULT_STORAGE = "L0B2U2"   # == d0c3a16 2U2B
_DEFAULT_PARTITION = "none"


def _impls_to_count_form(impls: Tuple[str, str, str, str]) -> str:
    """Re-emit a per-bank impl tuple as the canonical count form L<l>B<b>U<u>."""
    u = sum(1 for i in impls if i == "uram")
    b = sum(1 for i in impls if i == "bram")
    l = sum(1 for i in impls if i == "lutram")
    return "L%dB%dU%d" % (l, b, u)


def _normalize(groups, per_group_impls, per_group_partition) -> str:
    """Canonical schedule string: one token per maximal run of adjacent groups with an identical
    recipe; ranges ascending (groups pre-sorted)."""
    parts = []
    i, n = 0, len(groups)
    while i < n:
        j = i
        rec = (per_group_impls[i], per_group_partition[i])
        while j + 1 < n and (per_group_impls[j + 1], per_group_partition[j + 1]) == rec:
            j += 1
        lo, hi = groups[i][0], groups[j][1]
        rng = "%d" % lo if lo == hi else "%d-%d" % (lo, hi)
        parts.append("%s:%s:%s" % (rng, _impls_to_count_form(rec[0]), rec[1]))
        i = j + 1
    return ";".join(parts)


def parse_icbu_recipe_schedule(env: Optional[str], num_l_stage: int, depth: int,
                               precompute_boundary: int = 4,
                               groups: Optional[Tuple[Tuple[int, int], ...]] = None) -> ResolvedSchedule:
    """Parse + validate (C1-C6) + normalize + resolve the Layer-2 ICBU recipe schedule.

    env: raw STREAMNTT_ICBU_RECIPE_SCHEDULE (None/'' -> default uniform 2U2B/none).
    num_l_stage: struct['NUM_L_stage'] (index range + Layer-3 group set).
    depth: DEPTH (= N/(2*BU)); bank array length = depth//2 (C5 legality).
    precompute_boundary: hybrid boundary B (back-compat fallback when groups is None).
    groups: the ACTUAL StagePlan-derived (lo,hi) group tuples (Phase 3.4 D4) — the single source of
    truth; when provided it overrides the boundary fallback (no grouping logic lives here).
    Returns a VALID ResolvedSchedule and REPORTS uniformity (does NOT enforce R-uniform; the caller
    enforces realization limits). Raises ValueError on grammar / C1-C6 / legality violations."""
    if groups is None:
        groups = _lstage_groups(num_l_stage, precompute_boundary)
    depth_half = depth // 2
    group_los = {lo for (lo, _) in groups}
    group_his = {hi for (_, hi) in groups}

    if env is None or env.strip() == "":
        entries = [(0, num_l_stage - 1, _DEFAULT_STORAGE, _DEFAULT_PARTITION)]
    else:
        entries = []
        for raw in env.split(";"):
            if raw.strip() == "":
                continue
            m = _ENTRY_RE.match(raw)
            if not m:
                raise ValueError("malformed schedule entry %r (expected 'range:storage:partition')" % raw)
            lo = int(m.group(1))
            hi = int(m.group(2)) if m.group(2) is not None else lo
            if hi < lo:
                raise ValueError("schedule range %d-%d has hi<lo" % (lo, hi))
            entries.append((lo, hi, m.group(3), m.group(4)))

    # C1 indices in range
    for (lo, hi, _, _) in entries:
        if lo < 0 or hi >= num_l_stage:
            raise ValueError("schedule range %d-%d out of [0,%d) [C1]" % (lo, hi, num_l_stage))
    # C2 no overlap
    covered = set()
    for (lo, hi, _, _) in entries:
        rng = set(range(lo, hi + 1))
        if rng & covered:
            raise ValueError("schedule ranges overlap at %s [C2]" % sorted(rng & covered))
        covered |= rng
    # C6 each entry range = exact union of whole groups (lo is a group lo AND hi is a group hi)
    for (lo, hi, _, _) in entries:
        if lo not in group_los or hi not in group_his:
            raise ValueError(
                "schedule range %d-%d does not align to generated stage groups %s; a range must be an "
                "exact union of whole groups (it may not split a group) [C6]" % (lo, hi, list(groups))
            )
    # C4 storage sum + partition token + C5 legality -> per-entry recipe
    entry_recipe = {}
    for (lo, hi, storage_tok, part_tok) in entries:
        impls = parse_storage_count_form(storage_tok)        # C4
        part_type, factor = parse_partition_token(part_tok)  # DP3/DP4 whitelist
        for impl in impls:                                   # C5 per bank
            check_bank_legality(BankSpec(impl, part_type, factor), depth_half)
        entry_recipe[(lo, hi)] = (impls, part_tok)
    # C3 resolve per group (each group wholly inside one entry by C6, or uncovered -> default)
    default_impls = parse_storage_count_form(_DEFAULT_STORAGE)
    per_group_impls, per_group_partition = [], []
    for (glo, ghi) in groups:
        hit = next(((lo, hi) for (lo, hi, _, _) in entries if lo <= glo and ghi <= hi), None)
        if hit is None:
            per_group_impls.append(default_impls)
            per_group_partition.append(_DEFAULT_PARTITION)
        else:
            impls, part_tok = entry_recipe[hit]
            per_group_impls.append(impls)
            per_group_partition.append(part_tok)
    first = (per_group_impls[0], per_group_partition[0])
    uniform = all((per_group_impls[i], per_group_partition[i]) == first for i in range(len(groups)))
    return ResolvedSchedule(
        normalized=_normalize(groups, per_group_impls, per_group_partition),
        groups=groups,
        per_group_impls=tuple(per_group_impls),
        per_group_partition=tuple(per_group_partition),
        uniform=uniform,
        uniform_impls=first[0] if uniform else None,
        uniform_partition=first[1] if uniform else None,
    )

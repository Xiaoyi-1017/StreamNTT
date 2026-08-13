"""Hybrid table-driven / on-the-fly L-stage planner.

Single source of truth for the L-stage precompute/OTF partition (StageGroup/StagePlan): table view,
emission view, and the ICBU storage-schedule group view all derive from the same StagePlan.
Supported plans use a table-driven prefix with boundary 4 or 5 followed by at most
one on-the-fly suffix. P=32 evidence classification uses the same plan.

PRECOMP=N means s0..s(N-1) are table-driven singletons and the on-the-fly suffix starts at sN.

Stage-order names ONLY: sN / sAtoB / s_geN (no precomp/otf/recipe/storage labels in generated names).

Stdlib-only; imports nothing from the project (icbu_generator imports THIS module, not vice versa)."""
from __future__ import annotations

from typing import NamedTuple, Tuple

__all__ = [
    "PRECOMP_DEFAULT", "StageGroup", "StagePlan", "plan_from_precomp_boundary",
    "plan_from_schedule",
    "P32Evidence", "classify_p32_evidence",
]

_SCHED_ENV = "STREAMNTT_LSTAGE_PRECOMPUTE_SCHEDULE"   # error messages must name the env var

PRECOMP_DEFAULT = 4
_KINDS = ("precomp", "otf")


class StageGroup(NamedTuple):
    lo: int
    hi: int            # inclusive
    kind: str          # 'precomp' | 'otf'


class StagePlan(NamedTuple):
    num_l_stage: int
    groups: Tuple[StageGroup, ...]

    # ---- views -----------------------------------------------------------
    def lstage_groups(self) -> Tuple[Tuple[int, int], ...]:
        """Return (lo, hi) tuples for the ICBU storage schedule."""
        return tuple((g.lo, g.hi) for g in self.groups)

    def group_name(self, g: StageGroup) -> str:
        """Stage-order name: s_geN for an OTF suffix group (even when it holds a single stage),
        sN for other singletons,
        sAtoB for interior ranges."""
        if g.kind == "otf" and g.hi == self.num_l_stage - 1:
            return "s_ge%d" % g.lo
        if g.lo == g.hi:
            return "s%d" % g.lo
        return "s%dto%d" % (g.lo, g.hi)

    def precomp_stages(self) -> Tuple[int, ...]:
        return tuple(s for g in self.groups if g.kind == "precomp" for s in range(g.lo, g.hi + 1))

    def precomp_boundary(self) -> int:
        """First OTF stage index (== precomputed-stage count under the prefix invariant V3);
        num_l_stage when the plan has no OTF group."""
        for g in self.groups:
            if g.kind == "otf":
                return g.lo
        return self.num_l_stage

    def otf_groups(self) -> Tuple[StageGroup, ...]:
        return tuple(g for g in self.groups if g.kind == "otf")

    def precomp_flat_entries(self) -> int:
        """Flat tw_l_base_table entry count = sum(2^s) over precomputed stages (offset (1<<s)-1 layout)."""
        return sum(1 << s for s in self.precomp_stages())

    # ---- invariants (V1-V5) ----------------------------------------------
    def validate(self) -> "StagePlan":
        n = self.num_l_stage
        if n <= 0:
            raise ValueError("num_l_stage must be positive (got %d)" % n)
        if not self.groups:
            raise ValueError("StagePlan has no groups")
        cursor = 0
        for g in self.groups:
            if g.kind not in _KINDS:
                raise ValueError("StageGroup kind must be precomp|otf (got %r)" % (g.kind,))
            if g.lo > g.hi:
                raise ValueError("StageGroup %d-%d has lo>hi" % (g.lo, g.hi))
            if g.lo != cursor:
                raise ValueError(
                    "StagePlan groups must be ascending, disjoint and covering: expected lo=%d, got %d [V1/V2]"
                    % (cursor, g.lo))
            cursor = g.hi + 1
        if cursor != n:
            raise ValueError("StagePlan does not cover [0,%d): ends at %d [V1]" % (n, cursor))
        # Table-driven stages must form a prefix.
        seen_otf = False
        for g in self.groups:
            if g.kind == "otf":
                seen_otf = True
            elif seen_otf:
                raise ValueError(
                    "precomp group s%d..s%d after an otf group: only prefix-precomp plans are supported "
                    "by the active planner [V3]" % (g.lo, g.hi))
        # V5: names derivable (exercise the derivation; raises only on internal inconsistency).
        for g in self.groups:
            self.group_name(g)
        return self


def plan_from_precomp_boundary(num_l_stage: int, precompute_boundary: int = PRECOMP_DEFAULT) -> StagePlan:
    """Construct a supported table-driven-prefix plan.

    PRECOMP=4 yields s0,s1,s2,s3,s_ge4; PRECOMP=5 yields s0..s4,s_ge5.
    Other values are not supported by the active emitters.
    """
    if num_l_stage <= 0:
        raise ValueError("num_l_stage must be positive (got %d)" % num_l_stage)
    if precompute_boundary not in (4, 5):
        raise ValueError(
            "precompute_boundary must be 4 or 5 (got %r)" % (precompute_boundary,))
    b = precompute_boundary
    groups = [StageGroup(s, s, "precomp") for s in range(min(b, num_l_stage))]
    if num_l_stage > b:
        groups.append(StageGroup(b, num_l_stage - 1, "otf"))
    return StagePlan(num_l_stage, tuple(groups)).validate()


# ---------------------------------------------------------------------------
# Stage-schedule constructor (boundary-equivalent plans only)
# ---------------------------------------------------------------------------
# Grammar: "kind:lo-hi;kind:lo-hi;..." ; kinds {precomp, otf} ; "otf:N-" = open suffix to the last stage.
# DISTINCT from the ICBU recipe schedule grammar ("range:storage:partition" on
# STREAMNTT_ICBU_RECIPE_SCHEDULE) — never mixed; all errors here name STREAMNTT_LSTAGE_PRECOMPUTE_SCHEDULE.
# Normalization: precomp ranges EXPAND to singleton groups, so equivalence with the PRECOMP-boundary
# constructors is checked on the normalized StagePlan (NOT raw strings): the singleton-split spelling
# "precomp:0-0;precomp:1-1;precomp:2-2;precomp:3-3;otf:4-" equals PRECOMP=4.

def plan_from_schedule(env, num_l_stage: int):
    """Parse the stage-schedule environment value into a StagePlan.

    Returns None when env is None/empty/whitespace (= unset; the caller falls back to the boundary path).
    Only plans structurally equal to plan_from_precomp_boundary(num_l_stage, 4 or 5)
    are accepted; other parsed schedules fail with a named reason.
    """
    if env is None or env.strip() == "":
        return None
    groups = []
    for raw in env.split(";"):
        tok = raw.strip()
        if tok == "":
            continue
        if ":" not in tok:
            raise ValueError("%s: malformed entry %r (expected 'kind:lo-hi' or 'otf:N-')" % (_SCHED_ENV, raw))
        kind, rng = tok.split(":", 1)
        kind = kind.strip()
        if kind not in ("precomp", "otf"):
            raise ValueError("%s: illegal kind %r (use precomp|otf)" % (_SCHED_ENV, kind))
        rng = rng.strip()
        try:
            if rng.endswith("-"):                      # open suffix: "N-"
                lo = int(rng[:-1])
                hi = num_l_stage - 1
            elif "-" in rng:
                lo_s, hi_s = rng.split("-", 1)
                lo, hi = int(lo_s), int(hi_s)
            else:
                lo = hi = int(rng)
        except ValueError:
            raise ValueError("%s: malformed range %r (numeric 'lo-hi', 'N' or 'N-')" % (_SCHED_ENV, rng))
        if lo < 0 or hi >= num_l_stage or lo > hi:
            raise ValueError("%s: range %d-%d out of [0,%d) or inverted" % (_SCHED_ENV, lo, hi, num_l_stage))
        if kind == "precomp":                          # normalize: expand to singleton groups
            groups.extend(StageGroup(s, s, "precomp") for s in range(lo, hi + 1))
        else:
            groups.append(StageGroup(lo, hi, "otf"))
    groups.sort(key=lambda g: g.lo)
    if sum(1 for g in groups if g.kind == "otf") > 1:
        raise ValueError("%s: multiple OTF ranges are unsupported; use one suffix OTF group"
                         % _SCHED_ENV)
    try:
        plan = StagePlan(num_l_stage, tuple(groups)).validate()
    except ValueError as e:
        raise ValueError("%s: %s" % (_SCHED_ENV, e))
    # Boundary-equivalence gate uses structural plan equality, not raw strings.
    b = plan.precomp_boundary()
    if b in (4, 5) and plan == plan_from_precomp_boundary(num_l_stage, b):
        return plan
    if b > 5 and plan.otf_groups() and plan == StagePlan(
            num_l_stage,
            tuple([StageGroup(s, s, "precomp") for s in range(b)]
                  + [StageGroup(b, num_l_stage - 1, "otf")])).validate():
        raise ValueError(
            "%s: PRECOMP>5 / ge6 is unsupported by the active emitters (got boundary %d)"
            % (_SCHED_ENV, b))
    raise ValueError(
        "%s: only boundary-equivalent schedules are supported "
        "(precomp:0-3;otf:4- == PRECOMP=4 or precomp:0-4;otf:5- == PRECOMP=5); got groups %s"
        % (_SCHED_ENV, [(g.lo, g.hi, g.kind) for g in plan.groups]))


# ---------------------------------------------------------------------------
# P=32 structural classification
# ---------------------------------------------------------------------------
class P32Evidence(NamedTuple):
    classification: str            # 'illegal-period-lt-P' | 'no-otf' | 'degenerate' | 'non-degenerate'
    period_blocks: Tuple[int, ...]  # per OTF stage, ascending stage order
    detail: str


def classify_p32_evidence(plan: StagePlan, P: int) -> P32Evidence:
    """Classify a plan for P (tfg_seg_len) recurrence/timing-evidence purposes.

    illegal-period-lt-P : some OTF stage has period 2^s < P -> generation must fail-fast (not classification's
                          job to allow it; reported for planning).
    no-otf              : no OTF stages (nothing to segment).
    degenerate          : every OTF stage has period_blocks == 1 (for example, single-s5 at P=32),
                          so it is useful only as a functional smoke case.
    non-degenerate      : >= 2 OTF stages and max period_blocks >= 2 -> recurrence LIVE; valid
                          recurrence/timing-evidence candidate (e.g. ge5={s5,s6}: period_blocks {1,2}).
    """
    otf_stages = [s for g in plan.otf_groups() for s in range(g.lo, g.hi + 1)]
    if not otf_stages:
        return P32Evidence("no-otf", (), "plan has no OTF stages")
    if any((1 << s) < P for s in otf_stages):
        bad = [s for s in otf_stages if (1 << s) < P]
        return P32Evidence("illegal-period-lt-P", (),
                           "OTF stage(s) %s have period < P=%d (no period<P handling planned)" % (bad, P))
    pbs = tuple((1 << s) // P for s in otf_stages)
    if len(otf_stages) >= 2 and max(pbs) >= 2:
        return P32Evidence("non-degenerate", pbs,
                           "recurrence LIVE (period_blocks %s); valid recurrence/timing-evidence candidate"
                           % (list(pbs),))
    if all(pb == 1 for pb in pbs):
        return P32Evidence("degenerate", pbs,
                           "all OTF period_blocks == 1 (stages %s): degenerate FUNCTIONAL SMOKE ONLY; "
                           "not recurrence/timing evidence" % (otf_stages,))
    return P32Evidence("degenerate", pbs,
                       "single OTF stage (period_blocks %s): recurrence live locally but <2 OTF stages; "
                       "FUNCTIONAL SMOKE ONLY, not recurrence/timing evidence" % (list(pbs),))

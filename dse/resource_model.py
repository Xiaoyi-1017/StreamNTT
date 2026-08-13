"""Read-only pre-generation resource cost model.

Encodes structural formulas and the calibrated per-modular-reduction DSP
coefficient table D_MR for the StreamNTT segmented architecture.
Pure functions, no I/O, no generation, no apply. It NEVER changes a recipe/route/config.

Scope and evidence:
  * Structural MR counts are derived from the generator emission and cross-checked against
    on-disk csynth at N1024/BU8 (x_stages 43 group-shared; bf_unit_instances 48).
  * D_MR (DSP per modular-reduction unit) is a CALIBRATED table, not a closed form. The
    23-vs-30 Barrett-K62 difference is mechanistically supported by a byte-identical active-
    source comparison plus a prime-constant difference (actual max prime bit-length 61 vs 62),
    NOT a generator-version effect. D_MR therefore keys on the ACTUAL max prime bit-length,
    NOT the K container / PRIME_CLASS alone.
  * Barrett is a first-class / general route modelled directly; shiftadd=reduce and
    shiftadd=mul are route AXES (separate columns), never substitutes for Barrett.

Caveats:
  * Structural MR count vs per-MR D_MR are SEPARATE (TFS model changes the count, not D_MR).
  * csynth-module vs RapidStream-merged-group aggregation is a SEPARATE report layer.
  * D_MR values are csynth at N1024/BU8; per-unit DSP is assumed BU/N-invariant (verified only
    at BU8 -> generalization is an open item). Unobserved keys RAISE, never interpolate.
"""
import logging
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "TFS_GROUPSHARE", "TFS_IDSKIP",
    "ROUTE_BARRETT", "ROUTE_SHIFTADD_REDUCE", "ROUTE_SHIFTADD_MUL",
    "DEFAULT_DSP_GUARD",
    "UnknownCoefficientError",
    "DmrCoefficient", "XStageMrBreakdown", "IcbuMemory",
    "num_l_stage", "num_x_stage", "estimate_bf_unit_instances",
    "estimate_x_stage_mr_count", "x_stage_mr_breakdown", "estimate_x_stage_per_stage_mr",
    "lookup_d_mr", "lookup_d_mr_entry",
    "estimate_x_stage_dsp", "estimate_l_stage_bf_dsp",
    "estimate_icbu_memory", "guard_status_for_dsp",
    "SLOT_DSP_CAPACITY_PROXY_U55C",
    "GUARD_LAYER_DEVICE", "GUARD_LAYER_SLOT", "GUARD_LAYER_GROUP", "GUARD_LAYER_MODULE",
    "DspGuardResult", "classify_dsp_guard", "device_dsp_guard", "slot_dsp_guard",
    "estimate_rapidstream_group_dsp",
    "estimate_x_stage_per_stage_dsp", "estimate_x_stage_group_dsp", "estimate_x_stage_partition_guard",
]

# --- TFS models (X-stage twiddle-factor-scale completion sharing) ------------
TFS_GROUPSHARE = "groupshare"   # current mainline: one completion per non-identity GROUP
TFS_IDSKIP = "idskip"           # Reference lane-level model; non-default.

# --- reduction/mul routes (cost-model axes) ---------------------------------
ROUTE_BARRETT = "barrett"
ROUTE_SHIFTADD_REDUCE = "shiftadd_reduce"
ROUTE_SHIFTADD_MUL = "shiftadd_mul"

DEFAULT_DSP_GUARD = 0.80        # icbu_dse PROXY device guard (NOT a slot/SLR or physical guarantee)


class UnknownCoefficientError(ValueError):
    """Raised when a requested D_MR coefficient is not in the calibrated table.

    The model NEVER interpolates or guesses an unobserved coefficient.
    """


# --- structural formulas ----------------------------------------------------

def _ilog2(x: int) -> int:
    """Integer log2 of a positive power of two; raise otherwise."""
    if x < 1 or (x & (x - 1)) != 0:
        raise ValueError("expected a positive power of two, got %r" % (x,))
    return x.bit_length() - 1


def num_l_stage(N: int, BU: int) -> int:
    """L-stage count = log2(N) - log2(BU) - 1 (generate_code.py:231; ntt.h NUM_L_stage)."""
    n = _ilog2(N) - _ilog2(BU) - 1
    if n < 1:
        raise ValueError("num_l_stage < 1 for N=%d BU=%d (infeasible)" % (N, BU))
    return n


def num_x_stage(BU: int) -> int:
    """X-stage count = log2(BU) + 1 (generate_code.py:1297; ntt.h NUM_X_STAGE)."""
    return _ilog2(BU) + 1


def estimate_bf_unit_instances(N: int, BU: int, num_core: int = 1) -> int:
    """L-stage bf_unit instance count = num_l_stage * BU * NUM_CORE (icbu_dse.py:88).

    Verified == 48 at N1024/BU8/NUM_CORE=1 (RapidStream enumerated 48 vertices).
    """
    if num_core < 1:
        raise ValueError("num_core must be >= 1, got %r" % (num_core,))
    return num_l_stage(N, BU) * BU * num_core


@dataclass(frozen=True)
class XStageMrBreakdown:
    """X-stage modular-reduction unit decomposition for one TFS model."""
    butterfly_mr: int
    tfs_mr: int
    total_mr: int
    tfs_model: str
    per_stage_mr: Tuple[int, ...]


def _tfs_per_stage(BU: int, tfs_model: str) -> Tuple[int, ...]:
    """TFS completion MR units per X-stage s (s = 0 .. log2(BU))."""
    nx = num_x_stage(BU)
    if tfs_model == TFS_GROUPSHARE:
        # one completion per non-identity spatial GROUP: 2**s - 1
        return tuple((1 << s) - 1 for s in range(nx))
    if tfs_model == TFS_IDSKIP:
        # Lane-level reference: one per non-identity lane.
        return tuple(BU - (BU >> s) for s in range(nx))
    raise ValueError("unknown tfs_model %r (use %r or %r)"
                     % (tfs_model, TFS_GROUPSHARE, TFS_IDSKIP))


def x_stage_mr_breakdown(BU: int, tfs_model: str = TFS_GROUPSHARE) -> XStageMrBreakdown:
    """Full X-stage MR breakdown: num_x_stage*BU butterflies + per-model TFS completions."""
    nx = num_x_stage(BU)
    tfs_stage = _tfs_per_stage(BU, tfs_model)
    butterfly_mr = nx * BU
    tfs_mr = sum(tfs_stage)
    per_stage = tuple(BU + t for t in tfs_stage)
    return XStageMrBreakdown(
        butterfly_mr=butterfly_mr,
        tfs_mr=tfs_mr,
        total_mr=butterfly_mr + tfs_mr,
        tfs_model=tfs_model,
        per_stage_mr=per_stage,
    )


def estimate_x_stage_mr_count(BU: int, tfs_model: str = TFS_GROUPSHARE) -> int:
    """Total X-stage MR units. Default = current mainline group-shared (43 @ BU8);
    tfs_model=TFS_IDSKIP gives the lane-level reference count (49 @ BU8)."""
    return x_stage_mr_breakdown(BU, tfs_model).total_mr


def estimate_x_stage_per_stage_mr(BU: int, tfs_model: str = TFS_GROUPSHARE) -> Tuple[int, ...]:
    """Per-X-stage MR units (group-shared default = (8,9,11,15) @ BU8). For split planning."""
    return x_stage_mr_breakdown(BU, tfs_model).per_stage_mr


# --- calibrated D_MR table (csynth N1024/BU8; provenance-tagged) ------------

@dataclass(frozen=True)
class DmrCoefficient:
    """One calibrated DSP-per-modular-reduction-unit coefficient."""
    value: int
    route: str
    max_prime_bit_length: int
    prime_mode: str                # "single" | "pair"
    provenance: str                # on-disk evidence
    report_layer: str              # "csynth" | "csynth+rapidstream"


def _entry(value: int, route: str, blen: int, mode: str, prov: str, layer: str) -> DmrCoefficient:
    return DmrCoefficient(value, route, blen, mode, prov, layer)


# key = (route, max_prime_bit_length, prime_mode)
_DMR_TABLE: Dict[Tuple[str, int, str], DmrCoefficient] = {
    (ROUTE_BARRETT, 52, "single"): _entry(15, ROUTE_BARRETT, 52, "single", "single TII_33 measurement", "csynth"),
    (ROUTE_BARRETT, 52, "pair"):   _entry(17, ROUTE_BARRETT, 52, "pair", "52-bit pair; x_stages 731", "csynth+rapidstream"),
    (ROUTE_BARRETT, 61, "pair"):   _entry(23, ROUTE_BARRETT, 61, "pair", "61-bit pair; x_stages 989", "csynth"),
    (ROUTE_BARRETT, 62, "single"): _entry(26, ROUTE_BARRETT, 62, "single", "single TI_9 measurement", "csynth"),
    (ROUTE_BARRETT, 62, "pair"):   _entry(30, ROUTE_BARRETT, 62, "pair", "62-bit pair; x_stages 1290", "csynth+rapidstream"),
    # shiftadd routes: DSP is the shared U product only -> route-flat in prime count (single==pair)
    (ROUTE_SHIFTADD_REDUCE, 52, "single"): _entry(6, ROUTE_SHIFTADD_REDUCE, 52, "single", "TII_33 shift-add measurement", "csynth"),
    (ROUTE_SHIFTADD_REDUCE, 52, "pair"):   _entry(6, ROUTE_SHIFTADD_REDUCE, 52, "pair", "52-bit shift-add pair (route-flat)", "csynth"),
    (ROUTE_SHIFTADD_REDUCE, 62, "single"): _entry(11, ROUTE_SHIFTADD_REDUCE, 62, "single", "TI_9 shift-add measurement", "csynth"),
    (ROUTE_SHIFTADD_REDUCE, 62, "pair"):   _entry(11, ROUTE_SHIFTADD_REDUCE, 62, "pair", "62-bit shift-add pair; x_stages 473", "csynth+rapidstream"),
    (ROUTE_SHIFTADD_MUL, 52, "single"): _entry(6, ROUTE_SHIFTADD_MUL, 52, "single", "TII_33 shift-add multiply measurement", "csynth"),
    (ROUTE_SHIFTADD_MUL, 52, "pair"):   _entry(6, ROUTE_SHIFTADD_MUL, 52, "pair", "52-bit shift-add multiply pair (route-flat)", "csynth"),
    (ROUTE_SHIFTADD_MUL, 62, "single"): _entry(11, ROUTE_SHIFTADD_MUL, 62, "single", "TI_9 shift-add multiply measurement", "csynth"),
    (ROUTE_SHIFTADD_MUL, 62, "pair"):   _entry(11, ROUTE_SHIFTADD_MUL, 62, "pair", "62-bit shift-add multiply pair (route-flat)", "csynth"),
}

# documented explicit unknowns: a key that is intentionally NOT calibrated (no on-disk datapoint)
_DMR_KNOWN_UNKNOWN: Dict[Tuple[str, int, str], str] = {
    (ROUTE_BARRETT, 61, "single"): "61-bit single-prime Barrett has no measured coefficient",
}


def _prime_mode(prime_count: int) -> str:
    if prime_count < 1:
        raise ValueError("prime_count must be >= 1, got %r" % (prime_count,))
    return "single" if prime_count == 1 else "pair"


def lookup_d_mr_entry(route: str, max_prime_bit_length: int, prime_count: int) -> DmrCoefficient:
    """Return the calibrated DmrCoefficient for (route, actual max prime bit-length, count).

    Raises UnknownCoefficientError for any unobserved key; NEVER interpolates.
    """
    mode = _prime_mode(prime_count)
    key = (route, max_prime_bit_length, mode)
    entry = _DMR_TABLE.get(key)
    if entry is not None:
        return entry
    reason = _DMR_KNOWN_UNKNOWN.get(key)
    if reason is not None:
        raise UnknownCoefficientError("D_MR %r is an explicit UNKNOWN: %s" % (key, reason))
    raise UnknownCoefficientError(
        "D_MR %r not calibrated (no on-disk evidence); NOT interpolated. "
        "Calibrated keys: %s" % (key, sorted(_DMR_TABLE.keys())))


def lookup_d_mr(route: str, max_prime_bit_length: int, prime_count: int) -> int:
    """Calibrated DSP per modular-reduction unit. Raises UnknownCoefficientError if unobserved."""
    return lookup_d_mr_entry(route, max_prime_bit_length, prime_count).value


# --- composed DSP estimates (MR_count x D_MR) -------------------------------

def estimate_x_stage_dsp(BU: int, route: str, max_prime_bit_length: int, prime_count: int,
                         tfs_model: str = TFS_GROUPSHARE) -> int:
    """X-stage (x_stages module) DSP = X-stage MR count * D_MR.

    e.g. BU8 Barrett 62-bit pair group-shared = 43 * 30 = 1290.
    """
    mr = estimate_x_stage_mr_count(BU, tfs_model)
    return mr * lookup_d_mr(route, max_prime_bit_length, prime_count)


def estimate_l_stage_bf_dsp(N: int, BU: int, route: str, max_prime_bit_length: int,
                            prime_count: int, num_core: int = 1) -> int:
    """L-stage bf_unit DSP = bf_unit_instances * D_MR.

    e.g. N1024 BU8 NUM_CORE1 Barrett 62-bit pair = 48 * 30 = 1440.
    """
    inst = estimate_bf_unit_instances(N, BU, num_core)
    return inst * lookup_d_mr(route, max_prime_bit_length, prime_count)


# --- ICBU memory (reuses icbu_dse as the storage estimator plugin) ----------

@dataclass(frozen=True)
class IcbuMemory:
    """ICBU (bf_unit) memory estimate. Carries NO DSP field by design: bf_unit DSP is owned by
    resource_model D_MR (estimate_l_stage_bf_dsp), NOT icbu_dse.CALIB's hardcoded 30."""
    uram: int
    bram_18k: int
    recipe: str
    bram_packing: str              # "calibrated" | "uncalibrated/conservative" (from icbu_dse)


def estimate_icbu_memory(N: int, BU: int, CH: int, K: int, num_core: int = 1,
                         recipe: str = "2U2B") -> IcbuMemory:
    """ICBU bf_unit memory (URAM / BRAM_18K) by reusing icbu_dse as the storage plugin. Report-only.

    Reuses icbu_dse.arch_quantities + estimate_icbu (no logic / board-constant duplication; lazy import).
    DSP is intentionally not taken from icbu_dse here (its CALIB uses the measured 62-bit-pair
    case); authoritative bf_unit DSP = estimate_l_stage_bf_dsp (D_MR-keyed).

    Measured anchor (not a universal constant): N1024/BU8/CH4/2U2B -> URAM 96 / BRAM_18K 192
    (match measured RapidStream). BRAM packing is "calibrated" only for K62 bank_depth in {1024,2048,4096};
    at small N (e.g. bank_depth 32) it is "uncalibrated/conservative". An unknown recipe raises KeyError
    (icbu_dse behavior; not re-validated here).
    """
    from dse import icbu_dse  # lazy: keep the core model dependency-free
    arch = icbu_dse.arch_quantities(N, BU, CH, K, num_core)
    m = icbu_dse.estimate_icbu(arch, recipe)
    return IcbuMemory(uram=m["URAM"], bram_18k=m["BRAM_18K"], recipe=recipe, bram_packing=m["packing"])


# --- optional device-level guard wrapper (reuses icbu_dse cleanly) ----------

def guard_status_for_dsp(dsp_value: int, guard: float = DEFAULT_DSP_GUARD) -> str:
    """DEVICE-level DSP guard status via icbu_dse.guard_status (U55C; returns UNDER/NEAR/OVER).

    NOTE: this is the whole-device DSP capacity (8204). The real X-stage blocker is a
    SLR/SLOT-level capacity (~1228), a separate layer modelled by slot_dsp_guard.
    """
    from dse import icbu_dse  # lazy: keep the core model dependency-free
    return icbu_dse.guard_status(dsp_value, "DSP", icbu_dse.BOARD_U55C, guard)


# --- layered DSP guardrail ---------------------------------------------------
#
# Four DSP accounting layers, kept strictly separate (never compared cross-layer silently):
#   MODULE  - one csynth module's DSP (e.g. x_stages = estimate_x_stage_dsp; 1290 @ TI_9+TI_2 BU8)
#   GROUP   - RapidStream merged-vertex group (estimate_rapidstream_group_dsp; 1410 = 1290 + TFR-X)
#   SLOT    - SLR/slot floorplan capacity PROXY (slot_dsp_guard vs SLOT_DSP_CAPACITY_PROXY_U55C)
#   DEVICE  - whole-device capacity (device_dsp_guard vs icbu_dse.BOARD_U55C["DSP"] = 8204)

GUARD_LAYER_DEVICE = "device"
GUARD_LAYER_SLOT = "slot"
GUARD_LAYER_GROUP = "group"
GUARD_LAYER_MODULE = "module"

# Largest SLR/slot DSP in the visible repository artifacts (xcu55c). A CONFIGURABLE PROXY, not a universal
# truth: it already represents the guardrail capacity, so do NOT apply a second guard ratio to it
# (slot_dsp_guard defaults guard_ratio=1.0). NOTE: 1218 does not appear in the visible repo artifacts; if it
# ever surfaces it is from an external P&R/report layer and must never be silently
# equated to this proxy.
SLOT_DSP_CAPACITY_PROXY_U55C = 1228


@dataclass(frozen=True)
class DspGuardResult:
    """One DSP guardrail classification at an explicit capacity layer."""
    used: int
    capacity: int
    capacity_layer: str            # GUARD_LAYER_DEVICE | GUARD_LAYER_SLOT | GUARD_LAYER_GROUP | ...
    guard_ratio: float             # applied ONCE to `capacity` to form the threshold
    threshold: int                 # int(guard_ratio * capacity) -- the effective limit
    status: str                    # "UNDER" | "NEAR" | "OVER" (from icbu_dse.guard_status)
    headroom: int                  # threshold - used


def classify_dsp_guard(used: int, capacity: int, capacity_layer: str,
                       guard_ratio: float = 0.80) -> DspGuardResult:
    """Classify a DSP `used` value against `capacity` at an EXPLICIT guard_ratio and layer.

    Status reuses icbu_dse.guard_status (OVER >= guard_ratio*capacity; NEAR >= 0.95*that; else UNDER).
    The caller owns guard_ratio: pass 1.0 when `capacity` already IS a guardrail/proxy number (so the
    ratio is not applied twice); pass 0.80 (etc.) against a raw device capacity.
    """
    from dse import icbu_dse  # lazy; reuse the existing guard semantics, no board-constant duplication
    status = icbu_dse.guard_status(used, "DSP", {"DSP": capacity}, guard_ratio)
    threshold = int(guard_ratio * capacity)
    return DspGuardResult(used=used, capacity=capacity, capacity_layer=capacity_layer,
                          guard_ratio=guard_ratio, threshold=threshold, status=status,
                          headroom=threshold - used)


def device_dsp_guard(used: int, device_capacity: Optional[int] = None,
                     guard_ratio: float = 0.80) -> DspGuardResult:
    """DEVICE-level DSP guard. Raw capacity (default icbu_dse.BOARD_U55C["DSP"]=8204) -> guard_ratio applied
    ONCE. Anchor: total 3090 vs 8204 -> UNDER. This is whole-device, NOT a slot/SLR judgement."""
    if device_capacity is None:
        from dse import icbu_dse
        device_capacity = icbu_dse.BOARD_U55C["DSP"]
    return classify_dsp_guard(used, device_capacity, GUARD_LAYER_DEVICE, guard_ratio)


def slot_dsp_guard(used: int, slot_capacity_proxy: int = SLOT_DSP_CAPACITY_PROXY_U55C,
                   guard_ratio: float = 1.0) -> DspGuardResult:
    """SLOT/SLR-level DSP guard against a configurable PROXY. guard_ratio defaults to 1.0 because the proxy
    already IS the guardrail capacity (avoids applying 0.8 twice). Anchor: x_stages 1290 / group 1410 vs
    proxy 1228 -> OVER. A placement/risk judgement, NOT a board-capacity comparison or a math failure."""
    return classify_dsp_guard(used, slot_capacity_proxy, GUARD_LAYER_SLOT, guard_ratio)


def estimate_rapidstream_group_dsp(x_stages_dsp: int, tfr_x_dsp: int = 0,
                                   tfg_x_dsp: int = 0, extra_dsp: int = 0) -> int:
    """RapidStream merged-vertex GROUP DSP = x_stages module DSP + the modules RapidStream merges with it.

    GROUP-level is DISTINCT from MODULE-level x_stages DSP; never conflate. Audited anchor (TI_9+TI_2
    Barrett BU8): group 1410 = x_stages 1290 + TFR-X 120 (4*30). The TFG-X completions were SEPARATE
    vertices (NOT in that failing group) -> tfg_x_dsp defaults to 0; pass it explicitly only if a future
    case is observed to merge them. Additive model of the observed merge, not a derived guarantee.
    """
    return x_stages_dsp + tfr_x_dsp + tfg_x_dsp + extra_dsp


# --- per-stage / per-group X-stage DSP --------------------------------------

def estimate_x_stage_per_stage_dsp(BU: int, d_mr: int,
                                   tfs_model: str = TFS_GROUPSHARE) -> Tuple[int, ...]:
    """Per-X-stage DSP = per-stage MR count * d_mr (d_mr supplied EXPLICITLY -- caller picks the
    calibrated/hypothetical coefficient). BU8 group-shared @ d_mr=30 -> (240,270,330,450), a measured
    anchor, NOT a universal split result. D_MR BU-invariance is unproven, so any BU != 8 figure here is
    structural/formula output, not calibrated P&R truth. idskip (TFS_IDSKIP) is a non-default reference."""
    return tuple(mr * d_mr for mr in estimate_x_stage_per_stage_mr(BU, tfs_model))


def _validate_partition(partition, n_stages: int) -> None:
    """Fail-fast: partition must be a non-empty, cover-and-disjoint set of in-range stage indices."""
    if not partition:
        raise ValueError("partition must be a non-empty list of stage-index groups")
    flat = [i for group in partition for i in group]
    for i in flat:
        if not isinstance(i, int) or i < 0 or i >= n_stages:
            raise ValueError("partition index %r out of range [0,%d)" % (i, n_stages))
    if len(flat) != len(set(flat)):
        raise ValueError("partition groups overlap (a stage index appears twice): %r" % (partition,))
    if set(flat) != set(range(n_stages)):
        missing = sorted(set(range(n_stages)) - set(flat))
        raise ValueError("partition must cover all %d stages (fail-fast); missing %r"
                         % (n_stages, missing))


def estimate_x_stage_group_dsp(stage_dsp, partition) -> Tuple[int, ...]:
    """Sum per-stage DSP into user-supplied groups. partition = explicit local-index groups
    (cover-and-disjoint; fail-fast on out-of-range / overlap / missing). NEVER chooses a split.
    e.g. (240,270,330,450) over [[0,1,2],[3]] -> (840, 450)."""
    _validate_partition(partition, len(stage_dsp))
    return tuple(sum(stage_dsp[i] for i in group) for group in partition)


def estimate_x_stage_partition_guard(stage_dsp, partition,
                                     slot_capacity_proxy: int = SLOT_DSP_CAPACITY_PROXY_U55C,
                                     guard_ratio: float = 1.0) -> Tuple[DspGuardResult, ...]:
    """Return slot-guard status per partition group. A group over the
    slot proxy is a placement/RISK signal, NOT a math failure and NOT a chosen split."""
    return tuple(slot_dsp_guard(g, slot_capacity_proxy, guard_ratio)
                 for g in estimate_x_stage_group_dsp(stage_dsp, partition))

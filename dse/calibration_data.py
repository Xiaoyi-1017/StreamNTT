"""Observed-evidence data for the pre-generation resource cost model.

Data only: curated N1024 measurements (csynth module DSP plus RapidStream group/memory and the
guardrail slot proxy) and a loader for the N131072 RapidStream manifests (reusing icbu_dse.load_calibration --
no duplicate manifest parsing). Comparison and rendering are owned by calibration_compare.

Nothing is invented: every record carries provenance and an explicit report layer. The N131072 manifests are
shiftadd=mul evidence and calibrate the ICBU DSP/memory submodel for those recorded
cases ONLY -- they are NOT full-design exact validation and NOT P&R/Fmax validation. No P&R records are
created (no P&R data is visible in the repo).
"""
import logging
import os
from dataclasses import dataclass
from typing import Optional, Tuple

from dse import icbu_dse

logger = logging.getLogger(__name__)

__all__ = [
    "LAYER_CSYNTH_MODULE", "LAYER_RAPIDSTREAM_ICBU", "LAYER_RAPIDSTREAM_GROUP",
    "LAYER_RAPIDSTREAM_DESIGN", "LAYER_PNR", "LAYER_GUARDRAIL_PROXY",
    "ROUTE_BARRETT", "ROUTE_SHIFTADD_REDUCE", "ROUTE_SHIFTADD_MUL",
    "ObservedRecord", "curated_n1024_records", "curated_n131072_single_prime_records",
    "load_n131072_records", "all_records",
]

# Report layers are kept strictly separate.
LAYER_CSYNTH_MODULE = "csynth_module"
LAYER_RAPIDSTREAM_ICBU = "rapidstream_icbu"
LAYER_RAPIDSTREAM_GROUP = "rapidstream_group"
LAYER_RAPIDSTREAM_DESIGN = "rapidstream_design"
LAYER_PNR = "pnr"                 # reserved; NO records created (no P&R data in the repo)
LAYER_GUARDRAIL_PROXY = "guardrail_proxy"

# Route labels (metadata only). Values must equal resource_model.ROUTE_* so the comparison layer can
# match observed records to predictions; kept as LOCAL literals to avoid a data-layer -> model-layer import.
ROUTE_BARRETT = "barrett"
ROUTE_SHIFTADD_REDUCE = "shiftadd_reduce"
ROUTE_SHIFTADD_MUL = "shiftadd_mul"


@dataclass(frozen=True)
class ObservedRecord:
    case_id: str
    N: int
    BU: int
    CH: int
    K: int
    route: Optional[str]                 # ROUTE_* label; None for route-independent (memory) / proxy
    actual_bit_length: Optional[int]     # max prime bit-length; None for proxy
    num_primes: Optional[int]
    tfs_model: Optional[str]             # "groupshare" | "idskip" | None (n/a or not recorded)
    storage_recipe: Optional[str]
    report_layer: str
    metric: str                          # "DSP" | "URAM" | "BRAM_18K" | ...
    observed_value: int
    provenance: str
    confidence: str                      # "measured" | "user_asserted" | "derived"
    notes: str


def _rec(**kw) -> ObservedRecord:
    base = dict(N=1024, BU=8, CH=4, K=62, route=None, actual_bit_length=None, num_primes=None,
                tfs_model=None, storage_recipe="2U2B", confidence="measured", notes="")
    base.update(kw)
    return ObservedRecord(**base)


def curated_n1024_records() -> Tuple[ObservedRecord, ...]:
    """Curated N1024 csynth, RapidStream group/ICBU, and guardrail measurements."""
    B = ROUTE_BARRETT
    SA = ROUTE_SHIFTADD_REDUCE
    cs = LAYER_CSYNTH_MODULE
    return (
        _rec(case_id="N1024_BU8_barrett_61_pair_2U2B_groupshare", route=B, actual_bit_length=61,
             num_primes=2, tfs_model="groupshare", report_layer=cs, metric="DSP", observed_value=989,
             provenance="measured csynth groupshare result (43x23)",
             notes="Barrett 61-bit pair; per-MR D_MR 23"),
        _rec(case_id="N1024_BU8_barrett_62_pair_2U2B_groupshare", route=B, actual_bit_length=62,
             num_primes=2, tfs_model="groupshare", report_layer=cs, metric="DSP", observed_value=1290,
             provenance="measured csynth and RapidStream groupshare result (43x30)",
             notes="Barrett 62-bit (TI_9) pair; per-MR D_MR 30; csynth==RapidStream"),
        _rec(case_id="N1024_BU8_barrett_61_pair_2U2B_idskip", route=B, actual_bit_length=61,
             num_primes=2, tfs_model="idskip", report_layer=cs, metric="DSP", observed_value=1127,
             provenance="measured csynth idskip result (49x23; same 61-bit pair as groupshare)",
             notes="lane-level TFS reference model (49 MR units); per-MR D_MR 23 (61-bit pair)"),
        _rec(case_id="N1024_BU8_barrett_52_pair_2U2B_groupshare", route=B, actual_bit_length=52,
             num_primes=2, tfs_model="groupshare", report_layer=cs, metric="DSP", observed_value=731,
             provenance="measured csynth groupshare result (43x17)",
             notes="Barrett 52-bit pair; per-MR D_MR 17"),
        _rec(case_id="N1024_BU8_shiftadd_62_2U2B_groupshare", route=SA, actual_bit_length=62,
             num_primes=1, tfs_model="groupshare", report_layer=cs, metric="DSP", observed_value=473,
             provenance="measured csynth shift-add result (43x11)",
             notes="shiftadd reduce AND mul both 473 (per-MR D_MR 11)"),
        _rec(case_id="N1024_BU8_shiftadd_52_2U2B_groupshare", route=SA, actual_bit_length=52,
             num_primes=1, tfs_model="groupshare", report_layer=cs, metric="DSP", observed_value=258,
             provenance="measured csynth shift-add result (43x6)",
             notes="shiftadd reduce AND mul both 258 (per-MR D_MR 6)"),
        _rec(case_id="N1024_BU8_barrett_62_pair_2U2B_group", route=B, actual_bit_length=62, num_primes=2,
             tfs_model="groupshare", report_layer=LAYER_RAPIDSTREAM_GROUP, metric="DSP", observed_value=1410,
             provenance="measured RapidStream group = x_stages 1290 + TFR-X 4x30",
             notes="merged-vertex GROUP, NOT module DSP; additive merge observed once"),
        _rec(case_id="N1024_BU8_2U2B_icbu_uram", report_layer=LAYER_RAPIDSTREAM_ICBU, metric="URAM",
             observed_value=96, provenance="measured RapidStream 48 bf_unit instances (2U2B)",
             notes="ICBU bf_unit URAM (route-independent)"),
        _rec(case_id="N1024_BU8_2U2B_icbu_bram", report_layer=LAYER_RAPIDSTREAM_ICBU, metric="BRAM_18K",
             observed_value=192, provenance="measured RapidStream 48 bf_unit instances (2U2B)",
             notes="ICBU bf_unit BRAM_18K (route-independent)"),
        _rec(case_id="U55C_slot_dsp_proxy", storage_recipe=None, report_layer=LAYER_GUARDRAIL_PROXY,
             metric="DSP", observed_value=1228, confidence="derived",
             provenance="largest measured xcu55c SLR/slot DSP proxy",
             notes="guardrail/proxy capacity, NOT an observed module DSP; configurable; 1218 not in artifacts"),
    )


def curated_n131072_single_prime_records() -> Tuple[ObservedRecord, ...]:
    """Curated single-prime N131072 module and ICBU-submodel measurements.

    These records use Vitis HLS 2023.2.2, a 3U1B recipe, and a passing XO
    simulation. Placement-group and full-design totals are excluded because they
    are different report layers. Per-instance bf_unit DSP is retained only in
    notes; the comparison model predicts x_stages and aggregate ICBU quantities.
    """
    B = ROUTE_BARRETT
    SA = ROUTE_SHIFTADD_REDUCE
    cs = LAYER_CSYNTH_MODULE
    ic = LAYER_RAPIDSTREAM_ICBU
    return (
        # x_stages MODULE DSP -- layer-stable (csynth == RapidStream module: 1118 / 473) -> compares EXACT
        _rec(case_id="N131072_BU8_barrett_62_single_3U1B_xstages", N=131072, route=B, actual_bit_length=62,
             num_primes=1, tfs_model="groupshare", storage_recipe="3U1B", report_layer=cs, metric="DSP",
             observed_value=1118,
             provenance="measured csynth/RapidStream x_stages result (43x26); XO simulation passed",
             notes="Barrett 62-bit SINGLE-prime; per-instance bf_unit DSP = 26 (agrees with delta=26); "
                   "x_stages 1118 = 43x26; csynth==RapidStream module (layer-stable); current generator; "
                   "recipe 3U1B (DSP recipe-invariant); NOT a P&R/Fmax claim"),
        _rec(case_id="N131072_BU8_shiftadd_reduce_62_single_3U1B_xstages", N=131072, route=SA,
             actual_bit_length=62, num_primes=1, tfs_model="groupshare", storage_recipe="3U1B",
             report_layer=cs, metric="DSP", observed_value=473,
             provenance="measured csynth/RapidStream x_stages result (43x11); XO simulation passed",
             notes="shiftadd_reduce K62 SINGLE-prime; per-instance bf_unit DSP = 11 (agrees with delta=11); "
                   "x_stages 473 = 43x11; csynth==RapidStream module (layer-stable); current generator; "
                   "recipe 3U1B"),
        # ICBU bf_unit x104 total DSP (rapidstream_icbu SUBMODEL) -> compares CALIBRATED
        _rec(case_id="N131072_BU8_barrett_62_single_3U1B_icbu", N=131072, route=B, actual_bit_length=62,
             num_primes=1, tfs_model=None, storage_recipe="3U1B", report_layer=ic, metric="DSP",
             observed_value=2704,
             provenance="measured RapidStream bf_unit result: 26 DSP x 104 instances; XO simulation passed",
             notes="Barrett 62-bit single ICBU bf_unit total = 104 x 26 (per-instance bf_unit DSP = 26, "
                   "agrees with delta=26); SUBMODEL anchor (ICBU DSP only) -- NOT full-design, NOT P&R/Fmax; "
                   "recipe 3U1B"),
        _rec(case_id="N131072_BU8_shiftadd_reduce_62_single_3U1B_icbu", N=131072, route=SA,
             actual_bit_length=62, num_primes=1, tfs_model=None, storage_recipe="3U1B", report_layer=ic,
             metric="DSP", observed_value=1144,
             provenance="measured RapidStream bf_unit result: 11 DSP x 104 instances; XO simulation passed",
             notes="shiftadd_reduce K62 single ICBU bf_unit total = 104 x 11 (per-instance bf_unit DSP = 11, "
                   "agrees with delta=11); SUBMODEL anchor (ICBU DSP only); recipe 3U1B"),
    )


def _route_from_shiftadd(shiftadd: Optional[str]) -> str:
    s = (shiftadd or "").lower()
    if s == "mul":
        return ROUTE_SHIFTADD_MUL
    if s == "reduce":
        return ROUTE_SHIFTADD_REDUCE
    return ROUTE_BARRETT


def load_n131072_records(root: str = icbu_dse.CALIBRATION_ROOT) -> Tuple[ObservedRecord, ...]:
    """Load the N131072 RapidStream manifests under `root`, reusing icbu_dse.load_calibration (no duplicate
    manifest parsing). Emits ICBU DSP/URAM/BRAM_18K (rapidstream_icbu) + design DSP (rapidstream_design)
    records -- SUBMODEL anchors only. Missing root -> () (clean-checkout-safe). Deterministic (sorted dirs)."""
    if not os.path.isdir(root):
        return ()
    recs = []
    for name in sorted(os.listdir(root)):
        case_dir = os.path.join(root, name)
        if not os.path.exists(os.path.join(case_dir, "manifest.json")):
            continue
        man = icbu_dse.load_calibration(case_dir)["manifest"]      # reuse; no duplicate parsing
        route = _route_from_shiftadd(man.get("shiftadd"))
        vals = man.get("prime_set", {}).get("values") or []
        bitlen = max((int(v).bit_length() for v in vals), default=man["K"])
        common = dict(case_id=name, N=man["N"], BU=man["BU"], CH=man["CH"], K=man["K"], route=route,
                      actual_bit_length=bitlen, num_primes=man.get("prime_set", {}).get("count"),
                      tfs_model=None, storage_recipe=man.get("storage_recipe"),
                      provenance=os.path.join(case_dir, "manifest.json"), confidence="measured",
                      notes="N131072 RapidStream submodel measurement (shiftadd=%s); "
                            "ICBU DSP/memory only -- not full-design / P&R validation" % man.get("shiftadd"))
        icbu = man.get("measured_rapidstream_icbu_x104", {})
        for metric in ("DSP", "URAM", "BRAM_18K"):
            if metric in icbu:
                recs.append(ObservedRecord(report_layer=LAYER_RAPIDSTREAM_ICBU, metric=metric,
                                           observed_value=icbu[metric], **common))
        design = man.get("measured_rapidstream_design", {})
        if "DSP" in design:
            recs.append(ObservedRecord(report_layer=LAYER_RAPIDSTREAM_DESIGN, metric="DSP",
                                       observed_value=design["DSP"], **common))
    return tuple(recs)


def all_records(root: str = icbu_dse.CALIBRATION_ROOT) -> Tuple[ObservedRecord, ...]:
    """Curated N1024 anchors + curated N131072 single-prime anchors + loaded N131072 manifest records."""
    return (curated_n1024_records()
            + curated_n131072_single_prime_records()
            + load_n131072_records(root))

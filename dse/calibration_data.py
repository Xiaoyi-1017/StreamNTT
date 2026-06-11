"""Observed-evidence data layer for the pre-gen cost-model calibration harness (Phase 4.0 Slice 4a).

Data only: curated AUDITED N1024 anchors (csynth module DSP + RapidStream forensic group/memory + the
guardrail slot proxy) and a loader for the N131072 RapidStream manifests (reusing icbu_dse.load_calibration --
no duplicate manifest parsing). NO comparison, NO delta engine, NO rendering -- that is Slice 4b.

Nothing is invented: every record carries provenance and an explicit report layer. The N131072 manifests are
shiftadd=mul historical / near-mainline evidence and calibrate the ICBU DSP/memory SUBMODEL for those recorded
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

# Report layers -- kept strictly separate (the cross-layer comparison discipline lives in Slice 4b).
LAYER_CSYNTH_MODULE = "csynth_module"
LAYER_RAPIDSTREAM_ICBU = "rapidstream_icbu"
LAYER_RAPIDSTREAM_GROUP = "rapidstream_group"
LAYER_RAPIDSTREAM_DESIGN = "rapidstream_design"
LAYER_PNR = "pnr"                 # reserved; NO records created (no P&R data in the repo)
LAYER_GUARDRAIL_PROXY = "guardrail_proxy"

# Route labels (metadata only). Values MUST equal resource_model.ROUTE_* so the Slice-4b comparison layer can
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
    """Audited N1024 anchors (Slice 1 / 1b): csynth module DSP, RapidStream forensic group + ICBU memory,
    and the guardrail slot proxy -- each provenance-tagged with its explicit report layer."""
    B = ROUTE_BARRETT
    SA = ROUTE_SHIFTADD_REDUCE
    cs = LAYER_CSYNTH_MODULE
    return (
        _rec(case_id="N1024_BU8_barrett_61_pair_2U2B_groupshare", route=B, actual_bit_length=61,
             num_primes=2, tfs_model="groupshare", report_layer=cs, metric="DSP", observed_value=989,
             provenance="on-disk csynth tfs_groupshare_bu8_post (43x23); Slice 1b",
             notes="Barrett 61-bit pair; per-MR D_MR 23"),
        _rec(case_id="N1024_BU8_barrett_62_pair_2U2B_groupshare", route=B, actual_bit_length=62,
             num_primes=2, tfs_model="groupshare", report_layer=cs, metric="DSP", observed_value=1290,
             provenance="on-disk csynth stepc1_step_c1_2U2B (43x30); == RapidStream forensic 1290; Slice 1b",
             notes="Barrett 62-bit (TI_9) pair; per-MR D_MR 30; csynth==RapidStream"),
        _rec(case_id="N1024_BU8_barrett_61_pair_2U2B_idskip", route=B, actual_bit_length=61,
             num_primes=2, tfs_model="idskip", report_layer=cs, metric="DSP", observed_value=1127,
             provenance="on-disk csynth tfs_idskip_bu8 (49x23; SAME 61-bit pair as the 989 groupshare case); Slice 1b",
             notes="historical lane-level TFS model (49 MR units); per-MR D_MR 23 (61-bit pair)"),
        _rec(case_id="N1024_BU8_barrett_52_pair_2U2B_groupshare", route=B, actual_bit_length=52,
             num_primes=2, tfs_model="groupshare", report_layer=cs, metric="DSP", observed_value=731,
             provenance="on-disk csynth p32_xo_2u2b / s1b_cfg4 (43x17); Slice 1b",
             notes="Barrett 52-bit pair; per-MR D_MR 17"),
        _rec(case_id="N1024_BU8_shiftadd_62_2U2B_groupshare", route=SA, actual_bit_length=62,
             num_primes=1, tfs_model="groupshare", report_layer=cs, metric="DSP", observed_value=473,
             provenance="on-disk csynth s4_ti9 / step11_reduce (43x11); Slice 1b",
             notes="shiftadd reduce AND mul both 473 (per-MR D_MR 11)"),
        _rec(case_id="N1024_BU8_shiftadd_52_2U2B_groupshare", route=SA, actual_bit_length=52,
             num_primes=1, tfs_model="groupshare", report_layer=cs, metric="DSP", observed_value=258,
             provenance="on-disk csynth s4_tii33 (43x6); Slice 1b",
             notes="shiftadd reduce AND mul both 258 (per-MR D_MR 6)"),
        _rec(case_id="N1024_BU8_barrett_62_pair_2U2B_group", route=B, actual_bit_length=62, num_primes=2,
             tfs_model="groupshare", report_layer=LAYER_RAPIDSTREAM_GROUP, metric="DSP", observed_value=1410,
             provenance="RapidStream forensic group = x_stages 1290 + TFR-X 4x30 (d0c3a16); supplement",
             notes="merged-vertex GROUP, NOT module DSP; additive merge observed once"),
        _rec(case_id="N1024_BU8_2U2B_icbu_uram", report_layer=LAYER_RAPIDSTREAM_ICBU, metric="URAM",
             observed_value=96, provenance="RapidStream forensic 48 bf_unit instances (2U2B); supplement",
             notes="ICBU bf_unit URAM (route-independent)"),
        _rec(case_id="N1024_BU8_2U2B_icbu_bram", report_layer=LAYER_RAPIDSTREAM_ICBU, metric="BRAM_18K",
             observed_value=192, provenance="RapidStream forensic 48 bf_unit instances (2U2B); supplement",
             notes="ICBU bf_unit BRAM_18K (route-independent)"),
        _rec(case_id="U55C_slot_dsp_proxy", storage_recipe=None, report_layer=LAYER_GUARDRAIL_PROXY,
             metric="DSP", observed_value=1228, confidence="derived",
             provenance="largest xcu55c SLR/slot DSP in visible artifacts; supplement",
             notes="guardrail/proxy capacity, NOT an observed module DSP; configurable; 1218 not in artifacts"),
    )


def curated_n131072_single_prime_records() -> Tuple[ObservedRecord, ...]:
    """Phase 4.1 AUDITED single-prime N131072 anchors (current generator, Vitis HLS 2023.2.2, XO cosim
    PASSED, recipe 3U1B). DSP-only, D_MR-safe MODULE + ICBU-submodel anchors. Source dirs (read-only audit):
    Checkpoint/20260606_single_prime/N131072_BU8_CH4_QK62_bu8_ti9_{barrett,reduce}_3U1B (intake checkpoint
    phase4_1_single_prime_evidence_intake_checkpoint_20260608_2144.md).

    NOT recorded (intentional, see the 1218 group audit checkpoint 20260608_2156): the post-synth placement
    GROUP (Barrett 1218 = x_stages 1118 + 4xTFR-X 25; TFR-X is 25 DSP post-synth but 26 DSP at csynth) and the
    design TOTALS (Barrett 4368 / reduce 1903) are GROUP/DESIGN layers, NOT D_MR anchors. The per-instance
    bf_unit DSP (26 / 11) IS the resource_model D_MR coefficient and is preserved in notes only -- it is NOT a
    standalone record (the comparison layer predicts x_stages x43 and ICBU x104 aggregates, not a bare
    per-instance metric, so a standalone bf_unit row would mis-compare)."""
    B = ROUTE_BARRETT
    SA = ROUTE_SHIFTADD_REDUCE
    cs = LAYER_CSYNTH_MODULE
    ic = LAYER_RAPIDSTREAM_ICBU
    return (
        # x_stages MODULE DSP -- layer-stable (csynth == RapidStream module: 1118 / 473) -> compares EXACT
        _rec(case_id="N131072_BU8_barrett_62_single_3U1B_xstages", N=131072, route=B, actual_bit_length=62,
             num_primes=1, tfs_model="groupshare", storage_recipe="3U1B", report_layer=cs, metric="DSP",
             observed_value=1118,
             provenance="Checkpoint/20260606_single_prime/N131072_BU8_CH4_QK62_bu8_ti9_barrett_3U1B: "
                        "x_stages_csynth.rpt == rapidstream.log:1468 (43x26); XO cosim PASSED; "
                        "phase4_1 intake 2026-06-08",
             notes="Barrett 62-bit SINGLE-prime; per-instance bf_unit DSP = 26 (agrees with delta=26); "
                   "x_stages 1118 = 43x26; csynth==RapidStream module (layer-stable); current generator; "
                   "recipe 3U1B (DSP recipe-invariant); NOT a P&R/Fmax claim"),
        _rec(case_id="N131072_BU8_shiftadd_reduce_62_single_3U1B_xstages", N=131072, route=SA,
             actual_bit_length=62, num_primes=1, tfs_model="groupshare", storage_recipe="3U1B",
             report_layer=cs, metric="DSP", observed_value=473,
             provenance="Checkpoint/20260606_single_prime/N131072_BU8_CH4_QK62_bu8_ti9_reduce_3U1B: "
                        "x_stages_csynth.rpt == rapidstream.log:1240 (43x11); XO cosim PASSED; "
                        "phase4_1 intake 2026-06-08",
             notes="shiftadd_reduce K62 SINGLE-prime; per-instance bf_unit DSP = 11 (agrees with delta=11); "
                   "x_stages 473 = 43x11; csynth==RapidStream module (layer-stable); current generator; "
                   "recipe 3U1B"),
        # ICBU bf_unit x104 total DSP (rapidstream_icbu SUBMODEL) -> compares CALIBRATED
        _rec(case_id="N131072_BU8_barrett_62_single_3U1B_icbu", N=131072, route=B, actual_bit_length=62,
             num_primes=1, tfs_model=None, storage_recipe="3U1B", report_layer=ic, metric="DSP",
             observed_value=2704,
             provenance="Checkpoint/20260606_single_prime/N131072_BU8_CH4_QK62_bu8_ti9_barrett_3U1B: "
                        "rapidstream.log:1187 bf_unit per-instance 26 x 104 instances; XO cosim PASSED; "
                        "phase4_1 intake 2026-06-08",
             notes="Barrett 62-bit single ICBU bf_unit total = 104 x 26 (per-instance bf_unit DSP = 26, "
                   "agrees with delta=26); SUBMODEL anchor (ICBU DSP only) -- NOT full-design, NOT P&R/Fmax; "
                   "recipe 3U1B"),
        _rec(case_id="N131072_BU8_shiftadd_reduce_62_single_3U1B_icbu", N=131072, route=SA,
             actual_bit_length=62, num_primes=1, tfs_model=None, storage_recipe="3U1B", report_layer=ic,
             metric="DSP", observed_value=1144,
             provenance="Checkpoint/20260606_single_prime/N131072_BU8_CH4_QK62_bu8_ti9_reduce_3U1B: "
                        "rapidstream.log:1060 bf_unit per-instance 11 x 104 instances; XO cosim PASSED; "
                        "phase4_1 intake 2026-06-08",
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
                      notes="N131072 RapidStream SUBMODEL anchor (historical/near-mainline, shiftadd=%s); "
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

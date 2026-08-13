"""Report-only pre-generation resource assembler and renderer.

Assembles resource_model outputs (structural MR/DSP, calibrated D_MR, ICBU memory, layered DSP guardrail,
per-stage/per-group X-stage DSP) into a ResourceReport and renders a deterministic text report. NO file
I/O, NO generation coupling, NO apply / recommendation, NO config rewrite, NO X-stage split.

Unknown / unobserved D_MR coefficients are reported as explicit UNKNOWN rows (never interpolated, never a
guessed coefficient); strict=True re-raises resource_model.UnknownCoefficientError instead.

Barrett is a first-class / general route reported directly; shiftadd=reduce and shiftadd=mul are cost-model
AXES, never a Barrett replacement.
"""
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from dse import resource_model as rm

logger = logging.getLogger(__name__)

__all__ = ["ResourceReport", "build_report", "render_report", "REPORT_CAVEATS"]

# Fixed caveat footer (always rendered; deterministic order).
REPORT_CAVEATS = (
    "1228 is a CONFIGURABLE slot-capacity proxy, not universal truth.",
    "1410 = x_stages + TFR-X is a measured RapidStream group anchor, not a universal merge rule.",
    "1218 is outside visible repo artifacts; if from an external P&R/report layer it is out of scope.",
    "guardrail status is a placement/risk policy, not a mathematical failure.",
    "BU8 DSP profile (240,270,330,450) is a measured anchor, not a universal split result.",
    "BU32 (any BU!=8) MR counts are structural/formula output only unless calibrated; D_MR BU-invariance unproven.",
    "report-only: no automatic/physical split, no recommendation/apply, no generation change.",
)


@dataclass(frozen=True)
class ResourceReport:
    config: Dict[str, int]
    route: Dict[str, object]
    normalized_modes: Tuple[Tuple[str, str], ...]
    mr: Dict[str, object]
    dmr: Optional[rm.DmrCoefficient]                 # None when the coefficient is unknown (non-strict)
    dsp: Dict[str, Optional[int]]                    # values None when D_MR is unknown
    memory: Optional[rm.IcbuMemory]
    guards: Dict[str, rm.DspGuardResult]             # keys among {"module","group","device"}
    per_stage_dsp: Tuple[int, ...]
    group_dsp: Tuple[int, ...]
    group_guards: Tuple[rm.DspGuardResult, ...]
    unknown_rows: Tuple[Tuple[str, str], ...]        # (key, reason)


def build_report(N: int, BU: int, CH: int, K: int, route: str,
                 max_prime_bit_length: int, prime_count: int, *,
                 num_core: int = 1, recipe: str = "2U2B", tfs_model: str = rm.TFS_GROUPSHARE,
                 partition=None, tfr_x_dsp: int = 0, tfg_x_dsp: int = 0, extra_group_dsp: int = 0,
                 total_design_dsp: Optional[int] = None, normalized_modes=None,
                 slot_capacity_proxy: int = rm.SLOT_DSP_CAPACITY_PROXY_U55C,
                 device_guard_ratio: float = 0.80, strict: bool = False) -> ResourceReport:
    """Assemble a ResourceReport from resource_model estimators.

    D_MR is looked up ONCE; on UnknownCoefficientError, strict=True re-raises and strict=False records an
    UNKNOWN row and leaves the D_MR-dependent DSP/guard rows as None (never interpolated). Structural MR and
    ICBU memory (which need no D_MR) are always present. total_design_dsp is an explicit caller input for the
    device guard; tfr_x_dsp/tfg_x_dsp/extra_group_dsp feed the group estimate.
    """
    config = {
        "N": N, "BU": BU, "CH": CH, "K": K, "num_core": num_core,
        "num_l_stage": rm.num_l_stage(N, BU), "num_x_stage": rm.num_x_stage(BU),
        "bf_unit_instances": rm.estimate_bf_unit_instances(N, BU, num_core),
    }
    route_d = {"route": route, "max_prime_bit_length": max_prime_bit_length, "prime_count": prime_count}
    modes = tuple(normalized_modes or ())

    brk = rm.x_stage_mr_breakdown(BU, tfs_model)
    mr = {"butterfly": brk.butterfly_mr, "tfs": brk.tfs_mr, "total": brk.total_mr,
          "tfs_model": brk.tfs_model, "bf_unit_instances": config["bf_unit_instances"],
          "per_stage": brk.per_stage_mr}

    unknown_rows = []
    dmr = None
    mode = "single" if prime_count == 1 else "pair"
    try:
        dmr = rm.lookup_d_mr_entry(route, max_prime_bit_length, prime_count)
    except rm.UnknownCoefficientError as e:
        if strict:
            raise
        unknown_rows.append(("%s/%d/%s" % (route, max_prime_bit_length, mode), str(e)))

    dsp = {"x_stages": None, "group": None, "l_bf": None}
    per_stage_dsp: Tuple[int, ...] = ()
    group_dsp: Tuple[int, ...] = ()
    group_guards: Tuple[rm.DspGuardResult, ...] = ()
    guards: Dict[str, rm.DspGuardResult] = {}
    if dmr is not None:
        d_mr = dmr.value
        dsp["x_stages"] = mr["total"] * d_mr
        dsp["group"] = rm.estimate_rapidstream_group_dsp(dsp["x_stages"], tfr_x_dsp, tfg_x_dsp, extra_group_dsp)
        dsp["l_bf"] = config["bf_unit_instances"] * d_mr
        per_stage_dsp = rm.estimate_x_stage_per_stage_dsp(BU, d_mr, tfs_model)
        guards["module"] = rm.slot_dsp_guard(dsp["x_stages"], slot_capacity_proxy)
        guards["group"] = rm.slot_dsp_guard(dsp["group"], slot_capacity_proxy)
        if partition is not None:
            group_dsp = rm.estimate_x_stage_group_dsp(per_stage_dsp, partition)
            group_guards = rm.estimate_x_stage_partition_guard(per_stage_dsp, partition, slot_capacity_proxy)
    if total_design_dsp is not None:
        guards["device"] = rm.device_dsp_guard(total_design_dsp, guard_ratio=device_guard_ratio)

    memory = rm.estimate_icbu_memory(N, BU, CH, K, num_core, recipe)

    return ResourceReport(config=config, route=route_d, normalized_modes=modes, mr=mr, dmr=dmr,
                          dsp=dsp, memory=memory, guards=guards, per_stage_dsp=per_stage_dsp,
                          group_dsp=group_dsp, group_guards=group_guards,
                          unknown_rows=tuple(unknown_rows))


def _fmt_guard(g: rm.DspGuardResult) -> str:
    return "vs %s %d (gr=%s, thr=%d) -> %s" % (g.capacity_layer, g.capacity, g.guard_ratio, g.threshold, g.status)


def render_report(report: ResourceReport) -> str:
    """Render a ResourceReport to a deterministic, readable text report (pure string; no I/O)."""
    out = ["=== StreamNTT pre-gen resource report (report-only) ==="]
    c = report.config
    out.append("[config]   N=%d BU=%d CH=%d K=%d num_core=%d num_l_stage=%d num_x_stage=%d bf_unit_instances=%d"
               % (c["N"], c["BU"], c["CH"], c["K"], c["num_core"],
                  c["num_l_stage"], c["num_x_stage"], c["bf_unit_instances"]))
    r = report.route
    out.append("[route]    %s  max_prime_bit_length=%s  prime_count=%s"
               "  (Barrett first-class/general; shiftadd = cost axes)"
               % (r["route"], r["max_prime_bit_length"], r["prime_count"]))
    if report.normalized_modes:
        out.append("[modes]    " + "  ".join("%s=%s" % (k, v) for k, v in report.normalized_modes))
    m = report.mr
    out.append("[MR count] butterfly=%d tfs(%s)=%d total=%d bf_unit_instances=%d  per_stage=%s"
               % (m["butterfly"], m["tfs_model"], m["tfs"], m["total"], m["bf_unit_instances"], m["per_stage"]))
    if report.dmr is not None:
        d = report.dmr
        out.append("[D_MR]     %s/%d/%s = %d  [%s; %s]"
                   % (d.route, d.max_prime_bit_length, d.prime_mode, d.value, d.report_layer, d.provenance))
    else:
        out.append("[D_MR]     UNKNOWN (not calibrated; not interpolated) -- see [unknown]")

    def _v(x):
        return "UNKNOWN" if x is None else str(x)
    out.append("[DSP]      module x_stages=%s  group(x_stages+TFR-X)=%s  L-bf=%s"
               % (_v(report.dsp["x_stages"]), _v(report.dsp["group"]), _v(report.dsp["l_bf"])))
    if report.memory is not None:
        mem = report.memory
        out.append("[memory]   ICBU URAM=%d BRAM_18K=%d (%s; bram=%s; measured N1024/BU8/2U2B anchor)"
                   % (mem.uram, mem.bram_18k, mem.recipe, mem.bram_packing))
    if report.guards:
        out.append("[guard]")
        labels = {"module": "x_stages", "group": "group", "device": "total"}
        for key in ("module", "group", "device"):
            g = report.guards.get(key)
            if g is not None:
                out.append("           %-7s %s %d %s" % (key, labels[key], g.used, _fmt_guard(g)))
    if report.per_stage_dsp:
        out.append("[per-stage DSP]  %s" % (report.per_stage_dsp,))
    if report.group_dsp:
        out.append("[per-group DSP]")
        for i, gd in enumerate(report.group_dsp):
            status = report.group_guards[i].status if i < len(report.group_guards) else "n/a"
            out.append("           g%d dsp=%d slot=%s" % (i, gd, status))
    if report.unknown_rows:
        out.append("[unknown]")
        for key, reason in report.unknown_rows:
            out.append("           %s : %s" % (key, reason))
    out.append("[caveats]")
    for cav in REPORT_CAVEATS:
        out.append("           - " + cav)
    return "\n".join(out)

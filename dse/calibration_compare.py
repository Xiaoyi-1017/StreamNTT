"""Report-only calibration comparison for the pre-generation cost model.

Compares resource_model PREDICTIONS against calibration_data OBSERVED records, producing deterministic
ComparisonRecords and a text report. Dependency direction:
    calibration_compare -> calibration_data   (observed evidence)
    calibration_compare -> resource_model      (predictions)
(calibration_data must not import resource_model.) This module does not apply DSE decisions.

Layer discipline (never cross): each observed record is predicted with the resource_model quantity that
matches its report_layer + metric. The full RapidStream DESIGN total and the SLOT/guardrail proxy are NOT
value-predicted (the model has no full-design total, and the proxy is a capacity, not a module DSP). The
N131072 manifests calibrate the ICBU DSP/memory SUBMODEL only -- not full-design or P&R/Fmax validation.
Unknown coefficients -> UNKNOWN row (non-strict) or raise (strict). Nothing invented.

Barrett is a first-class / general route; shiftadd=reduce / shiftadd=mul are cost-model axes.
"""
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

from dse import calibration_data as cd
from dse import resource_model as rm

logger = logging.getLogger(__name__)

# status taxonomy
STATUS_EXACT = "EXACT"               # predicted == observed where exact comparison is valid
STATUS_CALIBRATED = "CALIBRATED"     # accepted match for a calibrated submodel/layer
STATUS_QUALITATIVE = "QUALITATIVE"   # only a risk/status classification is compared (no value equality)
STATUS_NO_OBSERVED = "NO_OBSERVED"   # observed value absent
STATUS_UNKNOWN = "UNKNOWN"           # model prediction unavailable (e.g. full-design total / unknown D_MR)
STATUS_MISMATCH = "MISMATCH"         # observed exists and predicted differs unexpectedly
STATUSES = (STATUS_EXACT, STATUS_CALIBRATED, STATUS_QUALITATIVE, STATUS_NO_OBSERVED,
            STATUS_UNKNOWN, STATUS_MISMATCH)

__all__ = [
    "STATUS_EXACT", "STATUS_CALIBRATED", "STATUS_QUALITATIVE", "STATUS_NO_OBSERVED",
    "STATUS_UNKNOWN", "STATUS_MISMATCH", "STATUSES",
    "ComparisonRecord", "compare", "render_comparison_report",
]


@dataclass(frozen=True)
class ComparisonRecord:
    case_id: str
    metric: str
    report_layer: str
    predicted_value: Optional[int]
    observed_value: Optional[int]
    delta: Optional[int]            # predicted - observed; None when not value-compared
    status: str
    provenance: str
    notes: str


def _predict(r) -> Tuple[Optional[int], str]:
    """Return (predicted_value or None, base_status) for an observed record, matching its layer + metric.
    Raises resource_model.UnknownCoefficientError if the D_MR coefficient is unobserved."""
    layer, metric = r.report_layer, r.metric
    if layer == cd.LAYER_CSYNTH_MODULE and metric == "DSP":
        return rm.estimate_x_stage_dsp(r.BU, r.route, r.actual_bit_length, r.num_primes, r.tfs_model), \
            STATUS_EXACT
    if layer == cd.LAYER_RAPIDSTREAM_GROUP and metric == "DSP":
        x = rm.estimate_x_stage_dsp(r.BU, r.route, r.actual_bit_length, r.num_primes, r.tfs_model)
        d_mr = rm.lookup_d_mr(r.route, r.actual_bit_length, r.num_primes)
        return rm.estimate_rapidstream_group_dsp(x, tfr_x_dsp=rm.num_x_stage(r.BU) * d_mr), STATUS_CALIBRATED
    if layer == cd.LAYER_RAPIDSTREAM_ICBU and metric == "DSP":
        return rm.estimate_l_stage_bf_dsp(r.N, r.BU, r.route, r.actual_bit_length, r.num_primes,
                                          num_core=1), STATUS_CALIBRATED
    if layer == cd.LAYER_RAPIDSTREAM_ICBU and metric in ("URAM", "BRAM_18K"):
        mem = rm.estimate_icbu_memory(r.N, r.BU, r.CH, r.K, 1, r.storage_recipe or "2U2B")
        val = mem.uram if metric == "URAM" else mem.bram_18k
        return val, (STATUS_EXACT if metric == "URAM" else STATUS_CALIBRATED)
    if layer == cd.LAYER_RAPIDSTREAM_DESIGN:
        return None, STATUS_UNKNOWN          # full-design total not modelled (ICBU + x_stages submodels only)
    if layer == cd.LAYER_GUARDRAIL_PROXY:
        return None, STATUS_QUALITATIVE      # capacity proxy, classified-against, not value-predicted
    return None, STATUS_UNKNOWN


def _note_for(r, base_status: str, extra: str = "") -> str:
    parts = [r.notes] if r.notes else []
    if base_status == STATUS_UNKNOWN and r.report_layer == cd.LAYER_RAPIDSTREAM_DESIGN:
        parts.append("full-design total NOT modelled (ICBU + x_stages submodels only); not full-design validation")
    if base_status == STATUS_QUALITATIVE and r.report_layer == cd.LAYER_GUARDRAIL_PROXY:
        parts.append("guardrail/slot capacity proxy; module/group DSP are CLASSIFIED against it (OVER), "
                     "NOT compared for value equality")
    if extra:
        parts.append(extra)
    return " | ".join(parts)


def compare(records, *, strict: bool = False) -> Tuple[ComparisonRecord, ...]:
    """Compare each observed record against the matching resource_model prediction. Deterministic (input
    order preserved). Unknown D_MR -> UNKNOWN row (non-strict) or raise (strict). No invented values."""
    out = []
    for r in records:
        if r.observed_value is None:
            out.append(ComparisonRecord(r.case_id, r.metric, r.report_layer, None, None, None,
                                        STATUS_NO_OBSERVED, r.provenance, _note_for(r, STATUS_NO_OBSERVED)))
            continue
        try:
            pred, base = _predict(r)
        except rm.UnknownCoefficientError as e:
            if strict:
                raise
            out.append(ComparisonRecord(r.case_id, r.metric, r.report_layer, None, r.observed_value, None,
                                        STATUS_UNKNOWN, r.provenance, _note_for(r, STATUS_UNKNOWN, str(e))))
            continue
        if pred is None:
            out.append(ComparisonRecord(r.case_id, r.metric, r.report_layer, None, r.observed_value, None,
                                        base, r.provenance, _note_for(r, base)))
            continue
        delta = pred - r.observed_value
        status = base if delta == 0 else STATUS_MISMATCH
        out.append(ComparisonRecord(r.case_id, r.metric, r.report_layer, pred, r.observed_value, delta,
                                    status, r.provenance, _note_for(r, status)))
    return tuple(out)


def render_comparison_report(comparisons) -> str:
    """Render comparisons to a deterministic, readable text report (pure string; no I/O)."""
    out = ["=== StreamNTT calibration comparison (report-only; model prediction vs observed evidence) ==="]
    counts = {s: sum(1 for c in comparisons if c.status == s) for s in STATUSES}
    out.append("[summary]  records=%d  " % len(comparisons)
               + "  ".join("%s=%d" % (s, counts[s]) for s in STATUSES))
    out.append("[rows]     case_id | metric | layer | predicted | observed | delta | status")

    def _v(x):
        return "-" if x is None else str(x)
    for c in comparisons:
        out.append("           %s | %s | %s | %s | %s | %s | %s"
                   % (c.case_id, c.metric, c.report_layer, _v(c.predicted_value),
                      _v(c.observed_value), _v(c.delta), c.status))
    out.append("[layers]   DSP layer-stable (csynth==RapidStream); URAM/BRAM RapidStream-calibrated; "
               "LUT/FF csynth-layer only (excluded); P&R not in repo")
    out.append("[caveats]  N131072 = shiftadd=mul submodel evidence (ICBU DSP/memory), not full-design or "
               "P&R/Fmax validation; guardrail status = placement/risk policy, not a math failure; "
               "Barrett first-class, shiftadd = cost axes")
    return "\n".join(out)

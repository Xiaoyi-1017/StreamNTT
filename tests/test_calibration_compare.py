#!/usr/bin/env python3
"""Tests for calibration_compare (Phase 4.0 Slice 4b: predicted-vs-observed comparison layer).

Run (no pytest needed):  ~/autontt-venv/bin/python3 tests/test_calibration_compare.py
Exit 0 = all pass. Compares resource_model predictions against calibration_data observed records."""
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
from dse import calibration_compare as cc
from dse import calibration_data as cd
from dse import resource_model as rm


def _expect_raises(fn, exc=Exception):
    try:
        fn()
    except exc:
        return
    raise AssertionError("expected %s, none raised" % getattr(exc, "__name__", exc))


def _cmp_for(comparisons, observed_value, metric="DSP"):
    hits = [c for c in comparisons if c.observed_value == observed_value and c.metric == metric]
    assert len(hits) == 1, "expected 1 comparison for observed=%r metric=%r, got %d" % (
        observed_value, metric, len(hits))
    return hits[0]


def _icbu_n131072_record(observed=1144, route="shiftadd_mul", count=2):
    return cd.ObservedRecord(
        case_id="fx_n131072", N=131072, BU=8, CH=4, K=62, route=route, actual_bit_length=62,
        num_primes=count, tfs_model=None, storage_recipe="2U2B",
        report_layer=cd.LAYER_RAPIDSTREAM_ICBU, metric="DSP", observed_value=observed,
        provenance="fixture", confidence="measured",
        notes="N131072 RapidStream SUBMODEL anchor (historical/near-mainline, shiftadd=mul); ICBU only")


# --- dependency direction ----------------------------------------------------

def test_compare_imports_data_and_model_not_generate_code():
    src = open(os.path.join(_REPO, "dse", "calibration_compare.py")).read()  # moved 2026-06-10
    assert "import calibration_data" in src
    assert "import resource_model" in src
    assert "import generate_code" not in src        # no generate_code COUPLING (docstring may mention it)


def test_calibration_data_still_does_not_import_resource_model():
    src = open(os.path.join(_REPO, "dse", "calibration_data.py")).read()  # moved 2026-06-10
    # the only allowed mention is the value-equality comment, never an import
    assert "import resource_model" not in src


# --- exact / calibrated anchor comparisons -----------------------------------

def test_module_989_barrett61_exact_delta0():
    c = _cmp_for(cc.compare(cd.curated_n1024_records()), 989)
    assert c.predicted_value == 989 and c.delta == 0 and c.status == cc.STATUS_EXACT
    assert c.report_layer == cd.LAYER_CSYNTH_MODULE


def test_module_1290_barrett62_exact_delta0():
    c = _cmp_for(cc.compare(cd.curated_n1024_records()), 1290)
    assert c.predicted_value == 1290 and c.delta == 0 and c.status == cc.STATUS_EXACT


def test_module_1127_idskip_exact_delta0():
    c = _cmp_for(cc.compare(cd.curated_n1024_records()), 1127)
    assert c.predicted_value == 1127 and c.delta == 0 and c.status == cc.STATUS_EXACT


def test_group_1410_calibrated_delta0():
    c = _cmp_for(cc.compare(cd.curated_n1024_records()), 1410)
    assert c.predicted_value == 1410 and c.delta == 0
    assert c.status == cc.STATUS_CALIBRATED and c.report_layer == cd.LAYER_RAPIDSTREAM_GROUP


def test_memory_uram96_bram192_delta0():
    comps = cc.compare(cd.curated_n1024_records())
    u = _cmp_for(comps, 96, metric="URAM")
    b = _cmp_for(comps, 192, metric="BRAM_18K")
    assert u.delta == 0 and u.status in (cc.STATUS_EXACT, cc.STATUS_CALIBRATED)
    assert b.delta == 0 and b.status in (cc.STATUS_EXACT, cc.STATUS_CALIBRATED)


# --- guardrail proxy is QUALITATIVE, not a value-equality ---------------------

def test_guardrail_1228_qualitative_not_value_equality():
    c = _cmp_for(cc.compare(cd.curated_n1024_records()), 1228)
    assert c.report_layer == cd.LAYER_GUARDRAIL_PROXY
    assert c.status == cc.STATUS_QUALITATIVE
    assert c.predicted_value is None      # NOT 1290; never compared as module-vs-observed equality
    assert c.delta is None
    assert c.status != cc.STATUS_MISMATCH


# --- N131072 = submodel, not full-design validation --------------------------

def test_n131072_icbu_dsp_calibrated_submodel():
    c = cc.compare([_icbu_n131072_record(observed=1144)])[0]
    assert c.predicted_value == 1144 and c.delta == 0      # 104 * 11
    assert c.status == cc.STATUS_CALIBRATED
    assert "submodel" in c.notes.lower() and "shiftadd=mul" in c.notes.lower()


def test_n131072_design_dsp_unknown_not_full_validation():
    rec = cd.ObservedRecord(case_id="fx_design", N=131072, BU=8, CH=4, K=62, route="shiftadd_mul",
                            actual_bit_length=62, num_primes=2, tfs_model=None, storage_recipe="2U2B",
                            report_layer=cd.LAYER_RAPIDSTREAM_DESIGN, metric="DSP", observed_value=1969,
                            provenance="fixture", confidence="measured", notes="design total")
    c = cc.compare([rec])[0]
    assert c.status == cc.STATUS_UNKNOWN          # full-design total not modelled (submodel only)
    assert c.predicted_value is None and c.delta is None


# --- unknown coefficient -----------------------------------------------------

def _unknown_record():
    return cd.ObservedRecord(case_id="fx_unknown", N=1024, BU=8, CH=4, K=62, route="barrett",
                             actual_bit_length=61, num_primes=1, tfs_model="groupshare",
                             storage_recipe="2U2B", report_layer=cd.LAYER_CSYNTH_MODULE, metric="DSP",
                             observed_value=999, provenance="fixture", confidence="measured", notes="")


def test_unknown_coefficient_nonstrict_unknown_row():
    c = cc.compare([_unknown_record()], strict=False)[0]
    assert c.status == cc.STATUS_UNKNOWN
    assert c.predicted_value is None and c.delta is None


def test_unknown_coefficient_strict_raises():
    _expect_raises(lambda: cc.compare([_unknown_record()], strict=True), rm.UnknownCoefficientError)


# --- mismatch detection ------------------------------------------------------

def test_mismatch_when_observed_differs():
    rec = cd.ObservedRecord(case_id="fx_mismatch", N=1024, BU=8, CH=4, K=62, route="barrett",
                            actual_bit_length=62, num_primes=2, tfs_model="groupshare",
                            storage_recipe="2U2B", report_layer=cd.LAYER_CSYNTH_MODULE, metric="DSP",
                            observed_value=9999, provenance="fixture", confidence="measured", notes="")
    c = cc.compare([rec])[0]
    assert c.predicted_value == 1290 and c.delta == 1290 - 9999
    assert c.status == cc.STATUS_MISMATCH


# --- rendering ---------------------------------------------------------------

def test_render_comparison_report_deterministic():
    comps = cc.compare(cd.curated_n1024_records())
    assert cc.render_comparison_report(comps) == cc.render_comparison_report(comps)


def test_render_comparison_report_has_headers_and_statuses():
    s = cc.render_comparison_report(cc.compare(cd.curated_n1024_records()))
    for token in ["calibration comparison", "EXACT", "CALIBRATED", "QUALITATIVE", "status"]:
        assert token in s, "missing %r" % token


def test_no_pnr_comparisons_fabricated():
    comps = cc.compare(cd.curated_n1024_records())
    assert all(c.report_layer != cd.LAYER_PNR for c in comps)


# --- Phase 4.1 single-prime N131072 anchors: module EXACT, ICBU CALIBRATED ---
# Scoped to curated_n131072_single_prime_records() (4 unique values); do NOT compare over all_records()
# here -- 473 (also N1024) and 1144 (also a real shiftadd_mul 2U2B manifest, if present) repeat there.

def test_n131072_barrett_xstages_1118_exact():
    c = _cmp_for(cc.compare(cd.curated_n131072_single_prime_records()), 1118)
    assert c.predicted_value == 1118 and c.delta == 0            # 43 * 26
    assert c.status == cc.STATUS_EXACT and c.report_layer == cd.LAYER_CSYNTH_MODULE


def test_n131072_reduce_xstages_473_exact():
    c = _cmp_for(cc.compare(cd.curated_n131072_single_prime_records()), 473)
    assert c.predicted_value == 473 and c.delta == 0             # 43 * 11
    assert c.status == cc.STATUS_EXACT and c.report_layer == cd.LAYER_CSYNTH_MODULE


def test_n131072_barrett_icbu_2704_calibrated():
    c = _cmp_for(cc.compare(cd.curated_n131072_single_prime_records()), 2704)
    assert c.predicted_value == 2704 and c.delta == 0            # 104 * 26
    assert c.status == cc.STATUS_CALIBRATED and c.report_layer == cd.LAYER_RAPIDSTREAM_ICBU


def test_n131072_reduce_icbu_1144_calibrated():
    c = _cmp_for(cc.compare(cd.curated_n131072_single_prime_records()), 1144)
    assert c.predicted_value == 1144 and c.delta == 0            # 104 * 11
    assert c.status == cc.STATUS_CALIBRATED and c.report_layer == cd.LAYER_RAPIDSTREAM_ICBU


def test_n131072_single_prime_set_has_no_mismatch():
    comps = cc.compare(cd.curated_n131072_single_prime_records())
    assert len(comps) == 4
    assert all(c.status != cc.STATUS_MISMATCH for c in comps)
    assert all(c.delta == 0 for c in comps if c.delta is not None)
    assert {c.status for c in comps} == {cc.STATUS_EXACT, cc.STATUS_CALIBRATED}


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print("PASS", t.__name__)
        except AssertionError as e:
            failed += 1
            print("FAIL", t.__name__, "--", e)
        except Exception as e:
            failed += 1
            print("ERROR", t.__name__, "--", type(e).__name__, e)
    print("\n%d/%d passed" % (len(tests) - failed, len(tests)))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

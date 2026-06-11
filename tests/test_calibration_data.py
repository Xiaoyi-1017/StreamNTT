#!/usr/bin/env python3
"""Tests for calibration_data (Phase 4.0 Slice 4a observed-evidence data layer).

Run (no pytest needed):  ~/autontt-venv/bin/python3 tests/test_calibration_data.py
Exit 0 = all pass. Clean-checkout-safe: N131072 manifest loading is exercised via /tmp fixtures
(the real Checkpoint/dse_calibration manifests are gitignored and only smoke-checked if present)."""
import json
import os
import shutil
import sys
import tempfile

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
from dse import calibration_data as cd
from dse import icbu_dse


def _one(records, **match):
    """Return the single record matching all kwargs; assert exactly one."""
    hits = [r for r in records if all(getattr(r, k) == v for k, v in match.items())]
    assert len(hits) == 1, "expected 1 record for %r, got %d" % (match, len(hits))
    return hits[0]


def _write_manifest(root, name, *, shiftadd="mul", recipe="2U2B", count=2,
                    values=None, icbu_dsp=1144, icbu_uram=208, icbu_bram=2912, design_dsp=1969):
    d = os.path.join(root, name)
    os.makedirs(d, exist_ok=True)
    man = {
        "N": 131072, "BU": 8, "CH": 4, "K": 62,
        "prime_set": {"count": count, "mode": "multi", "values": values or [2305843009224179713]},
        "shiftadd": shiftadd, "target_clock": "3.33ns", "source_phase": "fixture",
        "included_reports": [], "storage_recipe": recipe,
        "measured_rapidstream_icbu_x104": {"DSP": icbu_dsp, "URAM": icbu_uram, "BRAM_18K": icbu_bram,
                                           "LUT": 1, "FF": 1},
        "measured_rapidstream_design": {"DSP": design_dsp, "URAM": 1, "BRAM_18K": 1, "LUT": 1, "FF": 1},
    }
    with open(os.path.join(d, "manifest.json"), "w") as f:
        json.dump(man, f)
    return d


# --- ObservedRecord + curated N1024 anchors ----------------------------------

def test_observed_record_has_required_fields():
    r = cd.curated_n1024_records()[0]
    for field in ("case_id", "N", "BU", "CH", "K", "route", "actual_bit_length", "num_primes",
                  "tfs_model", "storage_recipe", "report_layer", "metric", "observed_value",
                  "provenance", "confidence", "notes"):
        assert hasattr(r, field), "missing field %r" % field


def test_curated_989_barrett_61_groupshare_csynth_module():
    r = _one(cd.curated_n1024_records(), observed_value=989, metric="DSP")
    assert r.report_layer == cd.LAYER_CSYNTH_MODULE
    assert r.route == "barrett" and r.actual_bit_length == 61 and r.num_primes == 2
    assert r.tfs_model == "groupshare"
    assert r.provenance


def test_curated_1290_barrett_62_groupshare_present():
    r = _one(cd.curated_n1024_records(), observed_value=1290, metric="DSP",
             report_layer=cd.LAYER_CSYNTH_MODULE)
    assert r.actual_bit_length == 62 and r.num_primes == 2 and r.tfs_model == "groupshare"


def test_curated_1127_idskip_present():
    r = _one(cd.curated_n1024_records(), observed_value=1127, metric="DSP")
    assert r.tfs_model == "idskip" and r.report_layer == cd.LAYER_CSYNTH_MODULE
    assert r.actual_bit_length == 61      # idskip uses the SAME 61-bit pair as the 989 groupshare case (49x23)


def test_curated_group_1410_tagged_rapidstream_group_not_module():
    r = _one(cd.curated_n1024_records(), observed_value=1410, metric="DSP")
    assert r.report_layer == cd.LAYER_RAPIDSTREAM_GROUP
    assert r.report_layer != cd.LAYER_CSYNTH_MODULE


def test_curated_slot_1228_tagged_guardrail_proxy_not_observed_dsp():
    r = _one(cd.curated_n1024_records(), observed_value=1228)
    assert r.report_layer == cd.LAYER_GUARDRAIL_PROXY
    assert r.report_layer not in (cd.LAYER_CSYNTH_MODULE, cd.LAYER_RAPIDSTREAM_GROUP)


def test_curated_memory_uram96_bram192_rapidstream_icbu():
    recs = cd.curated_n1024_records()
    u = _one(recs, metric="URAM", observed_value=96)
    b = _one(recs, metric="BRAM_18K", observed_value=192)
    assert u.report_layer == cd.LAYER_RAPIDSTREAM_ICBU and u.storage_recipe == "2U2B"
    assert b.report_layer == cd.LAYER_RAPIDSTREAM_ICBU


# --- N131072 manifest loader (reuses icbu_dse.load_calibration) ---------------

def test_load_n131072_from_fixture_icbu_dsp_record():
    root = tempfile.mkdtemp()
    try:
        _write_manifest(root, "case_2U2B", icbu_dsp=1144, design_dsp=1969)
        recs = cd.load_n131072_records(root)
        r = _one(recs, report_layer=cd.LAYER_RAPIDSTREAM_ICBU, metric="DSP")
        assert r.observed_value == 1144
        assert r.route == "shiftadd_mul" and r.storage_recipe == "2U2B" and r.N == 131072
    finally:
        shutil.rmtree(root)


def test_load_n131072_design_dsp_layer():
    root = tempfile.mkdtemp()
    try:
        _write_manifest(root, "case_2U2B", design_dsp=1969)
        recs = cd.load_n131072_records(root)
        r = _one(recs, report_layer=cd.LAYER_RAPIDSTREAM_DESIGN, metric="DSP")
        assert r.observed_value == 1969
    finally:
        shutil.rmtree(root)


def test_load_n131072_value_matches_icbu_dse_load_calibration():
    # proves reuse (not duplicate parsing): record value == icbu_dse.load_calibration manifest value
    root = tempfile.mkdtemp()
    try:
        case = _write_manifest(root, "case_2U2B", icbu_dsp=777)
        ref = icbu_dse.load_calibration(case)["manifest"]["measured_rapidstream_icbu_x104"]["DSP"]
        rec = _one(cd.load_n131072_records(root), report_layer=cd.LAYER_RAPIDSTREAM_ICBU, metric="DSP")
        assert rec.observed_value == ref == 777
    finally:
        shutil.rmtree(root)


def test_load_n131072_missing_root_returns_empty():
    assert cd.load_n131072_records("/no/such/calibration/root") == ()


def test_no_pnr_records_created():
    root = tempfile.mkdtemp()
    try:
        _write_manifest(root, "case_2U2B")
        allr = cd.curated_n1024_records() + cd.load_n131072_records(root)
        assert all(r.report_layer != cd.LAYER_PNR for r in allr)
    finally:
        shutil.rmtree(root)


def test_records_deterministic_order():
    root = tempfile.mkdtemp()
    try:
        _write_manifest(root, "b_case")
        _write_manifest(root, "a_case")
        assert cd.load_n131072_records(root) == cd.load_n131072_records(root)
    finally:
        shutil.rmtree(root)


def test_real_manifests_bu8_2u2b_icbu_dsp_1144_if_present():
    root = os.path.join(_REPO, "Checkpoint", "dse_calibration")
    if not os.path.isdir(root):
        print("  (skip: real calibration manifests not present)")
        return
    recs = cd.load_n131072_records(root)
    hits = [r for r in recs if r.report_layer == cd.LAYER_RAPIDSTREAM_ICBU and r.metric == "DSP"
            and r.BU == 8 and r.storage_recipe == "2U2B" and r.observed_value == 1144]
    assert hits, "expected a BU8/2U2B ICBU DSP=1144 record from the real manifests"


# --- Phase 4.1 single-prime N131072 anchors (curated; D_MR-safe) -------------

def test_curated_n131072_barrett_1118_xstages_csynth_module_dmr_safe():
    r = _one(cd.curated_n131072_single_prime_records(), observed_value=1118, metric="DSP")
    assert r.report_layer == cd.LAYER_CSYNTH_MODULE
    assert r.route == "barrett" and r.actual_bit_length == 62 and r.num_primes == 1
    assert r.N == 131072 and r.BU == 8 and r.CH == 4 and r.K == 62
    assert r.tfs_model == "groupshare" and r.storage_recipe == "3U1B"
    assert r.provenance and "26" in r.notes          # per-instance bf_unit DSP = 26 (agrees with delta=26)


def test_curated_n131072_reduce_473_xstages_csynth_module():
    r = _one(cd.curated_n131072_single_prime_records(), observed_value=473, metric="DSP")
    assert r.report_layer == cd.LAYER_CSYNTH_MODULE
    assert r.route == "shiftadd_reduce" and r.actual_bit_length == 62 and r.num_primes == 1
    assert r.N == 131072 and r.tfs_model == "groupshare" and r.storage_recipe == "3U1B"
    assert r.provenance and "11" in r.notes          # per-instance bf_unit DSP = 11 (agrees with delta=11)


def test_curated_n131072_barrett_icbu_2704_rapidstream_icbu():
    r = _one(cd.curated_n131072_single_prime_records(), observed_value=2704, metric="DSP")
    assert r.report_layer == cd.LAYER_RAPIDSTREAM_ICBU
    assert r.route == "barrett" and r.num_primes == 1 and r.N == 131072 and r.BU == 8
    assert r.provenance and "104" in r.notes and "26" in r.notes      # 104 x 26


def test_curated_n131072_reduce_icbu_1144_rapidstream_icbu():
    r = _one(cd.curated_n131072_single_prime_records(), observed_value=1144, metric="DSP")
    assert r.report_layer == cd.LAYER_RAPIDSTREAM_ICBU
    assert r.route == "shiftadd_reduce" and r.num_primes == 1 and r.storage_recipe == "3U1B"
    assert r.provenance and "104" in r.notes and "11" in r.notes      # 104 x 11


def test_curated_n131072_set_is_exactly_the_four_dmr_safe_anchors():
    recs = cd.curated_n131072_single_prime_records()
    assert sorted(r.observed_value for r in recs) == [473, 1118, 1144, 2704]
    for r in recs:
        assert r.metric == "DSP" and r.num_primes == 1
        assert r.report_layer in (cd.LAYER_CSYNTH_MODULE, cd.LAYER_RAPIDSTREAM_ICBU)
        assert r.report_layer not in (cd.LAYER_RAPIDSTREAM_GROUP, cd.LAYER_RAPIDSTREAM_DESIGN,
                                      cd.LAYER_GUARDRAIL_PROXY, cd.LAYER_PNR)


def test_curated_n131072_forbidden_group_and_design_values_absent():
    # 1218 (post-synth placement group), 4368 (Barrett design total), 1903 (reduce design total) are NOT
    # D_MR/module anchors and must never be recorded as calibration records.
    allr = cd.all_records("/no/such/calibration/root")    # () from loader -> isolates the curated sets
    forbidden = {1218, 4368, 1903}
    hits = [(r.case_id, r.observed_value) for r in allr if r.observed_value in forbidden]
    assert not hits, "forbidden non-D_MR values leaked into records: %r" % hits


def test_all_records_includes_n131072_single_prime():
    allr = cd.all_records("/no/such/calibration/root")    # () from loader -> isolates the curated sets
    cases = {r.case_id for r in allr}
    assert any("barrett_62_single_3U1B_xstages" in c for c in cases)
    assert any("shiftadd_reduce_62_single_3U1B_icbu" in c for c in cases)
    assert len(cd.curated_n1024_records()) + 4 == len(allr)    # exactly the 4 new curated records added


def test_calibration_data_has_no_model_or_generate_code_coupling():
    src = open(os.path.join(_REPO, "dse", "calibration_data.py")).read()  # moved 2026-06-10
    assert "import resource_model" not in src          # data layer must NOT import the model layer
    assert "import generate_code" not in src           # no generator coupling


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

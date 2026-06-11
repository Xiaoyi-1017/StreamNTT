#!/usr/bin/env python3
"""T0 unit tests for icbu_dse (Phase 3.3.4 ICBU bind_storage minor-DSE).

Run (no pytest needed):  ~/autontt-venv/bin/python3 tests/test_icbu_dse.py
Exit 0 = all pass; non-zero = a failure (prints the offending test)."""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dse import icbu_dse as dse


def _expect_raises(fn, needle=None):
    try:
        fn()
    except (ValueError, NotImplementedError) as e:
        if needle is not None:
            assert needle in str(e), "expected %r in error, got %r" % (needle, str(e))
        return
    raise AssertionError("expected an exception, none raised")


def test_closed_form_n1024_bu8():
    p = dse.arch_quantities(N=1024, BU=8, CH=4, K=62, NUM_CORE=1)
    assert p.num_l_stage == 6
    assert p.bf_unit_instances == 48          # 6*8*1
    assert p.bank_depth == 32                  # 1024/(4*8)
    assert p.icbu_bit_volume == 48 * 4 * 32 * 62


def test_packing_k62_depth32():
    assert dse.uram_per_bank(62, 32) == 1
    assert dse.bram18k_per_bank(62, 32) == 2


def test_icbu_self_validation_2u2b():
    p = dse.arch_quantities(N=1024, BU=8, CH=4, K=62, NUM_CORE=1)
    est = dse.estimate_icbu(p, "2U2B")
    assert est["URAM"] == 96                    # measured (design URAM)
    assert est["BRAM_18K"] == 192               # measured (ICBU BRAM portion)
    assert est["DSP"] == 48 * 30


def test_icbu_4u_4b():
    p = dse.arch_quantities(N=1024, BU=8, CH=4, K=62, NUM_CORE=1)
    assert dse.estimate_icbu(p, "4U")["URAM"] == 192
    assert dse.estimate_icbu(p, "4B")["BRAM_18K"] == 384


def test_icbu_3u1b_packing():
    p = dse.arch_quantities(N=1024, BU=8, CH=4, K=62, NUM_CORE=1)
    est = dse.estimate_icbu(p, "3U1B")
    assert est["URAM"] == 3 * 48                # 3 URAM banks * 1 * 48
    assert est["BRAM_18K"] == 1 * 2 * 48        # 1 BRAM bank * 2 * 48
    assert est["LUT"] is None and est["FF"] is None   # not yet calibrated


def test_guard_status_thresholds():
    assert dse.guard_status(96, "URAM", dse.BOARD_U55C, 0.80) == "UNDER"
    assert dse.guard_status(800, "URAM", dse.BOARD_U55C, 0.80) == "OVER"   # 800 >= 768


def test_recommend_d1_lower_bound_under_without_background_is_informational():
    p = dse.arch_quantities(N=1024, BU=8, CH=4, K=62, NUM_CORE=1)
    rec = dse.recommend(p, background=None, board=dse.BOARD_U55C, guard=0.80, configured="2U2B")
    assert rec.calibrated is False
    assert "lower bound under guard" in rec.summary and "background not calibrated" in rec.summary
    assert "informational" in rec.summary and "RECOMMEND" not in rec.summary


def test_recommend_d3_when_only_noncalibrated_fit():
    p = dse.arch_quantities(N=1024, BU=8, CH=4, K=62, NUM_CORE=1)
    # URAM background 700: both calibrated candidates (2U2B 96+700, 3U1B 144+700) >= 768 -> no STRONG;
    # only closed-form/exploratory (4B/4L) fit -> D3 informational, never "RECOMMEND 4B/4L".
    bg = {"URAM": 700, "BRAM_18K": 0, "DSP": 0, "LUT": 0, "FF": 0}
    rec = dse.recommend(p, background=bg, board=dse.BOARD_U55C, guard=0.80, configured="2U2B")
    assert "no single calibrated bind_storage recipe" in rec.summary
    assert "RECOMMEND" not in rec.summary and "deferred" in rec.summary


_BG_SEG = {"BRAM_18K": 228, "URAM": 256, "DSP": 825, "LUT": 155987, "FF": 203047}      # seg_phase1 measured
_BG_BU16 = {"BRAM_18K": 264, "URAM": 512, "DSP": 1449, "LUT": 219563, "FF": 242129}    # ti9 BU16 measured
_BG_BU32 = {"BRAM_18K": 14846, "URAM": 0, "DSP": 3016, "LUT": 663719, "FF": 929126}    # ti9 BU32 measured


def test_recommend_d2_bu8_archetype_2u2b_to_3u1b():
    p = dse.arch_quantities(N=131072, BU=8, CH=4, K=62, NUM_CORE=1)
    rec = dse.recommend(p, background=_BG_SEG, board=dse.BOARD_U55C, guard=0.80, configured="2U2B")
    assert "RECOMMEND 3U1B" in rec.summary
    assert "class calibrated" in rec.summary and "confidence calibrated historical" in rec.summary
    assert "BRAM_18K" in rec.summary                      # reason: configured over on BRAM


def test_recommend_d1_no_action_calibrated_under():
    p = dse.arch_quantities(N=131072, BU=8, CH=4, K=62, NUM_CORE=1)
    rec = dse.recommend(p, background=_BG_SEG, board=dse.BOARD_U55C, guard=0.80, configured="3U1B")
    assert "no action" in rec.summary and "calibrated" in rec.summary


def test_recommend_d3_bu16_both_over_no_strong_no_4l():
    p = dse.arch_quantities(N=131072, BU=16, CH=8, K=62, NUM_CORE=1)
    rec = dse.recommend(p, background=_BG_BU16, board=dse.BOARD_U55C, guard=0.80, configured="2U2B")
    assert "no single calibrated bind_storage recipe" in rec.summary
    assert "RECOMMEND" not in rec.summary
    assert "R-hetero" in rec.summary and "explicit validation" in rec.summary


def test_recommend_d4_bu32_background_alone_over():
    p = dse.arch_quantities(N=131072, BU=32, CH=16, K=62, NUM_CORE=1)
    rec = dse.recommend(p, background=_BG_BU32, board=dse.BOARD_U55C, guard=0.80, configured="2U2B")
    assert "not solvable by ICBU bind_storage" in rec.summary
    assert "suppressed" in rec.summary and "RECOMMEND" not in rec.summary


def test_recommend_d0_missing_background_over_lower_bound_informational():
    p = dse.arch_quantities(N=131072, BU=8, CH=4, K=62, NUM_CORE=1)   # 2U2B ICBU-only BRAM 2912/3552 > 0.8
    rec = dse.recommend(p, background=None, board=dse.BOARD_U55C, guard=0.80, configured="2U2B")
    assert "informational" in rec.summary and "LOWER BOUND" in rec.summary
    assert "no strong recommendation" in rec.summary and "RECOMMEND" not in rec.summary
    assert all(r["class"] == dse.RECIPE_CLASS[r["recipe"]] for r in rec.rows)


def test_load_calibration_missing_field():
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "manifest.json"), "w") as f:
        json.dump({"N": 131072, "BU": 8, "CH": 4}, f)   # missing K, prime_set, ...
    _expect_raises(lambda: dse.load_calibration(d), "K")


def _mk_case(root, name, tag, recipe, count=1, mode="single-prime cyclic", bg=None):
    d = os.path.join(root, name)
    os.makedirs(d)
    man = {"N": 131072, "BU": 8, "CH": 4, "K": 62, "NUM_CORE": 1,
           "prime_set": {"count": count, "mode": mode, "values": [3]},
           "shiftadd": "mul", "target_clock": "3.33ns", "source_phase": "x", "included_reports": [],
           "source_phase_tag": tag}
    if recipe is not None:
        man["storage_recipe"] = recipe
    if bg is not None:
        man["background_design_minus_icbu"] = bg
    with open(os.path.join(d, "manifest.json"), "w") as f:
        json.dump(man, f)


_BG = {"BRAM_18K": 1, "URAM": 2, "DSP": 3, "LUT": 4, "FF": 5}


def _bgkey(tag, recipe=None):
    bg = dse.BackgroundKey(131072, 8, 4, 62, tag, "single", 1, "mul")
    return bg if recipe is None else dse.RecipeKey(bg, recipe)


def test_keyed_recipe_same_nbuchk_different_recipe_resolve_distinct():
    root = tempfile.mkdtemp()
    _mk_case(root, "a_2u2b", "seg_phase1", "2U2B")
    _mk_case(root, "b_3u1b", "seg_phase1", "3U1B")
    k = dse.BackgroundKey(131072, 8, 4, 62, "seg_phase1", "single", 1, "mul")
    r2 = dse.lookup_recipe_keyed(dse.RecipeKey(k, "2U2B"), root)
    r3 = dse.lookup_recipe_keyed(dse.RecipeKey(k, "3U1B"), root)
    assert r2 and r3 and r2["case_dir"] != r3["case_dir"]


def test_keyed_background_no_mix_across_source_phase_tag():
    root = tempfile.mkdtemp()
    _mk_case(root, "segp", "seg_phase1", "2U2B", bg={"BRAM_18K": 228, "URAM": 256, "DSP": 825, "LUT": 1, "FF": 1})
    _mk_case(root, "ti9", "seg_single_prime", "2U2B", bg={"BRAM_18K": 142, "URAM": 256, "DSP": 759, "LUT": 2, "FF": 2})
    a = dse.lookup_background_keyed(_bgkey("seg_phase1"), root)
    b = dse.lookup_background_keyed(_bgkey("seg_single_prime"), root)
    assert a["background"]["BRAM_18K"] == 228 and b["background"]["BRAM_18K"] == 142


def test_keyed_ambiguous_duplicates_fail_fast():
    root = tempfile.mkdtemp()
    _mk_case(root, "dup1", "seg_phase1", "2U2B")
    _mk_case(root, "dup2", "seg_phase1", "2U2B")
    _expect_raises(lambda: dse.lookup_recipe_keyed(_bgkey("seg_phase1", "2U2B"), root), "ambiguous")
    _expect_raises(lambda: dse.lookup_background_keyed(_bgkey("seg_phase1"), root), "candidates")


def test_recipe_default_rule_missing_means_2u2b_only():
    root = tempfile.mkdtemp()
    _mk_case(root, "norec", "seg_phase1", None)        # storage_recipe absent -> 2U2B ONLY
    assert dse.lookup_recipe_keyed(_bgkey("seg_phase1", "2U2B"), root) is not None
    assert dse.lookup_recipe_keyed(_bgkey("seg_phase1", "3U1B"), root) is None


def test_coarse_lookup_ambiguity_returns_none_no_first_match():
    root = tempfile.mkdtemp()
    _mk_case(root, "a", "seg_phase1", "2U2B")
    _mk_case(root, "b", "seg_phase1", "3U1B")
    p = dse.arch_quantities(N=131072, BU=8, CH=4, K=62, NUM_CORE=1)
    assert dse.lookup_calibration(p, root) is None     # ambiguous -> None, never first match


def test_bram_bank_calibrated_values_and_fallback():
    assert dse.bram18k_per_bank_calibrated(62, 4096) == (14, "calibrated")
    assert dse.bram18k_per_bank_calibrated(62, 2048) == (7, "calibrated")
    assert dse.bram18k_per_bank_calibrated(62, 1024) == (4, "calibrated")
    v, tag = dse.bram18k_per_bank_calibrated(62, 32)                       # outside domain (N1024)
    assert (v, tag) == (2, "uncalibrated/conservative")
    assert dse.bram18k_per_bank_calibrated(52, 4096)[1] == "uncalibrated/conservative"   # K not calibrated
    assert dse.bram18k_per_bank_calibrated(62, 4096, NUM_CORE=2)[1] == "uncalibrated/conservative"


def test_estimate_icbu_uses_calibrated_packing_at_large_n():
    for BU, exp in ((8, 2912), (16, 2688), (32, 2816)):     # 2U2B = 2 B banks * calib/bank * inst
        p = dse.arch_quantities(N=131072, BU=BU, CH=4, K=62, NUM_CORE=1)
        est = dse.estimate_icbu(p, "2U2B")
        assert est["BRAM_18K"] == exp and est["packing"] == "calibrated"
        assert est["URAM"] == 2 * p.bf_unit_instances


def test_curated_layout_replica_keys():
    """Replicates the curated 5-case layout in a TEMP root (no dependency on gitignored Checkpoint/)."""
    root = tempfile.mkdtemp()
    _mk_case(root, "seg_2U2B", "seg_phase1", "2U2B", count=2, mode="multi-prime cyclic",
             bg={"BRAM_18K": 228, "URAM": 256, "DSP": 825, "LUT": 155987, "FF": 203047})
    _mk_case(root, "seg_3U1B", "seg_phase1", "3U1B", count=2, mode="multi-prime cyclic",
             bg={"BRAM_18K": 228, "URAM": 256, "DSP": 825, "LUT": 155987, "FF": 203047})
    _mk_case(root, "ti9_2U2B", "seg_single_prime", "2U2B",
             bg={"BRAM_18K": 142, "URAM": 256, "DSP": 759, "LUT": 98677, "FF": 103904})
    segbg = dse.BackgroundKey(131072, 8, 4, 62, "seg_phase1", "multi", 2, "mul")
    _expect_raises(lambda: dse.lookup_background_keyed(segbg, root), "ambiguous")   # 2U2B+3U1B share bg key
    r3 = dse.lookup_recipe_keyed(dse.RecipeKey(segbg, "3U1B"), root)                # recipe key disambiguates
    assert r3 is not None and r3["manifest"]["storage_recipe"] == "3U1B"
    b = dse.lookup_background_keyed(dse.BackgroundKey(131072, 8, 4, 62, "seg_single_prime", "single", 1, "mul"), root)
    assert b is not None and b["background"]["BRAM_18K"] == 142                     # never mixes with seg_phase1 228


def test_format_report_contains_table_and_recommendation():
    p = dse.arch_quantities(N=1024, BU=8, CH=4, K=62, NUM_CORE=1)
    out = dse.format_report(p, None, dse.BOARD_U55C, 0.80)
    assert "ICBU bind_storage minor-DSE" in out
    assert "recommendation:" in out
    assert "NOT calibrated" in out                # background None -> caveat
    assert "2U2B" in out and "4U" in out          # candidate rows present


def test_recommend_exposes_best_decision_reason():
    """DSE-apply slice: Recommendation carries machine-readable best/decision/reason fields."""
    p8 = dse.arch_quantities(N=131072, BU=8, CH=4, K=62, NUM_CORE=1)
    r2 = dse.recommend(p8, background=_BG_SEG, board=dse.BOARD_U55C, guard=0.80, configured="2U2B")
    assert r2.best == "3U1B" and r2.decision == "D2" and r2.reason
    r1 = dse.recommend(p8, background=_BG_SEG, board=dse.BOARD_U55C, guard=0.80, configured="3U1B")
    assert r1.best is None and r1.decision == "D1"
    p1k = dse.arch_quantities(N=1024, BU=8, CH=4, K=62, NUM_CORE=1)
    r0u = dse.recommend(p1k, background=None, board=dse.BOARD_U55C, guard=0.80, configured="2U2B")
    assert r0u.best is None and r0u.decision == "D0"
    r0o = dse.recommend(p8, background=None, board=dse.BOARD_U55C, guard=0.80, configured="2U2B")
    assert r0o.best is None and r0o.decision == "D0"
    p16 = dse.arch_quantities(N=131072, BU=16, CH=8, K=62, NUM_CORE=1)
    r3 = dse.recommend(p16, background=_BG_BU16, board=dse.BOARD_U55C, guard=0.80, configured="2U2B")
    assert r3.best is None and r3.decision == "D3"
    p32 = dse.arch_quantities(N=131072, BU=32, CH=16, K=62, NUM_CORE=1)
    r4 = dse.recommend(p32, background=_BG_BU32, board=dse.BOARD_U55C, guard=0.80, configured="2U2B")
    assert r4.best is None and r4.decision == "D4"


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

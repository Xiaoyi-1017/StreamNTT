#!/usr/bin/env python3
"""Tests for resource_model (Phase 4.0 Slice 2 pre-gen cost model: structural
formulas + calibrated D_MR coefficient table).

Run (no pytest needed):  ~/autontt-venv/bin/python3 tests/test_resource_model.py
Exit 0 = all pass; non-zero = a failure (prints the offending test).

Anchors come from Slice 1 / Slice 1b on-disk csynth evidence (N1024 BU8 CH4),
recorded in Checkpoint/phase4_0_pregen_cost_model_*audit_*.md."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dse import resource_model as rm


def _expect_raises(fn, exc=Exception, needle=None):
    try:
        fn()
    except exc as e:
        if needle is not None:
            assert needle in str(e), "expected %r in error, got %r" % (needle, str(e))
        return
    raise AssertionError("expected %s, none raised" % getattr(exc, "__name__", exc))


# --- structural counts -------------------------------------------------------

def test_num_l_stage_and_x_stage_bu8():
    assert rm.num_l_stage(N=1024, BU=8) == 6       # log2(1024)-log2(8)-1
    assert rm.num_x_stage(BU=8) == 4               # log2(8)+1


def test_bf_unit_instances_n1024_bu8_is_48():
    assert rm.estimate_bf_unit_instances(N=1024, BU=8, num_core=1) == 48   # 6*8*1


# --- X-stage MR count (structural) -------------------------------------------

def test_x_stage_mr_groupshare_bu8_is_43():
    # current mainline (group-shared TFS): 4*8 butterflies + (0+1+3+7) TFS
    assert rm.estimate_x_stage_mr_count(BU=8) == 43
    assert rm.estimate_x_stage_mr_count(BU=8, tfs_model=rm.TFS_GROUPSHARE) == 43


def test_x_stage_mr_idskip_bu8_is_49_historical_nondefault():
    # historical lane-level model: 4*8 + (0+4+6+7) = 49 ; must be opt-in, not default
    assert rm.estimate_x_stage_mr_count(BU=8, tfs_model=rm.TFS_IDSKIP) == 49
    assert rm.estimate_x_stage_mr_count(BU=8) == 43       # default stays group-shared


def test_x_stage_per_stage_groupshare_bu8():
    assert rm.estimate_x_stage_per_stage_mr(BU=8) == (8, 9, 11, 15)


def test_x_stage_mr_unknown_tfs_model_raises():
    _expect_raises(lambda: rm.estimate_x_stage_mr_count(BU=8, tfs_model="bogus"),
                   ValueError)


# --- calibrated D_MR lookup --------------------------------------------------

def test_dmr_barrett_62bit_pair_is_30():
    assert rm.lookup_d_mr(rm.ROUTE_BARRETT, max_prime_bit_length=62, prime_count=2) == 30


def test_dmr_barrett_61bit_pair_is_23():
    assert rm.lookup_d_mr(rm.ROUTE_BARRETT, max_prime_bit_length=61, prime_count=2) == 23


def test_dmr_barrett_52bit_single_is_15_and_pair_is_17():
    assert rm.lookup_d_mr(rm.ROUTE_BARRETT, 52, 1) == 15
    assert rm.lookup_d_mr(rm.ROUTE_BARRETT, 52, 2) == 17


def test_dmr_barrett_62bit_single_is_26():
    assert rm.lookup_d_mr(rm.ROUTE_BARRETT, 62, 1) == 26


def test_dmr_barrett_61bit_single_raises_unknown():
    # explicit unknown (no on-disk datapoint) -> error, NOT interpolation
    _expect_raises(lambda: rm.lookup_d_mr(rm.ROUTE_BARRETT, 61, 1),
                   rm.UnknownCoefficientError)


def test_dmr_shiftadd_reduce_k62_is_11():
    assert rm.lookup_d_mr(rm.ROUTE_SHIFTADD_REDUCE, 62, 1) == 11
    assert rm.lookup_d_mr(rm.ROUTE_SHIFTADD_REDUCE, 62, 2) == 11   # route-flat in NP


def test_dmr_shiftadd_mul_k62_is_11():
    assert rm.lookup_d_mr(rm.ROUTE_SHIFTADD_MUL, 62, 1) == 11


def test_dmr_shiftadd_reduce_k52_is_6():
    assert rm.lookup_d_mr(rm.ROUTE_SHIFTADD_REDUCE, 52, 1) == 6


def test_dmr_shiftadd_mul_k52_is_6():
    assert rm.lookup_d_mr(rm.ROUTE_SHIFTADD_MUL, 52, 1) == 6


def test_dmr_unknown_route_raises():
    _expect_raises(lambda: rm.lookup_d_mr("madeup_route", 62, 2),
                   rm.UnknownCoefficientError)


def test_dmr_unknown_bitlength_raises_not_interpolated():
    _expect_raises(lambda: rm.lookup_d_mr(rm.ROUTE_BARRETT, 60, 2),
                   rm.UnknownCoefficientError)


def test_dmr_entry_carries_provenance():
    e = rm.lookup_d_mr_entry(rm.ROUTE_BARRETT, 62, 2)
    assert e.value == 30
    assert e.provenance and isinstance(e.provenance, str)
    assert "rapidstream" in e.report_layer        # 1290 == RapidStream forensic


# --- composed DSP estimates (MR_count x D_MR) --------------------------------

def test_x_stage_dsp_groupshare_barrett61_pair_is_989():
    assert rm.estimate_x_stage_dsp(BU=8, route=rm.ROUTE_BARRETT,
                                   max_prime_bit_length=61, prime_count=2) == 989  # 43*23


def test_x_stage_dsp_groupshare_barrett62_pair_is_1290():
    assert rm.estimate_x_stage_dsp(BU=8, route=rm.ROUTE_BARRETT,
                                   max_prime_bit_length=62, prime_count=2) == 1290  # 43*30


def test_x_stage_dsp_idskip_barrett61_pair_is_1127():
    assert rm.estimate_x_stage_dsp(BU=8, route=rm.ROUTE_BARRETT,
                                   max_prime_bit_length=61, prime_count=2,
                                   tfs_model=rm.TFS_IDSKIP) == 1127  # 49*23


def test_l_stage_bf_dsp_n1024_bu8_barrett62_pair_is_1440():
    assert rm.estimate_l_stage_bf_dsp(N=1024, BU=8, route=rm.ROUTE_BARRETT,
                                      max_prime_bit_length=62, prime_count=2,
                                      num_core=1) == 1440  # 48*30


# --- optional device-level guard wrapper (reuses icbu_dse) -------------------

def test_guard_status_device_dsp_under():
    # 1290 DSP << 0.80 * 8204 device DSP -> UNDER (device-level only; slot-level NOT modeled here)
    assert rm.guard_status_for_dsp(1290) == "UNDER"


# --- 3a: ICBU memory wrapper (reuses icbu_dse; URAM/BRAM only, no DSP) --------

def test_icbu_memory_n1024_bu8_2u2b_uram_96():
    m = rm.estimate_icbu_memory(N=1024, BU=8, CH=4, K=62, num_core=1, recipe="2U2B")
    assert m.uram == 96            # audited N1024/BU8/2U2B anchor (matches measured RapidStream)


def test_icbu_memory_n1024_bu8_2u2b_bram_192():
    m = rm.estimate_icbu_memory(N=1024, BU=8, CH=4, K=62, recipe="2U2B")
    assert m.bram_18k == 192       # audited anchor (matches measured)


def test_icbu_memory_n1024_bu8_bram_packing_is_conservative():
    # bank_depth = 1024/(4*8) = 32 is OUTSIDE the calibrated BRAM domain {1024,2048,4096}
    m = rm.estimate_icbu_memory(N=1024, BU=8, CH=4, K=62, recipe="2U2B")
    assert m.bram_packing == "uncalibrated/conservative"


def test_icbu_memory_has_no_dsp_field():
    # DSP is owned by resource_model D_MR, NOT icbu_dse CALIB (hardcoded 30); memory object carries no DSP
    m = rm.estimate_icbu_memory(N=1024, BU=8, CH=4, K=62, recipe="2U2B")
    assert not hasattr(m, "dsp")
    assert "dsp" not in {f.lower() for f in m.__dataclass_fields__}


def test_icbu_memory_reuses_icbu_dse_estimate_not_duplicates():
    from dse import icbu_dse
    arch = icbu_dse.arch_quantities(1024, 8, 4, 62, 1)
    ref = icbu_dse.estimate_icbu(arch, "2U2B")
    m = rm.estimate_icbu_memory(N=1024, BU=8, CH=4, K=62, recipe="2U2B")
    assert (m.uram, m.bram_18k) == (ref["URAM"], ref["BRAM_18K"])     # same source, not re-derived


def test_icbu_memory_no_own_board_constant():
    # must not duplicate board constants; reuse icbu_dse.BOARD_U55C
    assert not hasattr(rm, "BOARD_U55C")


def test_icbu_memory_invalid_recipe_raises_like_icbu_dse():
    _expect_raises(lambda: rm.estimate_icbu_memory(N=1024, BU=8, CH=4, K=62, recipe="BOGUS"),
                   KeyError)


def test_icbu_memory_3u1b_uram_144():
    # recipe parameter is honored: 3U1B has 3 URAM banks -> 3*1*48 = 144
    m = rm.estimate_icbu_memory(N=1024, BU=8, CH=4, K=62, recipe="3U1B")
    assert m.uram == 144


# --- 3b: layered DSP guardrail (device / slot / group / module) --------------

def test_slot_dsp_capacity_proxy_default_is_1228():
    # PROXY (largest SLR/slot DSP in visible artifacts) -- NOT 1218 (outside visible repo artifacts)
    assert rm.SLOT_DSP_CAPACITY_PROXY_U55C == 1228


def test_device_dsp_guard_3090_under():
    r = rm.device_dsp_guard(3090)
    assert r.status == "UNDER"
    assert r.capacity == 8204
    assert r.capacity_layer == rm.GUARD_LAYER_DEVICE
    assert r.guard_ratio == 0.8


def test_device_dsp_guard_applies_guard_ratio_once():
    # raw device capacity -> 0.8 applied ONCE: threshold = int(0.8*8204) = 6563
    assert rm.device_dsp_guard(3090).threshold == int(0.8 * 8204)


def test_slot_dsp_guard_1290_over():
    r = rm.slot_dsp_guard(1290)
    assert r.status == "OVER"
    assert r.capacity == 1228
    assert r.capacity_layer == rm.GUARD_LAYER_SLOT


def test_slot_dsp_guard_group_1410_over():
    assert rm.slot_dsp_guard(1410).status == "OVER"


def test_slot_dsp_guard_no_double_guard_threshold_equals_proxy():
    # proxy ALREADY represents the guardrail capacity -> guard_ratio defaults to 1.0 -> NO second 0.8 derate
    assert rm.slot_dsp_guard(1290).threshold == 1228
    assert rm.slot_dsp_guard(1290).guard_ratio == 1.0


def test_slot_dsp_capacity_proxy_is_configurable():
    assert rm.slot_dsp_guard(1290, slot_capacity_proxy=2000).status == "UNDER"


def test_rapidstream_group_dsp_1410_equals_xstages_plus_tfrx():
    assert rm.estimate_rapidstream_group_dsp(1290, tfr_x_dsp=4 * 30) == 1410


def test_rapidstream_group_excludes_tfg_x_by_default():
    # TFG-X were SEPARATE RapidStream vertices (NOT in the audited failing group) -> tfg_x defaults to 0
    assert rm.estimate_rapidstream_group_dsp(1290, tfr_x_dsp=120) == 1410


def test_module_level_xstages_distinct_from_group_level():
    module = rm.estimate_x_stage_dsp(BU=8, route=rm.ROUTE_BARRETT,
                                     max_prime_bit_length=62, prime_count=2)   # module-level
    group = rm.estimate_rapidstream_group_dsp(module, tfr_x_dsp=4 * 30)        # group-level
    assert module == 1290
    assert group == 1410
    assert module != group


def test_classify_dsp_guard_explicit_inputs_no_hidden_derate():
    r = rm.classify_dsp_guard(used=1000, capacity=1228,
                              capacity_layer=rm.GUARD_LAYER_SLOT, guard_ratio=1.0)
    assert r.status == "UNDER"
    assert r.threshold == 1228
    assert r.headroom == 228


# --- 3c-1: per-stage / per-group X-stage DSP (Phase 5 split-planning DATA only) ----

def test_x_stage_per_stage_dsp_bu8_dmr30():
    assert rm.estimate_x_stage_per_stage_dsp(BU=8, d_mr=30) == (240, 270, 330, 450)


def test_x_stage_per_stage_dsp_bu8_sum_is_1290():
    assert sum(rm.estimate_x_stage_per_stage_dsp(BU=8, d_mr=30)) == 1290


def test_x_stage_per_stage_dsp_default_is_group_shared_not_idskip():
    gs = rm.estimate_x_stage_per_stage_dsp(BU=8, d_mr=30)                       # default
    idskip = rm.estimate_x_stage_per_stage_dsp(BU=8, d_mr=30, tfs_model=rm.TFS_IDSKIP)
    assert gs == (240, 270, 330, 450)
    assert idskip != gs                          # idskip is historical / opt-in, never the default


def test_x_stage_group_dsp_partition_012_3():
    sd = rm.estimate_x_stage_per_stage_dsp(BU=8, d_mr=30)
    assert rm.estimate_x_stage_group_dsp(sd, [[0, 1, 2], [3]]) == (840, 450)


def test_x_stage_group_dsp_partition_01_23():
    sd = rm.estimate_x_stage_per_stage_dsp(BU=8, d_mr=30)
    assert rm.estimate_x_stage_group_dsp(sd, [[0, 1], [2, 3]]) == (510, 780)


def test_x_stage_group_dsp_invalid_index_raises():
    sd = rm.estimate_x_stage_per_stage_dsp(BU=8, d_mr=30)
    _expect_raises(lambda: rm.estimate_x_stage_group_dsp(sd, [[0, 1, 2, 4]]), ValueError)


def test_x_stage_group_dsp_overlapping_partition_raises():
    sd = rm.estimate_x_stage_per_stage_dsp(BU=8, d_mr=30)
    _expect_raises(lambda: rm.estimate_x_stage_group_dsp(sd, [[0, 1, 2], [2, 3]]), ValueError)


def test_x_stage_group_dsp_missing_stage_raises_fail_fast():
    sd = rm.estimate_x_stage_per_stage_dsp(BU=8, d_mr=30)
    _expect_raises(lambda: rm.estimate_x_stage_group_dsp(sd, [[0, 1, 2]]), ValueError)  # stage 3 missing


def test_x_stage_per_stage_mr_bu32_structural_only():
    # BU32 num_x_stage=6 -> structural MR counts (formula-derived; D_MR / physical split NOT calibrated)
    assert rm.estimate_x_stage_per_stage_mr(BU=32) == (32, 33, 35, 39, 47, 63)


def test_x_stage_partition_guard_full_group_slot_over():
    sd = rm.estimate_x_stage_per_stage_dsp(BU=8, d_mr=30)
    guards = rm.estimate_x_stage_partition_guard(sd, [[0, 1, 2, 3]])
    assert len(guards) == 1
    assert guards[0].status == "OVER"            # group 1290 vs slot proxy 1228
    assert guards[0].capacity_layer == rm.GUARD_LAYER_SLOT


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

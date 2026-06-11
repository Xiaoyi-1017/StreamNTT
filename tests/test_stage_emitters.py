#!/usr/bin/env python3
"""T0 unit tests for the Phase 3.4 D2 generic StagePlan-driven emitters.

Oracle = the legacy byte-identity anchors in stage_generator (committed-template / db153f1 bytes).
Run (no pytest needed):  ~/autontt-venv/bin/python3 tests/test_stage_emitters.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from code_generator import stage_generator as sg
import generate_code as gc
from code_generator.stage_plan import plan_from_precomp_boundary

P4 = plan_from_precomp_boundary(6, 4)
P5 = plan_from_precomp_boundary(6, 5)


def test_region2_default_plan_equals_legacy_anchors():
    m = sg.region2_emission_map(P4)
    assert set(m) == set(sg.REGION2_WRAPPER_MARKERS)
    for k in sg.REGION2_WRAPPER_MARKERS:
        assert m[k] == sg.gen_region2_wrapper(k, {}), "anchor drift: %s" % k


def test_region2_precomp5_plan_equals_db153f1_anchors():
    m = sg.region2_emission_map(P5)
    for k, v in sg.B5_REGION2_OVERRIDES.items():
        assert m[k] == v, "db153f1 anchor drift: %s" % k
    for k in sg.REGION2_WRAPPER_MARKERS:
        if k not in sg.B5_REGION2_OVERRIDES:
            assert m[k] == sg.gen_region2_wrapper(k, {}), "non-override marker drift: %s" % k


def test_region1_default_and_precomp5_equal_legacy_accessors():
    for st in ({}, {"is_single": True}):
        r4 = dict(sg.region1_emission_map(P4, st, gc._gen_tw_gen_L_small))
        assert r4["tw_gen_L_base_s_ge4"] == sg.gen_tw_gen_L_base_ge4(st)
        assert r4["tw_complete_L_ratio_s_ge4"] == sg.gen_tw_complete_L_ratio_ge4(st)
        r5 = dict(sg.region1_emission_map(P5, st, gc._gen_tw_gen_L_small))
        assert r5["tw_gen_L_base_s_ge4"] == (
            gc._gen_tw_gen_L_small(st, 4) + "\n\n" + sg.gen_tw_gen_L_base_ge5(st))
        assert r5["tw_complete_L_ratio_s_ge4"] == sg.gen_tw_complete_L_ratio_ge5(st)


def test_generic_ge_producers_equal_named_accessors():
    for st in ({}, {"is_single": True}):
        assert sg.gen_tw_gen_L_base_ge(st, "s_ge4") == sg.gen_tw_gen_L_base_ge4(st)
        assert sg.gen_tw_gen_L_base_ge(st, "s_ge5") == sg.gen_tw_gen_L_base_ge5(st)
        assert sg.gen_tw_complete_L_ratio_ge(st, "s_ge4") == sg.gen_tw_complete_L_ratio_ge4(st)
        assert sg.gen_tw_complete_L_ratio_ge(st, "s_ge5") == sg.gen_tw_complete_L_ratio_ge5(st)


_CHAIN_DEFAULT_TAIL = (
    "#if NUM_L_STAGE_GE4 > 0\n"
    "\t\t.invoke<tapa::detach, NUM_L_STAGE_GE4>(l_stage_s_ge4, tapa::seq(), core_streams, core_streams)\n"
    "#endif\n"
    "\t\t.invoke<tapa::detach>(x_stages_top, core_streams, core_ostreams);")

_CHAIN_B5_TAIL = (
    "#if NUM_L_stage > 4\n"
    "\t\t.invoke<tapa::detach>(l_stage_s4, 4, core_streams, core_streams)\n"
    "#endif\n"
    "#if NUM_L_STAGE_GE5 > 0\n"
    "\t\t.invoke<tapa::detach, NUM_L_STAGE_GE5>(l_stage_s_ge5, tapa::seq(), core_streams, core_streams)\n"
    "#endif\n"
    "\t\t.invoke<tapa::detach>(x_stages_top, core_streams, core_ostreams);")


def test_chain_emitter_tails_byte_equal_for_both_plans():
    c4 = gc._gen_ntt_core_lstage_chain({"_stage_plan": P4})
    c5 = gc._gen_ntt_core_lstage_chain({"_stage_plan": P5})
    assert c4.endswith(_CHAIN_DEFAULT_TAIL) and "l_stage_s4" not in c4 and "GE5" not in c4
    assert c5.endswith(_CHAIN_B5_TAIL)
    assert c4.split("#if NUM_L_STAGE_GE4")[0] == c5.split("#if NUM_L_stage > 4\n\t\t.invoke<tapa::detach>(l_stage_s4")[0]
    assert gc._gen_ntt_core_lstage_chain({}).endswith(_CHAIN_DEFAULT_TAIL)   # no-plan fallback = legacy tail


def test_singleton_emitters_match_db153f1_s4_strings():
    g4 = next(g for g in P5.groups if g.lo == 4)
    assert sg.gen_bf_unit_for_group(P5, g4) == sg._BF_UNIT_S4
    assert sg.gen_l_stage_for_group(P5, g4) == sg._L_STAGE_S4
    ge5 = P5.otf_groups()[0]
    assert sg.gen_bf_unit_for_group(P5, ge5) == sg._BF_UNIT_S_GE5
    assert sg.gen_l_stage_for_group(P5, ge5) == sg._L_STAGE_S_GE5


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

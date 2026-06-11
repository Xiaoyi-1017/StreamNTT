#!/usr/bin/env python3
"""T0 unit tests for stage_plan (Phase 3.4 D1 planner core).

Run (no pytest needed):  ~/autontt-venv/bin/python3 tests/test_stage_plan.py
Exit 0 = all pass; non-zero = a failure (prints the offending test)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from code_generator import stage_plan as sp
from code_generator import icbu_generator as ig


def _expect_raises(fn, needle=None):
    try:
        fn()
    except ValueError as e:
        if needle is not None:
            assert needle in str(e), "expected %r in error, got %r" % (needle, str(e))
        return
    raise AssertionError("expected ValueError, none raised")


def test_precomp4_default_plan_matches_current_structure():
    p = sp.plan_from_precomp_boundary(6)                       # N1024 BU8
    assert p.lstage_groups() == ((0, 0), (1, 1), (2, 2), (3, 3), (4, 5))
    assert [p.group_name(g) for g in p.groups] == ["s0", "s1", "s2", "s3", "s_ge4"]
    assert [g.kind for g in p.groups] == ["precomp"] * 4 + ["otf"]
    assert p.precomp_flat_entries() == 15                       # 1+2+4+8


def test_precomp5_plan_matches_db153f1_structure():
    p = sp.plan_from_precomp_boundary(6, 5)
    assert p.lstage_groups() == ((0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5))
    assert [p.group_name(g) for g in p.groups] == ["s0", "s1", "s2", "s3", "s4", "s_ge5"]
    assert p.precomp_flat_entries() == 31                       # 1+2+4+8+16
    p7 = sp.plan_from_precomp_boundary(7, 5)                    # N1024 BU4 / N2048 BU8
    assert p7.lstage_groups()[-1] == (5, 6)
    assert p7.group_name(p7.groups[-1]) == "s_ge5"


def test_invalid_precomp_values_fail_clearly():
    for b in (0, 1, 2, 3, 6, 7):
        _expect_raises(lambda b=b: sp.plan_from_precomp_boundary(6, b), "precompute_boundary")
    _expect_raises(lambda: sp.plan_from_precomp_boundary(0), "num_l_stage")
    _expect_raises(lambda: sp.plan_from_precomp_boundary(6, 6), "Layer-3")   # PRECOMP>5 hint present


def test_invariants_reject_bad_plans():
    G = sp.StageGroup
    _expect_raises(lambda: sp.StagePlan(6, (G(0, 0, "precomp"), G(2, 5, "otf"))).validate(), "[V1/V2]")  # gap
    _expect_raises(lambda: sp.StagePlan(6, (G(0, 3, "precomp"), G(3, 5, "otf"))).validate(), "[V1/V2]")  # overlap
    _expect_raises(lambda: sp.StagePlan(6, (G(0, 3, "precomp"), G(4, 4, "otf"), G(5, 5, "precomp"))).validate(),
                   "[V3]")                                                                                # precomp after otf
    _expect_raises(lambda: sp.StagePlan(6, (G(0, 5, "weird"),)).validate(), "kind")
    _expect_raises(lambda: sp.StagePlan(6, (G(0, 4, "precomp"),)).validate(), "[V1]")                     # not covering


def test_interior_range_name_saTOb():
    G = sp.StageGroup
    p = sp.StagePlan(8, (G(0, 3, "precomp"), G(4, 6, "otf"), G(7, 7, "otf"))).validate()
    assert p.group_name(p.groups[1]) == "s4to6"                 # interior range -> sAtoB
    assert p.group_name(p.groups[2]) == "s_ge7"


def test_icbu_lstage_groups_same_source_of_truth():
    for n in (5, 6, 7, 9):
        for b in (4, 5):
            if n <= b and b == 5 and n < 5:
                continue
            assert ig._lstage_groups(n, b) == sp.plan_from_precomp_boundary(n, b).lstage_groups(), (n, b)


def test_p32_classification_degenerate_single_s5():
    p = sp.plan_from_precomp_boundary(6, 5)                     # N1024 BU8: ge5 = {s5} only
    ev = sp.classify_p32_evidence(p, 32)
    assert ev.classification == "degenerate" and ev.period_blocks == (1,)
    assert "FUNCTIONAL SMOKE ONLY" in ev.detail


def test_p32_classification_non_degenerate_two_stages():
    p = sp.plan_from_precomp_boundary(7, 5)                     # N1024 BU4 / N2048 BU8: ge5 = {s5,s6}
    ev = sp.classify_p32_evidence(p, 32)
    assert ev.classification == "non-degenerate" and ev.period_blocks == (1, 2)
    assert "evidence candidate" in ev.detail


def test_p32_classification_illegal_under_precomp4():
    p = sp.plan_from_precomp_boundary(6, 4)                     # ge4 contains s4: period 16 < 32
    ev = sp.classify_p32_evidence(p, 32)
    assert ev.classification == "illegal-period-lt-P" and "[4]" in ev.detail


def test_p32_classification_no_otf():
    G = sp.StageGroup
    p = sp.StagePlan(4, (G(0, 0, "precomp"), G(1, 1, "precomp"), G(2, 2, "precomp"), G(3, 3, "precomp"))).validate()
    assert sp.classify_p32_evidence(p, 32).classification == "no-otf"


def test_p16_two_stage_ge5_is_non_degenerate_for_p16():
    # sanity at P=16: ge5={s5,s6} -> period_blocks (2,4); classification machinery generalizes over P.
    ev = sp.classify_p32_evidence(sp.plan_from_precomp_boundary(7, 5), 16)
    assert ev.classification == "non-degenerate" and ev.period_blocks == (2, 4)


def test_schedule_boundary_equivalents():
    # Phase 3.4 D4 first slice: schedule -> StagePlan equals the PRECOMP-boundary constructors.
    assert sp.plan_from_schedule("precomp:0-3;otf:4-", 6) == sp.plan_from_precomp_boundary(6, 4)
    assert sp.plan_from_schedule("precomp:0-4;otf:5-", 6) == sp.plan_from_precomp_boundary(6, 5)
    assert sp.plan_from_schedule("precomp:0-4;otf:5-", 7) == sp.plan_from_precomp_boundary(7, 5)


def test_schedule_singleton_split_spelling_normalizes():
    split = "precomp:0-0;precomp:1-1;precomp:2-2;precomp:3-3;otf:4-"
    assert sp.plan_from_schedule(split, 6) == sp.plan_from_precomp_boundary(6, 4)  # plan equality, not strings


def test_schedule_unset_and_empty():
    assert sp.plan_from_schedule(None, 6) is None
    assert sp.plan_from_schedule("", 6) is None
    assert sp.plan_from_schedule("   ", 6) is None


def test_schedule_fail_fasts():
    E = sp.plan_from_schedule
    _expect_raises(lambda: E("precomp:0-3;otf:3-", 6), "STREAMNTT_LSTAGE_PRECOMPUTE_SCHEDULE")  # overlap (V1)
    _expect_raises(lambda: E("precomp:0-2;otf:4-", 6), "STREAMNTT_LSTAGE_PRECOMPUTE_SCHEDULE")  # gap (V1)
    _expect_raises(lambda: E("precomp:0-3;otf:4-9", 6), "out of [0,6)")                          # out-of-range
    _expect_raises(lambda: E("frozen:0-3;otf:4-", 6), "illegal kind")
    _expect_raises(lambda: E("otf:0-1;precomp:2-3;otf:4-", 6), "multiple OTF ranges deferred")
    _expect_raises(lambda: E("precomp:0-5;otf:6-", 8), "future generalized-emitter validation slice")  # PRECOMP>5
    _expect_raises(lambda: E("precomp:0-3;otf:4-4;otf:5-", 6), "multiple OTF ranges deferred")
    _expect_raises(lambda: E("precomp:0-1;otf:2-", 6), "boundary-equivalent")    # legal plan, non-4/5 boundary
    _expect_raises(lambda: E("precomp:0-3;otf:x-", 6), "malformed range")


def test_schedule_non_prefix_precomp_rejected():
    _expect_raises(lambda: sp.plan_from_schedule("otf:0-3;precomp:4-5", 6), "STREAMNTT_LSTAGE_PRECOMPUTE_SCHEDULE")


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

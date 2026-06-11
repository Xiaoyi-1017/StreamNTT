#!/usr/bin/env python3
"""Pure-Python T0 unit tests for icbu_generator (Phase 3.3.4 ICBU recipe-schedule parser).

Run (no pytest needed):  ~/autontt-venv/bin/python3 tests/test_icbu_recipe_schedule_parser.py
Exit 0 = all pass; non-zero = a failure (prints the offending test)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from code_generator import icbu_generator as ig

# d0c3a16 named scalar recipe -> (count form, per-bank tuple): the cross-module consistency anchor.
NAMED_TO_COUNT = {"2U2B": "L0B2U2", "4U": "L0B0U4", "4B": "L0B4U0", "4L": "L4B0U0", "3U1B": "L0B1U3"}
NAMED_TO_TUPLE = {
    "2U2B": ("uram", "uram", "bram", "bram"),
    "4U":   ("uram", "uram", "uram", "uram"),
    "4B":   ("bram", "bram", "bram", "bram"),
    "4L":   ("lutram", "lutram", "lutram", "lutram"),
    "3U1B": ("uram", "uram", "uram", "bram"),
}


def _expect_raises(fn, needle=None):
    try:
        fn()
    except ValueError as e:
        if needle is not None:
            assert needle in str(e), "expected %r in error, got %r" % (needle, str(e))
        return
    raise AssertionError("expected ValueError, none raised")


def test_count_form_subsumes_named():
    for name, count in NAMED_TO_COUNT.items():
        assert ig.parse_storage_count_form(count) == NAMED_TO_TUPLE[name], name


def test_count_form_fill_order_U_B_L():
    assert ig.parse_storage_count_form("L1B0U3") == ("uram", "uram", "uram", "lutram")
    assert ig.parse_storage_count_form("L2B1U1") == ("uram", "bram", "lutram", "lutram")


def test_count_form_sum_must_be_4():
    _expect_raises(lambda: ig.parse_storage_count_form("L0B2U1"), "sum to 4")
    _expect_raises(lambda: ig.parse_storage_count_form("L1B2U2"), "sum to 4")


def test_count_form_malformed():
    _expect_raises(lambda: ig.parse_storage_count_form("2U2B"), "malformed")
    _expect_raises(lambda: ig.parse_storage_count_form("L0B0U"), "malformed")


def test_partition_tokens():
    assert ig.parse_partition_token("none") == ("none", 1)
    assert ig.parse_partition_token("block2") == ("block", 2)
    assert ig.parse_partition_token("block4") == ("block", 4)
    assert ig.parse_partition_token("cyclic2") == ("cyclic", 2)
    assert ig.parse_partition_token("cyclic4") == ("cyclic", 4)


def test_partition_complete_and_abbrev_rejected():
    _expect_raises(lambda: ig.parse_partition_token("complete"), "deferred")
    _expect_raises(lambda: ig.parse_partition_token("p2"))
    _expect_raises(lambda: ig.parse_partition_token("c2"))


def test_legality_factor_divides_depth_half():
    ig.check_bank_legality(ig.BankSpec("uram", "block", 4), 32)   # 32 % 4 == 0 -> ok
    ig.check_bank_legality(ig.BankSpec("uram", "none", 1), 6)     # none -> always ok
    _expect_raises(lambda: ig.check_bank_legality(ig.BankSpec("uram", "block", 4), 6), "divide")


def test_lstage_groups():
    assert ig._lstage_groups(6) == ((0, 0), (1, 1), (2, 2), (3, 3), (4, 5))
    assert ig._lstage_groups(7) == ((0, 0), (1, 1), (2, 2), (3, 3), (4, 6))
    assert ig._lstage_groups(4) == ((0, 0), (1, 1), (2, 2), (3, 3))


def test_default_unset_is_uniform_2u2b():
    r = ig.parse_icbu_recipe_schedule(None, 6, 64)
    assert r.uniform is True
    assert r.uniform_impls == ("uram", "uram", "bram", "bram")
    assert r.uniform_partition == "none"
    assert r.normalized == "0-5:L0B2U2:none"


def test_empty_env_is_default():
    r = ig.parse_icbu_recipe_schedule("", 6, 64)
    assert r.uniform is True
    assert r.uniform_impls == ("uram", "uram", "bram", "bram")
    assert r.normalized == "0-5:L0B2U2:none"


def test_whitespace_env_is_default():
    r = ig.parse_icbu_recipe_schedule("   ", 6, 64)
    assert r.uniform is True
    assert r.normalized == "0-5:L0B2U2:none"


def test_explicit_uniform_merges_to_single_range():
    r = ig.parse_icbu_recipe_schedule("0-3:L0B2U2:none;4-5:L0B2U2:none", 6, 64)
    assert r.uniform is True
    assert r.normalized == "0-5:L0B2U2:none"


def test_uniform_via_full_range_4u():
    r = ig.parse_icbu_recipe_schedule("0-5:L0B0U4:none", 6, 64)
    assert r.uniform is True
    assert r.uniform_impls == ("uram", "uram", "uram", "uram")
    assert r.normalized == "0-5:L0B0U4:none"


def test_hetero_detected_not_raised():
    r = ig.parse_icbu_recipe_schedule("0-3:L0B0U4:none;4-5:L0B2U2:none", 6, 64)
    assert r.uniform is False
    assert r.normalized == "0-3:L0B0U4:none;4-5:L0B2U2:none"


def test_c3_uncovered_uses_default_then_hetero():
    r = ig.parse_icbu_recipe_schedule("0-3:L0B0U4:none", 6, 64)   # s_ge4 uncovered -> default 2U2B
    assert r.uniform is False
    assert r.per_group_impls[-1] == ("uram", "uram", "bram", "bram")
    assert r.normalized == "0-3:L0B0U4:none;4-5:L0B2U2:none"


def test_c1_index_out_of_range():
    _expect_raises(lambda: ig.parse_icbu_recipe_schedule("0-6:L0B2U2:none", 6, 64), "[C1]")


def test_c2_overlap():
    _expect_raises(lambda: ig.parse_icbu_recipe_schedule("0-3:L0B2U2:none;3-5:L0B2U2:none", 6, 64), "[C2]")


def test_c4_storage_sum():
    _expect_raises(lambda: ig.parse_icbu_recipe_schedule("0-5:L0B1U2:none", 6, 64), "sum to 4")


def test_c5_partition_factor_must_divide_depth_half():
    # depth=12 -> depth_half=6 ; cyclic4 factor 4 does not divide 6 (small bank chosen to trigger C5).
    _expect_raises(lambda: ig.parse_icbu_recipe_schedule("0-5:L0B2U2:cyclic4", 6, 12), "divide")


def test_c6_group_split_rejected():
    _expect_raises(lambda: ig.parse_icbu_recipe_schedule("4:L0B2U2:none", 6, 64), "[C6]")
    _expect_raises(lambda: ig.parse_icbu_recipe_schedule("0-4:L0B2U2:none", 6, 64), "[C6]")


def test_normalize_deterministic_sort():
    r = ig.parse_icbu_recipe_schedule("4-5:L0B2U2:none;0-3:L0B2U2:none", 6, 64)
    assert r.normalized == "0-5:L0B2U2:none"


def test_b5_groups_default_unchanged():
    # Phase 3.4 hybrid boundary: default B=4 groups byte-equal to the pre-boundary shape.
    assert ig._lstage_groups(6) == ig._lstage_groups(6, 4) == ((0, 0), (1, 1), (2, 2), (3, 3), (4, 5))


def test_b5_groups_boundary_5():
    assert ig._lstage_groups(6, 5) == ((0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5))
    assert ig._lstage_groups(7, 5) == ((0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 6))
    _expect_raises(lambda: ig._lstage_groups(6, 3), "precompute_boundary")


def test_b5_schedule_c6_follows_boundary():
    # num_l_stage=7. "4:..." splits the s_ge4 group (4,6) at B=4 -> C6 reject; at B=5 stage 4 is a
    # singleton group -> the SAME schedule string becomes legal. C6 therefore follows the boundary.
    _expect_raises(lambda: ig.parse_icbu_recipe_schedule("4:L0B0U4:none", 7, 64, 4), "[C6]")
    r5 = ig.parse_icbu_recipe_schedule("4:L0B0U4:none", 7, 64, 5)
    assert not r5.uniform                      # uncovered groups default to 2U2B -> heterogeneous
    # "5-6" hits the (5,6) ge5 group at B=5 but splits (4,6) at B=4.
    _expect_raises(lambda: ig.parse_icbu_recipe_schedule("5-6:L0B0U4:none", 7, 64, 4), "[C6]")
    r5b = ig.parse_icbu_recipe_schedule("0-3:L0B0U4:none;4-6:L0B0U4:none", 7, 64, 5)
    assert r5b.uniform and r5b.uniform_impls == ("uram", "uram", "uram", "uram")
    assert r5b.normalized == "0-6:L0B0U4:none"  # adjacent identical groups merge across the boundary


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
        except Exception as e:  # ImportError before the module exists, etc.
            failed += 1
            print("ERROR", t.__name__, "--", type(e).__name__, e)
    print("\n%d/%d passed" % (len(tests) - failed, len(tests)))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

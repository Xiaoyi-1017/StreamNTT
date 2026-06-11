#!/usr/bin/env python3
"""Phase 3.5 S3e — stdout-only ReduceRecipe route report.

The report is printed for shiftadd routes (reduce/mul) ONLY, after the S3d custom
fail-fast, derived from the ReduceRecipe. STDOUT-ONLY: no generated source / no
ntt.h / ntt_shiftadd.h metadata / no behavior change. (fold, correction) params
render as grep-friendly "fd:corr" (e.g. 2:positive_only), NOT Python tuple repr.
Run (no pytest needed):  ~/autontt-venv/bin/python3 tests/test_shiftadd_route_report.py"""
import contextlib
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import generate_code as gc
from code_generator.reduce_generator import classify_reduce_set
from code_generator.reduce_recipe import build_reduce_recipe

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GENERATED = os.path.join(ROOT, "generated")
TI9_Q = gc.KNOWN_PRIMES["TI_9"]


def _pairs(*names):
    return tuple((n, gc.KNOWN_PRIMES[n]) for n in names)


def _report(names, route):
    aq = _pairs(*names)
    info = classify_reduce_set(aq, None)
    recipe = build_reduce_recipe(aq, k_env=info.final_k, classify_info=info)
    return gc._format_shiftadd_route_report(recipe, route, info.final_k, info.mode)


def _cfg(primes, shiftadd):
    return dict(N=1024, BU=8, CH=4, RATE=0.5, tfg_seg_len=16,
                primes=list(primes), shiftadd=shiftadd, name="s3e_probe")


def _generated_dirs():
    return set(os.listdir(GENERATED)) if os.path.isdir(GENERATED) else set()


def _run_capture(cfg, suffix="s3etest"):
    before = _generated_dirs()
    buf = io.StringIO()
    raised = None
    try:
        with contextlib.redirect_stdout(buf):
            gc.generate_seg_case(cfg, suffix=suffix, force=True)
    except Exception as exc:  # noqa: BLE001
        raised = (type(exc).__name__, str(exc))
    return buf.getvalue(), raised, (_generated_dirs() - before)


# ---- formatter unit tests --------------------------------------------------
def test_f1_legal_same_bit_mul():
    s = _report(["TI_2", "TI_9"], "mul")
    assert "shiftadd-recipe: route=mul aliases=[TI_2,TI_9] final_K=62 " \
           "class_mode=same_bit provenance=known_table" in s, s
    assert "anchor: B=61 anchor_k_group=62 K_spread=1 batch_legal=True" in s, s
    assert "reduce: can_emit=True default_ok=True default=2:positive_only " \
           "resolver=2:positive_only" in s, s
    assert "mul:    can_emit=True corrections=1 t_match=2/2 delta_needed=0" in s, s
    assert "fail_fast:" not in s, s


def test_f2_legal_small_pair_reduce():
    s = _report(["TII_2", "TII_33"], "reduce")
    assert "route=reduce aliases=[TII_2,TII_33] final_K=52 class_mode=same_bit" in s, s
    assert "anchor: B=51 anchor_k_group=52 K_spread=1 batch_legal=True" in s, s
    # fixed-point FAILS but the live resolver emits via 3:signed.
    assert "reduce: can_emit=True default_ok=False default=2:positive_only " \
           "resolver=3:signed" in s, s
    assert "mul:    can_emit=True corrections=2 t_match=1/2 delta_needed=1" in s, s


def test_f3_mixed_kgroup_legal_has_cls_lines():
    """K_group-mixed {52,62} known sets are batch-LEGAL with per-class lines
    (cls0 = 52-class first); the batch resolver params live per class."""
    s = _report(["TI_9", "TII_33"], "reduce")
    assert "class_mode=mixed_bit" in s, s
    assert "batch_legal=True" in s, s
    assert "reduce: can_emit=True default_ok=True default=2:positive_only resolver=None" in s, s
    assert "mul:    can_emit=True corrections=2 t_match=2/2 delta_needed=0" in s, s
    assert "cls0:   K_group=52 B=51 n=1 reduce=True resolver=2:positive_only" in s, s
    assert "cls1:   K_group=62 B=61 n=1 reduce=True resolver=2:positive_only" in s, s
    assert "fail_fast:" not in s, s


def test_f4_anchor_k_group_not_bare_k_group():
    s = _report(["TI_2", "TI_9"], "reduce")
    assert "anchor_k_group=" in s, "must report anchor_k_group"
    assert " k_group=" not in s, "must NOT use bare k_group token"


def test_f5_no_python_tuple_repr():
    """Params must be grep-friendly fd:corr, never Python tuple repr."""
    for names, route in ((["TI_2", "TI_9"], "mul"), (["TII_2", "TII_33"], "reduce")):
        s = _report(names, route)
        assert "(2, 'positive_only')" not in s and "(3, 'signed')" not in s, s
        assert "default=2:positive_only" in s, s


# ---- generate-level stdout tests -------------------------------------------
def test_g1_known_reduce_prints_report():
    out, raised, _ = _run_capture(_cfg(["TI_9"], "reduce"))
    assert raised is None, raised
    assert "shiftadd-recipe: route=reduce" in out, out[-400:]


def test_g2_known_mul_prints_report():
    out, raised, _ = _run_capture(_cfg(["TI_2", "TI_9"], "mul"))
    assert raised is None, raised
    assert "shiftadd-recipe: route=mul" in out, out[-400:]


def test_g3_none_no_report():
    out, raised, _ = _run_capture(_cfg(["TI_2", "TI_9"], "none"))
    assert raised is None, raised
    assert "shiftadd-recipe:" not in out, "none/Barrett must not print a route report"


def test_g4_custom_raises_no_report_no_partial():
    out, raised, new_dirs = _run_capture(_cfg([TI9_Q], "reduce"))
    assert raised is not None and raised[0] == "ValueError", raised
    assert "custom" in raised[1], raised
    assert "shiftadd-recipe:" not in out, "custom must not print a report (S3d raises first)"
    assert new_dirs == set(), "custom must create no partial dir: %r" % (new_dirs,)


def test_g5_known_mixed_kgroup_generates_with_report():
    """K_group-mixed {52,62} known sets GENERATE (per-class mixed emitter);
    the report prints batch_legal=True with the per-class cls lines."""
    out, raised, _ = _run_capture(_cfg(["TI_9", "TII_33"], "reduce"))
    assert raised is None, "mixed K_group reduce must generate: %s" % (raised,)
    assert "shiftadd-recipe: route=reduce" in out and "batch_legal=True" in out, out[-400:]
    assert "cls0:   K_group=52 B=51" in out and "cls1:   K_group=62 B=61" in out, out[-600:]


def main():
    tests = [
        test_f1_legal_same_bit_mul,
        test_f2_legal_small_pair_reduce,
        test_f3_mixed_kgroup_legal_has_cls_lines,
        test_f4_anchor_k_group_not_bare_k_group,
        test_f5_no_python_tuple_repr,
        test_g1_known_reduce_prints_report,
        test_g2_known_mul_prints_report,
        test_g3_none_no_report,
        test_g4_custom_raises_no_report_no_partial,
        test_g5_known_mixed_kgroup_generates_with_report,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print("PASS", t.__name__)
        except AssertionError as e:
            failed += 1
            print("FAIL", t.__name__, "--", e)
        except Exception as e:  # noqa: BLE001
            failed += 1
            print("ERROR", t.__name__, "--", type(e).__name__, e)
    print("\n%d/%d passed" % (len(tests) - failed, len(tests)))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

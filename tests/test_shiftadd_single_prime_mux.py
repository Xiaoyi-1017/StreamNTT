#!/usr/bin/env python3
"""Phase 3.5 S4 — single-prime shiftadd mux removal (normalized mode).

STREAMNTT_SHIFTADD_SINGLE_PRIME_MODE=legacy|normalized (emit_shiftadd_header kwarg
single_prime_mode). legacy (default) = byte-identical to ef753da. normalized = ONLY when
NUM_PRIMES==1 and route in {reduce,mul}: the single-prime is_<alias>/mod_id recipe muxes are
resolved at generation time (NP==1 -> predicate always true) -> emitted candidates are bare.
Multi-prime stays legacy in BOTH modes. Function signatures keep ModId mod_id (ntt.cpp
byte-identical). These are EMIT-LEVEL tests (no full generate); the legacy full-generate byte
gate vs ef753da and the SW/csynth ladder are run separately in S4 validation.
Run (no pytest needed):  ~/autontt-venv/bin/python3 tests/test_shiftadd_single_prime_mux.py"""
import contextlib
import io
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import generate_code as gc
from code_generator.reduce_generator import classify_reduce_set
from code_generator.reduce_recipe import build_reduce_recipe
from code_generator.shiftadd_generator import emit_shiftadd_header


def _pairs(*names):
    return tuple((n, gc.KNOWN_PRIMES[n]) for n in names)


def _emit(names, shiftadd_mul, config_k=None, **kw):
    """Emit ntt_shiftadd.h to a throwaway dir; return its content. kw forwards
    single_prime_mode=. config_k pins K (raised-envelope tests)."""
    aq = _pairs(*names)
    info = classify_reduce_set(aq, config_k)
    recipe = build_reduce_recipe(aq, k_env=info.final_k, classify_info=info)
    with tempfile.TemporaryDirectory() as td:
        case = os.path.join(td, "case")
        os.makedirs(os.path.join(case, "src"))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            emit_shiftadd_header(case_dir=case, K_env=info.final_k, aliases_q=list(aq),
                                 enabled=True, shiftadd_mul=shiftadd_mul,
                                 balanced_selector_scope="all", recipe=recipe, **kw)
        with open(os.path.join(case, "src", "ntt_shiftadd.h")) as f:
            return f.read()


SINGLE = (("TI_9", "is_TI9"), ("TII_33", "is_TII33"))
ROUTES = ((False, "USE_REDUCE_SHIFTADD"), (True, "USE_REDUCE_SHIFTADD_MUL"))


def test_legacy_kwarg_byte_identical_to_default():
    """legacy mode (and unset default) == the unchanged path, single AND multi."""
    for names in (["TI_9"], ["TII_33"], ["TI_2", "TI_9"], ["TII_2", "TII_33"]):
        for mul, _def in ROUTES:
            base = _emit(names, mul)
            leg = _emit(names, mul, single_prime_mode="legacy")
            assert base == leg, "legacy != default for %s mul=%s" % (names, mul)


def test_normalized_single_prime_removes_mux():
    """normalized single-prime: no is_<alias>, no mod_id== mux, no ternary; route define +
    signature (ModId mod_id) retained."""
    for alias, is_tok in SINGLE:
        for mul, define in ROUTES:
            s = _emit([alias], mul, single_prime_mode="normalized")
            assert is_tok not in s, "%s residue in normalized %s mul=%s" % (is_tok, alias, mul)
            assert "(mod_id == (ModId)" not in s, "mod_id mux residue (%s mul=%s)" % (alias, mul)
            assert s.count("?") == 0, "ternary residue in normalized %s mul=%s" % (alias, mul)
            assert define in s, "route define lost (%s mul=%s)" % (alias, mul)
            sig = ("reduce_shiftadd_mul(Data2 X, ModId mod_id, Data q_cur)" if mul
                   else "reduce_shiftadd(Data2 X, ModId mod_id, Data q_cur)")
            assert sig in s, "signature (ModId mod_id) must be retained (%s mul=%s)" % (alias, mul)


def test_normalized_multi_prime_unchanged():
    """NP>=2 stays legacy even in normalized mode (resolution is NP==1 only)."""
    for names in (["TI_2", "TI_9"], ["TII_2", "TII_33"]):
        for mul, _d in ROUTES:
            leg = _emit(names, mul, single_prime_mode="legacy")
            norm = _emit(names, mul, single_prime_mode="normalized")
            assert leg == norm, "multi changed under normalized for %s mul=%s" % (names, mul)
            assert "is_TI" in norm or "is_TII" in norm, "multi must keep its mux"


def test_residue_count_contrast():
    """Concrete contrast: legacy single TI_9 reduce has is_TI9 residue; normalized has none."""
    leg = _emit(["TI_9"], False, single_prime_mode="legacy")
    norm = _emit(["TI_9"], False, single_prime_mode="normalized")
    assert leg.count("is_TI9") >= 1 and norm.count("is_TI9") == 0
    assert leg.count("mod_id == (ModId)") >= 1 and norm.count("mod_id == (ModId)") == 0
    # base/v_1/v_2 datapath survives (the candidate is folded in, not dropped).
    assert "reduce_shiftadd" in norm and "#pragma HLS INLINE" in norm


def test_raised_envelope_single_tii33_k62():
    """single TII_33 with explicit K=62: final_K=62 (container widths) but anchor B=51
    (anchor_k_group=52) must be preserved under normalized -> mux removed, base stays 51."""
    for mul in (True, False):
        s = _emit(["TII_33"], mul, config_k=62, single_prime_mode="normalized")
        assert "is_TII33" not in s and s.count("?") == 0, "mux not removed (mul=%s)" % mul
        if mul:
            assert "SHIFTADD_MUL_BASE = 51" in s, "mul base must stay 51 (anchor), not 62"
            assert "K = 62" in s, "K_env (container) should be 62"
        else:
            assert "B = 2^51" in s, "reduce base comment must stay 2^51"


def test_invalid_mode_raises():
    try:
        _emit(["TI_9"], False, single_prime_mode="weird")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid single_prime_mode must raise ValueError")


def main():
    tests = [
        test_legacy_kwarg_byte_identical_to_default,
        test_normalized_single_prime_removes_mux,
        test_normalized_multi_prime_unchanged,
        test_residue_count_contrast,
        test_raised_envelope_single_tii33_k62,
        test_invalid_mode_raises,
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

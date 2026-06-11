#!/usr/bin/env python3
"""Phase 3.5 S3c — emit-level byte-identity tests for shiftadd recipe consumption.

Core proof (no full generate_code run): for every config, the recipe-governed
emit path and the legacy (recipe=None) path produce the IDENTICAL OUTCOME —
either byte-identical ntt_shiftadd.h or the same exception. This proves:
  * recipe=None is byte-identical to legacy (the path is unchanged);
  * the recipe-governed path consumes recipe values yet emits identical bytes
    (the assert-shims pin equality);
  * small-family shiftadd=reduce still emits (NOT marked unsupported);
  * mixed K_group {52,62} emits via the per-class mixed emitter (recipe parity);
  * INT/custom behavior is unchanged (no fail-fast added in S3c).
The full ntt_shiftadd.h/ntt.cpp/ntt.h/ntt_primes.h byte gate vs 9ae7447 is run
separately in the S3c validation report.
Run (no pytest needed):  ~/autontt-venv/bin/python3 tests/test_shiftadd_recipe_consumption.py"""
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


def _recipe_for(aliases_q):
    info = classify_reduce_set(aliases_q, None)
    return build_reduce_recipe(aliases_q, k_env=info.final_k, classify_info=info), info.final_k


def _emit_outcome(aliases_q, k_env, shiftadd_mul, recipe):
    """Emit into a throwaway dir; return ('bytes', content) on success or
    ('raise', exc_type_name, message) on failure. stdout (self-check logs) is
    captured so it never pollutes the test run."""
    with tempfile.TemporaryDirectory() as td:
        case = os.path.join(td, "case")
        os.makedirs(os.path.join(case, "src"), exist_ok=True)
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                emit_shiftadd_header(
                    case_dir=case, K_env=k_env, aliases_q=list(aliases_q),
                    enabled=True, shiftadd_mul=shiftadd_mul,
                    balanced_selector_scope="all", recipe=recipe)
        except Exception as exc:  # noqa: BLE001 - parity check needs the type+msg
            return ("raise", type(exc).__name__, str(exc))
        with open(os.path.join(case, "src", "ntt_shiftadd.h")) as f:
            return ("bytes", f.read())


def _assert_ab_identical(label, aliases_q, shiftadd_mul):
    """recipe=None vs recipe-governed must give the IDENTICAL outcome."""
    recipe, k_env = _recipe_for(aliases_q)
    legacy = _emit_outcome(aliases_q, k_env, shiftadd_mul, recipe=None)
    governed = _emit_outcome(aliases_q, k_env, shiftadd_mul, recipe=recipe)
    assert legacy == governed, (
        "%s: recipe-governed outcome != legacy outcome\n legacy=%r\n governed=%r"
        % (label, legacy[:2], governed[:2]))
    return legacy


# ---- gate config matrix (emit-level) -------------------------------------
REDUCE_SETS = (("TI_9",), ("TI_2", "TI_9"), ("TII_33",), ("TII_2", "TII_33"))
MUL_SETS = (("TI_9",), ("TI_2", "TI_9"), ("TII_33",), ("TII_2", "TII_33"))


def test_kwarg_is_accepted():
    """RED trigger: emit_shiftadd_header must accept keyword-only recipe=."""
    recipe, k_env = _recipe_for(_pairs("TI_9"))
    out = _emit_outcome(_pairs("TI_9"), k_env, shiftadd_mul=False, recipe=recipe)
    assert out[0] == "bytes", "recipe= path did not emit: %r" % (out[:2],)


def test_reduce_routes_ab_identical():
    for names in REDUCE_SETS:
        out = _assert_ab_identical("reduce " + "+".join(names), _pairs(*names),
                                   shiftadd_mul=False)
        assert out[0] == "bytes", (names, out[:2])  # all currently emit
        assert "USE_REDUCE_SHIFTADD 1" in out[1], names


def test_mul_routes_ab_identical():
    for names in MUL_SETS:
        out = _assert_ab_identical("mul " + "+".join(names), _pairs(*names),
                                   shiftadd_mul=True)
        assert out[0] == "bytes", (names, out[:2])
        assert "USE_REDUCE_SHIFTADD_MUL 1" in out[1], names


def test_small_family_reduce_emits_not_unsupported():
    """B1/F1: small-family reduce (solo TII_33 + pair TII_2,TII_33) EMITS via the
    live resolver; S3c must not mark it unsupported."""
    for names in (("TII_33",), ("TII_2", "TII_33")):
        out = _assert_ab_identical("reduce " + "+".join(names), _pairs(*names),
                                   shiftadd_mul=False)
        assert out[0] == "bytes", "small-family reduce refused: %r" % (out[:2],)
        assert "USE_REDUCE_SHIFTADD 1" in out[1], names


def test_mixed_kgroup_parity():
    """F1/F2 (updated for K_group mixed support): mixed {TI_9,TII_33} on a
    shiftadd route EMITS via the per-class mixed emitter, with the IDENTICAL
    outcome with and without a recipe (the recipe's per-class params are
    assert-shimmed against the live per-class resolution)."""
    mixed = _pairs("TI_9", "TII_33")
    for shiftadd_mul in (False, True):
        out = _assert_ab_identical(
            "mixed " + ("mul" if shiftadd_mul else "reduce"), mixed, shiftadd_mul)
        assert out[0] == "bytes", "mixed K_group route must emit: %r" % (out[:2],)
        expected = ("USE_REDUCE_SHIFTADD_MUL 1" if shiftadd_mul
                    else "USE_REDUCE_SHIFTADD 1")
        assert expected in out[1], out[1][:200]
        assert "const int cls = PRIME_CLASS[mod_id];" in out[1]


def test_int_custom_in_family_unchanged():
    """INT/custom in-family (carrying a known 62-bit value): emits identically
    with and without a recipe -> S3c adds NO fail-fast (that is S3d)."""
    q = gc.KNOWN_PRIMES["TI_9"]
    aliases_q = (("INT_%d" % q, q),)
    for shiftadd_mul in (False, True):
        out = _assert_ab_identical("INT in-family", aliases_q, shiftadd_mul)
        assert out[0] == "bytes", "in-family INT should still emit: %r" % (out[:2],)


def test_int_custom_out_of_family_parity():
    """INT/custom out-of-family bit length: whatever the legacy emitter does
    (it proceeds via the <=52->51 quirk then fails the self-check), the recipe
    path does the SAME (recipe not batch_legal -> _resolve_anchor falls back to
    the legacy helper). No behavior change either way."""
    q = (1 << 39) + 7  # 40-bit custom prime
    aliases_q = (("INT_%d" % q, q),)
    recipe, _info_k = build_reduce_recipe(aliases_q), None
    k_env = 52
    legacy = _emit_outcome(aliases_q, k_env, shiftadd_mul=False, recipe=None)
    governed = _emit_outcome(aliases_q, k_env, shiftadd_mul=False, recipe=recipe)
    assert legacy == governed, "out-of-family INT parity broke:\n %r\n %r" % (
        legacy[:2], governed[:2])


def test_recipe_none_is_default():
    """Omitting recipe entirely == passing recipe=None (default legacy path)."""
    with tempfile.TemporaryDirectory() as td:
        for tag in ("a", "b"):
            os.makedirs(os.path.join(td, tag, "src"), exist_ok=True)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            emit_shiftadd_header(case_dir=os.path.join(td, "a"), K_env=62,
                                 aliases_q=list(_pairs("TI_2", "TI_9")),
                                 enabled=True, shiftadd_mul=False,
                                 balanced_selector_scope="all")           # no recipe kwarg
            emit_shiftadd_header(case_dir=os.path.join(td, "b"), K_env=62,
                                 aliases_q=list(_pairs("TI_2", "TI_9")),
                                 enabled=True, shiftadd_mul=False,
                                 balanced_selector_scope="all", recipe=None)
        with open(os.path.join(td, "a", "src", "ntt_shiftadd.h")) as f:
            a = f.read()
        with open(os.path.join(td, "b", "src", "ntt_shiftadd.h")) as f:
            b = f.read()
        assert a == b, "default arg != explicit recipe=None"


def main():
    tests = [
        test_kwarg_is_accepted,
        test_reduce_routes_ab_identical,
        test_mul_routes_ab_identical,
        test_small_family_reduce_emits_not_unsupported,
        test_mixed_kgroup_parity,
        test_int_custom_in_family_unchanged,
        test_int_custom_out_of_family_parity,
        test_recipe_none_is_default,
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

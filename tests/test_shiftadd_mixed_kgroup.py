#!/usr/bin/env python3
"""shiftadd K_group mixed-prime support — semantics lock.

User K_group semantics (binding):
  * "same-bit" = same K_group / arithmetic recipe group, NOT identical
    q.bit_length(). 51/52-bit TII primes are ONE K_group=52; 61/62-bit TI
    primes are the K_group=62 envelope.
  * A set is "mixed-bit" for generation ONLY when it spans multiple K_groups
    ({52,62}); such a set uses cls-based per-class shiftadd recipe selection
    (PRIME_CLASS / cls), never a blind single-anchor recipe and never a
    bit-length-spread rejection.

Locks: classification (TII 51/52 same; TI+TII mixed {52,62}); ReduceRecipe
two-family legality + per-class cls_recipes; emit-level mixed reduce/mul
emission with PRIME_CLASS/cls selection and per-class anchors; single-family
output free of mixed markers (structural regression); recipe-vs-None parity.
Run (no pytest needed):  ~/autontt-venv/bin/python3 tests/test_shiftadd_mixed_kgroup.py"""
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

TII_ALL = tuple("TII_%d" % i for i in range(1, 42))


def _pairs(*names):
    return tuple((n, gc.KNOWN_PRIMES[n]) for n in names)


def _recipe_for(aliases_q):
    info = classify_reduce_set(aliases_q, None)
    return build_reduce_recipe(aliases_q, k_env=info.final_k, classify_info=info), info.final_k


def _emit_outcome(aliases_q, k_env, shiftadd_mul, recipe):
    """('bytes', content) on success / ('raise', exc_type, msg) on failure."""
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
        except Exception as exc:  # noqa: BLE001 - outcome capture
            return ("raise", type(exc).__name__, str(exc))
        with open(os.path.join(case, "src", "ntt_shiftadd.h")) as f:
            return ("bytes", f.read())


# ---- 1. classification locks (user K_group semantics) ---------------------

def test_cls_tii_51_52_same_kgroup():
    """TII_1 is 51-bit, TII_33 is 52-bit: SAME K_group=52, never mixed."""
    q1, q33 = gc.KNOWN_PRIMES["TII_1"], gc.KNOWN_PRIMES["TII_33"]
    assert q1.bit_length() == 51 and q33.bit_length() == 52, (
        q1.bit_length(), q33.bit_length())
    info = classify_reduce_set(_pairs("TII_1", "TII_33"), None)
    assert info.mode == "same_bit", info
    assert info.class_map == (52,), info
    assert info.k_group == 52 and info.final_k == 52, info


def test_cls_tii_family_same_kgroup():
    info = classify_reduce_set(_pairs(*TII_ALL), None)
    assert info.mode == "same_bit" and info.class_map == (52,), info


def test_cls_ti1_plus_tii_mixed_52_62():
    """TI_1 (61-bit) + TII set = mixed K_group {52,62}; cls0=52, cls1=62."""
    assert gc.KNOWN_PRIMES["TI_1"].bit_length() == 61
    info = classify_reduce_set(_pairs("TI_1", "TII_1", "TII_33"), None)
    assert info.mode == "mixed_bit", info
    assert info.class_map == (52, 62), info
    # set-relative classes: TI_1 -> cls1 (62-family), TIIs -> cls0 (52-family)
    rel = {alias: r for alias, _q, _k, r in info.per_prime}
    assert rel["TI_1"] == 1 and rel["TII_1"] == 0 and rel["TII_33"] == 0, rel


# ---- 2. ReduceRecipe two-family legality + per-class structure ------------

def test_recipe_mixed_two_family_legal():
    aliases_q = _pairs("TI_1", "TII_1", "TII_33")
    recipe, _k = _recipe_for(aliases_q)
    assert recipe.batch_legal_shiftadd, recipe.fail_fast_reasons
    assert recipe.fail_fast_reasons == (), recipe.fail_fast_reasons
    assert recipe.can_emit_shiftadd_reduce, recipe.reduce_resolver_reason
    assert recipe.can_emit_shiftadd_mul
    assert recipe.cls_recipes is not None and len(recipe.cls_recipes) == 2
    c0, c1 = recipe.cls_recipes
    assert (c0.k_class, c0.b_anchor) == (52, 51), c0
    assert (c1.k_class, c1.b_anchor) == (62, 61), c1
    assert c0.aliases == ("TII_1", "TII_33") and c1.aliases == ("TI_1",)
    assert c0.reduce_resolver_emit_ok and c1.reduce_resolver_emit_ok
    assert c0.reduce_resolver_params is not None
    assert c1.reduce_resolver_params is not None
    assert c0.mul_ok and c1.mul_ok


def test_recipe_single_family_has_no_cls_recipes():
    """Single-family recipes keep the existing shape (cls_recipes is None)."""
    for names in (("TI_9",), ("TII_2", "TII_33"), TII_ALL):
        recipe, _k = _recipe_for(_pairs(*names))
        assert recipe.cls_recipes is None, names
        assert recipe.batch_legal_shiftadd, names


def test_recipe_custom_wide_mix_still_illegal():
    """A wide spread WITHOUT two known families (custom 40-bit + TI_9) keeps
    the old illegality (unknown anchor reasons; never the new mixed path)."""
    q = (1 << 39) + 7
    aliases_q = (("INT_%d" % q, q), ("TI_9", gc.KNOWN_PRIMES["TI_9"]))
    recipe = build_reduce_recipe(aliases_q)
    assert not recipe.batch_legal_shiftadd
    assert recipe.cls_recipes is None
    assert any("outside {51, 52, 61, 62}" in r for r in recipe.fail_fast_reasons), (
        recipe.fail_fast_reasons)


# ---- 3. emit-level mixed generation (both routes) --------------------------

def test_emit_mixed_reduce_generates_with_cls_selection():
    aliases_q = _pairs("TI_1", "TII_1", "TII_33")
    recipe, k_env = _recipe_for(aliases_q)
    out = _emit_outcome(aliases_q, k_env, shiftadd_mul=False, recipe=recipe)
    assert out[0] == "bytes", "mixed reduce must emit: %r" % (out[:3],)
    body = out[1]
    assert "USE_REDUCE_SHIFTADD 1" in body
    # cls-based recipe selection, not per-prime datapath duplication:
    assert "const int cls = PRIME_CLASS[mod_id];" in body, "cls read missing"
    assert body.count("inline Data reduce_shiftadd(") == 1, "one function only"
    # both class anchors present (per-class fold bases)
    assert "B = 2^51" in body and "B = 2^61" in body, "per-class anchors missing"
    # final per-class result select
    assert "cls ?" in body, "cls result select missing"
    # per-prime predicates exist once each (global mod_id indices)
    for i, (alias, _q) in enumerate(aliases_q):
        clean = "".join(ch for ch in alias if ch.isalnum())
        decl = "const bool is_%s = (mod_id == (ModId)%d);" % (clean, i)
        assert body.count(decl) == 1, decl


def test_emit_mixed_mul_generates_with_cls_selection():
    aliases_q = _pairs("TI_1", "TII_1", "TII_33")
    recipe, k_env = _recipe_for(aliases_q)
    out = _emit_outcome(aliases_q, k_env, shiftadd_mul=True, recipe=recipe)
    assert out[0] == "bytes", "mixed mul must emit: %r" % (out[:3],)
    body = out[1]
    assert "USE_REDUCE_SHIFTADD_MUL 1" in body
    assert "const int cls = PRIME_CLASS[mod_id];" in body
    assert body.count("inline Data reduce_shiftadd_mul(") == 1
    assert "cls ?" in body
    # per-class Barrett bases
    assert ">> 51" in body and ">> 61" in body, "per-class mul bases missing"


def test_emit_mixed_recipe_vs_none_parity():
    """The mixed dispatch is recipe-independent: recipe-governed and
    recipe=None paths must produce byte-identical output (assert-shim rule)."""
    aliases_q = _pairs("TI_1", "TII_1", "TII_33")
    recipe, k_env = _recipe_for(aliases_q)
    for shiftadd_mul in (False, True):
        legacy = _emit_outcome(aliases_q, k_env, shiftadd_mul, recipe=None)
        governed = _emit_outcome(aliases_q, k_env, shiftadd_mul, recipe=recipe)
        assert legacy == governed, (
            "mixed recipe parity broke (mul=%s):\n %r\n %r"
            % (shiftadd_mul, legacy[:2], governed[:2]))
        assert legacy[0] == "bytes", legacy[:3]


def test_emit_mixed_larger_set_ti1_tii_all():
    """TI_1 + TII_1..41 (the target case) emits on both routes."""
    aliases_q = _pairs("TI_1", *TII_ALL)
    recipe, k_env = _recipe_for(aliases_q)
    assert k_env == 62
    for shiftadd_mul in (False, True):
        out = _emit_outcome(aliases_q, k_env, shiftadd_mul, recipe=recipe)
        assert out[0] == "bytes", (
            "TI_1+TII_all (mul=%s) must emit: %r" % (shiftadd_mul, out[:3]))


# ---- 4. structural regression: single-family output has NO mixed markers --

def test_single_family_output_free_of_mixed_markers():
    for names, shiftadd_mul in ((("TI_2", "TI_9"), False),
                                (("TII_2", "TII_33"), True),
                                (TII_ALL, False)):
        recipe, k_env = _recipe_for(_pairs(*names))
        out = _emit_outcome(_pairs(*names), k_env, shiftadd_mul, recipe=recipe)
        assert out[0] == "bytes", (names, out[:3])
        body = out[1]
        assert "PRIME_CLASS" not in body, "single-family must not read PRIME_CLASS"
        assert "_c0" not in body and "_c1" not in body, (
            "single-family must not carry class-suffixed names")


def main():
    tests = [
        test_cls_tii_51_52_same_kgroup,
        test_cls_tii_family_same_kgroup,
        test_cls_ti1_plus_tii_mixed_52_62,
        test_recipe_mixed_two_family_legal,
        test_recipe_single_family_has_no_cls_recipes,
        test_recipe_custom_wide_mix_still_illegal,
        test_emit_mixed_reduce_generates_with_cls_selection,
        test_emit_mixed_mul_generates_with_cls_selection,
        test_emit_mixed_recipe_vs_none_parity,
        test_emit_mixed_larger_set_ti1_tii_all,
        test_single_family_output_free_of_mixed_markers,
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

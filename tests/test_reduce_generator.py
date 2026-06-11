#!/usr/bin/env python3
"""T0 unit tests for the Phase 3.5 S1a-0 reduce seam (reduce_generator.py).

Oracle = the committed template bytes: the [reduce] seam content must equal the
verbatim emitter constant, and marker-inclusive injection must reproduce the
template with the marker lines removed (byte-for-byte).
Run (no pytest needed):  ~/autontt-venv/bin/python3 tests/test_reduce_generator.py"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from code_generator import reduce_generator as rg
import generate_code as gc


def _pairs(*names):
    return tuple((n, gc.KNOWN_PRIMES[n]) for n in names)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TPL = os.path.join(ROOT, "templates", "ntt.cpp")  # flat template layout (2026-06-11)
GB = "// [reduce:begin]"
GE = "// [reduce:end]"


def _read_template():
    with open(TPL, "r") as f:
        return f.read()


def _inject(src):
    """The exact generate_code.py injection idiom (markers-inclusive replacement)."""
    out = src
    for mk in rg.REDUCE_MARKERS:
        gb = "// [%s:begin]" % mk
        ge = "// [%s:end]" % mk
        i = out.index(gb)
        j = out.index(ge) + len(ge)
        out = out[:i] + rg.gen_reduce_block({}) + out[j:]
    return out


def test_emitter_equals_template_span():
    src = _read_template()
    i = src.index(GB)
    j = src.index(GE) + len(GE)
    assert src[i:j] == GB + "\n" + rg._REDUCE_BLOCK_VERBATIM + "\n" + GE, \
        "template [reduce] seam != verbatim constant"


def test_whole_file_reconstruction_equals_marker_stripped_template():
    src = _read_template()
    stripped = "".join(
        line for line in src.splitlines(keepends=True)
        if line.rstrip("\n") not in (GB, GE)
    )
    assert _inject(src) == stripped, "injection != marker-line-stripped template"


def test_no_marker_residue_after_injection():
    out = _inject(_read_template())
    assert "// [reduce:" not in out, "marker residue in injected output"


def test_legacy_equivalence_all_today_modes():
    """S1c (supersedes the S1a struct-independence test): every TODAY-legal
    struct — and absent metadata — must render byte-equal to the verbatim anchor."""
    mixed = {"_reduce_class_mode": "mixed_bit", "_reduce_class_map": (52, 62),
             "_reduce_final_k": 62, "_reduce_k_group": 62}
    same62 = {"_reduce_class_mode": "same_bit", "_reduce_class_map": (62,),
              "_reduce_final_k": 62, "_reduce_k_group": 62}
    same52 = {"_reduce_class_mode": "same_bit", "_reduce_class_map": (52,),
              "_reduce_final_k": 52, "_reduce_k_group": 52}
    single = {"_reduce_class_mode": "single", "_reduce_class_map": (62,),
              "_reduce_final_k": 62, "_reduce_k_group": 62}
    for st in ({}, mixed, same62, same52, single):
        assert rg.gen_reduce_block(st) == rg._REDUCE_BLOCK_VERBATIM, st


def test_render_anchor_and_repeatability():
    a = rg._render_reduce_block(52, 62)
    b = rg._render_reduce_block(52, 62)
    assert a == b == rg._REDUCE_BLOCK_VERBATIM


def test_template_integrity():
    t = rg._REDUCE_BLOCK_TEMPLATE
    for tok in ("@KS@", "@KSm1@", "@KSp1@", "@KSp2@", "@KL@", "@KLm1@"):
        assert tok in t, "missing token %s" % tok
    for d in rg._CLASS_DIGITS:
        assert d not in t, "residual class numeral %s in template" % d
    assert "@K" not in rg._render_reduce_block(52, 62), "residual token after render"


def test_params_for_struct_table():
    mixed = {"_reduce_class_mode": "mixed_bit", "_reduce_class_map": (52, 62),
             "_reduce_final_k": 62, "_reduce_k_group": 62}
    assert rg._reduce_params_for_struct(mixed) == (52, 62)
    assert rg._reduce_params_for_struct({}) == (52, 62)
    assert rg._reduce_params_for_struct({"_reduce_class_mode": "same_bit",
                                         "_reduce_class_map": (52,)}) == (52, 62)
    assert rg._reduce_params_for_struct({"_reduce_class_mode": "single",
                                         "_reduce_class_map": (62,)}) == (52, 62)


def test_mixed_failfasts():
    base = {"_reduce_class_mode": "mixed_bit", "_reduce_class_map": (52, 62),
            "_reduce_final_k": 62, "_reduce_k_group": 62}
    bads = [dict(base, _reduce_final_k=52),                 # k_large != final_K
            dict(base, _reduce_k_group=52),                 # K_group mismatch
            dict(base, _reduce_class_map=(52,)),            # not 2 classes
            dict(base, _reduce_class_map=None),             # missing map
            dict(base, _reduce_class_map=(51, 61),
                 _reduce_final_k=61, _reduce_k_group=61)]   # outside domain
    for st in bads:
        try:
            rg.gen_reduce_block(st)
        except RuntimeError:
            pass
        else:
            raise AssertionError("no fail-fast for %r" % (st,))


def test_renderer_is_permissive_pure():
    alt = rg._render_reduce_block(51, 61)
    assert alt != rg._REDUCE_BLOCK_VERBATIM
    assert "U >> 50" in alt and "ap_uint<51>" in alt and "<< 60" in alt
    assert "@K" not in alt


def _st(mode, cmap, fk, kg, route="none", cls_mode="normalized"):
    return {"_alias_reduce_cls_mode": cls_mode, "_reduce_shiftadd_mode": route,
            "_reduce_class_mode": mode, "_reduce_class_map": cmap,
            "_reduce_final_k": fk, "_reduce_k_group": kg}


def test_s1d_legacy_mode_byte_frozen_incl_explicit():
    for cls_mode in ("legacy",):
        for st in (_st("same_bit", (62,), 62, 62, cls_mode=cls_mode),
                   _st("single", (52,), 52, 52, cls_mode=cls_mode),
                   _st("mixed_bit", (52, 62), 62, 62, cls_mode=cls_mode)):
            assert rg.gen_reduce_block(st) == rg._REDUCE_BLOCK_VERBATIM, st
    assert rg.gen_reduce_block({}) == rg._REDUCE_BLOCK_VERBATIM  # unset = legacy


def test_s1d_one_class_structural():
    for st, k_class, sym in ((_st("same_bit", (62,), 62, 62), 62, True),
                             (_st("single", (62,), 62, 62), 62, True),
                             (_st("same_bit", (52,), 52, 52), 52, False),
                             (_st("single", (52,), 52, 52), 52, False)):
        t = rg.gen_reduce_block(st)
        for residue in ("PRIME_CLASS", "_cls0", "cls ?", "const int cls"):
            assert residue not in t, (residue, st)
        assert "#ifndef SINGLE_PRIME" in t and "USE_REDUCE_SHIFTADD" in t
        assert "BARRETT_MUS[mod_id]" in t and "Data2 T = BARRETT_MU;" in t
        assert "ap_uint<62>" in t and "<< 61" in t, "shared superset widths kept"
        if sym:
            assert "(K - 1)" in t and "(K + 2)" in t, "one-class-62 = symbolic-K side"
        else:
            assert all(s in t for s in ("U >> 51", "ap_uint<52>", "<< 52", ">> 53", "<< 54")), \
                "one-class-52 = cls0-family constants from K_group"
            assert "(K - 1)" not in t and "(K + 2)" not in t


def test_s1d_k_group_not_final_k():
    a = rg.gen_reduce_block(_st("same_bit", (52,), 52, 52))
    b = rg.gen_reduce_block(_st("same_bit", (52,), 62, 52))  # raised envelope (cfg6)
    assert a == b, "class arithmetic must come from K_group, not final_K"


def test_s1d_mixed_and_shiftadd_unchanged_in_normalized():
    assert rg.gen_reduce_block(_st("mixed_bit", (52, 62), 62, 62)) == rg._REDUCE_BLOCK_VERBATIM
    for route in ("reduce", "mul"):
        assert rg.gen_reduce_block(_st("same_bit", (62,), 62, 62, route=route)) == \
            rg._REDUCE_BLOCK_VERBATIM, route


def test_s1d_failfasts():
    bads = [_st("same_bit", (62,), 62, 62, cls_mode="weird"),       # invalid mode
            _st("same_bit", (52, 62), 62, 62),                      # not a 1-class map
            _st("same_bit", (51,), 51, 51),                         # domain
            _st("same_bit", (62,), 62, 52),                         # K_group mismatch
            _st("same_bit", (62,), 52, 62)]                         # one-class-62 needs final 62
    for st in bads:
        try:
            rg.gen_reduce_block(st)
        except RuntimeError:
            pass
        else:
            raise AssertionError("no fail-fast for %r" % (st,))


def test_block_content_sanity():
    blk = rg.gen_reduce_block({})
    for tok in ("USE_REDUCE_SHIFTADD", "USE_REDUCE_SHIFTADD_MUL", "PRIME_CLASS[mod_id]",
                "mask_cls0", "ring_cls0", "mul_full_data_nonstd(A, B)", "#pragma HLS INLINE"):
        assert tok in blk, "expected token missing: %s" % tok
    assert "/*" not in blk, "dead commented reduce must stay outside the seam"
    assert "// [reduce:" not in blk, "emitter must not contain markers"
    assert not blk.endswith("\n"), "verbatim constant must have NO trailing newline"


def test_negative_self_check_uses_tmp_copy_only():
    with tempfile.TemporaryDirectory() as td:
        # (a) perturbed seam content -> RuntimeError
        bad = os.path.join(td, "ntt_perturbed.cpp")
        shutil.copy2(TPL, bad)
        with open(bad, "r") as f:
            src = f.read()
        # perturb one byte strictly INSIDE the seam span (index-based: content-agnostic;
        # the file's first "#pragma HLS INLINE" lives in bitrev(), OUTSIDE the seam)
        mid = (src.index(GB) + len(GB) + 1 + src.index(GE)) // 2
        with open(bad, "w") as f:
            f.write(src[:mid] + ("X" if src[mid] != "X" else "Y") + src[mid + 1:])
        try:
            rg._s1a_anchor_self_check(template_path=bad)
        except RuntimeError:
            pass
        else:
            raise AssertionError("self-check did not fail on a perturbed tmp copy")
        # (b) missing markers -> RuntimeError
        nomark = os.path.join(td, "ntt_nomarkers.cpp")
        with open(nomark, "w") as f:
            f.write("int x;\n")
        try:
            rg._s1a_anchor_self_check(template_path=nomark)
        except RuntimeError:
            pass
        else:
            raise AssertionError("self-check did not fail on missing markers")
    # real template untouched by this test
    rg._s1a_anchor_self_check()


def test_classify_no_k_modes_and_final_k():
    cases = [
        (_pairs("TI_9"), "single", (62,), 62),
        (_pairs("TII_33"), "single", (52,), 52),
        (_pairs("TI_2", "TI_9"), "same_bit", (62,), 62),
        (_pairs("TII_2", "TII_33"), "same_bit", (52,), 52),
        (_pairs("TI_9", "TII_33"), "mixed_bit", (52, 62), 62),
    ]
    for pairs, mode, cmap, fk in cases:
        info = rg.classify_reduce_set(pairs, None)
        assert info.mode == mode, (pairs, info.mode)
        assert info.class_map == cmap, (pairs, info.class_map)
        assert info.k_group == cmap[-1] and info.final_k == fk
        assert info.config_k is None and info.provenance == "inferred"


def test_classify_k_pinned_preserve_and_lift():
    tii = _pairs("TII_2", "TII_33")
    a = rg.classify_reduce_set(tii, 62)
    assert (a.final_k, a.provenance, a.mode, a.class_map) == (62, "preserved", "same_bit", (52,))
    b = rg.classify_reduce_set(tii, 52)
    assert (b.final_k, b.provenance) == (52, "preserved")
    c = rg.classify_reduce_set(_pairs("TI_9"), 52)
    assert (c.final_k, c.provenance, c.k_group) == (62, "lifted", 62)
    d = rg.classify_reduce_set(_pairs("TI_9"), 62)
    assert (d.final_k, d.provenance) == (62, "preserved")


def test_classify_mixed_relative_classes():
    info = rg.classify_reduce_set(_pairs("TI_9", "TII_33"), None)
    assert info.class_map == (52, 62), "cls0 must be the SMALLER class present"
    rel = {alias: r for alias, _q, _k, r in info.per_prime}
    ka = {alias: k for alias, _q, k, _r in info.per_prime}
    assert rel["TII_33"] == 0 and ka["TII_33"] == 52, "TII_33 = small class (cls0)"
    assert rel["TI_9"] == 1 and ka["TI_9"] == 62, "TI_9 = large class (cls1)"


def test_classify_rejects_wide_prime_and_empty_set():
    try:
        rg.classify_reduce_set((("X", 2 ** 63),), None)
    except RuntimeError:
        pass
    else:
        raise AssertionError("did not reject a >62-bit prime")
    try:
        rg.classify_reduce_set((), None)
    except ValueError:
        pass
    else:
        raise AssertionError("did not reject an empty prime set")


def test_validate_k_optional_and_range():
    base = {"primes": ["TI_9"], "tfg_seg_len": 16}
    gc.validate_config_seg(dict(base))  # no K: must validate cleanly now
    gc.validate_config_seg(dict(base, K=62))
    for bad in (0, 63, "62", True, 52.0):
        try:
            gc.validate_config_seg(dict(base, K=bad))
        except ValueError:
            pass
        else:
            raise AssertionError("validate accepted bad K=%r" % (bad,))


def main():
    tests = [
        test_emitter_equals_template_span,
        test_whole_file_reconstruction_equals_marker_stripped_template,
        test_no_marker_residue_after_injection,
        test_legacy_equivalence_all_today_modes,
        test_render_anchor_and_repeatability,
        test_template_integrity,
        test_params_for_struct_table,
        test_mixed_failfasts,
        test_renderer_is_permissive_pure,
        test_s1d_legacy_mode_byte_frozen_incl_explicit,
        test_s1d_one_class_structural,
        test_s1d_k_group_not_final_k,
        test_s1d_mixed_and_shiftadd_unchanged_in_normalized,
        test_s1d_failfasts,
        test_block_content_sanity,
        test_negative_self_check_uses_tmp_copy_only,
        test_classify_no_k_modes_and_final_k,
        test_classify_k_pinned_preserve_and_lift,
        test_classify_mixed_relative_classes,
        test_classify_rejects_wide_prime_and_empty_set,
        test_validate_k_optional_and_range,
    ]
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

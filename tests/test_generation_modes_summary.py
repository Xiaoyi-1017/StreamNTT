#!/usr/bin/env python3
"""Phase 3.5 S5 — stdout-only generation-modes summary line.

One compact "generation-modes:" line per generated case, printed after the
reduce-class/K lines and before the detailed S3e shiftadd-recipe block. Fields:
  shiftadd_route reduce_cls_mode shiftadd_single_prime_mode reduce_recipe
  output_mode default_flip=false
output_mode is DERIVED (a normalized env that is set-but-INERT for the config
reports legacy-byte-identical). STDOUT-ONLY: no generated-file change. These are
generate-level tests (capture stdout); the byte gate vs b5faadc is run in S5
validation.
Run (no pytest needed):  ~/autontt-venv/bin/python3 tests/test_generation_modes_summary.py"""
import contextlib
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import generate_code as gc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GENERATED = os.path.join(ROOT, "generated")


def _cfg(primes, shiftadd, K=None):
    c = dict(N=1024, BU=8, CH=4, RATE=0.5, tfg_seg_len=16,
             primes=list(primes), shiftadd=shiftadd, name="s5_probe")
    if K is not None:
        c["K"] = K
    return c


def _run(cfg, env=None, suffix="s5test"):
    """Generate (capturing stdout); return (stdout, raised, case_name|None).
    env: dict of STREAMNTT_* overrides for this call only (restored after)."""
    saved = {}
    for k, v in (env or {}).items():
        saved[k] = os.environ.get(k)
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    buf = io.StringIO()
    raised = None
    name = None
    try:
        with contextlib.redirect_stdout(buf):
            name, _d = gc.generate_seg_case(cfg, suffix=suffix, force=True)
    except Exception as exc:  # noqa: BLE001
        raised = (type(exc).__name__, str(exc))
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return buf.getvalue(), raised, name


def _modes_line(out):
    for ln in out.splitlines():
        if ln.startswith("generation-modes:"):
            return ln
    return None


def _clean_env():
    return dict(STREAMNTT_REDUCE_CLS_MODE=None, STREAMNTT_SHIFTADD_SINGLE_PRIME_MODE=None)


def test_appears_for_none_reduce_mul():
    for route in ("none", "reduce", "mul"):
        out, raised, _ = _run(_cfg(["TI_9"], route), env=_clean_env())
        assert raised is None, (route, raised)
        line = _modes_line(out)
        assert line is not None, "generation-modes line missing for shiftadd=%s" % route
        assert ("shiftadd_route=%s" % route) in line, line
        assert "default_flip=false" in line, line


def test_default_unset_is_legacy_byte_identical():
    out, _r, _n = _run(_cfg(["TI_9"], "reduce"), env=_clean_env())
    line = _modes_line(out)
    assert "reduce_cls_mode=legacy" in line and "shiftadd_single_prime_mode=legacy" in line, line
    assert "output_mode=legacy-byte-identical" in line, line
    assert "reduce_recipe=active" in line, line  # reduce route -> recipe constructed


def test_recipe_none_for_barrett():
    out, _r, _n = _run(_cfg(["TI_2", "TI_9"], "none"), env=_clean_env())
    line = _modes_line(out)
    assert "shiftadd_route=none" in line and "reduce_recipe=none" in line, line
    assert "output_mode=legacy-byte-identical" in line, line


def test_active_barrett_normalized():
    """reduce_cls_mode=normalized + shiftadd=none + class_mode single -> ACTIVE."""
    out, _r, _n = _run(_cfg(["TI_9"], "none"),
                       env=dict(STREAMNTT_REDUCE_CLS_MODE="normalized",
                                STREAMNTT_SHIFTADD_SINGLE_PRIME_MODE=None))
    line = _modes_line(out)
    assert "reduce_cls_mode=normalized" in line, line
    assert "output_mode=normalized-source-cleanup" in line, line
    assert "reduce_recipe=none" in line, line


def test_active_single_prime_shiftadd_normalized():
    """shiftadd_single_prime_mode=normalized + shiftadd=reduce + NP==1 -> ACTIVE."""
    out, _r, _n = _run(_cfg(["TI_9"], "reduce"),
                       env=dict(STREAMNTT_REDUCE_CLS_MODE=None,
                                STREAMNTT_SHIFTADD_SINGLE_PRIME_MODE="normalized"))
    line = _modes_line(out)
    assert "shiftadd_single_prime_mode=normalized" in line, line
    assert "output_mode=normalized-source-cleanup" in line, line
    assert "reduce_recipe=active" in line, line


def test_inert_single_prime_mode_on_multiprime():
    """SHIFTADD_SINGLE_PRIME_MODE=normalized on a SAME-BIT multi-prime shiftadd is INERT
    (NP>=2) -> output_mode must stay legacy-byte-identical even though the env is set."""
    out, _r, _n = _run(_cfg(["TI_2", "TI_9"], "reduce"),
                       env=dict(STREAMNTT_REDUCE_CLS_MODE=None,
                                STREAMNTT_SHIFTADD_SINGLE_PRIME_MODE="normalized"))
    line = _modes_line(out)
    assert "shiftadd_single_prime_mode=normalized" in line, line
    assert "output_mode=legacy-byte-identical" in line, "inert mode must not claim cleanup: %s" % line


def test_inert_cls_normalized_on_shiftadd_route():
    """REDUCE_CLS_MODE=normalized is INERT on a shiftadd route (applies to Barrett only)."""
    out, _r, _n = _run(_cfg(["TI_9"], "reduce"),
                       env=dict(STREAMNTT_REDUCE_CLS_MODE="normalized",
                                STREAMNTT_SHIFTADD_SINGLE_PRIME_MODE=None))
    line = _modes_line(out)
    assert "reduce_cls_mode=normalized" in line, line
    assert "output_mode=legacy-byte-identical" in line, line  # cls normalized inert on shiftadd


def test_line_not_in_generated_src():
    out, _r, name = _run(_cfg(["TI_9"], "reduce"), env=_clean_env())
    assert name is not None
    srcdir = os.path.join(GENERATED, name, "src")
    for fn in os.listdir(srcdir):
        with open(os.path.join(srcdir, fn)) as f:
            assert "generation-modes:" not in f.read(), \
                "summary leaked into generated file %s" % fn


def test_exactly_one_line():
    out, _r, _n = _run(_cfg(["TI_9"], "mul"), env=_clean_env())
    n = sum(1 for ln in out.splitlines() if ln.startswith("generation-modes:"))
    assert n == 1, "expected exactly 1 generation-modes line, got %d" % n


def main():
    tests = [
        test_appears_for_none_reduce_mul,
        test_default_unset_is_legacy_byte_identical,
        test_recipe_none_for_barrett,
        test_active_barrett_normalized,
        test_active_single_prime_shiftadd_normalized,
        test_inert_single_prime_mode_on_multiprime,
        test_inert_cls_normalized_on_shiftadd_route,
        test_line_not_in_generated_src,
        test_exactly_one_line,
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

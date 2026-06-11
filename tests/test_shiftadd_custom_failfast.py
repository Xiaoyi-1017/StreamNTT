#!/usr/bin/env python3
"""Phase 3.5 S3d — shiftadd custom/INT prime fail-fast (generate_code-level).

S3d behavior change: on the shiftadd routes (reduce/mul) ONLY, a prime set
containing any unsupported_custom (INT/custom) prime must FAIL-FAST with a clear
ValueError raised BEFORE any case directory / file is created. Barrett /
shiftadd=none is unaffected (recipe-free, universal). Known-prime shiftadd output
is byte-identical (the byte gate vs ecb9658 is run separately in validation).

The gate lives in generate_code (the shiftadd-route boundary); these tests drive
the real generate_seg_case path. INT value = KNOWN_PRIMES["TI_9"] (bit_length 62,
RECOGNIZED) -> proves the S3a-D gap (recognized-bit custom previously EMITTED).
Run (no pytest needed):  ~/autontt-venv/bin/python3 tests/test_shiftadd_custom_failfast.py"""
import contextlib
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import generate_code as gc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GENERATED = os.path.join(ROOT, "generated")
TI9_Q = gc.KNOWN_PRIMES["TI_9"]      # 2305843009224179713 (62-bit, recognized)
TII33_Q = gc.KNOWN_PRIMES["TII_33"]  # 2251799851958273 (52-bit, recognized)


def _cfg(primes, shiftadd):
    return dict(N=1024, BU=8, CH=4, RATE=0.5, tfg_seg_len=16,
                primes=list(primes), shiftadd=shiftadd, name="s3d_probe")


def _generated_dirs():
    return set(os.listdir(GENERATED)) if os.path.isdir(GENERATED) else set()


def _run(cfg, suffix="s3dtest"):
    """Return ('raise', ExcType, msg, new_dirs) or ('ok', case_name, new_dirs).
    new_dirs = generated/ entries created during the call (for no-partial-output)."""
    before = _generated_dirs()
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            case_name, _case_dir = gc.generate_seg_case(cfg, suffix=suffix, force=True)
    except Exception as exc:  # noqa: BLE001
        new_dirs = _generated_dirs() - before
        return ("raise", type(exc).__name__, str(exc), new_dirs)
    new_dirs = _generated_dirs() - before
    return ("ok", case_name, new_dirs)


def test_t1_int_reduce_failfast():
    out = _run(_cfg([TI9_Q], "reduce"))
    assert out[0] == "raise", "INT+reduce should fail-fast, got %r" % (out[:2],)
    assert out[1] == "ValueError", out[:3]
    assert "shiftadd=reduce" in out[2] and "custom" in out[2], out[2]
    # no-partial-output: the fail-fast precedes case_dir creation.
    assert out[3] == set(), "partial output created: %r" % (out[3],)


def test_t2_int_mul_failfast():
    out = _run(_cfg([TI9_Q], "mul"))
    assert out[0] == "raise" and out[1] == "ValueError", out[:3]
    assert "shiftadd=mul" in out[2] and "custom" in out[2], out[2]
    assert out[3] == set(), "partial output created: %r" % (out[3],)


def test_t3_int_none_still_generates():
    """Barrett/none with the SAME custom prime is UNAFFECTED (universal)."""
    out = _run(_cfg([TI9_Q], "none"))
    assert out[0] == "ok", "INT+none should still generate, got %r" % (out[:3],)
    # ntt_shiftadd.h must be the disabled stub (no recipe on the none route).
    case_dir = os.path.join(GENERATED, out[1])
    with open(os.path.join(case_dir, "src", "ntt_shiftadd.h")) as f:
        body = f.read()
    assert "shiftadd disabled: stub header" in body, "none route should emit the stub"


def test_t4_known_ti9_reduce_succeeds():
    out = _run(_cfg(["TI_9"], "reduce"))
    assert out[0] == "ok", "known TI_9 reduce should succeed, got %r" % (out[:3],)


def test_t5_known_pair_mul_succeeds():
    out = _run(_cfg(["TI_2", "TI_9"], "mul"))
    assert out[0] == "ok", "known TI_2,TI_9 mul should succeed, got %r" % (out[:3],)


def test_t6_known_mixed_kgroup_reduce_generates():
    """All-known K_group-mixed {52,62} sets GENERATE on shiftadd=reduce
    (per-class mixed emitter); the custom gate must not fire."""
    out = _run(_cfg(["TI_9", "TII_33"], "reduce"))
    assert out[0] == "ok", "known mixed K_group reduce must generate: %r" % (out[:3],)


def test_t7_known_mixed_kgroup_mul_generates():
    out = _run(_cfg(["TI_9", "TII_33"], "mul"))
    assert out[0] == "ok", "known mixed K_group mul must generate: %r" % (out[:3],)


def test_t8_custom_plus_mixed_custom_fires_first():
    """A set that is BOTH custom AND mixed-width -> the custom gate fires FIRST
    (earlier boundary), NOT the K_spread gate."""
    out = _run(_cfg(["TI_2", TII33_Q], "reduce"))
    assert out[0] == "raise" and out[1] == "ValueError", out[:3]
    assert "custom" in out[2], "custom gate should fire first: %s" % out[2]
    assert "K_spread" not in out[2], "K_spread must not pre-empt the custom gate: %s" % out[2]
    assert out[3] == set(), "partial output created: %r" % (out[3],)


def test_t9_message_content():
    out = _run(_cfg([TI9_Q], "reduce"))
    assert out[0] == "raise", out[:2]
    msg = out[2]
    for token in ("shiftadd=reduce", "INT_%d" % TI9_Q, "q=%d" % TI9_Q,
                  "custom", "KNOWN_PRIMES", 'shiftadd="none"'):
        assert token in msg, "missing %r in error message:\n%s" % (token, msg)


def main():
    tests = [
        test_t1_int_reduce_failfast,
        test_t2_int_mul_failfast,
        test_t3_int_none_still_generates,
        test_t4_known_ti9_reduce_succeeds,
        test_t5_known_pair_mul_succeeds,
        test_t6_known_mixed_kgroup_reduce_generates,
        test_t7_known_mixed_kgroup_mul_generates,
        test_t8_custom_plus_mixed_custom_fires_first,
        test_t9_message_content,
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

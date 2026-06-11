#!/usr/bin/env python3
"""Tests for the ICBU bind_storage DSE apply path (modes recommend/apply/apply_override).

Run (no pytest needed):  ~/autontt-venv/bin/python3 tests/test_dse_apply_mode.py
Exit 0 = all pass; non-zero = a failure (prints the offending test).

Generates real N1024 single-prime Barrett cases into generated/ (gitignored, suffix
"dseapply"). Calibration evidence comes from SMALL TEMP FIXTURE manifests injected via
STREAMNTT_ICBU_DSE_CALIBRATION_ROOT — no dependency on Checkpoint/ raw logs.

Fixture arithmetic (N=1024 BU8 CH4 K62; board U55C BRAM 3552 / URAM 960; guard 0.80):
  ICBU 2U2B = BRAM 192 / URAM 96 ; 3U1B = BRAM 96 / URAM 144 ; 4B = BRAM 384.
  D2 bg (BRAM 2700, URAM 256): 2U2B BRAM 2892 >= 2841.6 OVER ; 3U1B BRAM 2796 + URAM 400
    both under -> strong RECOMMEND 3U1B.
  D3 bg (BRAM 100, URAM 700): 2U2B URAM 796 OVER ; 3U1B URAM 844 OVER -> no calibrated
    candidate fits -> apply must FAIL-FAST (never silently emit a guard-busting recipe).
"""
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import generate_code as gc

_DSE_ENVS = (
    "STREAMNTT_ICBU_BIND_STORAGE_DSE_MODE",
    "STREAMNTT_ICBU_BIND_STORAGE_DSE_GUARD",
    "STREAMNTT_ICBU_DSE_CALIBRATION_ROOT",
    "STREAMNTT_BF_UNIT_STORAGE_RECIPE",
    "STREAMNTT_ICBU_RECIPE_SCHEDULE",
)


@contextlib.contextmanager
def _env(**kv):
    """Set exactly the given DSE env vars; all other knobs in _DSE_ENVS cleared; restore after."""
    old = {k: os.environ.get(k) for k in _DSE_ENVS}
    try:
        for k in _DSE_ENVS:
            os.environ.pop(k, None)
        for k, v in kv.items():
            os.environ[k] = v
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _mk_fixture_root(tmp, bg):
    """One coarse-unique manifest keyed (N=1024,BU=8,CH=4,K=62) with a measured background."""
    d = os.path.join(tmp, "N1024_BU8_CH4_QK62_fixture_2U2B")
    os.makedirs(d)
    man = {"N": 1024, "BU": 8, "CH": 4, "K": 62, "NUM_CORE": 1,
           "prime_set": {"count": 1, "mode": "single-prime cyclic", "values": [3]},
           "shiftadd": "none", "target_clock": "3.33ns", "source_phase": "fixture",
           "included_reports": [], "source_phase_tag": "fixture",
           "storage_recipe": "2U2B", "background_design_minus_icbu": bg}
    with open(os.path.join(d, "manifest.json"), "w") as f:
        json.dump(man, f)
    return tmp


_BG_D2 = {"BRAM_18K": 2700, "URAM": 256, "DSP": 0, "LUT": 0, "FF": 0}
_BG_D3 = {"BRAM_18K": 100, "URAM": 700, "DSP": 0, "LUT": 0, "FF": 0}

_BASE = {"N": 1024, "BU": 8, "CH": 4, "RATE": 0.5, "K": 62,
         "primes": ["TI_9"], "tfg_seg_len": 16, "shiftadd": "none"}


def _gen(name, **envkv):
    cfg = dict(_BASE)
    cfg["name"] = name
    buf = io.StringIO()
    with _env(**envkv):
        with contextlib.redirect_stdout(buf):
            case_name, case_dir = gc.generate_seg_case(cfg, suffix="dseapply", force=True)
    return case_dir, buf.getvalue()


def _ntt_cpp(case_dir):
    with open(os.path.join(case_dir, "src", "ntt.cpp")) as f:
        return f.read()


def _decision(case_dir):
    p = os.path.join(case_dir, "dse_decision.txt")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return f.read()


def test_report_default_no_decision_file_and_default_recipe():
    d, out = _gen("t1rep")  # all DSE envs unset -> default mode = report
    cpp = _ntt_cpp(d)
    assert "// storage recipe:" not in cpp, "default 2U2B must stay comment-free (byte anchor)"
    assert cpp.count("impl=uram ") == 2 and cpp.count("impl=bram ") == 2
    assert _decision(d) is None, "report mode must not add dse_decision.txt"
    assert "icbu-dse:" not in out, "report mode must not print the decision line"
    assert "ICBU bind_storage minor-DSE (report)" in out


def test_off_mode_no_report_no_decision():
    d, out = _gen("t2off", STREAMNTT_ICBU_BIND_STORAGE_DSE_MODE="off")
    assert "ICBU bind_storage minor-DSE" not in out
    assert _decision(d) is None
    cpp = _ntt_cpp(d)
    assert cpp.count("impl=uram ") == 2 and cpp.count("impl=bram ") == 2


def test_report_explicit_equals_default_bytes():
    d1, _ = _gen("t3rep")
    a = _ntt_cpp(d1)
    d2, _ = _gen("t3rep", STREAMNTT_ICBU_BIND_STORAGE_DSE_MODE="report")
    b = _ntt_cpp(d2)
    assert a == b, "explicit report mode must emit byte-identical ntt.cpp vs default"


def test_recommend_prints_best_but_does_not_change_emission():
    tmp = tempfile.mkdtemp(prefix="dse_t4_")
    try:
        root = _mk_fixture_root(tmp, _BG_D2)
        d, out = _gen("t4rec", STREAMNTT_ICBU_BIND_STORAGE_DSE_MODE="recommend",
                      STREAMNTT_ICBU_DSE_CALIBRATION_ROOT=root)
        cpp = _ntt_cpp(d)
        assert cpp.count("impl=uram ") == 2 and cpp.count("impl=bram ") == 2
        assert "// storage recipe:" not in cpp
        assert "icbu-dse: mode=recommend decision=D2 best=3U1B" in out
        dec = _decision(d)
        assert dec is not None, "recommend mode must write dse_decision.txt"
        assert "action: none" in dec and "best: 3U1B" in dec
        assert "source_emission_changed: no" in dec
    finally:
        shutil.rmtree(tmp)


def test_apply_d2_emits_best_recipe():
    tmp = tempfile.mkdtemp(prefix="dse_t5_")
    try:
        root = _mk_fixture_root(tmp, _BG_D2)
        d, out = _gen("t5app", STREAMNTT_ICBU_BIND_STORAGE_DSE_MODE="apply",
                      STREAMNTT_ICBU_DSE_CALIBRATION_ROOT=root)
        cpp = _ntt_cpp(d)
        assert "// storage recipe: mem0=URAM, mem1=URAM, mem2=URAM, mem3=BRAM" in cpp
        assert cpp.count("impl=uram ") == 3 and cpp.count("impl=bram ") == 1
        assert ("icbu-dse: mode=apply decision=D2 best=3U1B configured=2U2B "
                "action=applied emitted=3U1B") in out
        dec = _decision(d)
        assert "decision: D2" in dec and "action: applied" in dec
        assert "emitted: 3U1B" in dec and "source_emission_changed: yes" in dec
        assert "guard: 0.80" in dec and "mode: apply" in dec
    finally:
        shutil.rmtree(tmp)


def test_apply_user_scalar_recipe_wins():
    tmp = tempfile.mkdtemp(prefix="dse_t6_")
    try:
        root = _mk_fixture_root(tmp, _BG_D2)
        d, out = _gen("t6usr", STREAMNTT_ICBU_BIND_STORAGE_DSE_MODE="apply",
                      STREAMNTT_ICBU_DSE_CALIBRATION_ROOT=root,
                      STREAMNTT_BF_UNIT_STORAGE_RECIPE="4B")
        cpp = _ntt_cpp(d)
        assert cpp.count("impl=bram ") == 4 and cpp.count("impl=uram ") == 0
        assert "action=none" in out and "emitted=4B" in out
        dec = _decision(d)
        assert "action: none" in dec and "source_emission_changed: no" in dec
        assert "user_recipe: 4B" in dec
    finally:
        shutil.rmtree(tmp)


def test_apply_override_overrides_scalar_recipe():
    tmp = tempfile.mkdtemp(prefix="dse_t7_")
    try:
        root = _mk_fixture_root(tmp, _BG_D2)
        d, out = _gen("t7ovr", STREAMNTT_ICBU_BIND_STORAGE_DSE_MODE="apply_override",
                      STREAMNTT_ICBU_DSE_CALIBRATION_ROOT=root,
                      STREAMNTT_BF_UNIT_STORAGE_RECIPE="4B")
        cpp = _ntt_cpp(d)
        assert cpp.count("impl=uram ") == 3 and cpp.count("impl=bram ") == 1
        assert "action=applied" in out and "emitted=3U1B" in out
    finally:
        shutil.rmtree(tmp)


def test_apply_with_schedule_skips_and_keeps_schedule():
    tmp = tempfile.mkdtemp(prefix="dse_t8_")
    try:
        root = _mk_fixture_root(tmp, _BG_D2)
        d, out = _gen("t8sch", STREAMNTT_ICBU_BIND_STORAGE_DSE_MODE="apply",
                      STREAMNTT_ICBU_DSE_CALIBRATION_ROOT=root,
                      STREAMNTT_ICBU_RECIPE_SCHEDULE="0-5:L0B2U2:none")
        cpp = _ntt_cpp(d)
        assert "// icbu recipe schedule:" in cpp
        assert cpp.count("impl=uram ") == 2 and cpp.count("impl=bram ") == 2
        assert "action=none" in out and "emitted=(schedule)" in out
        dec = _decision(d)
        assert "schedule" in dec and "action: none" in dec
    finally:
        shutil.rmtree(tmp)


def test_apply_override_with_schedule_fails_fast():
    tmp = tempfile.mkdtemp(prefix="dse_t9_")
    try:
        root = _mk_fixture_root(tmp, _BG_D2)
        try:
            _gen("t9bad", STREAMNTT_ICBU_BIND_STORAGE_DSE_MODE="apply_override",
                 STREAMNTT_ICBU_DSE_CALIBRATION_ROOT=root,
                 STREAMNTT_ICBU_RECIPE_SCHEDULE="0-5:L0B2U2:none")
        except ValueError as e:
            assert "apply_override" in str(e) and "SCHEDULE" in str(e).upper()
        else:
            raise AssertionError("expected ValueError for apply_override + explicit schedule")
    finally:
        shutil.rmtree(tmp)


def test_apply_no_calibration_does_not_guess():
    tmp = tempfile.mkdtemp(prefix="dse_t10_")  # empty calibration root: no manifests
    try:
        d, out = _gen("t10d0", STREAMNTT_ICBU_BIND_STORAGE_DSE_MODE="apply",
                      STREAMNTT_ICBU_DSE_CALIBRATION_ROOT=tmp)
        cpp = _ntt_cpp(d)
        assert cpp.count("impl=uram ") == 2 and cpp.count("impl=bram ") == 2
        assert "decision=D0" in out and "best=none" in out and "action=none" in out
        dec = _decision(d)
        assert "decision: D0" in dec and "action: none" in dec
    finally:
        shutil.rmtree(tmp)


def test_apply_d3_over_guard_no_candidate_fails_fast():
    tmp = tempfile.mkdtemp(prefix="dse_t11_")
    try:
        root = _mk_fixture_root(tmp, _BG_D3)
        try:
            _gen("t11d3", STREAMNTT_ICBU_BIND_STORAGE_DSE_MODE="apply",
                 STREAMNTT_ICBU_DSE_CALIBRATION_ROOT=root)
        except ValueError as e:
            assert "D3" in str(e) and "guard" in str(e)
        else:
            raise AssertionError("expected ValueError for apply when no calibrated recipe fits")
    finally:
        shutil.rmtree(tmp)


def test_invalid_mode_fails_fast():
    try:
        _gen("t12bad", STREAMNTT_ICBU_BIND_STORAGE_DSE_MODE="auto")
    except ValueError as e:
        assert "STREAMNTT_ICBU_BIND_STORAGE_DSE_MODE" in str(e)
    else:
        raise AssertionError("expected ValueError for invalid DSE mode")


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

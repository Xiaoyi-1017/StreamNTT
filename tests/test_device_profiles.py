#!/usr/bin/env python3
"""Tests for the DeviceProfile interface (u55c calibrated/default; u250/u280 capacity-only).

Run (no pytest needed):  ~/autontt-venv/bin/python3 tests/test_device_profiles.py
Exit 0 = all pass; non-zero = a failure (prints the offending test).

HARD honesty rules locked here (Paper-A DSE API slice):
  u55c numbers == legacy icbu_dse.BOARD_U55C (existing behavior unchanged);
  u250/u280 are CAPACITY-ONLY (no calibration manifests / packing / CALIB evidence) -> the
  DSE report must say uncalibrated/conservative, label u55c packing as an explicit FALLBACK,
  and NEVER produce a calibrated strong recommendation for them (even when a u55c-keyed
  manifest exists in the calibration root); unknown devices FAIL FAST."""
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dse import icbu_dse
import generate_code as gc
from dse.device_profiles import (
    CALIBRATED_DEVICES, DEVICE_PROFILES, KNOWN_DEVICES, get_device_profile,
)

_DSE_ENVS = (
    "STREAMNTT_ICBU_BIND_STORAGE_DSE_MODE",
    "STREAMNTT_ICBU_BIND_STORAGE_DSE_GUARD",
    "STREAMNTT_ICBU_DSE_CALIBRATION_ROOT",
    "STREAMNTT_ICBU_DSE_DEVICE",
    "STREAMNTT_BF_UNIT_STORAGE_RECIPE",
    "STREAMNTT_ICBU_RECIPE_SCHEDULE",
)


@contextlib.contextmanager
def _env(**kv):
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

_BASE = {"N": 1024, "BU": 8, "CH": 4, "RATE": 0.5, "K": 62,
         "primes": ["TI_9"], "tfg_seg_len": 16, "shiftadd": "none"}


def _gen(name, icbu=None, **envkv):
    cfg = dict(_BASE)
    cfg["name"] = name
    if icbu is not None:
        cfg["icbu"] = icbu
    buf = io.StringIO()
    with _env(**envkv):
        with contextlib.redirect_stdout(buf):
            case_name, case_dir = gc.generate_seg_case(cfg, suffix="devprof", force=True)
    return case_dir, buf.getvalue()


def _ntt_cpp(case_dir):
    with open(os.path.join(case_dir, "src", "ntt.cpp")) as f:
        return f.read()


def test_u55c_profile_matches_legacy_board():
    p = get_device_profile("u55c")
    assert p.as_board() == icbu_dse.BOARD_U55C, "u55c profile must equal the legacy BOARD_U55C"
    assert p.calibrated is True and "u55c" in CALIBRATED_DEVICES
    assert "calibrated" in p.notes
    assert (p.lut, p.ff, p.bram18, p.dsp, p.uram) == (1045680, 2167360, 3552, 8204, 960)


def test_u250_u280_capacity_only_profiles():
    for name, dsp, uram, bram18 in (("u250", 12288, 1280, 5376), ("u280", 9024, 960, 4032)):
        p = get_device_profile(name)
        assert p.name == name and p.dsp == dsp and p.uram == uram and p.bram18 == bram18
        assert p.lut > 0 and p.ff > 0
        assert p.calibrated is False, "%s must NOT claim calibration" % name
        assert "capacity-only" in p.notes and "UNCALIBRATED" in p.notes
        b = p.as_board()
        assert set(b) == {"FF", "LUT", "BRAM_18K", "DSP", "URAM"}
    assert KNOWN_DEVICES == ("u250", "u280", "u55c") and len(DEVICE_PROFILES) == 3


def test_lookup_normalization_and_unknown_fails_fast():
    assert get_device_profile("U55C").name == "u55c"
    assert get_device_profile(" u250 ").name == "u250"
    for bad in ("u999", "vu13p", "", "  "):
        try:
            get_device_profile(bad)
        except ValueError as e:
            if bad.strip():
                assert "unknown device profile" in str(e) and "u55c" in str(e)
        else:
            raise AssertionError("expected ValueError for device %r" % bad)


def test_format_report_u55c_output_unchanged_by_device_param():
    arch = icbu_dse.arch_quantities(1024, 8, 4, 62, 1)
    legacy = icbu_dse.format_report(arch, None, icbu_dse.BOARD_U55C, 0.80)
    explicit = icbu_dse.format_report(arch, None, icbu_dse.BOARD_U55C, 0.80, device="u55c")
    assert legacy == explicit, "device='u55c' must be a byte-identical default"
    assert "board=U55C" in legacy and "CAPACITY-ONLY" not in legacy
    assert "u55c-fallback" not in legacy


def test_format_report_non_u55c_marks_uncalibrated_and_fallback():
    arch = icbu_dse.arch_quantities(131072, 8, 4, 62, 1)   # bank_depth 4096 -> packing 'calibrated'
    board = get_device_profile("u250").as_board()
    rep = icbu_dse.format_report(arch, None, board, 0.80, device="u250")
    assert "board=U250" in rep
    assert "CAPACITY-ONLY" in rep, "non-u55c report must state the capacity-only status"
    assert "calibrated(u55c-fallback)" in rep, \
        "u55c BRAM packing on another device must be labelled as a fallback"
    assert "informational" in rep, "non-u55c recommendation must stay informational"


def test_generation_config_device_u250_never_pretends_calibrated():
    tmp = tempfile.mkdtemp(prefix="devprof_t6_")
    try:
        root = _mk_fixture_root(tmp, _BG_D2)   # would be a strong D2 on u55c
        d, out = _gen("d6u250", icbu={"device": "u250", "dse_mode": "recommend"},
                      STREAMNTT_ICBU_DSE_CALIBRATION_ROOT=root)
        assert "board=U250" in out and "CAPACITY-ONLY" in out
        assert "decision=D0" in out and "best=none" in out, \
            "u55c manifests must NOT be reused as u250 calibration"
        cpp = _ntt_cpp(d)
        assert cpp.count("impl=uram ") == 2 and cpp.count("impl=bram ") == 2
    finally:
        shutil.rmtree(tmp)


def test_generation_env_device_overrides_config_device():
    tmp = tempfile.mkdtemp(prefix="devprof_t7_")
    try:
        root = _mk_fixture_root(tmp, _BG_D2)
        d, out = _gen("d7ovr", icbu={"device": "u250", "dse_mode": "recommend"},
                      STREAMNTT_ICBU_DSE_DEVICE="u55c",
                      STREAMNTT_ICBU_DSE_CALIBRATION_ROOT=root)
        assert ("icbu-config: device=u250 superseded by env "
                "STREAMNTT_ICBU_DSE_DEVICE=u55c") in out
        assert "board=U55C" in out and "decision=D2" in out and "best=3U1B" in out
    finally:
        shutil.rmtree(tmp)


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

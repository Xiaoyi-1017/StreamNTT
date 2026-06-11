#!/usr/bin/env python3
"""Tests for the config-facing ICBU DSE API (public `icbu` config block).

Run (no pytest needed):  ~/autontt-venv/bin/python3 tests/test_config_api.py
Exit 0 = all pass; non-zero = a failure (prints the offending test).

Covers (Paper-A DSE API slice):
  {"icbu": {"storage_recipe": "2U2B|3U1B|4U|4B|4L|auto", "dse_mode": "off|report|recommend|
  apply|apply_override", "device": "u55c|u250|u280"}}.
  PRIORITY RULE: a set STREAMNTT_* env knob supersedes the corresponding icbu.* config field
  (env = override / backward-compatible path) and prints one "icbu-config: ... superseded by
  env ..." stdout note. Configs WITHOUT the icbu block must stay byte-identical to legacy.
  "auto" = recipe DSE-eligible ONLY under dse_mode apply/apply_override with a strong D2.

Generates real N1024 single-prime Barrett cases into generated/ (gitignored, suffix
"cfgapi"). Calibration evidence comes from SMALL TEMP FIXTURE manifests injected via
STREAMNTT_ICBU_DSE_CALIBRATION_ROOT (same arithmetic as tests/test_dse_apply_mode.py:
D2 bg BRAM 2700 / URAM 256 -> strong RECOMMEND 3U1B for configured 2U2B)."""
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
    "STREAMNTT_ICBU_DSE_DEVICE",
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
            case_name, case_dir = gc.generate_seg_case(cfg, suffix="cfgapi", force=True)
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


def test_no_icbu_block_keeps_default_anchors():
    d, out = _gen("c1leg")  # no icbu block, no env -> legacy default path
    cpp = _ntt_cpp(d)
    assert "// storage recipe:" not in cpp, "default 2U2B must stay comment-free (byte anchor)"
    assert cpp.count("impl=uram ") == 2 and cpp.count("impl=bram ") == 2
    assert _decision(d) is None and "icbu-dse:" not in out
    assert "icbu-config:" not in out, "no supersede note without an icbu block"
    assert "ICBU bind_storage minor-DSE (report)" in out, "default dse_mode must stay report"


def test_empty_and_explicit_default_icbu_byte_identical():
    d, _ = _gen("c2def")
    legacy = _ntt_cpp(d)
    d, _ = _gen("c2def", icbu={})
    assert _ntt_cpp(d) == legacy, "empty icbu block must be a byte-identical no-op"
    d, out = _gen("c2def", icbu={"storage_recipe": "2U2B", "dse_mode": "report",
                                 "device": "u55c"})
    assert _ntt_cpp(d) == legacy, "explicit defaults must emit byte-identical ntt.cpp"
    assert "board=U55C" in out and "icbu-config:" not in out


def test_config_recipe_3u1b_without_env():
    d, _ = _gen("c3rcp", icbu={"storage_recipe": "3U1B"})
    cpp = _ntt_cpp(d)
    assert "// storage recipe: mem0=URAM, mem1=URAM, mem2=URAM, mem3=BRAM" in cpp
    assert cpp.count("impl=uram ") == 3 and cpp.count("impl=bram ") == 1


def test_config_dse_mode_apply_d2_adopts_best():
    tmp = tempfile.mkdtemp(prefix="cfgapi_t4_")
    try:
        root = _mk_fixture_root(tmp, _BG_D2)
        d, out = _gen("c4app", icbu={"dse_mode": "apply"},
                      STREAMNTT_ICBU_DSE_CALIBRATION_ROOT=root)
        cpp = _ntt_cpp(d)
        assert cpp.count("impl=uram ") == 3 and cpp.count("impl=bram ") == 1
        assert ("icbu-dse: mode=apply decision=D2 best=3U1B configured=2U2B "
                "action=applied emitted=3U1B") in out
        dec = _decision(d)
        assert "action: applied" in dec and "source_emission_changed: yes" in dec
    finally:
        shutil.rmtree(tmp)


def test_auto_recipe_apply_adopts_and_report_stays_default():
    tmp = tempfile.mkdtemp(prefix="cfgapi_t5_")
    try:
        root = _mk_fixture_root(tmp, _BG_D2)
        d, out = _gen("c5auto", icbu={"storage_recipe": "auto", "dse_mode": "apply"},
                      STREAMNTT_ICBU_DSE_CALIBRATION_ROOT=root)
        cpp = _ntt_cpp(d)
        assert cpp.count("impl=uram ") == 3 and cpp.count("impl=bram ") == 1, \
            "auto + apply + D2 must adopt the recommended recipe"
        assert "action=applied" in out and "user_recipe: (auto)" in _decision(d)
        d, out = _gen("c5auto", icbu={"storage_recipe": "auto"},
                      STREAMNTT_ICBU_DSE_CALIBRATION_ROOT=root)
        cpp = _ntt_cpp(d)
        assert cpp.count("impl=uram ") == 2 and cpp.count("impl=bram ") == 2, \
            "auto outside apply modes must emit the 2U2B default"
        assert "// storage recipe:" not in cpp and _decision(d) is None
    finally:
        shutil.rmtree(tmp)


def test_config_pinned_recipe_wins_under_apply_but_not_apply_override():
    tmp = tempfile.mkdtemp(prefix="cfgapi_t6_")
    try:
        root = _mk_fixture_root(tmp, _BG_D2)
        d, out = _gen("c6pin", icbu={"storage_recipe": "4B", "dse_mode": "apply"},
                      STREAMNTT_ICBU_DSE_CALIBRATION_ROOT=root)
        cpp = _ntt_cpp(d)
        assert cpp.count("impl=bram ") == 4 and cpp.count("impl=uram ") == 0, \
            "apply must keep a config-pinned recipe (use apply_override to let DSE win)"
        assert "action=none" in out and "emitted=4B" in out
        assert "user_recipe: 4B (config)" in _decision(d)
        d, out = _gen("c6pin", icbu={"storage_recipe": "4B", "dse_mode": "apply_override"},
                      STREAMNTT_ICBU_DSE_CALIBRATION_ROOT=root)
        cpp = _ntt_cpp(d)
        assert cpp.count("impl=uram ") == 3 and cpp.count("impl=bram ") == 1
        assert "action=applied" in out and "emitted=3U1B" in out
    finally:
        shutil.rmtree(tmp)


def test_env_recipe_overrides_config_recipe():
    d, out = _gen("c7ovr", icbu={"storage_recipe": "4B"},
                  STREAMNTT_BF_UNIT_STORAGE_RECIPE="4U")
    cpp = _ntt_cpp(d)
    assert cpp.count("impl=uram ") == 4 and cpp.count("impl=bram ") == 0
    assert ("icbu-config: storage_recipe=4B superseded by env "
            "STREAMNTT_BF_UNIT_STORAGE_RECIPE=4U") in out


def test_env_schedule_overrides_config_recipe():
    d, out = _gen("c8sch", icbu={"storage_recipe": "4B"},
                  STREAMNTT_ICBU_RECIPE_SCHEDULE="0-5:L0B2U2:none")
    cpp = _ntt_cpp(d)
    assert "// icbu recipe schedule:" in cpp
    assert cpp.count("impl=uram ") == 2 and cpp.count("impl=bram ") == 2
    assert ("icbu-config: storage_recipe=4B superseded by env "
            "STREAMNTT_ICBU_RECIPE_SCHEDULE") in out


def test_env_mode_overrides_config_mode():
    tmp = tempfile.mkdtemp(prefix="cfgapi_t9_")
    try:
        root = _mk_fixture_root(tmp, _BG_D2)
        d, out = _gen("c9mod", icbu={"dse_mode": "apply"},
                      STREAMNTT_ICBU_BIND_STORAGE_DSE_MODE="report",
                      STREAMNTT_ICBU_DSE_CALIBRATION_ROOT=root)
        cpp = _ntt_cpp(d)
        assert cpp.count("impl=uram ") == 2 and cpp.count("impl=bram ") == 2, \
            "env report must veto config apply (no adoption)"
        assert _decision(d) is None and "icbu-dse:" not in out
        assert ("icbu-config: dse_mode=apply superseded by env "
                "STREAMNTT_ICBU_BIND_STORAGE_DSE_MODE=report") in out
    finally:
        shutil.rmtree(tmp)


def test_config_dse_mode_off_silences_report():
    d, out = _gen("c10off", icbu={"dse_mode": "off"})
    assert "ICBU bind_storage minor-DSE" not in out and _decision(d) is None


def test_invalid_config_values_fail_fast():
    cases = (
        ({"storage_recipe": "5U"}, "icbu.storage_recipe"),
        ({"dse_mode": "auto"}, "icbu.dse_mode"),
        ({"device": "u999"}, "unknown device profile"),
        ({"recipe": "3U1B"}, "Unsupported icbu config field"),
        ("3U1B", "must be an object"),
    )
    for icbu, token in cases:
        try:
            _gen("c11bad", icbu=icbu)
        except ValueError as e:
            assert token in str(e), "expected %r in error, got: %s" % (token, e)
        else:
            raise AssertionError("expected ValueError for icbu=%r" % (icbu,))


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

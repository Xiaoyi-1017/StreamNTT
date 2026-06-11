#!/usr/bin/env python3
"""Tests for resource_report (Phase 4.0 Slice 3c-2 pre-gen resource report assembler/renderer).

Run (no pytest needed):  ~/autontt-venv/bin/python3 tests/test_resource_report.py
Exit 0 = all pass; non-zero = a failure (prints the offending test).

Anchors come from the committed resource_model (Slices 1/1b/2/3a/3b/3c-1)."""
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
from dse import resource_model as rm
from dse import resource_report as rr


def _expect_raises(fn, exc=Exception):
    try:
        fn()
    except exc:
        return
    raise AssertionError("expected %s, none raised" % getattr(exc, "__name__", exc))


def _bu8_barrett(**kw):
    base = dict(N=1024, BU=8, CH=4, K=62, route=rm.ROUTE_BARRETT,
                max_prime_bit_length=62, prime_count=2,
                tfr_x_dsp=4 * 30, total_design_dsp=3090)
    base.update(kw)
    return rr.build_report(**base)


# --- build_report structured values ------------------------------------------

def test_report_xstage_mr_total_43():
    assert _bu8_barrett().mr["total"] == 43


def test_report_xstage_module_dsp_1290():
    assert _bu8_barrett().dsp["x_stages"] == 1290


def test_report_group_dsp_1410_with_tfrx():
    assert _bu8_barrett().dsp["group"] == 1410


def test_report_module_slot_guard_over():
    g = _bu8_barrett().guards["module"]
    assert g.status == "OVER"
    assert g.capacity_layer == rm.GUARD_LAYER_SLOT
    assert g.capacity == 1228


def test_report_device_guard_under():
    g = _bu8_barrett().guards["device"]
    assert g.status == "UNDER"
    assert g.capacity_layer == rm.GUARD_LAYER_DEVICE
    assert g.capacity == 8204


def test_report_icbu_memory_anchor_uram96_bram192():
    m = _bu8_barrett().memory
    assert m.uram == 96
    assert m.bram_18k == 192


def test_report_per_stage_dsp_profile():
    assert _bu8_barrett().per_stage_dsp == (240, 270, 330, 450)


def test_report_partition_group_sums():
    r = _bu8_barrett(partition=[[0, 1, 2], [3]])
    assert r.group_dsp == (840, 450)


# --- unknown handling --------------------------------------------------------

def test_report_unknown_nonstrict_yields_unknown_row():
    r = rr.build_report(N=1024, BU=8, CH=4, K=62, route=rm.ROUTE_BARRETT,
                        max_prime_bit_length=61, prime_count=1, strict=False)
    assert r.dmr is None
    assert len(r.unknown_rows) >= 1
    assert r.dsp["x_stages"] is None              # no interpolation / no guessed coefficient
    assert "UNKNOWN" in rr.render_report(r)


def test_report_unknown_strict_raises():
    _expect_raises(lambda: rr.build_report(N=1024, BU=8, CH=4, K=62, route=rm.ROUTE_BARRETT,
                                           max_prime_bit_length=61, prime_count=1, strict=True),
                   rm.UnknownCoefficientError)


def test_report_unknown_nonstrict_still_has_structural_and_memory():
    r = rr.build_report(N=1024, BU=8, CH=4, K=62, route=rm.ROUTE_BARRETT,
                        max_prime_bit_length=61, prime_count=1, strict=False)
    assert r.mr["total"] == 43                    # structural still present
    assert r.memory.uram == 96                    # memory (no D_MR) still present


# --- renderer ----------------------------------------------------------------

def test_render_report_is_deterministic():
    r = _bu8_barrett(partition=[[0, 1, 2], [3]])
    assert rr.render_report(r) == rr.render_report(r)


def test_render_report_returns_string():
    assert isinstance(rr.render_report(_bu8_barrett()), str)


def test_render_report_has_all_section_headers():
    s = rr.render_report(_bu8_barrett(partition=[[0, 1, 2], [3]]))
    for h in ["[config]", "[route]", "[MR", "[D_MR]", "[DSP]", "[memory]", "[guard]",
              "[per-stage DSP]", "[per-group DSP]", "[caveats]"]:
        assert h in s, "missing section %r" % h


def test_render_report_has_layer_labels():
    s = rr.render_report(_bu8_barrett())
    for label in ["module", "group", "slot", "device"]:
        assert label in s, "missing layer label %r" % label


def test_render_report_preserves_caveats():
    s = rr.render_report(_bu8_barrett())
    assert "proxy" in s            # 1228 configurable slot proxy
    assert "1218" in s             # 1218 outside-artifacts caveat
    assert "risk" in s             # guardrail risk/policy, not failure
    assert "audited" in s          # BU8 / 1410 audited anchor
    assert "BU32" in s or "BU!=8" in s   # BU32 MR structural-only caveat


# --- no generation coupling --------------------------------------------------

def test_resource_report_does_not_import_generate_code():
    src = open(os.path.join(_REPO, "dse", "resource_report.py")).read()  # moved 2026-06-10
    assert "generate_code" not in src


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

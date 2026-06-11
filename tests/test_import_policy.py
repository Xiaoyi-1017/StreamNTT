#!/usr/bin/env python3
"""Public import-policy tests (replaces the private-history shim tests).

Run (no pytest needed):  python3 tests/test_import_policy.py
Exit 0 = all pass; non-zero = a failure (prints the offending test).

Locks the public layout contract:
  generate_code.py is the ONLY top-level Python module (the CLI); all library code
  lives in the code_generator/ and dse/ packages and is imported via package paths;
  no top-level compatibility shims exist; the frozen shift-add surface survives in
  code_generator.shiftadd_generator; the dse.icbu_dse import stays lazy; the active
  segmented template resolves at the repo-root templates/ tree.
"""
import importlib
import os
import subprocess
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

_PACKAGED = (
    ("twiddle_generator", "code_generator"),
    ("shiftadd_generator", "code_generator"),
    ("reduce_generator", "code_generator"),
    ("reduce_recipe", "code_generator"),
    ("stage_generator", "code_generator"),
    ("stage_plan", "code_generator"),
    ("icbu_generator", "code_generator"),
    ("icbu_dse", "dse"),
    ("resource_model", "dse"),
    ("resource_report", "dse"),
    ("calibration_data", "dse"),
    ("calibration_compare", "dse"),
    ("device_profiles", "dse"),
)


def test_no_top_level_shim_files():
    top_py = sorted(f for f in os.listdir(_ROOT) if f.endswith(".py"))
    assert top_py == ["generate_code.py"], \
        "generate_code.py must be the only top-level .py file, found: %s" % top_py


def test_old_top_level_names_are_not_importable():
    # Fresh interpreter so previously-imported packages cannot mask the check.
    names = ", ".join(repr(n) for n, _ in _PACKAGED)
    code = (
        "import sys, importlib; sys.path.insert(0, %r)\n"
        "for n in (%s):\n"
        "    try:\n"
        "        importlib.import_module(n)\n"
        "    except ModuleNotFoundError:\n"
        "        pass\n"
        "    else:\n"
        "        raise SystemExit('top-level shim still importable: ' + n)\n"
        % (_ROOT, names)
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=_ROOT)
    assert r.returncode == 0, "shim-free check failed:\n%s%s" % (r.stdout, r.stderr)


def test_every_module_importable_via_its_package():
    for name, pkg in _PACKAGED:
        mod = importlib.import_module("%s.%s" % (pkg, name))
        assert mod.__name__ == "%s.%s" % (pkg, name)


def test_frozen_shiftadd_surface_survives():
    from code_generator.shiftadd_generator import emit_shiftadd_header  # noqa: F401
    from code_generator.reduce_generator import classify_reduce_set, _k_arith_for  # noqa: F401
    from code_generator.stage_plan import plan_from_precomp_boundary  # noqa: F401
    from dse.device_profiles import get_device_profile, CALIBRATED_DEVICES  # noqa: F401
    from dse import icbu_dse
    assert callable(emit_shiftadd_header), "frozen shiftadd surface must survive"
    assert isinstance(icbu_dse.BOARD_U55C, dict)


def test_reduce_generator_template_path():
    from code_generator import reduce_generator as rg
    tpl = rg._SEG_NTT_CPP_TEMPLATE
    assert os.path.normpath(tpl) == os.path.normpath(
        os.path.join(_ROOT, "templates", "ntt.cpp")), \
        "template must live at the repo root templates/ntt.cpp (NOT inside code_generator/)"
    assert os.path.exists(tpl), "template must still resolve: %s" % tpl


def test_generate_code_stays_top_level_and_uses_packages():
    import generate_code as gc
    assert gc.__name__ == "generate_code", "generate_code must remain the real top-level module"
    assert os.path.normpath(gc.SEG_TEMPLATE_DIR) == os.path.normpath(
        os.path.join(_ROOT, "templates"))
    assert gc.gen_reduce_block.__module__ == "code_generator.reduce_generator"
    assert gc.get_device_profile.__module__ == "dse.device_profiles"


def test_cross_package_edge_board_u55c():
    from dse import icbu_dse
    from dse.device_profiles import get_device_profile
    assert icbu_dse.BOARD_U55C == get_device_profile("u55c").as_board()


def test_dse_icbu_dse_import_stays_lazy_in_generate_code():
    # Fresh interpreter: importing generate_code must NOT pull dse.icbu_dse (lazy DSE block).
    code = ("import sys; sys.path.insert(0, %r); import generate_code; "
            "assert 'dse.icbu_dse' not in sys.modules, 'icbu_dse must stay lazy'; "
            "assert 'code_generator.stage_plan' in sys.modules" % _ROOT)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=_ROOT)
    assert r.returncode == 0, "lazy-import check failed:\n%s%s" % (r.stdout, r.stderr)


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

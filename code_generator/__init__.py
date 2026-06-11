"""code_generator -- StreamNTT segmented generator package (repo cleanup slice, 2026-06-10).

Architecture / generation-planning + emission modules moved here from the repo root:
twiddle_generator, shiftadd_generator, reduce_generator, reduce_recipe, stage_generator,
stage_plan, icbu_generator. The top-level generate_code.py remains the public CLI entry.

INTENTIONALLY no eager submodule imports here: reduce_generator runs an import-time
template anchor self-check, and importing every submodule from the package __init__
would change import order/cost for callers that need a single module. Import submodules
explicitly (e.g. `from code_generator import stage_plan`). Top-level shim modules with
the old names re-export every moved module for backward compatibility (old tests /
scripts keep working; removal of a shim requires tests + byte gates proving it safe).
"""

__all__ = [
    "icbu_generator",
    "reduce_generator",
    "reduce_recipe",
    "shiftadd_generator",
    "stage_generator",
    "stage_plan",
    "twiddle_generator",
]

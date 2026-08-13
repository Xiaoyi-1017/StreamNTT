"""StreamNTT segmented generator package.

The package owns twiddle generation, arithmetic emission, stage planning, ICBU
configuration, and top/DRAM wrapper emission. The top-level generate_code.py remains
the public CLI entry.

INTENTIONALLY no eager submodule imports here: reduce_generator runs an import-time
template anchor self-check, and importing every submodule from the package __init__
would change import order and cost for callers that need a single module. Import
submodules explicitly (for example, `from code_generator import stage_plan`).
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

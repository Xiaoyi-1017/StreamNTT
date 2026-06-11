"""dse -- StreamNTT pre-generation DSE / cost-model / calibration package (2026-06-10).

Report / model / observed-evidence modules moved here from the repo root: icbu_dse,
resource_model, resource_report, calibration_data, calibration_compare, device_profiles.

Dependency direction: dse modules may import code_generator modules (icbu_dse ->
code_generator.icbu_generator for the storage count-form parser); generate_code.py
imports `dse.icbu_dse` LAZILY inside its DSE block (keep it lazy -- no import cycle,
and `off` mode must stay import-free). INTENTIONALLY no eager submodule imports here
(resource_model keeps its icbu_dse imports lazy on purpose). Import submodules
explicitly (e.g. `from dse import icbu_dse`). Top-level shim modules with the old
names re-export every moved module for backward compatibility (removal of a shim
requires tests + byte gates proving it safe).
"""

__all__ = [
    "calibration_compare",
    "calibration_data",
    "device_profiles",
    "icbu_dse",
    "resource_model",
    "resource_report",
]

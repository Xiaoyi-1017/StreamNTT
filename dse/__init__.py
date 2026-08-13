"""StreamNTT pre-generation DSE, cost-model, and calibration package.

Contains the ICBU DSE, resource model/report, calibration data/comparison, and
device profiles.

Dependency direction: DSE modules may import code_generator modules. The top-level
generator imports ``dse.icbu_dse`` lazily so DSE-off mode remains import-free and
no import cycle is introduced. Import submodules explicitly, for example
``from dse import icbu_dse``.
"""

__all__ = [
    "calibration_compare",
    "calibration_data",
    "device_profiles",
    "icbu_dse",
    "resource_model",
    "resource_report",
]

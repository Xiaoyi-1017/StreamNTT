"""Device capacity profiles for the resource / ICBU bind_storage DSE reports.

Single source of truth for per-device FPGA capacity numbers usable by the pre-gen
resource / ICBU bind_storage DSE reporting layers. Three profiles:

  u55c : the EXISTING calibrated/default flow profile. Its numbers ARE the historical
         icbu_dse.BOARD_U55C dict (post-shell AVAILABLE capacity, NOT raw datasheet
         totals); icbu_dse now derives BOARD_U55C from this profile, so existing U55C
         behavior is unchanged (locked by tests/test_device_profiles.py).
  u250 : CAPACITY-ONLY (XCU250 datasheet fabric totals; no shell subtraction).
  u280 : CAPACITY-ONLY (XCU280 datasheet fabric totals; no shell subtraction).

CALIBRATION HONESTY (HARD): only u55c has DSE calibration evidence (the gitignored
Checkpoint/dse_calibration manifests, BRAM_BANK_CALIB bank packing, CALIB LUT/FF rows --
all u55c-flow RapidStream/csynth measurements). u250/u280 carry CAPACITY numbers ONLY:
reports against them must say uncalibrated/conservative, may use u55c-derived packing
ONLY as an explicitly labelled fallback, and must never yield a calibrated strong
recommendation. SLOT-level guardrail capacities (resource_model SLOT_*) are u55c
RapidStream evidence and are NOT part of these profiles.

Stdlib-only LEAF module: imports nothing from other project modules (icbu_dse /
generate_code import this; never the reverse)."""
from __future__ import annotations

from typing import NamedTuple

__all__ = [
    "DeviceProfile", "DEVICE_PROFILES", "KNOWN_DEVICES", "CALIBRATED_DEVICES",
    "get_device_profile",
]


class DeviceProfile(NamedTuple):
    """Per-device capacity profile (physical resource counts; bram18 = RAMB18 units)."""
    name: str
    lut: int
    ff: int
    bram18: int
    dsp: int
    uram: int
    notes: str

    def as_board(self) -> dict:
        """icbu_dse-style board dict (same keys/shape as the legacy BOARD_U55C)."""
        return {"FF": self.ff, "LUT": self.lut, "BRAM_18K": self.bram18,
                "DSP": self.dsp, "URAM": self.uram}

    @property
    def calibrated(self) -> bool:
        """True only for devices with real DSE calibration evidence (currently u55c)."""
        return self.name in CALIBRATED_DEVICES


CALIBRATED_DEVICES = frozenset({"u55c"})

DEVICE_PROFILES = {
    "u55c": DeviceProfile(
        name="u55c", lut=1045680, ff=2167360, bram18=3552, dsp=8204, uram=960,
        notes="calibrated default: post-shell AVAILABLE capacity (== legacy "
              "icbu_dse.BOARD_U55C); the only device with DSE calibration evidence "
              "(manifests + BRAM bank packing + CALIB LUT/FF)."),
    "u250": DeviceProfile(
        name="u250", lut=1728000, ff=3456000, bram18=5376, dsp=12288, uram=1280,
        notes="capacity-only: XCU250 datasheet fabric totals (no shell subtraction); "
              "UNCALIBRATED (no manifests / packing / CALIB evidence) -> reports must "
              "stay conservative."),
    "u280": DeviceProfile(
        name="u280", lut=1303680, ff=2607360, bram18=4032, dsp=9024, uram=960,
        notes="capacity-only: XCU280 datasheet fabric totals (no shell subtraction); "
              "UNCALIBRATED (no manifests / packing / CALIB evidence) -> reports must "
              "stay conservative."),
}

KNOWN_DEVICES = tuple(sorted(DEVICE_PROFILES))


def get_device_profile(name: str) -> DeviceProfile:
    """Look up a device profile by name (case-insensitive; surrounding whitespace ignored).

    Unknown devices FAIL FAST -- never fall back to u55c silently (that would smuggle
    u55c calibration assumptions onto an uncalibrated device).
    """
    if not isinstance(name, str) or name.strip() == "":
        raise ValueError("device profile name must be a non-empty string, got %r" % (name,))
    key = name.strip().lower()
    if key not in DEVICE_PROFILES:
        raise ValueError("unknown device profile %r; known devices: %s"
                         % (name, ", ".join(KNOWN_DEVICES)))
    return DEVICE_PROFILES[key]

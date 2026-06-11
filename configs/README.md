# StreamNTT configuration reference

`generate_code.py --config <file>.json` is the public entry point. Configs are plain
JSON (no comments — JSON comments break the parser) validated against a **strict
whitelist**: any unknown top-level field is rejected, including a `description` field,
so explanatory text lives here instead of inside the JSON files.

## Minimal config format

```json
{
  "name": "single52_barrett",
  "N": 65536,
  "BU": 16,
  "CH": 8,
  "RATE": 0.5,
  "primes": ["TII_33"],
  "tfg_seg_len": 16,
  "shiftadd": "none",
  "icbu": {"storage_recipe": "2U2B", "dse_mode": "report", "device": "u55c"}
}
```

`K` and `icbu` are optional; `shiftadd` defaults to `"none"`; `name` tags the
generated directory (`generated/N<N>_BU<BU>_CH<CH>_QK<K>_<name>`).

## Fields

| Field | Meaning |
|---|---|
| `N` | polynomial degree / transform size (power of two) |
| `BU` | butterfly units (parallel lanes) per core (power of two) |
| `CH` | HBM/DRAM data channels |
| `RATE` | channel data-rate utilization factor. With the 64-bit data path the effective words per channel beat are `EffDataCHLen = round_pow2(8 x RATE)` (e.g. `0.5` -> 4); the design needs `CH x EffDataCHLen >= 2 x BU`, and the core count is `CH x EffDataCHLen / (2 x BU)` |
| `K` | container / arithmetic envelope width (bits) for `ap_uint<K>` data and tables. **Optional**: when omitted it is inferred from the prime set; when present, the final envelope is `max(K, K_group)` — an under-specified `K` is lifted with a warning, never silently truncated |
| `primes` | list of prime **aliases**, not raw values. `TI_1..TI_12` = the 61/62-bit class (envelope K=62); `TII_1..TII_41` = the 51/52-bit class (envelope K=52). Mixing classes is supported on the Barrett route |
| `tfg_seg_len` | segmented TFG period P. `16` is the canonical default; `8` is allowed (mainly with shift-add routes); `4` is experimental; `32` is **experimental** and needs the precompute-boundary control below. `N/(2*BU)` must be strictly greater than P |
| `shiftadd` | reduction route: `"none"` (Barrett, general default), `"reduce"` (shift-add full reduction for lightweight primes), `"mul"` (shift-add multiplication path) |
| `name` | case-name tag appended to the generated directory name |
| `icbu` | optional nested block — see below |

## The `icbu` block

| Sub-field | Values | Meaning |
|---|---|---|
| `storage_recipe` | `2U2B` (default), `3U1B`, `4U`, `4B`, `4L`, `auto` | per-bank storage mapping of the ICBU ping-pong banks (U=URAM, B=BRAM, L=LUTRAM). A concrete value **pins** the recipe; `auto` leaves it DSE-eligible |
| `dse_mode` | `off`, `report` (default), `recommend`, `apply`, `apply_override` | `report` prints the resource/guard table only; `recommend` also writes `dse_decision.txt`; `apply` adopts a recipe **only** on a calibrated strong (D2) recommendation and keeps any pinned recipe; `apply_override` may override a pinned scalar recipe (never a schedule). On uncalibrated data the apply modes never guess |
| `device` | `u55c` (default), `u250`, `u280` | capacity profile for the DSE report. Only `u55c` is calibrated; `u250`/`u280` are capacity-only and their reports state `UNCALIBRATED` |

Advanced note: the corresponding `STREAMNTT_*` environment variables still exist as an
override layer for experiments — a set environment knob supersedes the matching config
field and prints an `icbu-config: ... superseded by env ...` notice. Normal use needs
no environment variables.

## Examples (`configs/examples/`)

The standard examples below are fully config-driven — they run with the plain CLI and
**no environment variables**:

```bash
python3 generate_code.py --config configs/examples/<case>.json --force
```

| File | Route | Purpose |
|---|---|---|
| `n128k_bu8_mixed42_barrett_p16_u55c_auto.json` | mixed-bit Barrett | FHE-scale **resource-stress** case: 42 primes (TII_1..41 + TI_1), N=131072, K=62. Many-prime Barrett loses constant pruning (q/mu become ROM-loaded) and can hit the K62 DSP ceiling in `x_stages` — generate it for the DSE/model report, and expect the optimized implementation flow to need planning |
| `n128k_bu8_tii41_shiftadd_reduce_p16_u55c_auto.json` | shift-add reduce | 41-prime 52-bit TII-only lightweight-prime path (same-bit set, K=52): the DSP-saving alternative at the same scale |
| `n128k_bu8_ti1_barrett_p16_u55c_auto.json` | Barrett | single 61-bit `TI_1` baseline at N=131072; `K` omitted (inferred to 62) |
| `n64k_bu16_single52_barrett_p16_u55c.json` | Barrett | smaller-N, wider-BU single 52-bit prime; pins `storage_recipe: 2U2B` with `dse_mode: report` |
| `n64k_bu16_single52_shiftadd_reduce_p16_u55c.json` | shift-add reduce | same shape on the shift-add route; `dse_mode: recommend` shows the decision file without changing emission |

The first three examples use `"storage_recipe": "auto"` + `"dse_mode": "apply"`: on a
machine without calibration manifests for those coordinates the DSE stays
informational (decision D0) and emits the default recipe — `apply` changes the emitted
source only where calibrated strong evidence exists.

### Advanced / experimental example: P = 32

`n1024_bu4_p32_experimental_u55c.json` — small `tfg_seg_len = 32` segmented-TFG case
(N=1024, BU=4, single-prime Barrett). Unlike the standard examples above, this one
intentionally needs one extra control:

```bash
STREAMNTT_LSTAGE_PRECOMPUTE_BOUNDARY=5 \
python3 generate_code.py --config configs/examples/n1024_bu4_p32_experimental_u55c.json --force
```

P = 32 is **not** a matter of changing an HLS II pragma: it is a structural
architecture change that requires a legal precompute/OTF-period setup — with 5
precomputed L-stages (`PRECOMP=5`), every remaining on-the-fly stage has a twiddle
period >= 32, so a 32-cycle segment is well-defined. Running this config without the
environment variable fail-fasts with a message naming exactly this requirement (by
design — the generator refuses an illegal P/period combination rather than guessing).

## Device profiles

`u55c` is the calibrated default profile (the only device with DSE calibration
evidence). `u250` and `u280` are **capacity-only** profiles: selecting them produces a
report against their datasheet capacity with an explicit `CAPACITY-ONLY / UNCALIBRATED`
status, informational recommendations only, and clearly-labelled fallback packing
figures. Do not read a u250/u280 report as a calibrated recommendation.

## Validating a config

```bash
python3 generate_code.py --config configs/examples/<case>.json --force
cd generated/<case_dir>
make all TARGET=sw
make run TARGET=sw NUM=1
```

XO build + RTL cosimulation is optional and heavier (`make compile` builds the host
runner the XO simulation needs; add `SAVE_WAVEFORM=0` to skip the waveform database):

```bash
make all TARGET=xo
make compile
make run TARGET=xo NUM=10
```

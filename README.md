# StreamNTT — Segmented Multi-Prime NTT Generator

StreamNTT is a code generator for high-throughput, streaming FPGA NTT accelerators
(TAPA / Vitis HLS dataflow), targeted at FHE-oriented workloads: large transforms,
many NTT-friendly primes, and HBM-attached streaming cores. This extension generates
the **segmented** StreamNTT architecture: a multi-prime NTT pipeline whose twiddle
factors are produced on the fly instead of being fully precomputed.

## Architecture overview

Each generated core is a streaming pipeline of butterfly stages:

```text
L-stage: TFG + TFR + BU (with ICBU ping-pong storage)
X-stage: TFG + TFR + TFS + BU
```

| Unit | Role |
|---|---|
| TFG | segmented Twiddle Factor Generator — produces one segment **base** per `tfg_seg_len` cycles via a modular base recurrence |
| TFR | Twiddle Factor Reconstructor — temporally reconstructs one twiddle per cycle from the segment base using precomputed in-segment ratios |
| TFS | Twiddle Factor Scaler (X-stages only) — applies the spatial scale factors across butterfly lanes |
| BU / ICBU | Butterfly Unit / its Inter-stage Connection Buffer (ping-pong banks, URAM/BRAM/LUTRAM-mappable) |

The segmented on-the-fly scheme rests on one observation: the twiddle recurrence does
not need to run at the butterfly consumption rate. The TFG runs a **base recurrence**
once per segment (period `tfg_seg_len`), the TFR performs **temporal reconstruction**
of the per-cycle twiddles inside the segment, and — for X-stages — the TFS adds the
**spatial scaling** across lanes. Early L-stages with short twiddle periods stay
table-driven; later stages are generated on the fly, which removes the large
full-twiddle ROMs that otherwise dominate multi-prime NTT designs.

## Repository layout

```text
generate_code.py    public CLI — generates a complete, self-contained accelerator case
code_generator/     generator modules (stage planning/emission, twiddle tables,
                    reduction emission, shift-add recipes, ICBU storage schedules)
dse/                pre-generation DSE: resource/cost model, ICBU bind_storage DSE,
                    calibration layers, device capacity profiles
templates/          active flat templates: ntt.cpp, ntt.h.template, host.cpp,
                    Makefile, gen_config.py, impl_config.json
configs/            configuration reference (configs/README.md) + configs/examples/
generated/          generator output (one directory per case) — never edit by hand
```

## Quickstart

Requirements: Python 3 with `pip install -r requirements.txt` (numpy, sympy), and a
TAPA + Vitis HLS environment for the build/simulation flows.

```bash
#Generate a case (config-driven; no environment variables needed)
python3 generate_code.py --config configs/examples/n64k_bu16_single52_barrett_p16_u55c.json --force
cd generated/N65536_BU16_CH8_QK52_single52_barrett
```

You can use the Makefile for compilation & execution.
We recommend that you use Rapidstream to achieve higher clock frequency, but you could only use the traditional Vitis flow.

HOST Compilation
```bash
make all TARGET=sw
```
Synthesize and generate xo file
```bash
make all TARGET=xo
```
Generate bitstream from xo file (Vitis only, no Rapidstream)
```bash
make all TARGET=hw
```
Optimize xo file with Rapidstream and generate bitstream (recommended)
```bash
make all TARGET=hw-opt
```
Simulation / Onboard verification (NUM = number of polynomials)
```bash
make run TARGET={sw | xo | hw | hw-opt} NUM=10000
```

You should see **PASSED!** message after running on-board verification. Note: Waveform capture is on by default; append `SAVE_WAVEFORM=0` to `make run TARGET=xo`
for a faster run without a waveform database.

## Configuration-driven DSE / ICBU storage

The ICBU storage choice and the pre-generation DSE are controlled from the config
file via the nested `icbu` block — no environment variables required:

```json
{
  "icbu": {
    "storage_recipe": "auto",
    "dse_mode": "apply",
    "device": "u55c"
  }
}
```

- `storage_recipe`: `2U2B` | `3U1B` | `4U` | `4B` | `4L` | `auto`. Concrete values pin
  the per-bank URAM/BRAM/LUTRAM mapping of the ICBU ping-pong banks; `auto` leaves the
  recipe eligible for DSE selection.
- `dse_mode`: `off` | `report` | `recommend` | `apply` | `apply_override`.
  The **default is `report`**: the resource/guard table is printed and nothing about
  the emitted source changes. `recommend` additionally records a machine-readable
  decision (`dse_decision.txt`). `apply` adopts a recommended recipe **only** when a
  calibrated, strong (D2) recommendation exists — it never guesses on uncalibrated
  data, and a recipe you pinned explicitly always wins (`apply_override` may override
  a pinned scalar recipe, but never a per-stage-group schedule).
- `device`: `u55c` (calibrated default) | `u250` | `u280`. The u250/u280 profiles are
  **capacity-only** until calibration data is added for them: reports against these
  devices state their uncalibrated status and stay informational.

## Supported reduction routes

- `shiftadd: "none"` — **Barrett** (the default, general route). Works for any
  supported prime set, including **wide mixed-bit sets** (e.g. 52-bit and 62-bit
  classes in one design).
- `shiftadd: "reduce"` — shift-add **full reduction** replacement for lightweight
  (low Hamming-weight) primes: DSP-saving, LUT-heavier.
- `shiftadd: "mul"` — shift-add **multiplication** path for lightweight primes.
- Wide mixed-bit **Barrett** is supported today. Wide mixed-bit **shift-add** remains
  limited; per-class multi-recipe shift-add emission is future work.

## Known limitations / caveats

- **Many-prime mixed Barrett resource ceiling.** With many primes, per-prime constants
  (q, mu) become ROM-loaded arrays, so HLS can no longer constant-fold and value-prune
  the modular multipliers. A 42-prime mixed Barrett BU8 diagnostic reaches
  `x_stages = 1290 = 43x30` DSP and can fail slot-level placement in the optimized
  implementation flow. This is not a correctness bug — it is a resource-model / DSE
  caveat to plan around (the DSE report exists for exactly this).
- **`tfg_seg_len = 32` is experimental.** P=32 requires a valid precompute/OTF-period
  setup (5 precomputed L-stages so every on-the-fly stage period is >= 32); it is a
  real structural change, **not** equivalent to changing an HLS II pragma. See
  `configs/README.md` for the currently supported control.
- **DSE scope.** The pre-generation DSE is a feasibility guard over calibrated
  evidence (module/ICBU-level DSP and memory), not a full place-and-route predictor.
  Passing the guard does not guarantee timing closure; failing it early saves a build.
- **`generated/` is output.** Each case directory is fully regenerated from its
  config; manual edits there are overwritten by the next `--force` run.

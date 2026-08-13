# StreamNTT — Segmented Multi-Prime NTT Generator

StreamNTT generates high-throughput streaming FPGA accelerators for
number-theoretic transforms. The generated TAPA/Vitis HLS design supports
multi-prime execution, HBM-attached data movement, and segmented on-the-fly
twiddle generation.

## Generated architecture

Each generated core uses the following twiddle-supply paths:

```text
L-stage, early: table-driven twiddle supply -> BU / ICBU
L-stage, later: Feedback TFG -> TFR -> BU / ICBU
X-stage:        Feedback TFG -> TFR -> TFS -> BU
```

| Unit | Role |
|---|---|
| Feedback TFG | Advances the segmented base recurrence |
| TFR | Reconstructs per-cycle twiddles within each recurrence segment |
| TFS | Reconstructs spatial twiddle factors across X-stage butterfly lanes |
| BU / ICBU | Performs butterfly arithmetic and provides inter-stage ping-pong storage |

`tfg_seg_len` is the public recurrence-segment period. Early L-stages use compact
tables; later L-stages and all X-stages use the segmented recurrence path. A generated
multi-prime design switches its prime-indexed constants at polynomial boundaries.
Channel geometry can replicate independent cores while preserving the same per-core
pipeline.

## Arithmetic support

StreamNTT supports Barrett, `shiftadd_reduce`, and `shiftadd_mul` for single-K
groups and eligible mixed-K {52,62} prime groups.

Shift-add support applies to eligible prime families.

## Configuration

Generation is driven by JSON. A representative single-case configuration is:

```json
{
  "name": "barrett",
  "N": 65536,
  "BU": 4,
  "CH": 2,
  "RATE": 0.5,
  "primes": ["TI_1"],
  "tfg_seg_len": 16,
  "shiftadd": "none"
}
```

The public fields are `N`, `BU`, `CH`, `RATE`, `K`, `primes`, `tfg_seg_len`,
`shiftadd`, `name`, and `icbu`. `K` is optional and is inferred from the selected
prime set when omitted. Named primes come from `KNOWN_PRIMES`; Barrett configurations
may also use compatible integer prime values. The `icbu` block controls the storage
recipe and resource-exploration mode. See [configs/README.md](configs/README.md) for
the complete schema and retained examples.

The generator includes configuration-driven ICBU storage and resource exploration
utilities.

## Generate a case

Install the Python dependencies, then invoke the config-only interface:

```bash
python3 -m pip install -r requirements.txt
python3 generate_code.py --config <config.json>
```

`--suffix <tag>` can distinguish generated directories, and `--force` can replace an
existing generated case with the same name.

## Build and run

The generator prints the output directory. From that directory, build and run the
software target with:

```bash
cd generated/<case-directory>
make all TARGET=sw
make run TARGET=sw NUM=1
```

The generated Makefile also provides `xo`, `hw`, and `hw-opt` targets for environments
with the corresponding TAPA, Vitis, board, and RapidStream setup:

```bash
make all TARGET=xo
make all TARGET=hw
make all TARGET=hw-opt
```

Use the matching target with `make run` when that execution environment is available.

## Repository organization

```text
generate_code.py    public generator entry point
code_generator/     stage, twiddle, arithmetic, ICBU, and top-level emitters
dse/                ICBU storage and resource exploration utilities
templates/          generated source, host, build, and floorplan templates
configs/            public schema reference and formal examples
generated/          generated case directories
```

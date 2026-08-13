# StreamNTT configuration reference

`python3 generate_code.py --config <config.json>` is the public generation interface.
Configuration files are JSON without comments. Unknown fields are rejected.

## Single-case schema

```json
{
  "name": "barrett",
  "N": 65536,
  "BU": 4,
  "CH": 2,
  "RATE": 0.5,
  "K": 62,
  "primes": ["TI_1"],
  "tfg_seg_len": 16,
  "shiftadd": "none",
  "icbu": {
    "storage_recipe": "2U2B",
    "dse_mode": "off",
    "device": "u55c"
  }
}
```

Required fields are `N`, `BU`, `CH`, `RATE`, `primes`, and `tfg_seg_len`.
`K`, `shiftadd`, `name`, and `icbu` are optional.

| Field | Meaning |
|---|---|
| `N` | Transform size; a power of two |
| `BU` | Butterfly units per core; a power of two |
| `CH` | HBM/DRAM data-channel count |
| `RATE` | Channel data-rate utilization factor; the design requires `CH * round_pow2(8 * RATE) >= 2 * BU` |
| `K` | Optional arithmetic/container width in bits; explicit values use `1..62`, and an omitted value is inferred from the prime set |
| `primes` | Non-empty list of aliases from `KNOWN_PRIMES` or integer custom primes; integer custom primes use the Barrett route |
| `tfg_seg_len` | Segmented recurrence period |
| `shiftadd` | `"none"` for Barrett, `"reduce"` for `shiftadd_reduce`, or `"mul"` for `shiftadd_mul`; defaults to `"none"` |
| `name` | Optional tag appended to the generated directory name |
| `icbu` | Optional ICBU storage and resource-exploration block |

Accepted `tfg_seg_len` values are `4`, `8`, `16`, and `32`. The value must divide
`N / (2 * BU)`, and `N / (2 * BU)` must be greater than `tfg_seg_len`. Values `4`
and `8` are used with shift-add routes in the public interface. A value of `32`
selects five table-driven L-stages and requires at least six L-stages in the generated
pipeline.

Arithmetic route support is summarized in the top-level README. Shift-add routes accept
eligible named-prime families; Barrett also accepts compatible integer primes.

## ICBU block

| Sub-field | Accepted values | Meaning |
|---|---|---|
| `storage_recipe` | `2U2B`, `3U1B`, `4U`, `4B`, `4L`, `auto` | Selects the ICBU bank-storage recipe; `auto` leaves selection to the configured exploration mode |
| `dse_mode` | `off`, `report`, `recommend`, `apply`, `apply_override` | Controls resource reporting, recommendation output, and recipe application |
| `device` | `u55c`, `u250`, `u280` | Selects the device-capacity profile used by resource exploration |

The defaults are `storage_recipe: "2U2B"`, `dse_mode: "report"`, and
`device: "u55c"`.

## Batch configs

A config file can generate several cases by combining a shared `common` object with a
`cases` list:

```json
{
  "common": {
    "N": 65536,
    "BU": 4,
    "CH": 2,
    "RATE": 0.5,
    "tfg_seg_len": 16
  },
  "cases": [
    {"name": "barrett", "primes": ["TI_1"], "shiftadd": "none"},
    {"name": "shiftadd_mul", "primes": ["TII_33", "TII_34"], "shiftadd": "mul"}
  ]
}
```

Each merged case is validated against the single-case schema.

## Formal examples

The public example set contains two transform sizes and four representative cases:

| File | Transform size | Route | Prime shape |
|---|---:|---|---|
| `n65536_barrett.json` | 65536 | Barrett default | Single 62-bit K-group |
| `n65536_mixed_shiftadd_reduce.json` | 65536 | `shiftadd_reduce` | Mixed K-group {52,62} |
| `n131072_mixed_barrett.json` | 131072 | Barrett | Mixed K-group {52,62} |
| `n131072_multiprime_shiftadd_mul.json` | 131072 | `shiftadd_mul` | Two primes in one 52-bit K-group |

Generate any retained example with:

```bash
python3 generate_code.py --config configs/examples/<example.json>
```

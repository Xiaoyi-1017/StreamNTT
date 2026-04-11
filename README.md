# StreamNTT
Code generator for high-throughput HLS FPGA NTT accelerator

### Prerequisites

- AMD Vitis (2023.2) - https://www.amd.com/en/products/software/adaptive-socs-and-fpgas/vitis.html

- AMD Alveo U55C Platform - https://www.amd.com/content/dam/amd/en/documents/products/accelerators/alveo/u55c/alveo-u55c-product-brief.pdf

- TAPA (0.1.20250803) - https://tapa.readthedocs.io/en/main/

- Rapidstream (2025.1.0807, recommended) - https://docs.rapidstream-da.com/

### Install Python Requirements
```bash
pip install -r requirements.txt 
```

## Generating an NTT project 

Please use generate_code.py to automatically generate an NTT project.

| Argument | Description | Supported Values | Default |
|----------|-------------|-----------------|-----------------|
| **N** | Transform size (polynomial degree) | 256 - 131072  | 131072 |
| **q** | Prime modulus | Arbitrary NTT-friendly Prime (up to 2^64-1) | 2305843009146585089 | 
| **BU** | Number of butterfly units per stage | 1, 2, 4, 8, 16, 32 | 16 |
| **CH** | Number of input HBM channels <br> (Total of 2×CH channels are used for input & output) | 1, 2, 4, 8, 16 | 8 |
| **RATE** | Effective data transfer rate of HBM channel | 0.5, 1.0 | 0.5 |
| **TFG_II** | Temporary twiddle generation II (l times II=l TFGs to achieve II=1 downstream BU). | 1, 2, 3, 4 | 4 |

Example command (with default values):
```bash
./generate_code.py -N 131072 -q 2305843009146585089 -BU 16 -CH 8 -RATE 0.5 -TFG_II 4
```

Example console message:
```
Values used -> N: 131072, q: 2305843009146585089, HostData: uint64_t, BU: 16, CH: 8, RATE: 0.5, veclen: 8, TFG_II: 4
Creating a new folder: N131072_BU16_CH8_q2305843009146585089_TFG_II4
```

## Compilation & Execution 

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
make run TARGET={sw | xo | hw | hw-opt} NUM=10
```

You should see **PASSED!** message after running on-board verification.



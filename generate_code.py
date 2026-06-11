#!/usr/bin/env python3
"""
Segmented TFG Phase 1 generator.

Generates StreamNTT cases with the segmented L-stage TFG architecture.
- Stages 0..3: flattened precomputed twiddle table (tw_l_base_table[15][NUM_PRIMES]).
- Stages >= 4:  segmented base recurrence + ratio completion.

Usage:
    ./generate_code.py --config configs/segmented_phase1_minimal.json
    ./generate_code.py --config <path> --suffix <tag> --force
    # --config is the supported entry point. (--preset needs a presets.json at the repo
    # root, which is not currently provided.)
"""

import argparse
import json
import math
import os
import shutil
import sys

from code_generator.twiddle_generator import (
    get_nth_root_of_unity_and_psi_fast,
    twiddle_generator_BR,
    twiddle_base_generator_lanes,
)

# Reuse stable helpers from generate_code without modification.
from code_generator.shiftadd_generator import emit_shiftadd_header
# Phase 3.2/3.4: stage/control generator framework. D2: region-1/2 emission goes through the
# StagePlan-driven PRODUCTION maps (legacy per-name accessors remain in stage_generator as
# byte-identity anchors / test oracle only).
from code_generator.stage_generator import (
    region1_emission_map,
    region2_emission_map,
    REGION2_WRAPPER_MARKERS,
    gen_icbu_write_limit,
    ICBU_WRITE_LIMIT_MARKERS,
    gen_bf_unit_storage,
    BF_UNIT_STORAGE_MARKERS,
    BF_UNIT_STORAGE_RECIPES,
)
from code_generator.icbu_generator import parse_icbu_recipe_schedule
# Paper-A DSE API slice: stdlib-only device capacity profiles (u55c calibrated/default;
# u250/u280 capacity-only). Used by validate_config_seg (icbu.device) + the ICBU DSE block.
from dse.device_profiles import get_device_profile
from code_generator.stage_plan import plan_from_precomp_boundary, plan_from_schedule, classify_p32_evidence
# Phase 3.5 S1a-0: reduce-family emission seam (VERBATIM this slice; import runs the
# read-only fail-fast anchor self-check pinning the template [reduce] block).
# Phase 3.5 S1b: final_K inference + reduce-set classification (metadata/stdout only;
# _k_arith_for moved into reduce_generator and imported back for compute_all_params).
from code_generator.reduce_generator import (
    gen_reduce_block,
    REDUCE_MARKERS,
    classify_reduce_set,
    _k_arith_for,
)
# Phase 3.5 S3c: ReduceRecipe composition layer (built ONLY for active shiftadd
# routes; the Barrett/none path stays recipe-free). shiftadd_generator consumes
# the instance by duck-typing -- it never imports reduce_recipe (no cycle).
from code_generator.reduce_recipe import build_reduce_recipe

ROOT = os.path.dirname(os.path.abspath(__file__))
# Template flatten slice (2026-06-11): the active segmented+shared templates live FLAT in
# templates/ (ntt.cpp, ntt.h.template, host.cpp, Makefile, gen_config.py, impl_config.json);
# the legacy multi/ and single/ template trees were removed (archived under Checkpoint/ + git history).
SEG_TEMPLATE_DIR = os.path.join(ROOT, "templates")


# ----------------------------------------------------------------------
# Local segmented generator helpers
# Extracted from legacy generate_code.py.
# Behavior-preserving. Do not edit arithmetic here.
# ----------------------------------------------------------------------

KNOWN_PRIMES = {
    # Table I (61/62-bit class)
    "TI_1":  2**61 - 2**26 + 1,
    "TI_2":  2**61 - 2**24 - 2**20 + 1,
    "TI_3":  2**61 - 2**24 + 1,
    "TI_4":  2**61 - 2**22 + 2**19 + 1,
    "TI_5":  2**61 - 2**21 + 1,
    "TI_6":  2**61 - 2**21 + 2**16 + 1,
    "TI_7":  2**61 + 2**22 + 2**20 + 1,
    "TI_8":  2**61 + 2**23 - 2**18 + 1,
    "TI_9":  2**61 + 2**23 + 2**21 + 1,
    "TI_10": 2**61 + 2**24 - 2**19 + 1,
    "TI_11": 2**61 + 2**25 + 2**23 + 1,
    "TI_12": 2**61 + 2**26 + 2**16 + 1,
    # Table II (51/52-bit class)
    "TII_1":  2**51 - 2**29 - 2**19 + 1,
    "TII_2":  2**51 - 2**28 - 2**23 + 2**19 + 1,
    "TII_3":  2**51 - 2**28 - 2**22 + 1,
    "TII_4":  2**51 - 2**28 + 2**26 + 2**20 + 1,
    "TII_5":  2**51 - 2**27 - 2**24 + 2**22 + 1,
    "TII_6":  2**51 - 2**27 - 2**19 + 1,
    "TII_7":  2**51 - 2**27 + 2**24 + 1,
    "TII_8":  2**51 - 2**26 - 2**22 - 2**19 + 1,
    "TII_9":  2**51 - 2**26 + 2**23 + 2**20 + 1,
    "TII_10": 2**51 - 2**26 + 2**24 - 2**18 + 1,
    "TII_11": 2**51 - 2**25 - 2**23 + 2**19 + 1,
    "TII_12": 2**51 - 2**25 - 2**22 - 2**20 + 1,
    "TII_13": 2**51 - 2**25 - 2**22 + 2**18 + 1,
    "TII_14": 2**51 - 2**25 - 2**21 - 2**18 + 1,
    "TII_15": 2**51 - 2**25 + 2**20 - 2**18 + 1,
    "TII_16": 2**51 - 2**25 + 2**23 - 2**19 + 1,
    "TII_17": 2**51 - 2**25 + 2**23 + 2**20 + 1,
    "TII_18": 2**51 - 2**24 + 2**19 + 1,
    "TII_19": 2**51 - 2**30 - 2**20 + 1,
    "TII_20": 2**51 - 2**29 - 2**22 + 2**18 + 1,
    "TII_21": 2**51 - 2**24 + 2**21 + 2**18 + 1,
    "TII_22": 2**51 - 2**23 + 2**20 + 1,
    "TII_23": 2**51 - 2**23 + 2**21 - 2**18 + 1,
    "TII_24": 2**51 + 2**21 - 2**18 + 1,
    "TII_25": 2**51 + 2**22 - 2**20 - 2**18 + 1,
    "TII_26": 2**51 + 2**22 + 2**20 + 1,
    "TII_27": 2**51 + 2**23 + 2**20 + 2**18 + 1,
    "TII_28": 2**51 + 2**25 - 2**23 - 2**21 + 1,
    "TII_29": 2**51 + 2**25 - 2**22 + 2**20 + 1,
    "TII_30": 2**51 + 2**25 + 1,
    "TII_31": 2**51 + 2**25 + 2**19 + 1,
    "TII_32": 2**51 + 2**25 + 2**21 + 1,
    "TII_33": 2**51 + 2**25 + 2**22 + 2**19 + 1,
    "TII_34": 2**51 + 2**26 - 2**20 - 2**18 + 1,
    "TII_35": 2**51 + 2**26 + 2**21 + 2**18 + 1,
    "TII_36": 2**51 + 2**27 - 2**24 - 2**21 + 1,
    "TII_37": 2**51 + 2**28 + 2**18 + 1,
    "TII_38": 2**51 + 2**28 + 2**20 + 1,
    "TII_39": 2**51 + 2**29 + 1,
    "TII_40": 2**51 - 2**30 + 2**19 + 1,
    "TII_41": 2**51 + 2**29 + 2**21 + 2**18 + 1,
    # Table III (61/62-bit class)
    "TIII_1":  2**61 - 2**24 + 1,
    "TIII_2":  2**61 - 2**21 + 1,
    "TIII_3":  2**61 + 2**23 - 2**18 + 1,
    "TIII_4":  2**61 + 2**24 - 2**19 + 1,
    "TIII_5":  2**61 + 2**27 + 2**21 + 1,
    "TIII_6":  2**61 + 2**29 - 2**22 + 1,
    "TIII_7":  2**61 + 2**30 + 2**18 + 1,
    "TIII_8":  2**61 + 2**30 + 2**26 + 1,
    "TIII_9":  2**61 - 2**22 + 2**19 + 1,
    "TIII_10": 2**61 + 2**22 + 2**20 + 1,
    "TIII_11": 2**61 + 2**23 + 2**21 + 1,
    "TIII_12": 2**61 + 2**25 + 2**23 + 1,
    "TIII_13": 2**61 + 2**28 - 2**25 + 1,
    "TIII_14": 2**61 + 2**29 + 2**19 + 1,
    "TIII_15": 2**61 + 2**30 + 2**20 + 1,
    "TIII_16": 2**61 + 2**30 + 2**28 + 1,
}

def resolve_primes(prime_list):
    """Resolve a list of prime specs (aliases or integers) to integer q values."""
    resolved = []
    for p in prime_list:
        if isinstance(p, int):
            resolved.append(p)
        elif isinstance(p, str):
            if p in KNOWN_PRIMES:
                resolved.append(KNOWN_PRIMES[p])
            else:
                raise RuntimeError(
                    f"Unknown prime alias '{p}'. Known aliases: "
                    f"{', '.join(sorted(KNOWN_PRIMES.keys()))}. "
                    f"Use an integer q value for custom primes."
                )
        else:
            raise RuntimeError(f"Invalid prime entry: {p}. Must be a string alias or an integer.")
    return resolved

def resolve_primes_with_aliases(prime_list):
    """Resolve to a list of (alias, q) pairs. Integer entries get a synthetic
    alias 'INT_<value>' so downstream code (e.g., the shift-add ntt_shiftadd.h
    emitter) can label per-prime entries."""
    out = []
    for p in prime_list:
        if isinstance(p, int):
            out.append((f"INT_{p}", p))
        elif isinstance(p, str):
            if p in KNOWN_PRIMES:
                out.append((p, KNOWN_PRIMES[p]))
            else:
                raise RuntimeError(
                    f"Unknown prime alias '{p}'. Known aliases: "
                    f"{', '.join(sorted(KNOWN_PRIMES.keys()))}. "
                    f"Use an integer q value for custom primes."
                )
        else:
            raise RuntimeError(f"Invalid prime entry: {p}. Must be a string alias or an integer.")
    return out

def sanitize_name(s):
    """Replace spaces with underscores and strip leading/trailing underscores."""
    return s.replace(" ", "_").strip("_")

def build_case_name(cfg, mode, suffix=None, k_value=None):
    N = cfg["N"]
    BU = cfg["BU"]
    CH = cfg["CH"]

    if mode == "single":
        TFG_II = cfg["TFG_II"]
        base = f"N{N}_BU{BU}_CH{CH}_Q{cfg['q']}_{TFG_II}TFG"
    else:
        # Phase 3.5 S1b: the QK tag must reflect final_K (inferred/lifted included),
        # never a stale/absent cfg["K"]; callers pass k_value explicitly.
        K_tag = k_value if k_value is not None else cfg["K"]
        base = f"N{N}_BU{BU}_CH{CH}_QK{K_tag}"

    parts = [base]
    name = sanitize_name(cfg.get("name", ""))
    if name:
        parts.append(name)
    cli_suffix = sanitize_name(suffix or "")
    if cli_suffix:
        parts.append(cli_suffix)
    return "_".join(parts)

def round_to_nearest_power_of_2(n):
    if n < 1:
        return 1
    low = 2 ** math.floor(math.log2(n))
    high = 2 ** math.ceil(math.log2(n))
    return low if (n - low) < (high - n) else high

def compute_structural(cfg, K):
    N = cfg["N"]
    BU = cfg["BU"]
    CH = cfg["CH"]
    RATE = cfg["RATE"]
    TFG_II = cfg.get("TFG_II", 3)

    logN = int(math.log2(N))
    logBU = int(math.log2(BU))
    DEPTH = N // (2 * BU)
    logDEPTH = int(math.log2(DEPTH))
    NUM_L_stage = logN - logBU - 1

    bits = 64
    DATA_BSIZE = bits // 8
    DataCHLen = 64 // DATA_BSIZE
    EffDataCHLen = round_to_nearest_power_of_2(DataCHLen * RATE)

    NUM_CORE = CH * EffDataCHLen // (2 * BU)
    if NUM_CORE < 1:
        raise RuntimeError(f"CH*{EffDataCHLen} must be >= 2*BU={2*BU}. Design not feasible.")

    GROUP_NUM = min(CH, NUM_CORE)
    GROUP_CH_NUM = CH // GROUP_NUM
    MCH = 1 if GROUP_CH_NUM > 1 else 0

    return {
        "K": K, "N": N, "logN": logN, "BU": BU, "logBU": logBU,
        "logDEPTH": logDEPTH, "NUM_L_stage": NUM_L_stage,
        "CH": CH, "TFG_II": TFG_II,
        "DATA_BSIZE": DATA_BSIZE, "EffDataCHLen": EffDataCHLen,
        "GROUP_NUM": GROUP_NUM, "GROUP_CH_NUM": GROUP_CH_NUM,
        "MCH": MCH, "DataCHLen": DataCHLen,
    }

def _generate_mul_full_data_nonstd_body(K):
    """Emit the body of mul_full_data_nonstd specialised for a compile-time K.

    The helper realizes the universal K-bit x K-bit unsigned full product
    with 26x17 DSP48E2 tiling. Asymmetric split: B as 26-bit chunks (last
    may be narrower), A as 17-bit chunks for normal rows; if the last B
    chunk is at most 17 bits, that single 'tail' row instead splits A
    into 26-bit chunks so b_tail fits on the 17-bit DSP side.

    Returns a triple (body_str, dsp_estimate, summary).
    """
    if K < 17 or K > 64:
        raise RuntimeError(
            f"K={K} outside the supported manual-tiling range [17, 64]. "
            f"Multi-prime container width must be representable inside two "
            f"DSP48E2 tile widths."
        )

    def split_widths(K, chunk):
        widths = []
        rem = K
        while rem > 0:
            w = min(chunk, rem)
            widths.append(w)
            rem -= w
        return widths

    b_widths = split_widths(K, 26)         # B split: 26-bit chunks
    a_widths = split_widths(K, 17)         # default A split: 17-bit chunks
    c_widths = split_widths(K, 26)         # tail-row A split: 26-bit chunks

    last_b_w = b_widths[-1]
    use_tail = (len(b_widths) >= 2) and (last_b_w <= 17)

    def chunk_decl(name_prefix, widths, src_name):
        lines = []
        pos = 0
        for i, w in enumerate(widths):
            lo = pos
            hi = pos + w - 1
            lines.append(
                f"\tap_uint<{w}> {name_prefix}{i} = "
                f"{src_name}.range({hi}, {lo});"
            )
            pos += w
        return lines

    body = []
    body.append("\tap_uint<{K}> A = (ap_uint<{K}>)A_in;".format(K=K))
    body.append("\tap_uint<{K}> B = (ap_uint<{K}>)B_in;".format(K=K))
    body.append("")
    body.append("\t// B split into 26-bit chunks (last chunk may be narrower)")
    body.extend(chunk_decl("B_b", b_widths, "B"))
    body.append("")
    body.append("\t// Default A split: 17-bit chunks (last chunk may be narrower)")
    body.extend(chunk_decl("A_a", a_widths, "A"))
    if use_tail:
        body.append("")
        body.append("\t// Tail row uses a 26-bit split of A (b_last <= 17 fits on "
                    "the 17-bit DSP side)")
        body.extend(chunk_decl("A_c", c_widths, "A"))
    body.append("")

    # Emit partial products
    n_dsp = 0
    n_b_normal_rows = len(b_widths) - 1 if use_tail else len(b_widths)

    body.append("\t// Partial products (one DSP48E2 per p_ij)")
    for i in range(n_b_normal_rows):
        bw = b_widths[i]
        for j, aw in enumerate(a_widths):
            res_w = bw + aw
            body.append(
                f"\tap_uint<{res_w}> p{i}{j} = (ap_uint<{res_w}>)"
                f"((ap_uint<{bw}>)B_b{i} * (ap_uint<{aw}>)A_a{j});"
            )
            body.append(
                f"#pragma HLS BIND_OP variable=p{i}{j} op=mul impl=dsp"
            )
            n_dsp += 1
    if use_tail:
        i = len(b_widths) - 1
        bw = b_widths[i]  # tail b width <= 17
        for j, cw in enumerate(c_widths):
            res_w = bw + cw
            body.append(
                f"\tap_uint<{res_w}> p{i}{j} = (ap_uint<{res_w}>)"
                f"((ap_uint<{cw}>)A_c{j} * (ap_uint<{bw}>)B_b{i});"
            )
            body.append(
                f"#pragma HLS BIND_OP variable=p{i}{j} op=mul impl=dsp"
            )
            n_dsp += 1
    body.append("")

    # Reassembly
    body.append("\tData2 acc = 0;")
    # b normal rows
    a_offsets = []
    pos = 0
    for w in a_widths:
        a_offsets.append(pos)
        pos += w
    b_offsets = []
    pos = 0
    for w in b_widths:
        b_offsets.append(pos)
        pos += w
    c_offsets = []
    pos = 0
    for w in c_widths:
        c_offsets.append(pos)
        pos += w

    for i in range(n_b_normal_rows):
        for j in range(len(a_widths)):
            shift = b_offsets[i] + a_offsets[j]
            body.append(f"\tacc += ((Data2)p{i}{j}) << {shift};")
    if use_tail:
        i = len(b_widths) - 1
        for j in range(len(c_widths)):
            shift = b_offsets[i] + c_offsets[j]
            body.append(f"\tacc += ((Data2)p{i}{j}) << {shift};")

    body.append("\treturn acc;")
    body_str = "\n".join(body)

    summary = (
        f"K={K}, B split = {b_widths}, default A split = {a_widths}, "
        f"tail-opt = {use_tail}"
        + (f", tail A split = {c_widths}" if use_tail else "")
        + f", DSPs = {n_dsp}"
    )
    return body_str, n_dsp, summary

def _split_widths(K, chunk):
    widths = []
    rem = K
    while rem > 0:
        w = min(chunk, rem)
        widths.append(w)
        rem -= w
    return widths

def _generate_mul_low_kplus2_body(K):
    """Emit the body of mul_low_data_data_kplus2_nonstd specialised for K.

    Same 26x17 tile as the full helper, but only emit partial products
    whose row+col offset is < K+2; the helper masks the result to the
    low K+2 bits. Used inside reduce() where X0 = (W_l*q) is read only
    via `(X0 + X1) & mask` with mask = (1<<(K+2))-1.
    """
    if K < 17 or K > 64:
        raise RuntimeError(f"K={K} outside [17, 64] manual-tiling range.")

    keep_threshold = K + 2

    b_widths = _split_widths(K, 26)
    a_widths = _split_widths(K, 17)
    c_widths = _split_widths(K, 26)
    last_b_w = b_widths[-1]
    use_tail = (len(b_widths) >= 2) and (last_b_w <= 17)

    a_offsets = []
    pos = 0
    for w in a_widths:
        a_offsets.append(pos)
        pos += w
    b_offsets = []
    pos = 0
    for w in b_widths:
        b_offsets.append(pos)
        pos += w
    c_offsets = []
    pos = 0
    for w in c_widths:
        c_offsets.append(pos)
        pos += w

    def chunk_decl(name_prefix, widths, src_name):
        lines = []
        pos = 0
        for i, w in enumerate(widths):
            lo = pos
            hi = pos + w - 1
            lines.append(
                f"\tap_uint<{w}> {name_prefix}{i} = "
                f"{src_name}.range({hi}, {lo});"
            )
            pos += w
        return lines

    body = []
    body.append("\tap_uint<{K}> A = (ap_uint<{K}>)A_in;".format(K=K))
    body.append("\tap_uint<{K}> B = (ap_uint<{K}>)B_in;".format(K=K))
    body.append("")
    body.append("\t// B split into 26-bit chunks (last chunk may be narrower)")
    body.extend(chunk_decl("B_b", b_widths, "B"))
    body.append("")
    body.append("\t// A 17-bit split for normal rows")
    body.extend(chunk_decl("A_a", a_widths, "A"))
    if use_tail:
        body.append("")
        body.append("\t// Tail row: 26-bit A split (b_last <= 17 fits the 17-bit DSP side)")
        body.extend(chunk_decl("A_c", c_widths, "A"))
    body.append("")

    # Determine kept partials
    kept = []  # list of (i, j, bw, aw, shift, ident, src_a_name, src_b_name)
    n_b_normal_rows = len(b_widths) - 1 if use_tail else len(b_widths)
    n_dsp = 0
    for i in range(n_b_normal_rows):
        for j, aw in enumerate(a_widths):
            shift = b_offsets[i] + a_offsets[j]
            if shift >= keep_threshold:
                continue
            kept.append((i, j, b_widths[i], aw, shift,
                         f"p{i}{j}", f"A_a{j}", f"B_b{i}"))
            n_dsp += 1
    if use_tail:
        i = len(b_widths) - 1
        for j, cw in enumerate(c_widths):
            shift = b_offsets[i] + c_offsets[j]
            if shift >= keep_threshold:
                continue
            kept.append((i, j, b_widths[i], cw, shift,
                         f"p{i}{j}", f"A_c{j}", f"B_b{i}"))
            n_dsp += 1

    body.append(
        f"\t// Kept partial products (shift < K+2={keep_threshold}); "
        f"{n_dsp} DSPs"
    )
    for (i, j, bw, aw, shift, ident, src_a, src_b) in kept:
        res_w = bw + aw
        # Tail row decomposes A on 26-bit side; emit "A * b" form for clarity
        if use_tail and i == len(b_widths) - 1:
            body.append(
                f"\tap_uint<{res_w}> {ident} = (ap_uint<{res_w}>)"
                f"((ap_uint<{aw}>){src_a} * (ap_uint<{bw}>){src_b});"
            )
        else:
            body.append(
                f"\tap_uint<{res_w}> {ident} = (ap_uint<{res_w}>)"
                f"((ap_uint<{bw}>){src_b} * (ap_uint<{aw}>){src_a});"
            )
        body.append(
            f"#pragma HLS BIND_OP variable={ident} op=mul impl=dsp"
        )
    body.append("")

    body.append("\tData2 acc = 0;")
    for (i, j, bw, aw, shift, ident, src_a, src_b) in kept:
        body.append(f"\tacc += ((Data2){ident}) << {shift};")

    body.append("")
    body.append(f"\treturn acc & ((((Data2)1) << (K + 2)) - 1);")

    summary = (f"K={K}, X0 low-(K+2) helper: kept {n_dsp} of "
               f"{len(b_widths)*len(a_widths)} partials")
    return "\n".join(body), n_dsp, summary

def _generate_mul_thigh_body(K):
    """Emit the body of mul_data_by_thigh_nonstd specialised for K.

    T_h is bounded by ap_uint<17> at the call site (caller's contract).
    Splits V_l into 26-bit chunks and lets HLS map each (chunk x T_h) to
    one DSP48E2 tile. ceil(K/26) partial products.
    """
    if K < 17 or K > 64:
        raise RuntimeError(f"K={K} outside [17, 64] manual-tiling range.")

    v_widths = _split_widths(K, 26)
    v_offsets = []
    pos = 0
    for w in v_widths:
        v_offsets.append(pos)
        pos += w

    body = []
    body.append("\tap_uint<{K}> V_l = (ap_uint<{K}>)V_l_in;".format(K=K))
    body.append("")
    pos = 0
    for i, w in enumerate(v_widths):
        lo = pos
        hi = pos + w - 1
        body.append(f"\tap_uint<{w}> v{i} = V_l.range({hi}, {lo});")
        pos += w
    body.append("")

    body.append("\t// Each partial: 26 (or last-chunk) bits x ap_uint<17> T_h")
    n_dsp = 0
    for i, w in enumerate(v_widths):
        res_w = w + 17
        body.append(
            f"\tap_uint<{res_w}> q{i} = (ap_uint<{res_w}>)"
            f"((ap_uint<{w}>)v{i} * (ap_uint<17>)T_h);"
        )
        body.append(f"#pragma HLS BIND_OP variable=q{i} op=mul impl=dsp")
        n_dsp += 1
    body.append("")

    body.append("\tData2 acc = 0;")
    for i, off in enumerate(v_offsets):
        body.append(f"\tacc += ((Data2)q{i}) << {off};")
    body.append("\treturn acc;")

    summary = f"K={K}, W1 (V_l*T_h with T_h<=17 bits) helper: {n_dsp} DSPs"
    return "\n".join(body), n_dsp, summary

def _compute_per_prime_thigh(all_params):
    """Per-prime T_h = mu >> K_arith.

    Used to decide whether the W1 tiny-high-word optimization (Route A)
    can be enabled at codegen time, and to embed the verified bound
    into ntt_mul.h.
    """
    rows = []
    for i, p in enumerate(all_params):
        T_h = p["mu"] >> p["K_arith"]
        rows.append((i, p["q"], p["K_arith"], p["mu"], T_h))
    max_T_h = max((r[4] for r in rows), default=0)
    return rows, max_T_h

def generate_multi_mul_helper(K, case_dir, all_params):
    """Emit src/ntt_mul.h with all four K-specific helpers used by reduce().

    Helpers emitted:
      - mul_full_data_nonstd            : universal K x K -> 2K full product
      - mul_low_data_data_kplus2_nonstd : (A*B) & ((1<<(K+2))-1)
      - mul_data_by_thigh_nonstd        : V_l * T_h (T_h <= ap_uint<17>) (fallback, 3 DSP @ K=62)
      - mul_data_by_thigh_tiny_nonstd   : V_l * T_h (T_h <= ap_uint<2>)  (Route A, 0 DSP)
      - mul_w1_compiletime              : codegen-chosen dispatcher; binds to
                                          tiny when max(T_h) <= 3, else fallback.

    The kernel calls these helpers by name regardless of K; only the
    bodies below are K-specialised. The W1 dispatcher is also chosen
    here at codegen time based on the per-prime T_h bound.
    """
    body_full, n_full, sum_full = _generate_mul_full_data_nonstd_body(K)
    body_low,  n_low,  sum_low  = _generate_mul_low_kplus2_body(K)
    body_th,   n_th,   sum_th   = _generate_mul_thigh_body(K)

    rows, max_T_h = _compute_per_prime_thigh(all_params)
    route_a_enabled = max_T_h <= 3

    if route_a_enabled:
        w1_dispatch_body = (
            "\t// Route A: max(T_h) <= 3 verified at codegen time -> 0-DSP\n"
            "\t// shift/mux helper. Truncating T_h to ap_uint<2> is bit-exact\n"
            "\t// for the verified prime set.\n"
            "\treturn mul_data_by_thigh_tiny_nonstd(V_l_in, (ap_uint<2>)T_h_in);"
        )
        sum_w1 = (f"W1 dispatcher = mul_data_by_thigh_tiny_nonstd "
                  f"(Route A, 0 DSP) — verified max(T_h)={max_T_h} <= 3")
    else:
        w1_dispatch_body = (
            "\t// Route A unavailable: max(T_h) > 3 across the generated\n"
            "\t// prime set. Falling back to the 26x17 DSP-aware helper.\n"
            "\treturn mul_data_by_thigh_nonstd(V_l_in, (ap_uint<17>)T_h_in);"
        )
        sum_w1 = (f"W1 dispatcher = mul_data_by_thigh_nonstd "
                  f"(fallback, ~3 DSP @ K=62) — observed max(T_h)={max_T_h}")

    text = (
        "#ifndef NTT_MUL_H\n"
        "#define NTT_MUL_H\n"
        "\n"
        "// Non-standard tiled multiplication helpers generated for this K\n"
        "\n"
        "static Data2 mul_full_data_nonstd(Data A_in, Data B_in) {\n"
        "#pragma HLS INLINE\n\n"
        f"{body_full}\n"
        "}\n"
        "\n"
        "static Data2 mul_low_data_data_kplus2_nonstd(Data A_in, Data B_in) {\n"
        "#pragma HLS INLINE\n\n"
        f"{body_low}\n"
        "}\n"
        "\n"
        "static Data2 mul_data_by_thigh_nonstd(Data V_l_in, ap_uint<17> T_h) {\n"
        "#pragma HLS INLINE\n\n"
        f"{body_th}\n"
        "}\n"
        "\n"
        "// Route A: 0-DSP shift/mux for T_h in {0,1,2,3}\n"
        "static Data2 mul_data_by_thigh_tiny_nonstd(Data A, ap_uint<2> h) {\n"
        "#pragma HLS INLINE\n"
        "\tData2 Aw = (Data2)A;\n"
        "\tData2 r0 = h[0] ? Aw : (Data2)0;\n"
        "\tData2 r1 = h[1] ? (Aw << 1) : (Data2)0;\n"
        "\treturn r0 + r1;\n"
        "}\n"
        "\n"
        "// W1 dispatcher (codegen-chosen at generator time)\n"
        "static Data2 mul_w1_compiletime(Data V_l_in, Data T_h_in) {\n"
        "#pragma HLS INLINE\n"
        f"{w1_dispatch_body}\n"
        "}\n"
        "\n"
        "#endif // NTT_MUL_H\n"
    )
    src_dir = os.path.join(case_dir, "src")
    with open(os.path.join(src_dir, "ntt_mul.h"), "w") as f:
        f.write(text)
    print(f"  ntt_mul.h: {sum_full}")
    print(f"             {sum_low}")
    print(f"             {sum_th}")
    print(f"             {sum_w1}")

def generate_build_infra(struct, case_dir):
    """Generate Makefile, link_config, and copy shared config files."""
    CH = struct["CH"]
    GROUP_NUM = struct["GROUP_NUM"]
    GROUP_CH_NUM = struct["GROUP_CH_NUM"]

    # Flattened template dir (2026-06-11): Makefile/gen_config.py/impl_config.json live
    # directly in templates/ (formerly under the shared/ template subdir).
    shared = os.path.join(ROOT, "templates")

    # Makefile
    with open(os.path.join(shared, "Makefile"), "r") as f:
        mk = f.read()
    mk = mk.replace("{CH}", str(CH)).replace("{GROUP_NUM}", str(GROUP_NUM)) \
            .replace("{GROUP_CH_NUM}", str(GROUP_CH_NUM))
    with open(os.path.join(case_dir, "Makefile"), "w") as f:
        f.write(mk)

    # link_config
    lines = ["[connectivity]"]
    lines_1 = ["[connectivity]"]
    for i in range(2 * CH):
        lines.append(f"sp=ntt.hbm_ch_{i}:HBM[{i}]")
        lines_1.append(f"sp=ntt_1.hbm_ch_{i}:HBM[{i}]")
    with open(os.path.join(case_dir, "link_config.ini"), "w") as f:
        f.write("\n".join(lines))
    with open(os.path.join(case_dir, "link_config_1.ini"), "w") as f:
        f.write("\n".join(lines_1))

    # Copy shared config files
    for fname in ("gen_config.py", "impl_config.json"):
        src = os.path.join(shared, fname)
        if os.path.exists(src):
            shutil.copy2(src, case_dir)

# ---------------------------------------------------------------------------
# BU-visible twiddle reference sequence
# ---------------------------------------------------------------------------
# The reference order is derived from the previously validated legacy
# 4-lane recurrence in the old multi-template ntt.cpp (the 4-lane variant, lines
# ~619-697; tree removed 2026-06-11 -- archived under Checkpoint/ and in git
# history). Both the 3-lane and 4-lane legacy variants produce
# the same mathematical NTT twiddle sequence to the butterfly unit; the
# 4-lane variant is chosen here because it has the simpler Python simulation.
# ---------------------------------------------------------------------------

def simulate_l_stage_reference_order(local_s: int, tw_l0: list, tw_l1: list,
                                      tw_l2: list, tw_l3: list, mod: int,
                                      DEPTH: int, logDEPTH: int,
                                      num_l_stage: int) -> list:
    """
    Produce the BU-visible reference twiddle sequence T_s[0..DEPTH-1] for
    L-stage s = local_s + 3.

    Uses the 4-lane recurrence structure from the validated legacy multi-prime
    kernel to derive the correct per-stage twiddle order. The result defines
    the ground truth for the segmented L-stage tables.

    Returns T_s[0..DEPTH-1] for one polynomial.
    """
    L_BASE = [
        tw_l0[local_s + 3],
        tw_l1[local_s + 2],
        tw_l2[local_s + 1],
        tw_l3[local_s + 1],
    ]
    R_s = tw_l0[local_s]

    shift_bits = logDEPTH - 3
    shift = (1 << (num_l_stage - 4 - local_s)) & ((1 << shift_bits) - 1)

    bundle_bits = logDEPTH - 2
    tw_local = list(L_BASE)
    read_idx = 0
    poly_bundle_ptr = 0

    seq = []
    for _ in range(DEPTH // 4):
        for lane in range(4):
            seq.append(tw_local[lane] % mod)
        nxt_val = [(tw_local[i] * R_s) % mod for i in range(4)]
        read_idx = (read_idx + shift) & ((1 << shift_bits) - 1)
        do_step_next = (read_idx != 0)
        poly_bundle_ptr = (poly_bundle_ptr + 1) & ((1 << bundle_bits) - 1)
        is_poly_boundary = (poly_bundle_ptr == 0)
        if is_poly_boundary:
            tw_local = list(L_BASE)
            read_idx = 0
        else:
            tw_local = [
                nxt_val[i] if (do_step_next and not is_poly_boundary) else L_BASE[i]
                for i in range(4)
            ]
    return seq[:DEPTH]


# ---------------------------------------------------------------------------
# Config validation (segmented-specific)
# ---------------------------------------------------------------------------

# Exact set of allowed public fields for the segmented Phase 1 schema.
_SEG_ALLOWED_FIELDS = frozenset({
    "N", "BU", "CH", "RATE", "K", "primes", "tfg_seg_len", "shiftadd", "name", "icbu"
})

# Paper-A DSE API slice: public `icbu` config block (config-facing twin of the ICBU env knobs).
#   icbu.storage_recipe : one of BF_UNIT_STORAGE_RECIPES, or "auto" (= recipe DSE-eligible: adopted
#                         ONLY under dse_mode apply/apply_override with a strong D2 recommendation;
#                         otherwise the 2U2B default is emitted -- never a guess).
#   icbu.dse_mode       : off/report/recommend/apply/apply_override; the DEFAULT stays "report"
#                         (no automatic source change without an explicit apply opt-in).
#   icbu.device         : device profile name (u55c default/calibrated; u250/u280 capacity-only).
# PRIORITY RULE (documented, tested): a SET STREAMNTT_* env knob ALWAYS supersedes the matching
# icbu.* config field -- env stays the experimental override / backward-compatible path; a
# superseded set config field prints one "icbu-config: ... superseded by env ..." stdout note.
# Configs WITHOUT the icbu block resolve exactly as before (byte-identical legacy path).
_ICBU_ALLOWED_SUBFIELDS = ("storage_recipe", "dse_mode", "device")
_ICBU_CONFIG_RECIPES = tuple(BF_UNIT_STORAGE_RECIPES) + ("auto",)
_DSE_MODES = ("off", "report", "recommend", "apply", "apply_override")


def validate_config_seg(cfg: dict) -> None:
    """Validate config for segmented TFG generator. Raises ValueError on error."""
    # Reject any field not in the approved schema (forbidden aliases and unknowns alike).
    for key in cfg:
        if key not in _SEG_ALLOWED_FIELDS:
            raise ValueError(
                f"Unsupported config field in segmented mainline schema: {key}"
            )
    if "primes" not in cfg:
        raise ValueError("Field 'primes' is required for segmented multi-prime generation.")
    if "tfg_seg_len" not in cfg:
        raise ValueError("Field 'tfg_seg_len' is required (use 16; 4/8 are experimental).")
    P = cfg["tfg_seg_len"]
    # tfg_seg_len (segment length P) timing policy on the current segmented mainline.
    # All of 4/8/16 GENERATE correctly (P must still divide DEPTH=N/(2*BU), checked later).
    # Timing criteria: 300 MHz is the general hard floor / minimum-pass criterion; the 390+ MHz
    # figures below are current P16/QK62 reference-quality evidence, NOT a universal requirement:
    #   P=16 : main / default choice. Reference-quality timing (~390+ MHz in XO even at QK62).
    #   P=8  : functionally valid; reaches reference-quality timing only with shiftadd="reduce"/"mul"
    #          (the Barrett / "none" backend at P=8 trails on timing).
    #   P=4  : experimental / test entry only; weakest timing, may not clear the floor in XO.
    #   P=32 : EXPERIMENTAL (Phase 3.4 D3); requires the hybrid precompute boundary PRECOMP=5 exactly
    #          (STREAMNTT_LSTAGE_PRECOMPUTE_BOUNDARY=5: s0..s4 precomputed -> OTF ge5 periods >= 32,
    #          32-entry tw_ratio segments; pairing enforced at the alias block). P=32 under the default
    #          PRECOMP=4 is INFEASIBLE (s4 period 16 < P; no period<P handling planned). Recurrence/
    #          timing EVIDENCE additionally requires a NON-DEGENERATE ge5 range (>=2 OTF stages, e.g.
    #          N1024/BU4 ge5={s5,s6}); single-OTF-stage cases generate but are classified DEGENERATE.
    if P not in (4, 8, 16, 32):
        raise ValueError(
            f"tfg_seg_len={P} is not supported. "
            "Use one of: 16 (default), 8, 4, or 32 (experimental; requires PRECOMP=5)."
        )
    shiftadd_mode = cfg.get("shiftadd", "none")
    if shiftadd_mode not in ("none", "reduce", "mul"):
        raise ValueError(
            f"shiftadd={shiftadd_mode!r} is not valid. "
            "Use one of: 'none' (Barrett), 'reduce' (shiftadd_reduce), 'mul' (shiftadd_mul)."
        )
    # Phase 3.5 S1b: "K" is OPTIONAL. When omitted, final_K is inferred from the prime
    # set (K_group = max per-prime K_arith); when present, final_K = max(K, K_group)
    # (under-spec lifts with a stdout warning). Only type/range is validated here.
    if "K" in cfg:
        _K_cfg = cfg["K"]
        if isinstance(_K_cfg, bool) or not isinstance(_K_cfg, int) or not (1 <= _K_cfg <= 62):
            raise ValueError(
                f"K={_K_cfg!r} is not valid. Use an integer in 1..62, or omit K to infer "
                "it from the prime set (final_K = max(config_K, K_group))."
            )
    # DEPTH = N/(2*BU) must be STRICTLY greater than the segment length P, not merely divisible by it.
    # At the boundary DEPTH == P (e.g. N=1024, BU=32, P=16 -> DEPTH=16) the later divisibility check
    # (DEPTH % P == 0) passes, but the X-stage seg-table builder then indexes seq[P] on a length-DEPTH
    # sequence (compute_x_seg_tables_one_prime) -> IndexError. Fail fast here with a clear message.
    # README documents the rule: DEPTH = N/(2*BU) > tfg_seg_len.
    # Guard order: only run this check once BU>0 and N is divisible by 2*BU, so it does NOT mask the
    # existing invalid-BU (math.log2(BU) in compute_structural) or N/(2*BU) divisibility errors and
    # never divides by zero for BU<=0. (Presence-guarded: configs missing N/BU keep downstream behavior.)
    if "N" in cfg and "BU" in cfg:
        N = cfg["N"]
        BU = cfg["BU"]
        if BU > 0 and N % (2 * BU) == 0:
            DEPTH = N // (2 * BU)
            if DEPTH <= P:
                raise ValueError(
                    f"DEPTH={DEPTH} (= N/(2*BU) = {N}/(2*{BU})) must be GREATER than "
                    f"tfg_seg_len={P}, not just divisible by it. DEPTH <= tfg_seg_len hits the segmented "
                    f"X-stage boundary (seq[P] is out of range in compute_x_seg_tables_one_prime). "
                    f"Use a larger N or a smaller tfg_seg_len. NEEDS_USER_DECISION."
                )
    # Paper-A DSE API slice: validate the OPTIONAL public `icbu` block (see the schema note at
    # _ICBU_ALLOWED_SUBFIELDS). Values are validated here (fail-fast, before any case dir is
    # created); resolution against the env override knobs happens in generate_seg_case.
    if "icbu" in cfg:
        icbu = cfg["icbu"]
        if not isinstance(icbu, dict):
            raise ValueError(
                "Config field 'icbu' must be an object, got %s." % type(icbu).__name__)
        for sub in icbu:
            if sub not in _ICBU_ALLOWED_SUBFIELDS:
                raise ValueError(
                    "Unsupported icbu config field: %s (allowed: %s)"
                    % (sub, ", ".join(_ICBU_ALLOWED_SUBFIELDS)))
        _sr = icbu.get("storage_recipe")
        if _sr is not None and _sr not in _ICBU_CONFIG_RECIPES:
            raise ValueError(
                "icbu.storage_recipe=%r is not valid. Use one of %s."
                % (_sr, list(_ICBU_CONFIG_RECIPES)))
        _dm = icbu.get("dse_mode")
        if _dm is not None and _dm not in _DSE_MODES:
            raise ValueError(
                "icbu.dse_mode=%r is not valid. Use one of %s."
                % (_dm, "/".join(_DSE_MODES)))
        _dv = icbu.get("device")
        if _dv is not None:
            get_device_profile(_dv)   # fail-fast: unknown device -> ValueError listing known names


# ---------------------------------------------------------------------------
# Table generation for stages 0..3 (flattened tw_l_base_table[15])
# ---------------------------------------------------------------------------

def compute_tw_l_base_table(mod: int, psi: int, N: int, BU: int,
                              logN: int, logBU: int,
                              tw_l0: list, tw_l1: list,
                              tw_l2: list, tw_l3: list,
                              boundary: int = 4) -> list:
    """
    Returns a flat list of twiddle values for the precomputed L-stages 0..boundary-1.
    Index = (1<<s) - 1 + phase, for s=0..boundary-1, phase=0..2^s-1.
    boundary=4 (default): 15 entries (current shape, byte-identical).
    boundary=5 (Phase 3.4 hybrid B=5): 31 entries (s4 appended; period 16).

    Stages 0..2: from legacy lane tables (matches tw_gen_L_s0/s1/s2 direct lookups).
    Stage 3:    from simulate_l_stage_reference_order (first 8 values, period=8).
    Stage 4:    (boundary=5 only) reference order local_s=1, first 16 values, period=16.
    """
    DEPTH = N // (2 * BU)
    num_l_stage = logN - logBU - 1
    logDEPTH = int(math.log2(DEPTH))

    result = []

    # Stage 0: single value = tw_l0[0]
    result.append(tw_l0[0])

    # Stage 1: [tw_l0[1], tw_l1[0]]
    result.append(tw_l0[1])
    result.append(tw_l1[0])

    # Stage 2: [tw_l0[2], tw_l1[1], tw_l2[0], tw_l3[0]]
    result.append(tw_l0[2])
    result.append(tw_l1[1])
    result.append(tw_l2[0])
    result.append(tw_l3[0])

    # Stage 3: first 8 twiddles from the reference order (local_s=0 → global stage 3).
    seq_s3 = simulate_l_stage_reference_order(0, tw_l0, tw_l1, tw_l2, tw_l3,
                                        mod, DEPTH, logDEPTH, num_l_stage)
    # Period = 8 (verified by caller via self-check); take first 8 values
    result.extend(seq_s3[:8])

    if boundary == 5:
        # Stage 4 precomputed (hybrid B=5): reference order local_s=1 (global stage 4), period 16.
        seq_s4 = simulate_l_stage_reference_order(1, tw_l0, tw_l1, tw_l2, tw_l3,
                                            mod, DEPTH, logDEPTH, num_l_stage)
        # Verify period 16 exactly over DEPTH before precomputing (fail-fast, not silent).
        for t in range(DEPTH):
            if seq_s4[t] != seq_s4[t % 16]:
                raise ValueError(
                    f"stage-4 precompute period check FAILED: q={mod} t={t}: "
                    f"expected period 16. NEEDS_USER_DECISION")
        result.extend(seq_s4[:16])

    assert len(result) == (31 if boundary == 5 else 15)
    return result


# ---------------------------------------------------------------------------
# Segmented table generation and self-check for stages >= 4
# ---------------------------------------------------------------------------

def compute_seg_tables_and_check(mod: int, psi: int, N: int, BU: int,
                                  logN: int, logBU: int, P: int,
                                  num_l_stage: int,
                                  tw_l0: list, tw_l1: list,
                                  tw_l2: list, tw_l3: list,
                                  start_stage: int = 4) -> tuple:
    """
    For each L-stage s in [start_stage, num_l_stage), derive the BU-visible twiddle
    sequence via simulate_l_stage_reference_order, then compute:
      base_s   = T_s[0]
      step_s   = T_s[P] * inv(T_s[0]) mod q   (valid within one period of 2^s)
      ratio_s  = [T_s[r] * inv(T_s[0]) mod q  for r in 0..P-1]
      period_blocks_s = (1 << s) // P

    Self-check uses period_blocks to reset base at each period boundary.

    Returns (bases, steps, ratios, period_blocks_list).
    Raises ValueError on self-check failure.
    """
    DEPTH = N // (2 * BU)
    logDEPTH = int(math.log2(DEPTH))
    bases, steps, ratios, period_blocks_list = [], [], [], []

    for s in range(start_stage, num_l_stage):
        local_s = s - 3  # local index 0=global3, 1=global4, ...
        seq = simulate_l_stage_reference_order(local_s, tw_l0, tw_l1, tw_l2, tw_l3,
                                        mod, DEPTH, logDEPTH, num_l_stage)

        period = 1 << s
        period_blocks = period // P
        base = seq[0]
        inv_base = pow(base, -1, mod)
        ratio = [(seq[r] * inv_base) % mod for r in range(P)]

        assert ratio[0] == 1, (
            f"ratio[0] != 1 for prime q={mod} stage={s}: got {ratio[0]}"
        )
        if len(ratio) != P:
            raise ValueError(
                f"L-stage ratio length mismatch: q={mod} stage={s}: got {len(ratio)}, "
                f"need tfg_seg_len={P}. NEEDS_USER_DECISION")

        # Step: T_s[P] * inv(T_s[0]) — valid within one period
        step = (seq[P] * inv_base) % mod

        # Self-check: uses sub_block = block % period_blocks to reset base
        for t in range(DEPTH):
            block = t // P
            phase = t % P
            sub_block = block % period_blocks
            recon_base = (base * pow(step, sub_block, mod)) % mod
            got = (recon_base * ratio[phase]) % mod
            if got != seq[t]:
                raise ValueError(
                    f"Segmentation self-check FAILED: prime q={mod} stage={s} "
                    f"P={P} t={t} block={block} sub_block={sub_block} phase={phase}: "
                    f"expected={seq[t]} got={got}. "
                    f"base={base} step={step} ratio[{phase}]={ratio[phase]}. "
                    f"NEEDS_USER_DECISION"
                )

        bases.append(base)
        steps.append(step)
        ratios.append(ratio)
        period_blocks_list.append(period_blocks)

    return bases, steps, ratios, period_blocks_list


# ---------------------------------------------------------------------------
# Work 2: X-stage per-stage segmented TFG tables and self-check
# ---------------------------------------------------------------------------

def simulate_x_stage_reference_order(s_x: int, tw_x0: list, tw_x1: list,
                                      tw_x2: list, tw_l1: list,
                                      mod: int, DEPTH: int,
                                      num_l_stage: int) -> list:
    """
    Reproduce the Work 1.2 BU-visible BASE sequence T_xs[t] for X-stage s_x.

    Mirrors the 3-lane II=3 producer + 3-phase II=1 merge for mod_id=0:
      tw_gen_X_s0to2 (s_x=0,1,2) / tw_gen_X_s3/s4/s5 (s_x>=3)
      + tw_merge_X_s0to2 / tw_merge_X_s3/s4/s5

    Formula derived from the II=3 producer:
      - Each producer iteration k emits three lane values (lane0/1/2).
      - Each lane advances by R_s once per iteration.
      - The merge demuxes lane(t%3) of iteration(t//3) as cycle t's output.

    So: T_xs[t] = lane<t%3>_init * R_s^(t//3) mod q

    Returns seq[0..DEPTH-1].
    """
    base_off = (1 << s_x) - 1
    lane0_init = int(tw_x0[base_off])
    lane1_init = int(tw_x1[base_off])
    lane2_init = int(tw_x2[base_off])

    if s_x == 0:
        # R_s[0] = tw_l_base_lane1_tab[num_l_stage_ge1 - 1] = tw_l1[num_l_stage - 2]
        R_s = int(tw_l1[num_l_stage - 2])
    else:
        # R_s = tw_x_base_lane1_tab[(1 << (s_x - 1)) - 1]
        R_s = int(tw_x1[(1 << (s_x - 1)) - 1])

    lanes = [lane0_init, lane1_init, lane2_init]
    seq = []
    for t in range(DEPTH):
        prod_iter = t // 3
        phase_3 = t % 3
        val = (lanes[phase_3] * pow(R_s, prod_iter, mod)) % mod
        seq.append(val)
    return seq


def compute_x_seg_tables_one_prime(mod: int, P: int, DEPTH: int,
                                    num_x_stage: int,
                                    tw_x0: list, tw_x1: list, tw_x2: list,
                                    tw_l1: list, num_l_stage: int) -> tuple:
    """
    For one prime q, compute per-stage Work 2 X-stage tables.

    For each X-stage s_x:
      - Simulate T_xs[0..DEPTH-1] via simulate_x_stage_reference_order.
      - Try recurrence: T[t] = base * step^block * ratio[phase]
      - Fallback to base table: T[t] = base_table[block] * ratio[phase]

    Returns (bases, steps, ratios, base_tables) each a list of length num_x_stage.
      - bases[s_x]  = base value (recurrence) or None (base_table mode)
      - steps[s_x]  = step value (recurrence) or None
      - base_tables[s_x] = list[NUM_SEG_BLOCKS] (base_table mode) or None
      - ratios[s_x] = list[P] (both modes)
    """
    if DEPTH % P != 0:
        raise ValueError(
            f"DEPTH={DEPTH} is not divisible by P={P}. NEEDS_USER_DECISION."
        )
    NUM_SEG_BLOCKS = DEPTH // P

    bases, steps, ratios_list, base_tables = [], [], [], []

    for s_x in range(num_x_stage):
        seq = simulate_x_stage_reference_order(s_x, tw_x0, tw_x1, tw_x2, tw_l1,
                                               mod, DEPTH, num_l_stage)
        assert len(seq) == DEPTH

        T0 = seq[0]
        inv_T0 = pow(T0, -1, mod)
        ratio = [(seq[r] * inv_T0) % mod for r in range(P)]
        assert ratio[0] == 1, (
            f"X-stage Work 2: ratio[0] != 1 for q={mod} s_x={s_x}: got {ratio[0]}"
        )
        if len(ratio) != P:
            raise ValueError(
                f"X-stage ratio length mismatch: q={mod} s_x={s_x}: got {len(ratio)}, "
                f"need tfg_seg_len={P}. NEEDS_USER_DECISION")

        # Try recurrence mode first
        step = (seq[P] * inv_T0) % mod
        ok_rec = True
        for t in range(DEPTH):
            block = t // P
            phase = t % P
            B = (T0 * pow(step, block, mod)) % mod
            recon = (B * ratio[phase]) % mod
            if recon != seq[t]:
                ok_rec = False
                break

        if ok_rec:
            bases.append(T0)
            steps.append(step)
            ratios_list.append(ratio)
            base_tables.append(None)
        else:
            # Base-table fallback: precomputed base per segment
            btab = [seq[k * P] for k in range(NUM_SEG_BLOCKS)]
            ok_tab = True
            for t in range(DEPTH):
                block = t // P
                phase = t % P
                recon = (btab[block] * ratio[phase]) % mod
                if recon != seq[t]:
                    ok_tab = False
                    break
            if ok_tab:
                bases.append(None)
                steps.append(None)
                ratios_list.append(ratio)
                base_tables.append(btab)
            else:
                raise ValueError(
                    f"X-stage Work 2 factorization FAILED for q={mod} s_x={s_x}: "
                    f"both recurrence and base-table modes fail self-check. "
                    f"NEEDS_USER_DECISION"
                )

    return bases, steps, ratios_list, base_tables


def determine_x_stage_modes(all_params: list, num_x_stage: int) -> list:
    """
    Determine mode ('recurrence' or 'base_table') for each X-stage across all primes.
    Mode is 'recurrence' if recurrence works for ALL primes at that stage;
    otherwise 'base_table' (the base_table must work for all primes — guaranteed
    by compute_x_seg_tables_one_prime raising ValueError otherwise).
    Returns list[num_x_stage] of mode strings.
    """
    modes = []
    for s_x in range(num_x_stage):
        all_rec = all(p['x_seg_bases'][s_x] is not None for p in all_params)
        modes.append('recurrence' if all_rec else 'base_table')
    return modes


# ---------------------------------------------------------------------------
# Work 2: HLS code generation helpers
# ---------------------------------------------------------------------------

def _gen_tw_gen_X_base_seg(s_x: int, P: int, is_single: bool = False, base_ii: int = None) -> str:
    """Generate tw_gen_X_base_seg_s<s_x> HLS function (recurrence mode, II=base_ii; default=P=tfg_seg_len)."""
    base_ii = P if base_ii is None else base_ii   # Phase 3.3.3: decoupled base-recurrence pipeline II
    fname = f"tw_gen_X_base_seg_s{s_x}"
    table_base = f"tw_x_s{s_x}_seg_base"
    table_step = f"tw_x_s{s_x}_seg_step"
    if is_single:
        # Single-prime: tw_x_s*_seg_base/_step are SCALARS (Task A); no cur_mod_id.
        return f"""void {fname}(tapa::ostream<Data>& base_fifo) {{
#pragma HLS INLINE off
\tData base = {table_base};
\tData step = {table_step};
\tap_uint<logDEPTH> seg_ctr = 0;
\tfor(;;) {{
#pragma HLS PIPELINE II={base_ii}
\t\tbase_fifo.write(base);
\t\tseg_ctr++;
\t\tbool poly_end = ((int)seg_ctr == NUM_SEG_BLOCKS);
\t\tif (poly_end) {{
\t\t\tseg_ctr = 0;
\t\t\tbase = {table_base};
\t\t\tstep = {table_step};
\t\t}} else {{
\t\t\tData nxt; reduce(base, step, nxt);
\t\t\tbase = nxt;
\t\t}}
\t}}
}}
"""
    return f"""void {fname}(tapa::ostream<Data>& base_fifo) {{
#pragma HLS INLINE off
\tModId cur_mod_id = 0;
\tData base = {table_base}[0];
\tData step = {table_step}[0];
\tap_uint<logDEPTH> seg_ctr = 0;
\tfor(;;) {{
#pragma HLS PIPELINE II={base_ii}
\t\tbase_fifo.write(base);
\t\tseg_ctr++;
\t\tbool poly_end = ((int)seg_ctr == NUM_SEG_BLOCKS);
\t\tif (poly_end) {{
\t\t\tseg_ctr = 0;
\t\t\tcur_mod_id = (cur_mod_id + 1 >= NUM_PRIMES) ? (ModId)0 : (ModId)(cur_mod_id + 1);
\t\t\tbase = {table_base}[cur_mod_id];
\t\t\tstep = {table_step}[cur_mod_id];
\t\t}} else {{
\t\t\tData nxt; reduce(base, step, nxt, (int)cur_mod_id);
\t\t\tbase = nxt;
\t\t}}
\t}}
}}
"""


def _gen_tw_gen_X_base_tab(s_x: int, P: int, is_single: bool = False) -> str:
    """Generate tw_gen_X_base_tab_s<s_x> HLS function (base-table mode, II=P)."""
    fname = f"tw_gen_X_base_tab_s{s_x}"
    table = f"tw_x_s{s_x}_base_table"
    if is_single:
        # Single-prime: tw_x_s*_base_table is [NUM_SEG_BLOCKS] (Task A); no cur_mod_id.
        return f"""void {fname}(tapa::ostream<Data>& base_fifo) {{
#pragma HLS INLINE off
\tap_uint<logDEPTH> seg_ctr = 0;
\tfor(;;) {{
#pragma HLS PIPELINE II={P}
\t\tData base = {table}[(int)seg_ctr];
\t\tbase_fifo.write(base);
\t\tseg_ctr++;
\t\tif ((int)seg_ctr == NUM_SEG_BLOCKS) {{
\t\t\tseg_ctr = 0;
\t\t}}
\t}}
}}
"""
    return f"""void {fname}(tapa::ostream<Data>& base_fifo) {{
#pragma HLS INLINE off
\tModId cur_mod_id = 0;
\tap_uint<logDEPTH> seg_ctr = 0;
\tfor(;;) {{
#pragma HLS PIPELINE II={P}
\t\tData base = {table}[cur_mod_id][(int)seg_ctr];
\t\tbase_fifo.write(base);
\t\tseg_ctr++;
\t\tif ((int)seg_ctr == NUM_SEG_BLOCKS) {{
\t\t\tseg_ctr = 0;
\t\t\tcur_mod_id = (cur_mod_id + 1 >= NUM_PRIMES) ? (ModId)0 : (ModId)(cur_mod_id + 1);
\t\t}}
\t}}
}}
"""


def _gen_tw_complete_X_ratio(s_x: int, P: int, is_single: bool = False) -> str:
    """Generate tw_complete_X_ratio_s<s_x> HLS function (II=1, ratio + Wide broadcast)."""
    fname = f"tw_complete_X_ratio_s{s_x}"
    table_ratio = f"tw_x_s{s_x}_seg_ratio"
    # Use ap_uint<logDEPTH> for phase to avoid wrap-before-TFG_SEG_LEN
    # (ap_uint<log2(P)> wraps to 0 when reaching P, so == TFG_SEG_LEN never fires)
    if is_single:
        # Single-prime: tw_x_s*_seg_ratio is [TFG_SEG_LEN] (Task A); no cur_mod_id / cnt cycling.
        return f"""void {fname}(tapa::istream<Data>& base_fifo,
\t\t\t\t\ttapa::ostream<Wide>& tw_X_W_s) {{
#pragma HLS INLINE off
\tData base = base_fifo.read();
\tap_uint<logDEPTH> phase = 0;
\tfor(;;) {{
#pragma HLS PIPELINE II=1
\t\tData tw_scalar;
\t\tif (phase == 0) {{
\t\t\ttw_scalar = base;
\t\t}} else {{
\t\t\treduce(base, {table_ratio}[(int)phase], tw_scalar);
\t\t}}
\t\tWide w = 0;
\t\tfor (int b = 0; b < BU; b++) {{
#pragma HLS UNROLL
\t\t\tw.range((b + 1) * K - 1, b * K) = tw_scalar;
\t\t}}
\t\ttw_X_W_s.write(w);
\t\tphase = (ap_uint<logDEPTH>)(phase + 1);
\t\tif ((int)phase == TFG_SEG_LEN) {{
\t\t\tphase = 0;
\t\t\tbase = base_fifo.read();
\t\t}}
\t}}
}}
"""
    return f"""void {fname}(tapa::istream<Data>& base_fifo,
\t\t\t\t\ttapa::ostream<Wide>& tw_X_W_s) {{
#pragma HLS INLINE off
\tModId cur_mod_id = 0;
\tData base = base_fifo.read();
\tap_uint<logDEPTH> phase = 0;
\tap_uint<logDEPTH> cnt = 0;
\tfor(;;) {{
#pragma HLS PIPELINE II=1
\t\tData tw_scalar;
\t\tif (phase == 0) {{
\t\t\ttw_scalar = base;
\t\t}} else {{
\t\t\treduce(base, {table_ratio}[(int)cur_mod_id][(int)phase], tw_scalar, (int)cur_mod_id);
\t\t}}
\t\tWide w = 0;
\t\tfor (int b = 0; b < BU; b++) {{
#pragma HLS UNROLL
\t\t\tw.range((b + 1) * K - 1, b * K) = tw_scalar;
\t\t}}
\t\ttw_X_W_s.write(w);
\t\tphase = (ap_uint<logDEPTH>)(phase + 1);
\t\tif ((int)phase == TFG_SEG_LEN) {{
\t\t\tphase = 0;
\t\t\tbase = base_fifo.read();
\t\t}}
\t\tcnt = (ap_uint<logDEPTH>)(cnt + 1);
\t\tif (cnt == 0) {{
\t\t\t// Wrapped: completed DEPTH cycles (polynomial boundary).
\t\t\tcur_mod_id = (cur_mod_id + 1 >= NUM_PRIMES) ? (ModId)0 : (ModId)(cur_mod_id + 1);
\t\t}}
\t}}
}}
"""


def _gen_x_stage_butterfly_dispatch(struct: dict) -> str:
    """Emit the X-stage BU-side butterfly dispatch (replaces the static template BUTTERFLY_LOOP).

    Keeps the existing ``if (s == N)`` per-stage selection, but unrolls the BU lanes explicitly.
    Spatial group 0 lanes (scale == 1, the TFS identity scalar) are emitted as a plain
    ``butterfly(...)`` with NO MR1 tw_scale reduce; non-identity lanes keep the exact
    ``butterfly_x_tw_scale(..., tw_scale_x_s<s>_tab[cur_mod_id][group], ...)`` call (the general
    path; USE_XSTAGE_S3_TW_SCALE_BU_COMPLETION is never defined, so no butterfly_x_s3_tw_scale).
    The per-stage scalars (shift/mask/next_mask/stride/next_stride) and ``s`` remain the STAGE_LOOP
    variables; only the dispatch body changes.
    """
    BU = struct["BU"]
    logBU = struct["logBU"]
    num_x_stage = logBU + 1
    is_single = struct.get("is_single", False)
    NL = chr(10)
    T = chr(9)

    def _shared(s: int) -> bool:
        # Sharing helps iff lanes-per-group = 2**(logBU - s) > 1 AND non-identity groups exist.
        return 1 <= s < logBU

    def _prologue(s: int) -> str:
        # One MR1 completion per non-identity spatial group g = 1 .. 2**s - 1 (shared by the group).
        rows = [T * 5 + "// [group-shared] s%d: %d non-identity group(s); base broadcast across BU lanes." % (s, (1 << s) - 1),
                T * 5 + "Data base_s%d = tw_local[s][0];" % s]
        for g in range(1, 1 << s):
            if is_single:
                rows.append(T * 5 + "Data ctw_s%d_g%d; reduce(base_s%d, tw_scale_x_s%d_tab[%d], ctw_s%d_g%d);"
                            % (s, g, s, s, g, s, g))
            else:
                rows.append(T * 5 + "Data ctw_s%d_g%d; reduce(base_s%d, tw_scale_x_s%d_tab[cur_mod_id][%d], ctw_s%d_g%d, cur_mod_id);"
                            % (s, g, s, s, g, s, g))
        return NL.join(rows) + NL

    def _lane(s: int, idx: int) -> str:
        group = idx >> (logBU - s)
        if s == 0 or group == 0:
            twdecl = "Data twf = tw_local[s][idx]; Data out_even, out_odd;"
            if is_single:
                call = ("butterfly(mem[s][2*idx], mem[s][2*idx+1], twf, "
                        "&out_even, &out_odd);  // s=%d idx=%d group0 scale==1 (identity skip)") % (s, idx)
            else:
                call = ("butterfly(mem[s][2*idx], mem[s][2*idx+1], twf, "
                        "&out_even, &out_odd, cur_mod_id);  // s=%d idx=%d group0 scale==1 (identity skip)") % (s, idx)
        elif _shared(s):
            twdecl = "Data out_even, out_odd;"
            if is_single:
                call = ("butterfly(mem[s][2*idx], mem[s][2*idx+1], ctw_s%d_g%d, "
                        "&out_even, &out_odd);  // s=%d idx=%d group%d (shared)") % (s, group, s, idx, group)
            else:
                call = ("butterfly(mem[s][2*idx], mem[s][2*idx+1], ctw_s%d_g%d, "
                        "&out_even, &out_odd, cur_mod_id);  // s=%d idx=%d group%d (shared)") % (s, group, s, idx, group)
        else:
            twdecl = "Data twf = tw_local[s][idx]; Data out_even, out_odd;"
            if is_single:
                call = ("butterfly_x_tw_scale(mem[s][2*idx], mem[s][2*idx+1], twf, "
                        "tw_scale_x_s%d_tab[%d], &out_even, &out_odd);"
                        "  // s=%d idx=%d group%d") % (s, group, s, idx, group)
            else:
                call = ("butterfly_x_tw_scale(mem[s][2*idx], mem[s][2*idx+1], twf, "
                        "tw_scale_x_s%d_tab[cur_mod_id][%d], &out_even, &out_odd, cur_mod_id);"
                        "  // s=%d idx=%d group%d") % (s, group, s, idx, group)
        rows = [
            T * 5 + "{ const int idx = %d;" % idx,
            T * 6 + "int j = idx >> shift; int k = idx & mask;",
            T * 6 + "int ind_even = (next_stride == 0) ? ( j << (shift+1) )",
            T * 7 + ": ( j << (shift+1) ) + (k & next_mask) * 2 + (k >> (shift-1));",
            T * 6 + "int ind_odd = ind_even + stride;",
            T * 6 + twdecl,
            T * 6 + call,
            T * 6 + "mem[s+1][ind_even] = out_even; mem[s+1][ind_odd] = out_odd; }",
        ]
        return NL.join(rows) + NL

    out = [
        T * 4 + "// [generated] X-stage BU-side butterfly dispatch: TFS identity skip + group-shared completion." + NL,
        T * 4 + "// Group 0 (scale==1) -> plain butterfly; non-identity groups share one completed_tw per group." + NL,
    ]
    for s in range(num_x_stage):
        if s == 0:
            out.append(T * 4 + "if (s == %d) {" % s + NL)
        else:
            out.append(T * 4 + "else if (s == %d) {" % s + NL)
        if _shared(s):
            out.append(_prologue(s))
        for idx in range(BU):
            out.append(_lane(s, idx))
        out.append(T * 4 + "}" + NL)
    return "".join(out)


def _gen_tw_gen_L_small(struct: dict, local_s: int) -> str:
    """Emit tw_gen_L_s<local_s> (small/direct L-stage TFG producer) BYTE-IDENTICALLY to the static
    template. The four stages differ only by table offset (1<<s)-1 and phase width s; s0 has no phase.
    Scaffold-only: the generated output stays four separate functions (no C++ consolidation). ``struct``
    is reserved for emitter-idiom consistency / a future approved 4-to-1 consolidation; not used for
    branching here."""
    offset = (1 << local_s) - 1
    is_single = struct.get("is_single", False)
    cmid = "" if is_single else "[(int)cur_mod_id]"
    if local_s == 0:
        decl_phase = ""
        idx = "tw_l_base_table[0]" + cmid
        inc_phase = ""
    else:
        decl_phase = "\tap_uint<%d> phase = 0;\n" % local_s
        idx = ("tw_l_base_table[%d + (int)phase]" % offset) + cmid
        inc_phase = "\t\tphase++;\n"
    if is_single:
        # Single-prime: tw_l_base_table is [15] (Task A); no cur_mod_id / poly_ctr cycling.
        return (
            "void tw_gen_L_s%d(const int stage, tapa::ostreams<Data, BU>& tw_L) {\n" % local_s
            + "#pragma HLS INLINE off\n"
            + decl_phase
            + "\tfor(;;) {\n"
            + "#pragma HLS PIPELINE II=1\n"
            + "\t\tData tw_val = " + idx + ";\n"
            + "\t\tfor (int j = 0; j < BU; ++j) {\n"
            + "#pragma HLS UNROLL\n"
            + "\t\t\ttw_L[j].write(tw_val);\n"
            + "\t\t}\n"
            + inc_phase
            + "\t}\n"
            + "}"
        )
    return (
        "void tw_gen_L_s%d(const int stage, tapa::ostreams<Data, BU>& tw_L) {\n" % local_s
        + "#pragma HLS INLINE off\n"
        + "\tModId cur_mod_id = 0;\n"
        + decl_phase
        + "\tap_uint<logDEPTH> poly_ctr = 0;\n"
        + "\tfor(;;) {\n"
        + "#pragma HLS PIPELINE II=1\n"
        + "\t\tData tw_val = " + idx + ";\n"
        + "\t\tfor (int j = 0; j < BU; ++j) {\n"
        + "#pragma HLS UNROLL\n"
        + "\t\t\ttw_L[j].write(tw_val);\n"
        + "\t\t}\n"
        + inc_phase
        + "\t\tpoly_ctr++;\n"
        + "\t\tif (poly_ctr == 0) {  // ap_uint<logDEPTH> wraps at DEPTH\n"
        + "\t\t\tcur_mod_id++;\n"
        + "\t\t\tif (cur_mod_id >= NUM_PRIMES) cur_mod_id = 0;\n"
        + "\t\t}\n"
        + "\t}\n"
        + "}"
    )


def _gen_ntt_core_lstage_chain(struct: dict) -> str:
    """Emit the ntt_core L-stage TAPA invoke chain (input -> l_stage_s0..s3 -> l_stage_s_ge4 ->
    x_stages_top). BYTE-IDENTICAL to the static template chain: the C-preprocessor #if guards
    (HOST_PREPERMUTE_INPUT / NUM_L_stage / NUM_L_STAGE_GE4) are reproduced verbatim, so the present
    stages are still selected at C-compile time exactly as before. ``struct`` selects ONLY the
    Phase 3.4 hybrid precompute boundary tail (B=4 default string is verbatim/byte-identical;
    B=5 inserts l_stage_s4 and switches the OTF tail to l_stage_s_ge5 / NUM_L_STAGE_GE5)."""
    _head = (
        "tapa::task()\n"
        "#if HOST_PREPERMUTE_INPUT\n"
        "\t\t.invoke<tapa::detach, BU>(input_stage, core_istreams, core_streams)\n"
        "#else\n"
        "\t\t.invoke<tapa::detach, BU>(input_mem_stage, core_istreams, core_streams)\n"
        "#endif\n"
        "\t\t.invoke<tapa::detach>(l_stage_s0, 0, core_streams, core_streams)\n"
        "\t\t.invoke<tapa::detach>(l_stage_s1, 1, core_streams, core_streams)\n"
        "#if NUM_L_stage > 2\n"
        "\t\t.invoke<tapa::detach>(l_stage_s2, 2, core_streams, core_streams)\n"
        "#endif\n"
        "#if NUM_L_stage > 3\n"
        "\t\t.invoke<tapa::detach>(l_stage_s3, 3, core_streams, core_streams)\n"
        "#endif\n"
    )
    # D2: tail built generically from the StagePlan (s0..s3 head is fixed; C-preprocessor #if guards
    # prune absent stages exactly as before). A plan with no OTF group emits the legacy GE4 tail
    # (historical small-n behavior). Byte-identical for PRECOMP=4 default and PRECOMP=5/db153f1.
    plan = struct.get("_stage_plan")
    tail = ""
    if plan is not None:
        for s in plan.precomp_stages():
            if s >= 4:
                tail += ("#if NUM_L_stage > %d\n"
                         "\t\t.invoke<tapa::detach>(l_stage_s%d, %d, core_streams, core_streams)\n"
                         "#endif\n" % (s, s, s))
    otf = plan.otf_groups()[0] if (plan is not None and plan.otf_groups()) else None
    ge_lo = otf.lo if otf is not None else 4
    tail += ("#if NUM_L_STAGE_GE%d > 0\n"
             "\t\t.invoke<tapa::detach, NUM_L_STAGE_GE%d>(l_stage_s_ge%d, tapa::seq(), core_streams, core_streams)\n"
             "#endif\n" % (ge_lo, ge_lo, ge_lo))
    return _head + tail + "\t\t.invoke<tapa::detach>(x_stages_top, core_streams, core_ostreams);"


def generate_work2_x_stage_cpp(struct: dict, modes: list, P: int) -> str:
    """
    Generate the full C++ string for Work 2 X-stage functions and x_stages_top.
    Returns a string that replaces [WORK2_X_STAGE_FUNCTIONS_HERE] and
    [WORK2_X_STAGE_TOP_BEGIN]..[WORK2_X_STAGE_TOP_END] in ntt.cpp.
    """
    num_x_stage = struct['logBU'] + 1

    # Build TFG+TFR function strings
    lines = [
        "// ---------------------------------------------------------------------------",
        "// Work 2: per-stage X-stage segmented TFG (base) + TFR (ratio completion)",
        "// ---------------------------------------------------------------------------",
        "",
    ]
    is_single = struct.get("is_single", False)
    base_ii = struct.get("_alias_tfg_base_recurrence_ii", P)  # Phase 3.3.3: decoupled base-recurrence II
    for s_x in range(num_x_stage):
        mode = modes[s_x]
        if mode == 'recurrence':
            lines.append(_gen_tw_gen_X_base_seg(s_x, P, is_single, base_ii=base_ii))
        else:
            lines.append(_gen_tw_gen_X_base_tab(s_x, P, is_single))  # base-table mode: no reduce; stays II=P
        lines.append(_gen_tw_complete_X_ratio(s_x, P, is_single))

    functions_str = "\n".join(lines)

    # Build x_stages_top wiring
    top_lines = [
        "void x_stages_top(tapa::istreams<Data2, BU>& input_streams,"
        " tapa::ostreams<Data2, BU>& output_streams){",
        "\ttapa::streams<Wide, num_x_stage, 2> tw_X_W(\"tw_X_W\");",
        "",
    ]
    # Declare per-stage base FIFOs
    for s_x in range(num_x_stage):
        top_lines.append(f'\ttapa::streams<Data, 1, 2> base_fifo_s{s_x}("base_fifo_s{s_x}");')
    top_lines.append("\ttapa::task()")
    # Invoke per-stage TFG+TFR
    for s_x in range(num_x_stage):
        mode = modes[s_x]
        gen_fn = f"tw_gen_X_base_seg_s{s_x}" if mode == 'recurrence' else f"tw_gen_X_base_tab_s{s_x}"
        cmp_fn = f"tw_complete_X_ratio_s{s_x}"
        top_lines.append(f"\t\t.invoke<tapa::detach>({gen_fn}, base_fifo_s{s_x})")
        top_lines.append(f"\t\t.invoke<tapa::detach>({cmp_fn}, base_fifo_s{s_x}, tw_X_W[{s_x}])")
    top_lines.append("\t\t.invoke<tapa::detach>(x_stages, input_streams, output_streams, tw_X_W);")
    top_lines.append("}")

    top_str = "\n".join(top_lines)

    return functions_str, top_str


# ---------------------------------------------------------------------------
# ntt_primes.h emission
# ---------------------------------------------------------------------------

def generate_seg_primes_header(case_dir: str, K: int, cfg: dict,
                                all_params: list, struct: dict) -> None:
    """Emit src/ntt_primes.h for segmented generator."""
    P = cfg["tfg_seg_len"]
    N = struct["N"]
    BU = struct["BU"]
    logN = struct["logN"]
    logBU = struct["logBU"]
    num_l_stage = struct["NUM_L_stage"]
    NP = len(all_params)
    LOGNP = max(1, math.ceil(math.log2(NP))) if NP > 1 else 1

    def fmt_arr(arr):
        return ", ".join(f"(Data)0x{int(x):016X}ULL" for x in arr)

    def fmt_val(v):
        return f"(Data)0x{int(v):016X}ULL"

    # ===== SINGLE-PRIME (len(primes)==1) dimension-reduced emission — Task A =====
    # Emits scalar datapath constants (MOD/BARRETT_MU/PRIME_CLASS) and prime-axis-
    # collapsed tables. MODS[1]/PSIS[1] are TEMPORARY host-compatibility shims only
    # (host.cpp still reads MODS[0]/PSIS[0]); the datapath must use the scalars.
    # The whole multi-prime path below is left untouched (byte-identical for NP>=2).
    if NP == 1:
        p0 = all_params[0]
        _pb_s = struct.get("_lstage_precompute_boundary", 4)
        _flat_n_s = 31 if _pb_s == 5 else 15
        num_ge4_s = max(0, num_l_stage - _pb_s)
        num_x_stage_s = logBU + 1
        DEPTH_s = N // (2 * BU)
        NUM_SEG_BLOCKS_s = DEPTH_s // P
        x_modes_s = determine_x_stage_modes(all_params, num_x_stage_s)

        s = [
            "#ifndef NTT_PRIMES_H", "#define NTT_PRIMES_H", "",
            f"// Auto-generated by generate_code_seg.py for 1 prime at K={K}",
            f"// Segmented TFG Phase 1, tfg_seg_len={P}",
            "// SINGLE_PRIME shape (len(primes)==1): scalar modular constants for the datapath.",
            "// MODS[1]/PSIS[1] below are HOST-COMPATIBILITY SHIMS ONLY (temporary, until host.cpp",
            "// is updated). The datapath must use MOD/BARRETT_MU/PRIME_CLASS, NOT MODS[0]/MODS[mod_id].",
            "",
            "#define NUM_PRIMES 1",
            "#define SINGLE_PRIME 1",
            "",
            "#define LOGNP 1",
            "using ModId = ap_uint<LOGNP>;",
            "",
            "// --- Datapath scalar modular constants (single-prime target shape) ---",
            f"static const Data MOD = (Data){p0['q']}ULL;",
            f"static const Data2 BARRETT_MU = (Data2)({p0['mu']}ULL);",
            f"static const int PRIME_CLASS = {p0['prime_class']};",
            "",
            "// --- Host-compatibility shims ONLY: temporary, until host.cpp is updated. ---",
            "// host.cpp reads MODS[0]/PSIS[0]; these size-1 arrays satisfy it. Do NOT use for datapath.",
            f"static const Data MODS[1] = {{(Data){p0['q']}ULL}};",
            f"static const HostData PSIS[1] = {{{p0['psi']}ULL}};",
            "",
        ]
        for name, dim2, key in [
            ("tw_l_base_lane0_tab", "NUM_L_stage",     "tw_l0"),
            ("tw_l_base_lane1_tab", "NUM_L_stage - 1", "tw_l1"),
            ("tw_l_base_lane2_tab", "NUM_L_stage - 2", "tw_l2"),
            ("tw_l_base_lane3_tab", "NUM_L_stage - 2", "tw_l3"),
        ]:
            s.append(f"static const Data {name}[{dim2}] = {{{fmt_arr(p0[key])}}};")
        s.append("")
        s.append(f"// Flattened stage-0..{_pb_s - 1} twiddle table: {_flat_n_s} entries (single-prime, prime axis dropped).")
        s.append(f"static const Data tw_l_base_table[{_flat_n_s}] = {{")
        for entry_idx in range(_flat_n_s):
            comma = "," if entry_idx < _flat_n_s - 1 else ""
            s.append(f"    {fmt_val(p0['tw_l_base_flat'][entry_idx])}{comma}  // entry {entry_idx}")
        s.append("};")
        s.append("")
        if num_ge4_s > 0:
            pb_str = ", ".join(str(v) for v in p0["seg_period_blocks"])
            s.append(f"static const int tw_l_period_blocks_tab[{num_ge4_s}] = {{{pb_str}}};")
            s.append("")
            s.append(f"static const Data tw_l_seg_base_table[{num_ge4_s}] = {{")
            for si in range(num_ge4_s):
                comma = "," if si < num_ge4_s - 1 else ""
                s.append(f"    {fmt_val(p0['seg_bases'][si])}{comma}")
            s.append("};")
            s.append("")
            s.append(f"static const Data tw_l_seg_step_table[{num_ge4_s}] = {{")
            for si in range(num_ge4_s):
                comma = "," if si < num_ge4_s - 1 else ""
                s.append(f"    {fmt_val(p0['seg_steps'][si])}{comma}")
            s.append("};")
            s.append("")
            total_ratio_entries_s = num_ge4_s * P
            for si in range(num_ge4_s):
                if len(p0["seg_ratios"][si]) != P:
                    raise ValueError(
                        f"L ratio emission mismatch (single): ge-stage {si}: "
                        f"{len(p0['seg_ratios'][si])} entries != tfg_seg_len={P}. NEEDS_USER_DECISION")
            s.append(f"static const Data tw_l_ratio_table[{total_ratio_entries_s}] = {{")
            for si in range(num_ge4_s):
                for phase in range(P):
                    entry = si * P + phase
                    comma = "," if entry < total_ratio_entries_s - 1 else ""
                    s.append(f"    {fmt_val(p0['seg_ratios'][si][phase])}{comma}  // s={si + _pb_s} phase={phase}")
            s.append("};")
            s.append("")
        s.append("// X-stage s1/s2 spatial tw_scale (single-prime, prime axis dropped).")
        s.append("#if NUM_X_STAGE > 1")
        s.append(f"static const Data tw_scale_x_s1_tab[2] = {{{fmt_arr(p0['tw_scale_s1'])}}};")
        s.append("#endif // NUM_X_STAGE > 1")
        s.append("")
        s.append("#if NUM_X_STAGE > 2")
        s.append(f"static const Data tw_scale_x_s2_tab[4] = {{{fmt_arr(p0['tw_scale_s2'])}}};")
        s.append("#endif // NUM_X_STAGE > 2")
        s.append("")
        s.append("#define USE_XSTAGE_TW_SCALE_BU_COMPLETION 1")
        if logBU >= 3:
            s.append("#if NUM_X_STAGE > 3")
            s.append(f"static const Data tw_scale_x_s3_tab[8] = {{{fmt_arr(p0['tw_scale_s3'])}}};")
            s.append("#endif // NUM_X_STAGE > 3")
        if logBU >= 4:
            s.append("#if NUM_X_STAGE > 4")
            s.append(f"static const Data tw_scale_x_s4_tab[16] = {{{fmt_arr(p0['tw_scale_s4'])}}};")
            s.append("#endif // NUM_X_STAGE > 4")
        if logBU >= 5:
            s.append("#if NUM_X_STAGE > 5")
            s.append(f"static const Data tw_scale_x_s5_tab[32] = {{{fmt_arr(p0['tw_scale_s5'])}}};")
            s.append("#endif // NUM_X_STAGE > 5")
        s.append("")
        s.append("// Work 2: X-stage per-stage segmented TFG tables (single-prime).")
        s.append(f"// NUM_SEG_BLOCKS = {NUM_SEG_BLOCKS_s} (= DEPTH/TFG_SEG_LEN, defined in ntt.h)")
        s.append("")
        for s_x in range(num_x_stage_s):
            mode = x_modes_s[s_x]
            s.append(f"// X-stage s{s_x}: mode={mode}")
            if mode == 'recurrence':
                s.append(f"static const Data tw_x_s{s_x}_seg_base = {fmt_val(p0['x_seg_bases'][s_x])};")
                s.append(f"static const Data tw_x_s{s_x}_seg_step = {fmt_val(p0['x_seg_steps'][s_x])};")
            else:
                s.append(f"static const Data tw_x_s{s_x}_base_table[{NUM_SEG_BLOCKS_s}] = {{{fmt_arr(p0['x_seg_base_tables'][s_x])}}};")
            if len(p0["x_seg_ratios"][s_x]) != P:
                raise ValueError(
                    f"X ratio emission mismatch (single): s_x={s_x}: "
                    f"{len(p0['x_seg_ratios'][s_x])} entries != tfg_seg_len={P}. NEEDS_USER_DECISION")
            s.append(f"static const Data tw_x_s{s_x}_seg_ratio[TFG_SEG_LEN] = {{{fmt_arr(p0['x_seg_ratios'][s_x])}}};")
            s.append("")
        s.extend(["", "#endif // NTT_PRIMES_H", ""])

        src_dir = os.path.join(case_dir, "src")
        with open(os.path.join(src_dir, "ntt_primes.h"), "w") as f:
            f.write("\n".join(s))
        return

    lines = [
        "#ifndef NTT_PRIMES_H", "#define NTT_PRIMES_H", "",
        f"// Auto-generated by generate_code_seg.py for {NP} primes at K={K}",
        f"// Segmented TFG Phase 1, tfg_seg_len={P}", "",
        f"#define NUM_PRIMES {NP}", "",
        f"#define LOGNP {LOGNP}",
        "using ModId = ap_uint<LOGNP>;", "",
    ]

    def scalar_block(name, typ, vals, compact=True):
        if compact and NP <= 32:
            s = ", ".join(vals)
            lines.append(f"static const {typ} {name}[NUM_PRIMES] = {{{s}}};")
        else:
            lines.append(f"static const {typ} {name}[NUM_PRIMES] = {{")
            for i, v in enumerate(vals):
                comma = "," if i < NP - 1 else ""
                lines.append(f"    {v}{comma}")
            lines.append("};")

    scalar_block("MODS", "Data",
                 [f"(Data){p['q']}ULL" for p in all_params])
    lines.append("")
    scalar_block("PSIS", "HostData",
                 [f"{p['psi']}ULL" for p in all_params])
    lines.append("")
    scalar_block("BARRETT_MUS", "Data2",
                 [f"(Data2)({p['mu']}ULL)" for p in all_params])
    lines.append("")
    scalar_block("PRIME_CLASS", "int",
                 [str(p["prime_class"]) for p in all_params])
    lines.append("")

    # ── Legacy tables still needed by L-stages 0..2 and all X-stages ──
    NUM_L_stage = struct["NUM_L_stage"]
    # L-stage legacy lane tables still needed by tw_gen_L_s0/s1/s2/s3 and
    # tw_gen_L_base_s_ge4 (which uses tw_l_base_lane1_tab).
    # X-stage legacy lane tables (tw_x_base_lane*_tab, tw_x_gamma_tab) removed:
    # Work 2 per-stage TFG modules use tw_x_s*_seg_* tables instead.
    for name, dim2, key in [
        ("tw_l_base_lane0_tab", "NUM_L_stage",     "tw_l0"),
        ("tw_l_base_lane1_tab", "NUM_L_stage - 1", "tw_l1"),
        ("tw_l_base_lane2_tab", "NUM_L_stage - 2", "tw_l2"),
        ("tw_l_base_lane3_tab", "NUM_L_stage - 2", "tw_l3"),
    ]:
        lines.append(f"static const Data {name}[NUM_PRIMES][{dim2}] = {{")
        for i, p in enumerate(all_params):
            comma = "," if i < NP - 1 else ""
            lines.append(f"    {{{fmt_arr(p[key])}}}{comma}")
        lines.append("};")
        lines.append("")

    # ── New: flattened stage-0..3 table, shape [15][NUM_PRIMES] ──
    lines.append("// Flattened stage-0..3 twiddle table: 15 entries x NUM_PRIMES.")
    _pb_m = struct.get("_lstage_precompute_boundary", 4)
    _flat_n_m = 31 if _pb_m == 5 else 15
    lines.append("// Offset(s) = (1<<s)-1; tw_l_base_table[offset+phase][mod_id] = T_s[phase].")
    lines.append(f"static const Data tw_l_base_table[{_flat_n_m}][NUM_PRIMES] = {{")
    for entry_idx in range(_flat_n_m):
        vals = [fmt_val(p["tw_l_base_flat"][entry_idx]) for p in all_params]
        comma = "," if entry_idx < _flat_n_m - 1 else ""
        lines.append(f"    {{{', '.join(vals)}}}{comma}  // entry {entry_idx}")
    lines.append("};")
    lines.append("")

    # ── New: segmented tables for stages >= boundary (4 default; 5 hybrid B=5) ──
    num_ge4 = max(0, num_l_stage - _pb_m)
    if num_ge4 > 0:
        # period_blocks: per-stage period-reset threshold for base recurrence.
        # period_blocks[s-4] = (1 << (s+0_based_s+4)) / TFG_SEG_LEN (same for all primes).
        period_blocks_vals = all_params[0]["seg_period_blocks"]
        pb_str = ", ".join(str(v) for v in period_blocks_vals)
        lines.append(f"// Blocks per twiddle period for each ge{_pb_m} stage (same for all primes).")
        lines.append(f"static const int tw_l_period_blocks_tab[{num_ge4}] = {{{pb_str}}};")
        lines.append("")

        lines.append(f"// Segmented base table [NUM_L_STAGE-{_pb_m}={num_ge4}][NUM_PRIMES].")
        lines.append(f"static const Data tw_l_seg_base_table[{num_ge4}][NUM_PRIMES] = {{")
        for si in range(num_ge4):
            vals = [fmt_val(p["seg_bases"][si]) for p in all_params]
            comma = "," if si < num_ge4 - 1 else ""
            lines.append(f"    {{{', '.join(vals)}}}{comma}")
        lines.append("};")
        lines.append("")

        lines.append(f"// Segmented step table [NUM_L_STAGE-{_pb_m}={num_ge4}][NUM_PRIMES].")
        lines.append(f"static const Data tw_l_seg_step_table[{num_ge4}][NUM_PRIMES] = {{")
        for si in range(num_ge4):
            vals = [fmt_val(p["seg_steps"][si]) for p in all_params]
            comma = "," if si < num_ge4 - 1 else ""
            lines.append(f"    {{{', '.join(vals)}}}{comma}")
        lines.append("};")
        lines.append("")

        total_ratio_entries = num_ge4 * P
        for p in all_params:
            for si in range(num_ge4):
                if len(p["seg_ratios"][si]) != P:
                    raise ValueError(
                        f"L ratio emission mismatch: q={p['q']} ge-stage {si}: "
                        f"{len(p['seg_ratios'][si])} entries != tfg_seg_len={P}. NEEDS_USER_DECISION")
        lines.append(f"// Segmented ratio table [(NUM_L_STAGE-{_pb_m})*TFG_SEG_LEN={total_ratio_entries}][NUM_PRIMES].")
        lines.append(f"// Index = (s-{_pb_m})*TFG_SEG_LEN + phase.")
        lines.append(f"static const Data tw_l_ratio_table[{total_ratio_entries}][NUM_PRIMES] = {{")
        for si in range(num_ge4):
            for phase in range(P):
                entry = si * P + phase
                vals = [fmt_val(p["seg_ratios"][si][phase]) for p in all_params]
                comma = "," if entry < total_ratio_entries - 1 else ""
                lines.append(f"    {{{', '.join(vals)}}}{comma}  // s={si + _pb_m} phase={phase}")
        lines.append("};")
        lines.append("")

    # ── X-stage s1/s2 tw_scale tables (always emitted; used by BU-side butterfly_x_tw_scale) ──
    lines.append("// X-stage s1/s2 spatial tw_scale (BU-side, reference-style).")
    lines.append("#if NUM_X_STAGE > 1")
    lines.append(f"static const Data tw_scale_x_s1_tab[NUM_PRIMES][2] = {{")
    for i, p in enumerate(all_params):
        comma = "," if i < NP - 1 else ""
        lines.append(f"    {{{fmt_arr(p['tw_scale_s1'])}}}{comma}")
    lines.append("};")
    lines.append("#endif // NUM_X_STAGE > 1")
    lines.append("")
    lines.append("#if NUM_X_STAGE > 2")
    lines.append(f"static const Data tw_scale_x_s2_tab[NUM_PRIMES][4] = {{")
    for i, p in enumerate(all_params):
        comma = "," if i < NP - 1 else ""
        lines.append(f"    {{{fmt_arr(p['tw_scale_s2'])}}}{comma}")
    lines.append("};")
    lines.append("#endif // NUM_X_STAGE > 2")
    lines.append("")

    # ── X-stage BU-side spatial completion: always enabled in segmented Phase 1. ──
    # The Work 2 TFG generates ONE twiddle per cycle (broadcast to all BU lanes);
    # butterfly_x_tw_scale applies the per-BU spatial correction for s>=1.
    # tw_scale_x_s1_tab and tw_scale_x_s2_tab are always emitted above.
    # Emit s3/s4 tables and the control define unconditionally.
    lines.append("#define USE_XSTAGE_TW_SCALE_BU_COMPLETION 1")
    if struct["logBU"] >= 3:
        lines.append("#if NUM_X_STAGE > 3")
        lines.append(f"static const Data tw_scale_x_s3_tab[NUM_PRIMES][8] = {{")
        for i, p in enumerate(all_params):
            comma = "," if i < NP - 1 else ""
            lines.append(f"    {{{fmt_arr(p['tw_scale_s3'])}}}{comma}")
        lines.append("};")
        lines.append("#endif // NUM_X_STAGE > 3")
    if struct["logBU"] >= 4:
        lines.append("#if NUM_X_STAGE > 4")
        lines.append(f"static const Data tw_scale_x_s4_tab[NUM_PRIMES][16] = {{")
        for i, p in enumerate(all_params):
            comma = "," if i < NP - 1 else ""
            lines.append(f"    {{{fmt_arr(p['tw_scale_s4'])}}}{comma}")
        lines.append("};")
        lines.append("#endif // NUM_X_STAGE > 4")
    if struct["logBU"] >= 5:
        lines.append("#if NUM_X_STAGE > 5")
        lines.append(f"static const Data tw_scale_x_s5_tab[NUM_PRIMES][32] = {{")
        for i, p in enumerate(all_params):
            comma = "," if i < NP - 1 else ""
            lines.append(f"    {{{fmt_arr(p['tw_scale_s5'])}}}{comma}")
        lines.append("};")
        lines.append("#endif // NUM_X_STAGE > 5")
    lines.append("")

    # ── Work 2: per-stage X-stage segmented TFG tables ──
    num_x_stage = logBU + 1
    DEPTH = N // (2 * BU)
    NUM_SEG_BLOCKS = DEPTH // P
    x_modes = determine_x_stage_modes(all_params, num_x_stage)

    lines.append("// Work 2: X-stage per-stage segmented TFG tables.")
    lines.append(f"// NUM_SEG_BLOCKS = {NUM_SEG_BLOCKS} (= DEPTH/TFG_SEG_LEN, defined in ntt.h)")
    lines.append("")

    for s_x in range(num_x_stage):
        mode = x_modes[s_x]
        lines.append(f"// X-stage s{s_x}: mode={mode}")
        if mode == 'recurrence':
            lines.append(f"static const Data tw_x_s{s_x}_seg_base[NUM_PRIMES] = {{")
            for i, p in enumerate(all_params):
                comma = "," if i < NP - 1 else ""
                lines.append(f"    {fmt_val(p['x_seg_bases'][s_x])}{comma}")
            lines.append("};")
            lines.append(f"static const Data tw_x_s{s_x}_seg_step[NUM_PRIMES] = {{")
            for i, p in enumerate(all_params):
                comma = "," if i < NP - 1 else ""
                lines.append(f"    {fmt_val(p['x_seg_steps'][s_x])}{comma}")
            lines.append("};")
        else:
            lines.append(f"static const Data tw_x_s{s_x}_base_table[NUM_PRIMES][{NUM_SEG_BLOCKS}] = {{")
            for i, p in enumerate(all_params):
                comma = "," if i < NP - 1 else ""
                lines.append(f"    {{{fmt_arr(p['x_seg_base_tables'][s_x])}}}{comma}")
            lines.append("};")
        for p in all_params:
            if len(p["x_seg_ratios"][s_x]) != P:
                raise ValueError(
                    f"X ratio emission mismatch: q={p['q']} s_x={s_x}: "
                    f"{len(p['x_seg_ratios'][s_x])} entries != tfg_seg_len={P}. NEEDS_USER_DECISION")
        lines.append(f"static const Data tw_x_s{s_x}_seg_ratio[NUM_PRIMES][TFG_SEG_LEN] = {{")
        for i, p in enumerate(all_params):
            comma = "," if i < NP - 1 else ""
            lines.append(f"    {{{fmt_arr(p['x_seg_ratios'][s_x])}}}{comma}")
        lines.append("};")
        lines.append("")

    lines.extend(["", "#endif // NTT_PRIMES_H", ""])

    src_dir = os.path.join(case_dir, "src")
    with open(os.path.join(src_dir, "ntt_primes.h"), "w") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# ntt.h generation (from segmented template)
# ---------------------------------------------------------------------------

def generate_seg_ntt_header(case_dir: str, K: int, cfg: dict,
                             struct: dict, all_params: list) -> None:
    P = cfg["tfg_seg_len"]
    shiftadd_mode = cfg.get("shiftadd", "none")

    if shiftadd_mode == "reduce":
        shiftadd_include_block = (
            "#define STREAMNTT_SHIFTADD_CONFIG 1\n"
            "#include \"ntt_shiftadd.h\"\n"
            "#define USE_REDUCE_SHIFTADD 1"
        )
    elif shiftadd_mode == "mul":
        shiftadd_include_block = (
            "#define STREAMNTT_SHIFTADD_CONFIG 1\n"
            "#include \"ntt_shiftadd.h\"\n"
            "#define USE_REDUCE_SHIFTADD_MUL 1"
        )
    else:
        shiftadd_include_block = (
            "// STREAMNTT_SHIFTADD_CONFIG == 0 (Barrett/Shoup baseline)\n"
            "#define STREAMNTT_SHIFTADD_CONFIG 0"
        )

    tpl_path = os.path.join(SEG_TEMPLATE_DIR, "ntt.h.template")
    with open(tpl_path) as f:
        template = f.read()

    replacements = {
        "{K}":             str(K),
        "{N}":             str(struct["N"]),
        "{logN}":          str(struct["logN"]),
        "{BU}":            str(struct["BU"]),
        "{logBU}":         str(struct["logBU"]),
        "{logDEPTH}":      str(struct["logDEPTH"]),
        "{NUM_L_stage}":   str(struct["NUM_L_stage"]),
        "{CH}":            str(struct["CH"]),
        "{MCH}":           str(struct["MCH"]),
        "{EffDataCHLen}":  str(struct["EffDataCHLen"]),
        "{GROUP_NUM}":     str(struct["GROUP_NUM"]),
        "{GROUP_CH_NUM}":  str(struct["GROUP_CH_NUM"]),
        "{TFG_SEG_LEN}":   str(P),
        "{SHIFTADD_INCLUDE_BLOCK}": shiftadd_include_block,
    }

    for k, v in replacements.items():
        template = template.replace(k, v)

    # Phase 3.4 B=5: insert NUM_L_STAGE_GE5 right after the GE4 define in the GENERATED header text
    # (generator-side; templates/ untouched; B=4 header bytes unchanged -> byte-identical default).
    if struct.get("_lstage_precompute_boundary", 4) == 5:
        _ge4_line = "#define NUM_L_STAGE_GE4 (%s - 4)\n" % struct["NUM_L_stage"]
        if _ge4_line not in template:
            raise ValueError(
                "B=5 header insertion anchor missing: expected %r in generated ntt.h" % _ge4_line)
        template = template.replace(
            _ge4_line,
            _ge4_line + "#define NUM_L_STAGE_GE5 (%s - 5)\n" % struct["NUM_L_stage"], 1)

    src_dir = os.path.join(case_dir, "src")
    with open(os.path.join(src_dir, "ntt.h"), "w") as f:
        f.write(template)


# ---------------------------------------------------------------------------
# Per-prime parameter computation
# ---------------------------------------------------------------------------

# _k_arith_for moved to reduce_generator.py (Phase 3.5 S1b; imported above) — the
# per-prime arithmetic class is reduce-family logic and classify_reduce_set owns it.


def compute_all_params(cfg: dict, struct: dict, q_values: list) -> list:
    """Compute per-prime tables for all primes."""
    K = struct["K"]
    N = struct["N"]
    BU = struct["BU"]
    logN = struct["logN"]
    logBU = struct["logBU"]
    num_l_stage = struct["NUM_L_stage"]
    P = cfg["tfg_seg_len"]
    all_params = []
    for q in q_values:
        K_arith = _k_arith_for(q)
        omega, psi = get_nth_root_of_unity_and_psi_fast(N, q)
        mu = (1 << (2 * K_arith)) // q
        # X-stage legacy lane tables: computed with the canonical 3-lane recurrence parameters.
        tw_l0, tw_l1, tw_l2, tw_l3, tw_l_gamma, \
            tw_x0, tw_x1, tw_x2, tw_x3, tw_x_gamma, delta = \
            twiddle_base_generator_lanes(q, psi, N, BU, logN, logBU, K_arith, 3)

        prime_class = 0 if K_arith == 52 else 1

        # ── Flattened precomputed-stage table (stage 3/(4 at B=5) via simulation) ──
        _pb = struct.get("_lstage_precompute_boundary", 4)
        tw_l_base_flat = compute_tw_l_base_table(
            q, psi, N, BU, logN, logBU, tw_l0, tw_l1, tw_l2, tw_l3, boundary=_pb)

        # ── Segmented tables for stages >= boundary (4 default; 5 hybrid B=5) ──
        if num_l_stage > _pb:
            seg_bases, seg_steps, seg_ratios, seg_period_blocks = \
                compute_seg_tables_and_check(
                    q, psi, N, BU, logN, logBU, P, num_l_stage,
                    tw_l0, tw_l1, tw_l2, tw_l3, start_stage=_pb)
        else:
            seg_bases, seg_steps, seg_ratios, seg_period_blocks = [], [], [], []

        # ── X-stage spatial tw_scale (s1/s2/s3/s4). Each X-stage s exists iff logBU >= s,
        #    mirrored by the C-side `#if NUM_X_STAGE > k` guards (primes header + ntt.cpp
        #    dispatch). Guard s1/s2 like s3/s4 below so BU=2 (num_x_stage=2) does not index
        #    the non-existent s2 entry of tw_x0 (len = 2*BU-1). ──
        tw_scale_s1, tw_scale_s2 = [], []
        s1s2_checks = []
        # s1: base_offset = (1<<1)-1 = 1, num_tw_base = 2  (X-stage s1 exists when logBU >= 1)
        if logBU >= 1:
            base_off_s1 = (1 << 1) - 1
            inv0_s1 = pow(int(tw_x0[base_off_s1]), -1, int(q))
            tw_scale_s1 = [(int(tw_x0[base_off_s1 + g]) * inv0_s1) % int(q)
                           for g in range(2)]
            assert tw_scale_s1[0] == 1, f"tw_scale_s1[0] must be 1 for q={q}"
            s1s2_checks.append((1, base_off_s1, tw_scale_s1))
        # s2: base_offset = (1<<2)-1 = 3, num_tw_base = 4  (X-stage s2 exists when logBU >= 2)
        if logBU >= 2:
            base_off_s2 = (1 << 2) - 1
            inv0_s2 = pow(int(tw_x0[base_off_s2]), -1, int(q))
            tw_scale_s2 = [(int(tw_x0[base_off_s2 + g]) * inv0_s2) % int(q)
                           for g in range(4)]
            assert tw_scale_s2[0] == 1, f"tw_scale_s2[0] must be 1 for q={q}"
            s1s2_checks.append((2, base_off_s2, tw_scale_s2))

        # Self-check: spatial factorization for the X-stages actually present.
        # For each group g: expected = tw_x0[base_off + g]; got = base * scale[g] mod q.
        for (s_check, base_off, scale_tab) in s1s2_checks:
            base_val = int(tw_x0[base_off])
            for g, scale_g in enumerate(scale_tab):
                expected = int(tw_x0[base_off + g])
                got = (base_val * scale_g) % int(q)
                if got != expected:
                    raise ValueError(
                        f"X-stage spatial factorization FAILED: "
                        f"q={q} s={s_check} g={g}: "
                        f"expected={expected} got={got}. "
                        f"NEEDS_USER_DECISION"
                    )

        tw_scale_s3, tw_scale_s4, tw_scale_s5 = [], [], []
        if logBU >= 3:
            base_off = (1 << 3) - 1
            inv0 = pow(int(tw_x0[base_off]), -1, int(q))
            tw_scale_s3 = [(int(tw_x0[base_off + g]) * inv0) % int(q)
                           for g in range(8)]
        if logBU >= 4:
            base_off = (1 << 4) - 1
            inv0 = pow(int(tw_x0[base_off]), -1, int(q))
            tw_scale_s4 = [(int(tw_x0[base_off + g]) * inv0) % int(q)
                           for g in range(16)]
        if logBU >= 5:
            base_off = (1 << 5) - 1
            inv0 = pow(int(tw_x0[base_off]), -1, int(q))
            tw_scale_s5 = [(int(tw_x0[base_off + g]) * inv0) % int(q)
                           for g in range(32)]

        # ── Work 2: X-stage per-stage segmented TFG tables ──
        num_x_stage = logBU + 1
        DEPTH = N // (2 * BU)
        x_seg_bases, x_seg_steps, x_seg_ratios, x_seg_base_tables = \
            compute_x_seg_tables_one_prime(
                int(q), P, DEPTH, num_x_stage,
                tw_x0, tw_x1, tw_x2, tw_l1, num_l_stage)

        all_params.append(dict(
            q=q, K_arith=K_arith, psi=psi, mu=mu, delta=delta,
            prime_class=prime_class,
            tw_l0=tw_l0, tw_l1=tw_l1, tw_l2=tw_l2, tw_l3=tw_l3,
            tw_l_gamma=tw_l_gamma,
            tw_x0=tw_x0, tw_x1=tw_x1, tw_x2=tw_x2, tw_x3=tw_x3,
            tw_x_gamma=tw_x_gamma,
            tw_l_base_flat=tw_l_base_flat,
            seg_bases=seg_bases,
            seg_steps=seg_steps,
            seg_ratios=seg_ratios,
            seg_period_blocks=seg_period_blocks,
            tw_scale_s1=tw_scale_s1,
            tw_scale_s2=tw_scale_s2,
            tw_scale_s3=tw_scale_s3,
            tw_scale_s4=tw_scale_s4,
            tw_scale_s5=tw_scale_s5,
            # Work 2 X-stage tables
            x_seg_bases=x_seg_bases,
            x_seg_steps=x_seg_steps,
            x_seg_ratios=x_seg_ratios,
            x_seg_base_tables=x_seg_base_tables,
        ))

    return all_params


# ---------------------------------------------------------------------------
# Case generation
# ---------------------------------------------------------------------------

def _format_shiftadd_route_report(recipe, route, final_k, class_mode):
    """Phase 3.5 S3e: STDOUT-ONLY route report from a ReduceRecipe. PURE -- returns a
    multi-line string; no file/source/behavior effect. Called ONLY for shiftadd routes
    (reduce/mul) after the S3d custom fail-fast (so every prime here is KNOWN-table, and
    t_match is a bool). (fold, correction) params render as grep-friendly "fd:corr"
    (e.g. 2:positive_only), never Python tuple repr; None renders "None"."""
    def _pp(params):
        return "None" if params is None else "%d:%s" % (params[0], params[1])
    aliases = ",".join(recipe.aliases)
    n_total = len(recipe.primes)
    n_t = sum(1 for f in recipe.primes if f.t_match is True)
    delta_needed = sum(1 for f in recipe.primes if f.t_match is False)
    lines = [
        "shiftadd-recipe: route=%s aliases=[%s] final_K=%s class_mode=%s provenance=%s" % (
            route, aliases, final_k, class_mode, recipe.provenance),
        "  anchor: B=%s anchor_k_group=%s K_spread=%s batch_legal=%s" % (
            recipe.b_anchor, recipe.anchor_k_group, recipe.k_spread,
            recipe.batch_legal_shiftadd),
        "  reduce: can_emit=%s default_ok=%s default=%s resolver=%s" % (
            recipe.can_emit_shiftadd_reduce, recipe.reduce_default_self_check_ok,
            _pp(recipe.reduce_default_self_check_params),
            _pp(recipe.reduce_resolver_params)),
        "  mul:    can_emit=%s corrections=%s t_match=%d/%d delta_needed=%d" % (
            recipe.can_emit_shiftadd_mul, recipe.mul_corrections, n_t, n_total, delta_needed),
    ]
    # K_group-mixed two-family batches: one per-class line (cls order; the
    # per-class resolver params are what the mixed emitter consumes).
    if getattr(recipe, "cls_recipes", None):
        for i, cr in enumerate(recipe.cls_recipes):
            lines.append(
                "  cls%d:   K_group=%d B=%d n=%d reduce=%s resolver=%s "
                "mul_ok=%s corrections=%s" % (
                    i, cr.k_class, cr.b_anchor, len(cr.aliases),
                    cr.reduce_resolver_emit_ok, _pp(cr.reduce_resolver_params),
                    cr.mul_ok, cr.mul_corrections))
    if recipe.fail_fast_reasons:
        lines.append("  fail_fast: %s" % "; ".join(recipe.fail_fast_reasons))
    return "\n".join(lines)


def generate_seg_case(cfg: dict, suffix=None, force=False):
    """Generate one segmented TFG case. Returns (case_name, case_dir)."""
    validate_config_seg(cfg)

    # Phase 3.5 S1b: K is DERIVED. final_K = max(config_K, K_group); "K" may be omitted
    # entirely. Classification (single / same_bit / mixed_bit; cls0 = the SMALLER class
    # PRESENT in the set) is metadata + stdout ONLY this slice — the emitted PRIME_CLASS
    # values, dual candidates and the reduce() block are untouched (S1c/S1d scope).
    _aliases_q = resolve_primes_with_aliases(cfg["primes"])
    q_values = [q for _alias, q in _aliases_q]
    _rc = classify_reduce_set(_aliases_q, cfg.get("K"))
    K = _rc.final_k
    # Phase 3.5 S1d: reduce cls-mode ENUM (internal _alias_reduce_cls_mode).
    # unset/empty/whitespace = legacy = byte-identical 9813216 output; "normalized"
    # enables one-class same-bit / single-prime Barrett emission (shiftadd routes
    # and mixed-bit stay byte-identical in BOTH modes). NOT in validate_config_seg.
    _cls_mode_env = os.environ.get("STREAMNTT_REDUCE_CLS_MODE")
    if _cls_mode_env is None or _cls_mode_env.strip() == "":
        _cls_mode = "legacy"
    elif _cls_mode_env.strip() in ("legacy", "normalized"):
        _cls_mode = _cls_mode_env.strip()
    else:
        raise ValueError(
            "STREAMNTT_REDUCE_CLS_MODE=%r is not valid. Use 'legacy' (default) "
            "or 'normalized'." % _cls_mode_env)
    # Phase 3.5 S4: shiftadd single-prime cls-mode ENUM. unset/empty/default = legacy =
    # byte-identical ef753da; "normalized" resolves the redundant single-prime is_<alias>/
    # mod_id shiftadd muxes at generation time (NP==1 only; multi-prime + Barrett unaffected;
    # passed to emit_shiftadd_header). NOT in validate_config_seg. NO default flip.
    _sa_sp_env = os.environ.get("STREAMNTT_SHIFTADD_SINGLE_PRIME_MODE")
    if _sa_sp_env is None or _sa_sp_env.strip() == "":
        _shiftadd_single_prime_mode = "legacy"
    elif _sa_sp_env.strip() in ("legacy", "normalized"):
        _shiftadd_single_prime_mode = _sa_sp_env.strip()
    else:
        raise ValueError(
            "STREAMNTT_SHIFTADD_SINGLE_PRIME_MODE=%r is not valid. Use 'legacy' "
            "(default) or 'normalized'." % _sa_sp_env)
    _cls_txt = "classes=[%s]" % ",".join(str(c) for c in _rc.class_map)
    if len(_rc.class_map) == 2:
        _cls_txt += " (cls0=%d cls1=%d)" % (_rc.class_map[0], _rc.class_map[1])
    print("reduce-class: mode=%s %s cls-mode=%s" % (_rc.mode, _cls_txt, _cls_mode))
    print("K: config=%s K_group=%d final=%d (%s)" % (
        "absent" if _rc.config_k is None else _rc.config_k,
        _rc.k_group, _rc.final_k, _rc.provenance))
    if _rc.provenance == "lifted":
        print("WARNING: config K=%d < K_group=%d -- envelope LIFTED to %d "
              "(under-spec guard; do not cite pre-lift outputs)" % (
                  _rc.config_k, _rc.k_group, _rc.final_k))
    # Phase 3.5 S5: STDOUT-ONLY one-line generation-modes summary (no generated-file carriage;
    # printed after reduce-class/K and before the detailed S3e shiftadd-recipe block). output_mode
    # is the DERIVED byte-effect verdict: a normalized env that is INERT for this config (cls
    # normalized off the Barrett single/same_bit path, or single-prime mode on NP>=2) reports
    # legacy-byte-identical. Exact mode names only (no final_K/K_group/anchor_k_group/b_anchor).
    _s5_route = cfg.get("shiftadd", "none")
    _s5_recipe = "active" if _s5_route in ("reduce", "mul") else "none"
    _s5_cls_norm_active = (_cls_mode == "normalized" and _s5_route == "none"
                           and _rc.mode in ("single", "same_bit"))
    _s5_sp_norm_active = (_shiftadd_single_prime_mode == "normalized"
                          and _s5_route in ("reduce", "mul") and len(_rc.per_prime) == 1)
    _s5_output_mode = ("normalized-source-cleanup"
                       if (_s5_cls_norm_active or _s5_sp_norm_active)
                       else "legacy-byte-identical")
    print("generation-modes: shiftadd_route=%s reduce_cls_mode=%s "
          "shiftadd_single_prime_mode=%s reduce_recipe=%s output_mode=%s default_flip=false"
          % (_s5_route, _cls_mode, _shiftadd_single_prime_mode, _s5_recipe, _s5_output_mode))
    struct = compute_structural(cfg, K)
    struct["NUM_L_stage"] = struct["NUM_L_stage"]  # alias already correct
    # S1b metadata carriage = struct + stdout ONLY (no generated-file fields: explicit-K
    # byte-identity and the no-K==explicit-K equivalence gates both forbid header carriage).
    struct["_reduce_config_k"] = _rc.config_k
    struct["_reduce_k_group"] = _rc.k_group
    struct["_reduce_final_k"] = _rc.final_k
    struct["_reduce_k_provenance"] = _rc.provenance
    struct["_reduce_class_mode"] = _rc.mode
    struct["_reduce_class_map"] = _rc.class_map
    struct["_alias_reduce_cls_mode"] = _cls_mode
    struct["_reduce_shiftadd_mode"] = cfg.get("shiftadd", "none")

    N = struct["N"]
    BU = struct["BU"]
    logN = struct["logN"]
    logBU = struct["logBU"]
    num_l_stage = struct["NUM_L_stage"]
    P = cfg["tfg_seg_len"]
    shiftadd_mode = cfg.get("shiftadd", "none")
    use_shiftadd_reduce = shiftadd_mode == "reduce"
    use_shiftadd_mul = shiftadd_mode == "mul"
    use_barrett = shiftadd_mode == "none"

    # Phase 3.5 S3d: shiftadd routes (reduce/mul) do NOT support custom/INT primes.
    # Build the ReduceRecipe EARLY (shiftadd routes only, per S3c OD-s3c-1) and
    # FAIL-FAST on unsupported_custom provenance BEFORE any case dir / file is
    # created (no partial output). Barrett/none stays recipe-free (recipe=None) and
    # unaffected. The SAME recipe instance is reused at emit_shiftadd_header below
    # (byte-identical for known primes). The gate is provenance == "unsupported_custom"
    # SPECIFICALLY (not "!= known_table") so the RESERVED "future_recognized" is never
    # blocked. Custom-first ordering: this precedes the K_spread fail-fast at emit, so
    # a set that is BOTH custom AND mixed-width reports the custom reason first.
    shiftadd_recipe = (
        build_reduce_recipe(_aliases_q, k_env=K, classify_info=_rc)
        if shiftadd_mode in ("reduce", "mul") else None)
    if shiftadd_recipe is not None:
        _custom_facts = [f for f in shiftadd_recipe.primes
                         if f.provenance == "unsupported_custom"]
        if _custom_facts:
            raise ValueError(
                "shiftadd=%s does not support custom/INT primes: %s. Custom/INT "
                "primes are not in KNOWN_PRIMES; the shift-add recipe is only "
                "validated for known prime families. Use a KNOWN_PRIMES alias, or "
                "set shiftadd=\"none\" (Barrett is universal)." % (
                    shiftadd_mode,
                    ", ".join("%s (q=%d)" % (f.alias, f.q) for f in _custom_facts)))
        # Phase 3.5 S3e: STDOUT-ONLY route report (shiftadd routes only; custom already
        # raised above; none/Barrett is recipe-free -> never enters this block). A known
        # mixed (batch_legal=False) set prints here THEN the emit-time K_spread fail-fast
        # still fires. No generated source / no behavior change.
        print(_format_shiftadd_route_report(
            shiftadd_recipe, shiftadd_mode, K, _rc.mode))

    DEPTH = N // (2 * BU)

    # Validate P divides DEPTH
    if DEPTH % P != 0:
        raise ValueError(
            f"tfg_seg_len={P} does not divide DEPTH={DEPTH}. "
            "Choose P that divides DEPTH."
        )

    # Build output directory
    case_name = build_case_name(cfg, "multi", suffix, k_value=K)
    case_dir = os.path.join(ROOT, "generated", case_name)
    print(f"Output: {case_dir}")

    if os.path.exists(case_dir):
        if not force:
            raise RuntimeError(
                f"Directory '{case_dir}' already exists. Use --force."
            )
        shutil.rmtree(case_dir)

    os.makedirs(os.path.join(case_dir, "src"))

    # Compute all per-prime params (includes self-check)
    # Phase 3.4: internal/experimental hybrid precompute-boundary alias. NOT a public config field
    # and NOT in validate_config_seg. Unset/empty/whitespace -> B=4 (current shape; byte-identical).
    # B=5 = experimental precomputed s0..s4 + OTF ge5 (structural prerequisite for clean future P=32;
    # P=16 ONLY this slice). Read BEFORE compute_all_params (tables depend on B).
    # Phase 3.4 D4 (first slice): Layer-3 stage-schedule env. Grammar "kind:lo-hi;..." (precomp|otf;
    # "otf:N-" open suffix) — DISTINCT from the ICBU recipe-schedule grammar. Boundary-equivalent plans
    # ONLY this slice (== PRECOMP=4/5). Mutually exclusive with the boundary env; empty/whitespace = unset.
    _pb_env = os.environ.get("STREAMNTT_LSTAGE_PRECOMPUTE_BOUNDARY")
    _lsched_env = os.environ.get("STREAMNTT_LSTAGE_PRECOMPUTE_SCHEDULE")
    _pb_set = _pb_env is not None and _pb_env.strip() != ""
    _lsched_set = _lsched_env is not None and _lsched_env.strip() != ""
    if _pb_set and _lsched_set:
        raise ValueError(
            "STREAMNTT_LSTAGE_PRECOMPUTE_SCHEDULE and STREAMNTT_LSTAGE_PRECOMPUTE_BOUNDARY cannot both be "
            "set (no precedence; the boundary is the uniform-schedule shortcut). Set only one.")
    _lsched_plan = plan_from_schedule(_lsched_env, struct["NUM_L_stage"]) if _lsched_set else None
    if _lsched_plan is not None:
        struct["_alias_lstage_precompute_schedule"] = _lsched_env.strip()
        _pb = _lsched_plan.precomp_boundary()
    elif not _pb_set:
        _pb = 4
    else:
        try:
            _pb = int(_pb_env)
        except ValueError:
            raise ValueError(
                "STREAMNTT_LSTAGE_PRECOMPUTE_BOUNDARY must be 4 or 5, got %r" % _pb_env)
    if _pb not in (4, 5):
        raise ValueError(
            "STREAMNTT_LSTAGE_PRECOMPUTE_BOUNDARY must be 4 or 5, got %d "
            "(PRECOMP>5 not implemented; future Layer-3 planner)" % _pb)
    if _pb == 5:
        if P not in (16, 32):
            raise ValueError(
                "STREAMNTT_LSTAGE_PRECOMPUTE_BOUNDARY=5 supports tfg_seg_len in {16,32} "
                "(got tfg_seg_len=%d)" % P)
        if struct["NUM_L_stage"] < 6:
            raise ValueError(
                "STREAMNTT_LSTAGE_PRECOMPUTE_BOUNDARY=5 needs num_l_stage >= 6 (>=1 OTF ge5 stage); "
                "got num_l_stage=%d" % struct["NUM_L_stage"])
    # Phase 3.4 D3: REAL P=32 pairing. P=32 is legal ONLY with PRECOMP=5 exactly (s0..s4 precomputed ->
    # every OTF stage s>=5 has period >= 32 = P). PRECOMP=4 would hit s4 period 16 < P (no period<P
    # handling planned); PRECOMP>5 is not implemented (future Layer-3 planner; rejected above).
    if P == 32 and _pb != 5:
        raise ValueError(
            "tfg_seg_len=32 requires STREAMNTT_LSTAGE_PRECOMPUTE_BOUNDARY=5 (PRECOMP=5 exactly: "
            "s0..s4 precomputed + OTF ge5). PRECOMP=4 cannot host P=32 (s4 period 16 < P); "
            "PRECOMP>5 not implemented (future Layer-3 planner).")
    struct["_lstage_precompute_boundary"] = _pb
    # D2: the StagePlan is the grouping single source of truth (emission maps + chain + C6 derive
    # from it). The scalar above remains the table-LAYOUT knob (flat-table size, seg start stage).
    # D4: a schedule-built plan (boundary-equivalent by construction this slice) is used directly.
    struct["_stage_plan"] = (_lsched_plan if _lsched_plan is not None
                             else plan_from_precomp_boundary(struct["NUM_L_stage"], _pb))
    # D3: P=32 evidence classification — STDOUT/REPORT-ONLY (generated artifacts unchanged by this).
    # DEGENERATE cases (e.g. single-s5 ge5 @ N1024/BU8) still generate, but are functional smoke only
    # and are NEVER recurrence/timing evidence (the 877769b lesson). NON-DEGENERATE needs >=2 OTF stages.
    if P == 32:
        _ev = classify_p32_evidence(struct["_stage_plan"], P)
        print("P32 evidence classification: %s — %s" % (_ev.classification, _ev.detail))

    print(f"Computing twiddle tables for {len(q_values)} primes, P={P} ...")
    all_params = compute_all_params(cfg, struct, q_values)
    print("  Self-check PASSED for all (prime, stage, block, phase) tuples.")
    # Task C: single-prime is a generation shape (len(primes)==1); emitters drop cur_mod_id.
    struct["is_single"] = (len(all_params) == 1)

    # Phase 3.3.1: internal/experimental ICBU write-limit delay knob. NOT a public config field
    # and NOT in validate_config_seg. Unset -> 1; valid {1,2,3,4}; invalid -> fail-fast.
    _wld_env = os.environ.get("STREAMNTT_ICBU_WRITE_LIMIT_DELAY")
    if _wld_env is None:
        struct["_alias_icbu_write_limit_delay"] = 1
    else:
        try:
            _wld = int(_wld_env)
        except ValueError:
            raise ValueError(
                "STREAMNTT_ICBU_WRITE_LIMIT_DELAY must be an integer in {1,2,3,4}, "
                f"got {_wld_env!r}"
            )
        if _wld not in (1, 2, 3, 4):
            raise ValueError(
                f"STREAMNTT_ICBU_WRITE_LIMIT_DELAY must be in {{1,2,3,4}}, got {_wld}"
            )
        struct["_alias_icbu_write_limit_delay"] = _wld

    # Phase 3.3.3: internal/experimental TFG base-recurrence pipeline-II knob. NOT a public config field
    # and NOT in validate_config_seg. Unset -> P (= tfg_seg_len; byte-identical); experimental {P,24,32}.
    # Decouples the loop-carried base recurrence II (tw_gen_L_base_s_ge4 / tw_gen_X_base_seg_s*) from
    # TFG_SEG_LEN so the reduce gets more pipeline slack; segmentation math/tables/counters unchanged.
    _brii_env = os.environ.get("STREAMNTT_TFG_BASE_RECURRENCE_II")
    if _brii_env is not None and P == 32:
        # Frozen knob (pragma-only; INVALID as P evidence). Real P=32 uses the DEFAULT II=P=32 pacing;
        # decoupling II from P is exactly the invalid substitution -> reject the combination outright.
        raise ValueError(
            "STREAMNTT_TFG_BASE_RECURRENCE_II cannot be combined with tfg_seg_len=32 "
            "(frozen pragma-only knob, not a P substitute; unset it -> II=P=32).")
    if _brii_env is None:
        struct["_alias_tfg_base_recurrence_ii"] = P
    else:
        try:
            _brii = int(_brii_env)
        except ValueError:
            raise ValueError(
                "STREAMNTT_TFG_BASE_RECURRENCE_II must be an integer in {%d,24,32}, got %r"
                % (P, _brii_env)
            )
        if _brii not in (P, 24, 32):
            raise ValueError(
                "STREAMNTT_TFG_BASE_RECURRENCE_II must be in {%d,24,32}, got %d" % (P, _brii)
            )
        struct["_alias_tfg_base_recurrence_ii"] = _brii
    base_ii = struct["_alias_tfg_base_recurrence_ii"]

    # Phase 3.3.3 Step C / Phase 3.3.4: bf_unit storage recipe via TWO mutually-exclusive interfaces:
    #   (a) GLOBAL scalar shortcut STREAMNTT_BF_UNIT_STORAGE_RECIPE {2U2B,4U,4B,4L,3U1B} (d0c3a16), and
    #   (b) Layer-2 schedule STREAMNTT_ICBU_RECIPE_SCHEDULE (Phase 3.3.4; per-stage-group recipes).
    # Empty / whitespace-only env values are treated as UNSET (-> byte-identical default/scalar path).
    # Both interfaces set -> fail-fast (DP5; no precedence override). Neither -> default 2U2B (byte-
    # identical). First slice realizes R-UNIFORM storage only: partition tokens parse + validate but only
    # `none` is emitted; heterogeneous schedules parse OK but are rejected here. NOT in validate_config_seg.
    # Paper-A DSE API slice adds the config-facing twin icbu.storage_recipe (validated in
    # validate_config_seg; see _ICBU_ALLOWED_SUBFIELDS). PRIORITY: a set env knob (scalar OR
    # schedule) supersedes the config field (env = override / backward-compatible path; one stdout
    # note); "auto" pins nothing (DSE-eligible under apply modes only; default 2U2B otherwise);
    # an absent icbu block resolves exactly as before (byte-identical legacy path).
    _icbu_cfg = cfg.get("icbu") or {}
    _cfg_recipe = _icbu_cfg.get("storage_recipe")
    _cfg_recipe_concrete = _cfg_recipe if _cfg_recipe not in (None, "auto") else None
    _bsr_env = os.environ.get("STREAMNTT_BF_UNIT_STORAGE_RECIPE")
    _sched_env = os.environ.get("STREAMNTT_ICBU_RECIPE_SCHEDULE")
    _bsr_set = _bsr_env is not None and _bsr_env.strip() != ""
    _sched_set = _sched_env is not None and _sched_env.strip() != ""
    if _bsr_set and _sched_set:
        raise ValueError(
            "STREAMNTT_ICBU_RECIPE_SCHEDULE and STREAMNTT_BF_UNIT_STORAGE_RECIPE cannot both be set "
            "(no precedence override; the scalar is a uniform-schedule shortcut). Set only one."
        )
    if _cfg_recipe is not None and (_bsr_set or _sched_set):
        print("icbu-config: storage_recipe=%s superseded by env %s"
              % (_cfg_recipe,
                 "STREAMNTT_BF_UNIT_STORAGE_RECIPE=%s" % _bsr_env if _bsr_set
                 else "STREAMNTT_ICBU_RECIPE_SCHEDULE"))
    if _sched_set:
        _depth = struct["N"] // (2 * struct["BU"])
        _resolved = parse_icbu_recipe_schedule(_sched_env, struct["NUM_L_stage"], _depth,
                                               groups=struct["_stage_plan"].lstage_groups())
        struct["_alias_icbu_recipe_schedule"] = _resolved.normalized
        if not _resolved.uniform:
            raise ValueError("heterogeneous ICBU recipe schedule not yet implemented")
        if _resolved.uniform_partition != "none":
            raise ValueError("partition emission not yet implemented")
        struct["_icbu_recipe_schedule_resolved"] = _resolved
    else:
        if not _bsr_set:
            # Config value (already validated; "auto"/absent -> 2U2B default) -- env unset.
            struct["_alias_bf_unit_storage_recipe"] = _cfg_recipe_concrete or "2U2B"
        else:
            if _bsr_env not in BF_UNIT_STORAGE_RECIPES:
                raise ValueError(
                    "STREAMNTT_BF_UNIT_STORAGE_RECIPE must be one of %s, got %r"
                    % (list(BF_UNIT_STORAGE_RECIPES), _bsr_env)
                )
            struct["_alias_bf_unit_storage_recipe"] = _bsr_env
        struct["_icbu_recipe_schedule_resolved"] = None
    # A recipe counts as USER-PINNED when set via the env scalar OR a concrete (non-"auto")
    # config value: plain `apply` keeps a pinned recipe; apply_override may override scalars
    # (a schedule is handled separately and is never overridden).
    _user_recipe_pinned = _bsr_set or (_cfg_recipe_concrete is not None and not _sched_set)

    # Phase 3.3.4 follow-up + DSE-apply slice: ICBU bind_storage minor-DSE.
    # Modes (STREAMNTT_ICBU_BIND_STORAGE_DSE_MODE):
    #   off            : no DSE computation at all (silent kill-switch).
    #   report (DEFAULT): print the resource/guard/recommendation table ONLY; artifacts byte-identical.
    #   recommend      : report + one greppable "icbu-dse:" decision line + dse_decision.txt; NO emission change.
    #   apply          : additionally ADOPT a strong D2 recommendation into
    #                    struct["_alias_bf_unit_storage_recipe"] BEFORE pragma emission, ONLY when the user
    #                    set no explicit recipe (STREAMNTT_BF_UNIT_STORAGE_RECIPE) and no schedule
    #                    (STREAMNTT_ICBU_RECIPE_SCHEDULE). Explicit user recipe always wins under `apply`.
    #   apply_override : like apply, but a strong D2 recommendation overrides an explicit SCALAR recipe.
    #                    An explicit recipe SCHEDULE is NEVER overridden (precision instrument) -> fail-fast.
    #   No strong recommendation (D0 uncalibrated / D1 within guard): apply modes never guess.
    #   Configured over guard with NO calibrated candidate (D3) or background alone over (D4): apply modes
    #   FAIL-FAST (never silently emit a recipe known to bust the guard); report/recommend only print.
    # DEFAULT stays "report" ON PURPOSE: apply outcomes depend on LOCAL calibration manifests
    # (Checkpoint/dse_calibration, gitignored), so an auto/apply default would make generated source
    # machine-dependent and break the byte-identity anchors. Hidden/experimental env; NOT in
    # validate_config_seg. STREAMNTT_ICBU_DSE_CALIBRATION_ROOT (optional) points the calibration
    # lookup at an alternate manifest root (tests / users with manifests elsewhere).
    # Paper-A DSE API slice: effective mode = env STREAMNTT_ICBU_BIND_STORAGE_DSE_MODE (if set)
    # > icbu.dse_mode (config; pre-validated) > "report" default. _DSE_MODES is module-level now
    # (shared with validate_config_seg).
    _dse_mode_env = os.environ.get("STREAMNTT_ICBU_BIND_STORAGE_DSE_MODE")
    _dse_mode_env_set = _dse_mode_env is not None and _dse_mode_env.strip() != ""
    _cfg_dse_mode = _icbu_cfg.get("dse_mode")
    if _cfg_dse_mode is not None and _dse_mode_env_set:
        print("icbu-config: dse_mode=%s superseded by env STREAMNTT_ICBU_BIND_STORAGE_DSE_MODE=%s"
              % (_cfg_dse_mode, _dse_mode_env.strip()))
    _dse_mode = ((_dse_mode_env if _dse_mode_env_set else None)
                 or _cfg_dse_mode or "report").strip().lower()
    if _dse_mode not in _DSE_MODES:
        raise ValueError("STREAMNTT_ICBU_BIND_STORAGE_DSE_MODE must be one of %s, got %r"
                         % ("/".join(_DSE_MODES), _dse_mode))
    if _dse_mode != "off":
        # Lazy ON PURPOSE (kept lazy across the package split): `off` stays import-free and
        # there is no import cycle (dse.icbu_dse imports code_generator.icbu_generator).
        from dse import icbu_dse
        _dse_guard_env = os.environ.get("STREAMNTT_ICBU_BIND_STORAGE_DSE_GUARD")
        if _dse_guard_env is None:
            _dse_guard = icbu_dse.DEFAULT_GUARD
        else:
            try:
                _dse_guard = float(_dse_guard_env)
            except ValueError:
                raise ValueError(
                    "STREAMNTT_ICBU_BIND_STORAGE_DSE_GUARD must be a float in (0,1], got %r" % _dse_guard_env)
            if not (0.0 < _dse_guard <= 1.0):
                raise ValueError(
                    "STREAMNTT_ICBU_BIND_STORAGE_DSE_GUARD must be in (0,1], got %r" % _dse_guard_env)
        _dse_root_env = os.environ.get("STREAMNTT_ICBU_DSE_CALIBRATION_ROOT")
        _dse_root = (_dse_root_env.strip() if (_dse_root_env is not None and _dse_root_env.strip() != "")
                     else icbu_dse.CALIBRATION_ROOT)
        # Paper-A DSE API slice: device selection (env STREAMNTT_ICBU_DSE_DEVICE > icbu.device >
        # "u55c" default). The calibration manifests / BRAM packing / CALIB rows are u55c-flow
        # evidence: for ANY other device the manifest lookup is SKIPPED (background=None ->
        # recommendations stay informational/D0, never a pretend-calibrated strong one; the report
        # labels the device CAPACITY-ONLY and u55c packing as explicit fallback). The u55c default
        # path is byte-identical (board dict == legacy BOARD_U55C; same report strings).
        _dev_env = os.environ.get("STREAMNTT_ICBU_DSE_DEVICE")
        _dev_env_set = _dev_env is not None and _dev_env.strip() != ""
        _cfg_dev = _icbu_cfg.get("device")
        if _cfg_dev is not None and _dev_env_set:
            print("icbu-config: device=%s superseded by env STREAMNTT_ICBU_DSE_DEVICE=%s"
                  % (_cfg_dev, _dev_env.strip()))
        _dse_device = get_device_profile(
            (_dev_env if _dev_env_set else None) or _cfg_dev or "u55c")
        _dse_board = _dse_device.as_board()
        _dse_nc = struct["CH"] * struct["EffDataCHLen"] // (2 * struct["BU"])
        _dse_arch = icbu_dse.arch_quantities(
            N=struct["N"], BU=struct["BU"], CH=struct["CH"], K=K, NUM_CORE=max(1, _dse_nc))
        _dse_configured = struct.get("_alias_bf_unit_storage_recipe", "2U2B")
        if _dse_configured not in icbu_dse.RECIPE_COUNTFORM:
            _dse_configured = "2U2B"
        _dse_calib = (icbu_dse.lookup_calibration(_dse_arch, root=_dse_root)
                      if _dse_device.name == "u55c" else None)
        print(icbu_dse.format_report(_dse_arch, _dse_calib, _dse_board, _dse_guard,
                                     _dse_configured, device=_dse_device.name))
        if _dse_mode in ("recommend", "apply", "apply_override"):
            _dse_bg = _dse_calib["background"] if _dse_calib else None
            _dse_rec = icbu_dse.recommend(_dse_arch, _dse_bg, _dse_board, _dse_guard,
                                          _dse_configured)
            _dse_action = "none"
            _dse_note = _dse_rec.reason
            if _dse_mode in ("apply", "apply_override"):
                if _sched_set:
                    if _dse_mode == "apply_override":
                        raise ValueError(
                            "STREAMNTT_ICBU_BIND_STORAGE_DSE_MODE=apply_override cannot override an "
                            "explicit STREAMNTT_ICBU_RECIPE_SCHEDULE (a recipe schedule is a precise "
                            "instrument and is never DSE-overridden); unset one of them")
                    _dse_note = "explicit recipe schedule set; apply skipped (schedule wins)"
                elif _dse_rec.best is not None:
                    if _user_recipe_pinned and _dse_mode == "apply":
                        if _bsr_set:
                            _dse_note = ("explicit STREAMNTT_BF_UNIT_STORAGE_RECIPE=%s set; apply keeps the "
                                         "user recipe (use apply_override to let DSE override)"
                                         % _dse_configured)
                        else:
                            _dse_note = ("explicit icbu.storage_recipe=%s set (config); apply keeps the "
                                         "user recipe (use apply_override to let DSE override)"
                                         % _dse_configured)
                    else:
                        if _dse_rec.best not in BF_UNIT_STORAGE_RECIPES:
                            raise ValueError(
                                "ICBU DSE recommended recipe %r is not an emittable bf_unit storage "
                                "recipe %s" % (_dse_rec.best, list(BF_UNIT_STORAGE_RECIPES)))
                        struct["_alias_bf_unit_storage_recipe"] = _dse_rec.best
                        _dse_action = "applied"
                elif _dse_rec.decision in ("D3", "D4"):
                    raise ValueError(
                        "ICBU bind_storage DSE %s: configured recipe %r does not satisfy the guard and "
                        "no calibrated recipe does (decision %s: %s). Refusing to silently emit a recipe "
                        "known to bust the guard; use mode=report, relax "
                        "STREAMNTT_ICBU_BIND_STORAGE_DSE_GUARD, or set a recipe explicitly."
                        % (_dse_mode, _dse_configured, _dse_rec.decision, _dse_rec.reason))
                # else D0/D1: nothing strong to adopt and nothing known-broken; never guess.
            _dse_emitted = ("(schedule)" if _sched_set
                            else struct.get("_alias_bf_unit_storage_recipe", "2U2B"))
            if _sched_set:
                _dse_user_recipe = "(schedule: %s)" % (_sched_env or "").strip()
            elif _bsr_set:
                _dse_user_recipe = _bsr_env
            elif _cfg_recipe_concrete is not None:
                _dse_user_recipe = "%s (config)" % _cfg_recipe_concrete
            elif _cfg_recipe == "auto":
                _dse_user_recipe = "(auto)"
            else:
                _dse_user_recipe = "(default)"
            print("icbu-dse: mode=%s decision=%s best=%s configured=%s action=%s emitted=%s"
                  % (_dse_mode, _dse_rec.decision or "n/a", _dse_rec.best or "none",
                     _dse_configured, _dse_action, _dse_emitted))
            with open(os.path.join(case_dir, "dse_decision.txt"), "w") as _dse_f:
                _dse_f.write(
                    "=== ICBU bind_storage DSE decision ===\n"
                    "mode: %s\n"
                    "guard: %.2f\n"
                    "user_recipe: %s\n"
                    "configured: %s\n"
                    "decision: %s\n"
                    "best: %s\n"
                    "action: %s\n"
                    "emitted: %s\n"
                    "source_emission_changed: %s\n"
                    "reason: %s\n"
                    % (_dse_mode, _dse_guard, _dse_user_recipe, _dse_configured,
                       _dse_rec.decision or "n/a", _dse_rec.best or "none", _dse_action,
                       _dse_emitted, "yes" if _dse_action == "applied" else "no", _dse_note))

    # Copy segmented ntt.cpp with Work 2 X-stage injection
    ntt_cpp_path = os.path.join(SEG_TEMPLATE_DIR, "ntt.cpp")
    with open(ntt_cpp_path) as f:
        ntt_src = f.read()
    # Phase 3.5 S1a-0: inject the reduce() block through the [reduce] seam — VERBATIM emitter
    # (byte-identical; gen_reduce_block ignores struct this slice). Markers-inclusive
    # replacement: generated output carries NO markers.
    for _mk in REDUCE_MARKERS:
        _gb = "// [%s:begin]" % _mk
        _ge = "// [%s:end]" % _mk
        _gi = ntt_src.index(_gb)
        _gj = ntt_src.index(_ge) + len(_ge)
        ntt_src = ntt_src[:_gi] + gen_reduce_block(struct) + ntt_src[_gj:]
    # Inject the L-stage TFG-L / TFR-L producers (D2: StagePlan-driven PRODUCTION maps; the s>=4
    # singleton producers ride the base-producer SITE so declarations stay use-before-def). Injected
    # BEFORE the II=TFG_SEG_LEN -> II=P replace below so the emitted base producer's pragma is
    # substituted uniformly with the rest of the template.
    _plan = struct["_stage_plan"]
    for _mk, _emit in region1_emission_map(_plan, struct, _gen_tw_gen_L_small):
        _gb = "// [%s:begin]" % _mk
        _ge = "// [%s:end]" % _mk
        _gi = ntt_src.index(_gb)
        _gj = ntt_src.index(_ge) + len(_ge)
        ntt_src = ntt_src[:_gi] + _emit + ntt_src[_gj:]
    # Region-2 L-stage wrappers (bf shells + l_stage task shells) from the SAME StagePlan map; the
    # shared bf_unit child body + write_limit / mem bind_storage pragmas stay static template code.
    _r2map = region2_emission_map(_plan)
    for _mk in REGION2_WRAPPER_MARKERS:
        _gb = "// [%s:begin]" % _mk
        _ge = "// [%s:end]" % _mk
        _gi = ntt_src.index(_gb)
        _gj = ntt_src.index(_ge) + len(_ge)
        ntt_src = ntt_src[:_gi] + _r2map[_mk] + ntt_src[_gj:]
    # Phase 3.3.1: inject ICBU write-limit delay micro-regions (decl / write_safe / shift) — global
    # knob struct["_alias_icbu_write_limit_delay"]; delay=1 emits the verbatim current bytes
    # (byte-identical), delay>=2 emits the named-scalar delay chain. bf_unit child otherwise static.
    for _mk in ICBU_WRITE_LIMIT_MARKERS:
        _gb = "// [%s:begin]" % _mk
        _ge = "// [%s:end]" % _mk
        _gi = ntt_src.index(_gb)
        _gj = ntt_src.index(_ge) + len(_ge)
        ntt_src = ntt_src[:_gi] + gen_icbu_write_limit(_mk, struct) + ntt_src[_gj:]
    # Phase 3.3.3: inject the bf_unit ping-pong storage block (mem0..3 decl + bind_storage) —
    # generator-owned storage shell; emits the default 2U2B recipe byte-identically. storage-recipe /
    # array_partition knobs come later (Phase 3.3.4). bf_unit compute / loop / write_limit untouched.
    for _mk in BF_UNIT_STORAGE_MARKERS:
        _gb = "// [%s:begin]" % _mk
        _ge = "// [%s:end]" % _mk
        _gi = ntt_src.index(_gb)
        _gj = ntt_src.index(_ge) + len(_ge)
        ntt_src = ntt_src[:_gi] + gen_bf_unit_storage(struct) + ntt_src[_gj:]
    # Substitute TFG_SEG_LEN pragma (HLS cannot expand macros in pragmas). Phase 3.3.3: the L-base
    # recurrence II is driven by base_ii (= tfg_seg_len by default -> byte-identical; {P,24,32} experimental).
    # This II=TFG_SEG_LEN literal originates ONLY from the tw_gen_L_base_s_ge4 producer (stage_generator.py).
    ntt_src = ntt_src.replace(
        "#pragma HLS PIPELINE II=TFG_SEG_LEN",
        f"#pragma HLS PIPELINE II={base_ii}"
    )
    # Inject Work 2 X-stage per-stage TFG+TFR functions and rewire x_stages_top
    num_x_stage = struct["logBU"] + 1
    x_modes = determine_x_stage_modes(all_params, num_x_stage)
    work2_functions_str, work2_top_str = generate_work2_x_stage_cpp(struct, x_modes, P)
    # Replace the functions placeholder
    ntt_src = ntt_src.replace("// [WORK2_X_STAGE_FUNCTIONS_HERE]", work2_functions_str)
    # Replace the x_stages_top body (between BEGIN and END markers inclusive)
    import re
    ntt_src = re.sub(
        r"// \[WORK2_X_STAGE_TOP_BEGIN\].*?// \[WORK2_X_STAGE_TOP_END\]",
        work2_top_str,
        ntt_src,
        flags=re.DOTALL
    )
    # Inject the generated X-stage butterfly dispatch (TFS identity-scalar skip): replace the
    # template fallback BUTTERFLY_LOOP between the markers so the generated case has exactly one
    # active dispatch body (spatial-group-0 / identity lanes carry no tw_scale reduce).
    _bd_begin = "// [WORK2_X_STAGE_BUTTERFLY_DISPATCH_BEGIN]"
    _bd_end = "// [WORK2_X_STAGE_BUTTERFLY_DISPATCH_END]"
    _b = ntt_src.index(_bd_begin)
    _e = ntt_src.index(_bd_end) + len(_bd_end)
    ntt_src = ntt_src[:_b] + _gen_x_stage_butterfly_dispatch(struct) + ntt_src[_e:]
    # Inject the ntt_core L-stage invoke chain (generator-owned; byte-identical to the static chain).
    _lc_begin = "// [WORK3_NTT_CORE_LSTAGE_CHAIN_BEGIN]"
    _lc_end = "// [WORK3_NTT_CORE_LSTAGE_CHAIN_END]"
    _lb = ntt_src.index(_lc_begin)
    _le = ntt_src.index(_lc_end) + len(_lc_end)
    ntt_src = ntt_src[:_lb] + _gen_ntt_core_lstage_chain(struct) + ntt_src[_le:]
    # Inject the small-stage L-TFG producers (generator-owned; byte-identical to the static defs).
    for _s in (0, 1, 2, 3):
        _tb = "// [WORK3_TW_GEN_L_S%d_BEGIN]" % _s
        _te = "// [WORK3_TW_GEN_L_S%d_END]" % _s
        _ti = ntt_src.index(_tb)
        _tj = ntt_src.index(_te) + len(_te)
        ntt_src = ntt_src[:_ti] + _gen_tw_gen_L_small(struct, _s) + ntt_src[_tj:]
    with open(os.path.join(case_dir, "src", "ntt.cpp"), "w") as f:
        f.write(ntt_src)
    # Copy host.cpp unchanged
    shutil.copy2(os.path.join(SEG_TEMPLATE_DIR, "host.cpp"),
                 os.path.join(case_dir, "src", "host.cpp"))

    # Generate ntt.h (fills template with case-specific values)
    generate_seg_ntt_header(case_dir, K, cfg, struct, all_params)

    # Generate ntt_shiftadd.h (for shiftadd paths; stub for Barrett baseline)
    aliases_q = resolve_primes_with_aliases(cfg["primes"])
    # Phase 3.5 S3c/S3d: REUSE the recipe built EARLY above (shiftadd routes only;
    # custom/INT already failed-fast pre-makedirs). recipe=None on the none route ->
    # production path byte-identical and recipe-free. On the batch_legal path the
    # recipe is the decision source for the shift-add anchor + resolved (fold,
    # correction), with the legacy helpers retained as asserts (S3c).
    emit_shiftadd_header(
        case_dir=case_dir,
        K_env=K,
        aliases_q=aliases_q,
        enabled=(shiftadd_mode != "none"),
        shiftadd_mul=use_shiftadd_mul,
        balanced_selector_scope="all",
        recipe=shiftadd_recipe,
        single_prime_mode=_shiftadd_single_prime_mode,
    )

    # Generate ntt_primes.h (all per-prime tables)
    generate_seg_primes_header(case_dir, K, cfg, all_params, struct)

    # Generate ntt_mul.h (same as upstream, reused)
    generate_multi_mul_helper(K, case_dir, all_params)

    # Generate Makefile, link_config, shared configs
    generate_build_infra(struct, case_dir)

    # Save config for reproducibility
    with open(os.path.join(case_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"Generated: {case_name}")
    print(f"  src/: {sorted(os.listdir(os.path.join(case_dir, 'src')))}")
    return case_name, case_dir


# ---------------------------------------------------------------------------
# Batch mode
# ---------------------------------------------------------------------------

def run_batch_seg(cases, common, suffix=None, force=False):
    total = len(cases)
    results = []
    for i, case_cfg in enumerate(cases):
        label = case_cfg.get("name", f"case_{i}")
        merged = dict(common)
        merged.update(case_cfg)
        print(f"\n{'='*60}\n[Batch {i+1}/{total}] {label}\n{'='*60}")
        try:
            case_name, _ = generate_seg_case(merged, suffix=suffix, force=force)
            results.append((i, label, "PASS", case_name))
        except (RuntimeError, ValueError) as e:
            print(f"FAILED: {e}")
            results.append((i, label, "FAIL", str(e)))
    passed = sum(1 for r in results if r[2] == "PASS")
    print(f"\nBatch Summary: {passed}/{total} passed")
    for _, label, status, detail in results:
        print(f"  [{'OK' if status == 'PASS' else '!!'}] {label}: {detail}")
    return passed, total - passed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Segmented TFG Phase 1 NTT case generator.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preset", type=str, help="Use a built-in preset")
    group.add_argument("--config", type=str, help="Path to JSON config file")
    parser.add_argument("--suffix", type=str, default="",
                        help="Optional suffix appended to output directory name")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing output directory")
    args = parser.parse_args()

    suffix = args.suffix or None

    if args.preset:
        presets_path = os.path.join(ROOT, "presets.json")
        with open(presets_path) as f:
            presets = json.load(f)
        if args.preset not in presets:
            print(f"ERROR: Unknown preset '{args.preset}'.", file=sys.stderr)
            sys.exit(1)
        cfg = presets[args.preset]
        print(f"Using preset: {args.preset}")
        generate_seg_case(cfg, suffix=suffix, force=args.force)
    else:
        with open(args.config) as f:
            cfg = json.load(f)
        if "cases" in cfg and isinstance(cfg["cases"], list):
            common = cfg.get("common", {})
            cases = cfg["cases"]
            print(f"Batch config: {args.config} ({len(cases)} cases)")
            passed, failed = run_batch_seg(cases, common, suffix=suffix, force=args.force)
            if failed > 0:
                sys.exit(1)
        else:
            print(f"Using config: {args.config}")
            generate_seg_case(cfg, suffix=suffix, force=args.force)
            print("Done.")


if __name__ == "__main__":
    main()

"""StreamNTT stage/control emission.

This module owns generator-controlled L-stage TFG/TFR producers, butterfly and
L-stage wrappers, the ICBU write-limit micro-regions, and ICBU storage mapping.
``generate_code.py`` assembles these regions into the active template. Reduction
and shift-add arithmetic are owned by their dedicated generator modules.

Naming contract: emitted C++ function names stay unified across shapes (no
single_/multi_ names). Single- vs multi-prime is a generation SHAPE selected by
struct["is_single"], not a name. The ge4 producers stay single C++ functions
(parameterized by the runtime `stage` arg) -- they are NOT fanned out to
per-stage names, which would change call sites and TAPA task/instance naming.
"""

__all__ = [
    "emit_tw_gen_L_small",
    "emit_tw_complete_L_ratio",
    "gen_tw_gen_L_base_ge4",
    "gen_tw_complete_L_ratio_ge4",
    "gen_tw_gen_L_base_ge5",
    "gen_tw_complete_L_ratio_ge5",
    "gen_tw_gen_L_base_ge",
    "gen_tw_complete_L_ratio_ge",
    "gen_bf_unit_shell",
    "gen_l_stage_singleton",
    "gen_l_stage_ge",
    "gen_bf_unit_for_group",
    "gen_l_stage_for_group",
    "region1_emission_map",
    "region2_emission_map",
    "B5_REGION2_OVERRIDES",
    "gen_region2_wrapper",
    "REGION2_WRAPPER_MARKERS",
    "gen_icbu_write_limit",
    "ICBU_WRITE_LIMIT_MARKERS",
    "gen_bf_unit_storage",
    "BF_UNIT_STORAGE_MARKERS",
    "BF_UNIT_STORAGE_RECIPES",
]


# ---------------------------------------------------------------------------
# Region 1: TFG-L / TFR-L (ge4) producers
# ---------------------------------------------------------------------------
# The emitted C++ keeps the base
# producer's `#pragma HLS PIPELINE II=TFG_SEG_LEN` verbatim; generate_code.py
# performs the II=TFG_SEG_LEN -> II=P substitution (the ge4 injection runs
# BEFORE that replace), exactly as for the static template today.

_TW_GEN_L_BASE_GE4_MULTI = """void tw_gen_L_base_s_ge4(const int stage, tapa::ostream<Data>& base_fifo) {
#pragma HLS INLINE off
\tModId cur_mod_id = 0;
\tData base = tw_l_seg_base_table[stage][(int)cur_mod_id];
\tData step = tw_l_seg_step_table[stage][(int)cur_mod_id];
\tint period_blocks = tw_l_period_blocks_tab[stage];
\tap_uint<logDEPTH> seg_ctr = 0;
\tint sub_block = 0;

\tfor(;;) {
#pragma HLS PIPELINE II=TFG_SEG_LEN
\t\tbase_fifo.write(base);
\t\tseg_ctr++;
\t\tsub_block++;
\t\tbool poly_end = ((int)seg_ctr == NUM_SEG_BLOCKS);
\t\tbool period_end = (sub_block == period_blocks);
\t\tif (poly_end) {
\t\t\tseg_ctr = 0; sub_block = 0;
\t\t\tcur_mod_id++;
\t\t\tif (cur_mod_id >= NUM_PRIMES) cur_mod_id = 0;
\t\t\tbase = tw_l_seg_base_table[stage][(int)cur_mod_id];
\t\t\tstep = tw_l_seg_step_table[stage][(int)cur_mod_id];
\t\t\tperiod_blocks = tw_l_period_blocks_tab[stage];
\t\t} else if (period_end) {
\t\t\tsub_block = 0;
\t\t\tbase = tw_l_seg_base_table[stage][(int)cur_mod_id];
\t\t} else {
\t\t\tData base_next;
\t\t\treduce(base, step, base_next, (int)cur_mod_id);
\t\t\tbase = base_next;
\t\t}
\t}
}"""

_TW_GEN_L_BASE_GE4_SINGLE = """void tw_gen_L_base_s_ge4(const int stage, tapa::ostream<Data>& base_fifo) {
#pragma HLS INLINE off
\tData base = tw_l_seg_base_table[stage];
\tData step = tw_l_seg_step_table[stage];
\tint period_blocks = tw_l_period_blocks_tab[stage];
\tap_uint<logDEPTH> seg_ctr = 0;
\tint sub_block = 0;

\tfor(;;) {
#pragma HLS PIPELINE II=TFG_SEG_LEN
\t\tbase_fifo.write(base);
\t\tseg_ctr++;
\t\tsub_block++;
\t\tbool poly_end = ((int)seg_ctr == NUM_SEG_BLOCKS);
\t\tbool period_end = (sub_block == period_blocks);
\t\tif (poly_end) {
\t\t\tseg_ctr = 0; sub_block = 0;
\t\t\tbase = tw_l_seg_base_table[stage];
\t\t\tstep = tw_l_seg_step_table[stage];
\t\t\tperiod_blocks = tw_l_period_blocks_tab[stage];
\t\t} else if (period_end) {
\t\t\tsub_block = 0;
\t\t\tbase = tw_l_seg_base_table[stage];
\t\t} else {
\t\t\tData base_next;
\t\t\treduce(base, step, base_next);
\t\t\tbase = base_next;
\t\t}
\t}
}"""

_TW_COMPLETE_L_RATIO_GE4_MULTI = """void tw_complete_L_ratio_s_ge4(const int stage, tapa::istream<Data>& base_fifo, tapa::ostreams<Data, BU>& tw_L) {
#pragma HLS INLINE off
\tModId cur_mod_id = 0;
\tap_uint<8> phase = 0;
\tap_uint<logDEPTH> seg_ctr = 0;
\tData base = base_fifo.read();

\tfor(;;) {
#pragma HLS PIPELINE II=1
\t\tint ratio_idx = stage * TFG_SEG_LEN + (int)phase;
\t\tData ratio = tw_l_ratio_table[ratio_idx][(int)cur_mod_id];
\t\tData completed;
\t\treduce(base, ratio, completed, (int)cur_mod_id);
\t\tfor (int j = 0; j < BU; ++j) {
#pragma HLS UNROLL
\t\t\ttw_L[j].write(completed);
\t\t}
\t\tphase++;
\t\tif ((int)phase == TFG_SEG_LEN) {
\t\t\tphase = 0;
\t\t\tseg_ctr++;
\t\t\tif ((int)seg_ctr == NUM_SEG_BLOCKS) {
\t\t\t\tseg_ctr = 0;
\t\t\t\tcur_mod_id++;
\t\t\t\tif (cur_mod_id >= NUM_PRIMES) cur_mod_id = 0;
\t\t\t}
\t\t\tbase = base_fifo.read();
\t\t}
\t}
}"""

_TW_COMPLETE_L_RATIO_GE4_SINGLE = """void tw_complete_L_ratio_s_ge4(const int stage, tapa::istream<Data>& base_fifo, tapa::ostreams<Data, BU>& tw_L) {
#pragma HLS INLINE off
\tap_uint<8> phase = 0;
\tap_uint<logDEPTH> seg_ctr = 0;
\tData base = base_fifo.read();

\tfor(;;) {
#pragma HLS PIPELINE II=1
\t\tint ratio_idx = stage * TFG_SEG_LEN + (int)phase;
\t\tData ratio = tw_l_ratio_table[ratio_idx];
\t\tData completed;
\t\treduce(base, ratio, completed);
\t\tfor (int j = 0; j < BU; ++j) {
#pragma HLS UNROLL
\t\t\ttw_L[j].write(completed);
\t\t}
\t\tphase++;
\t\tif ((int)phase == TFG_SEG_LEN) {
\t\t\tphase = 0;
\t\t\tseg_ctr++;
\t\t\tif ((int)seg_ctr == NUM_SEG_BLOCKS) {
\t\t\t\tseg_ctr = 0;
\t\t\t}
\t\t\tbase = base_fifo.read();
\t\t}
\t}
}"""


def emit_tw_gen_L_small(func_name: str, local_s: int, is_single: bool = False, out_width_expr: str = "BU") -> str:
    """Emit a small/direct L-stage TFG producer with configurable output stream width."""
    offset = (1 << local_s) - 1
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
        return (
            "void %s(const int stage, tapa::ostreams<Data, %s>& tw_L) {\n" % (func_name, out_width_expr)
            + "#pragma HLS INLINE off\n"
            + decl_phase
            + "\tfor(;;) {\n"
            + "#pragma HLS PIPELINE II=1\n"
            + "\t\tData tw_val = " + idx + ";\n"
            + "\t\tfor (int j = 0; j < %s; ++j) {\n" % out_width_expr
            + "#pragma HLS UNROLL\n"
            + "\t\t\ttw_L[j].write(tw_val);\n"
            + "\t\t}\n"
            + inc_phase
            + "\t}\n"
            + "}"
        )
    return (
        "void %s(const int stage, tapa::ostreams<Data, %s>& tw_L) {\n" % (func_name, out_width_expr)
        + "#pragma HLS INLINE off\n"
        + "\tModId cur_mod_id = 0;\n"
        + decl_phase
        + "\tap_uint<logDEPTH> poly_ctr = 0;\n"
        + "\tfor(;;) {\n"
        + "#pragma HLS PIPELINE II=1\n"
        + "\t\tData tw_val = " + idx + ";\n"
        + "\t\tfor (int j = 0; j < %s; ++j) {\n" % out_width_expr
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


def emit_tw_complete_L_ratio(func_name: str, is_single: bool = False, out_width_expr: str = "BU") -> str:
    """Emit segmented ge L-stage TFR with configurable output stream width."""
    if is_single:
        return (
            "void %s(const int stage, tapa::istream<Data>& base_fifo, tapa::ostreams<Data, %s>& tw_L) {\n"
            % (func_name, out_width_expr)
            + "#pragma HLS INLINE off\n"
            + "\tap_uint<8> phase = 0;\n"
            + "\tap_uint<logDEPTH> seg_ctr = 0;\n"
            + "\tData base = base_fifo.read();\n"
            + "\n"
            + "\tfor(;;) {\n"
            + "#pragma HLS PIPELINE II=1\n"
            + "\t\tint ratio_idx = stage * TFG_SEG_LEN + (int)phase;\n"
            + "\t\tData ratio = tw_l_ratio_table[ratio_idx];\n"
            + "\t\tData completed;\n"
            + "\t\treduce(base, ratio, completed);\n"
            + "\t\tfor (int j = 0; j < %s; ++j) {\n" % out_width_expr
            + "#pragma HLS UNROLL\n"
            + "\t\t\ttw_L[j].write(completed);\n"
            + "\t\t}\n"
            + "\t\tphase++;\n"
            + "\t\tif ((int)phase == TFG_SEG_LEN) {\n"
            + "\t\t\tphase = 0;\n"
            + "\t\t\tseg_ctr++;\n"
            + "\t\t\tif ((int)seg_ctr == NUM_SEG_BLOCKS) {\n"
            + "\t\t\t\tseg_ctr = 0;\n"
            + "\t\t\t}\n"
            + "\t\t\tbase = base_fifo.read();\n"
            + "\t\t}\n"
            + "\t}\n"
            + "}"
        )
    return (
        "void %s(const int stage, tapa::istream<Data>& base_fifo, tapa::ostreams<Data, %s>& tw_L) {\n"
        % (func_name, out_width_expr)
        + "#pragma HLS INLINE off\n"
        + "\tModId cur_mod_id = 0;\n"
        + "\tap_uint<8> phase = 0;\n"
        + "\tap_uint<logDEPTH> seg_ctr = 0;\n"
        + "\tData base = base_fifo.read();\n"
        + "\n"
        + "\tfor(;;) {\n"
        + "#pragma HLS PIPELINE II=1\n"
        + "\t\tint ratio_idx = stage * TFG_SEG_LEN + (int)phase;\n"
        + "\t\tData ratio = tw_l_ratio_table[ratio_idx][(int)cur_mod_id];\n"
        + "\t\tData completed;\n"
        + "\t\treduce(base, ratio, completed, (int)cur_mod_id);\n"
        + "\t\tfor (int j = 0; j < %s; ++j) {\n" % out_width_expr
        + "#pragma HLS UNROLL\n"
        + "\t\t\ttw_L[j].write(completed);\n"
        + "\t\t}\n"
        + "\t\tphase++;\n"
        + "\t\tif ((int)phase == TFG_SEG_LEN) {\n"
        + "\t\t\tphase = 0;\n"
        + "\t\t\tseg_ctr++;\n"
        + "\t\t\tif ((int)seg_ctr == NUM_SEG_BLOCKS) {\n"
        + "\t\t\t\tseg_ctr = 0;\n"
        + "\t\t\t\tcur_mod_id++;\n"
        + "\t\t\t\tif (cur_mod_id >= NUM_PRIMES) cur_mod_id = 0;\n"
        + "\t\t\t}\n"
        + "\t\t\tbase = base_fifo.read();\n"
        + "\t\t}\n"
        + "\t}\n"
        + "}"
    )


def gen_tw_gen_L_base_ge4(struct: dict) -> str:
    """Emit the L-stage TFG-L base/recurrence producer for stages >=4.

    Unified C++ name `tw_gen_L_base_s_ge4` for both shapes. Multi-prime keeps the
    `ModId cur_mod_id` declaration + cycling and `[stage][(int)cur_mod_id]` table
    indexing; single-prime drops cur_mod_id and indexes `[stage]` directly. The
    body is a faithful verbatim extraction of the committed branches (no new
    logic).

    Args:
        struct: assembly state dict; ``struct["is_single"]`` selects the shape.

    Returns:
        The C++ function source (no trailing newline; no #ifndef scaffolding).
    """
    if struct.get("is_single", False):
        return _TW_GEN_L_BASE_GE4_SINGLE
    return _TW_GEN_L_BASE_GE4_MULTI


def gen_tw_complete_L_ratio_ge4(struct: dict, out_width_expr: str = "BU") -> str:
    """Emit the L-stage TFR-L ratio completion producer for stages >=4.

    Unified C++ name `tw_complete_L_ratio_s_ge4` for both shapes. Multi-prime
    keeps `cur_mod_id` + cycling and `tw_l_ratio_table[ratio_idx][(int)cur_mod_id]`;
    single-prime drops cur_mod_id and indexes `[ratio_idx]` directly. The body is
    a faithful verbatim extraction of the committed branches (no new logic).

    Args:
        struct: assembly state dict; ``struct["is_single"]`` selects the shape.

    Returns:
        The C++ function source (no trailing newline; no #ifndef scaffolding).
    """
    return emit_tw_complete_L_ratio(
        "tw_complete_L_ratio_s_ge4",
        is_single=struct.get("is_single", False),
        out_width_expr=out_width_expr,
    )


# ---------------------------------------------------------------------------
# Region 2: bf_unit / l_stage wrappers
# ---------------------------------------------------------------------------
# The bf_unit_s*/ge* stage wrappers and l_stage_s* task shells are
# shape-independent (identical for single- and
# multi-prime: the only shape-dependence in region 2 lives in the shared `bf_unit`
# CHILD, which stays static template code and is NOT emitted here). The wrappers
# carry only naming + stage offset + TAPA task wiring.
#
# The shared `bf_unit` child body remains in the template; its write-limit and
# storage micro-regions are emitted separately below.

_BF_UNIT_S0 = """void bf_unit_s0(int stage_local, const int bf_id, tapa::istream<Data2>& input_stream, tapa::ostream<Data2>& output_stream, tapa::istream<Data>& tw_i) {
#pragma HLS INLINE off
  bf_unit(stage_local, bf_id, input_stream, output_stream, tw_i);
}"""

_BF_UNIT_S1 = """void bf_unit_s1(int stage_local, const int bf_id, tapa::istream<Data2>& input_stream, tapa::ostream<Data2>& output_stream, tapa::istream<Data>& tw_i) {
#pragma HLS INLINE off
  bf_unit(stage_local, bf_id, input_stream, output_stream, tw_i);
}"""

_BF_UNIT_S2 = """void bf_unit_s2(int stage_local, const int bf_id, tapa::istream<Data2>& input_stream, tapa::ostream<Data2>& output_stream, tapa::istream<Data>& tw_i) {
#pragma HLS INLINE off
  bf_unit(stage_local, bf_id, input_stream, output_stream, tw_i);
}"""

_BF_UNIT_GE3 = """void bf_unit_ge3(int stage_local, const int bf_id, tapa::istream<Data2>& input_stream, tapa::ostream<Data2>& output_stream, tapa::istream<Data>& tw_i) {
#pragma HLS INLINE off
  int global_stage = stage_local + 3;
  bf_unit(global_stage, bf_id, input_stream, output_stream, tw_i);
}"""

_BF_UNIT_GE4 = """void bf_unit_ge4(int stage_local, const int bf_id, tapa::istream<Data2>& input_stream, tapa::ostream<Data2>& output_stream, tapa::istream<Data>& tw_i) {
#pragma HLS INLINE off
  int global_stage = stage_local + 4;
  bf_unit(global_stage, bf_id, input_stream, output_stream, tw_i);
}"""

_L_STAGE_S0 = """void l_stage_s0(const int stage, tapa::istreams<Data2, BU>& input_stream, tapa::ostreams<Data2, BU>& output_stream){
\ttapa::streams<Data, BU, 2> tw_L("twL");
\ttapa::task()
\t\t.invoke<tapa::detach>(tw_gen_L_s0, stage, tw_L)
\t\t.invoke<tapa::detach, BU>(bf_unit_s0, stage, tapa::seq(), input_stream, output_stream, tw_L);
}"""

_L_STAGE_S1 = """void l_stage_s1(const int stage, tapa::istreams<Data2, BU>& input_stream, tapa::ostreams<Data2, BU>& output_stream){
\ttapa::streams<Data, BU, 2> tw_L("twL");
\ttapa::task()
\t\t.invoke<tapa::detach>(tw_gen_L_s1, stage, tw_L)
\t\t.invoke<tapa::detach, BU>(bf_unit_s1, stage, tapa::seq(), input_stream, output_stream, tw_L);
}"""

_L_STAGE_S2 = """void l_stage_s2(const int stage, tapa::istreams<Data2, BU>& input_stream, tapa::ostreams<Data2, BU>& output_stream){
\ttapa::streams<Data, BU, 2> tw_L("twL");
\ttapa::task()
\t\t.invoke<tapa::detach>(tw_gen_L_s2, stage, tw_L)
\t\t.invoke<tapa::detach, BU>(bf_unit_s2, stage, tapa::seq(), input_stream, output_stream, tw_L);
}"""

_L_STAGE_S3 = """void l_stage_s3(const int stage, tapa::istreams<Data2, BU>& input_stream, tapa::ostreams<Data2, BU>& output_stream) {
#pragma HLS INLINE off
\ttapa::streams<Data, BU, 2> tw_L("twL");
\ttapa::task()
\t\t.invoke<tapa::detach>(tw_gen_L_s3, stage, tw_L)
\t\t.invoke<tapa::detach, BU>(bf_unit_ge3, 0, tapa::seq(), input_stream, output_stream, tw_L);
}"""

_L_STAGE_S_GE4 = """void l_stage_s_ge4(int stage, tapa::istreams<Data2, BU>& input_stream, tapa::ostreams<Data2, BU>& output_stream) {
#pragma HLS INLINE off
\ttapa::stream<Data, 2> base_fifo("base_fifo");
\ttapa::streams<Data, BU, 2> tw_L("twL");
\ttapa::task()
\t\t.invoke<tapa::detach>(tw_gen_L_base_s_ge4, stage, base_fifo)
\t\t.invoke<tapa::detach>(tw_complete_L_ratio_s_ge4, stage, base_fifo, tw_L)
\t\t.invoke<tapa::detach, BU>(bf_unit_ge4, stage, tapa::seq(), input_stream, output_stream, tw_L);
}"""

# ---------------------------------------------------------------------------
# Reference anchors for the B=5 stage-order variants.
# These are a test oracle only; the production path is the generic StagePlan-driven
# emitters below (see `region1_emission_map` / `region2_emission_map`). The
# import-time self-check asserts generic output == these bytes.
# ---------------------------------------------------------------------------

_BF_UNIT_S4 = """void bf_unit_s4(int stage_local, const int bf_id, tapa::istream<Data2>& input_stream, tapa::ostream<Data2>& output_stream, tapa::istream<Data>& tw_i) {
#pragma HLS INLINE off
  int global_stage = stage_local + 4;
  bf_unit(global_stage, bf_id, input_stream, output_stream, tw_i);
}"""

_L_STAGE_S4 = """void l_stage_s4(const int stage, tapa::istreams<Data2, BU>& input_stream, tapa::ostreams<Data2, BU>& output_stream) {
#pragma HLS INLINE off
\ttapa::streams<Data, BU, 2> tw_L("twL");
\ttapa::task()
\t\t.invoke<tapa::detach>(tw_gen_L_s4, stage, tw_L)
\t\t.invoke<tapa::detach, BU>(bf_unit_s4, 0, tapa::seq(), input_stream, output_stream, tw_L);
}"""

_BF_UNIT_S_GE5 = (_BF_UNIT_GE4
                  .replace("bf_unit_ge4", "bf_unit_s_ge5")
                  .replace("stage_local + 4", "stage_local + 5"))

_L_STAGE_S_GE5 = (_L_STAGE_S_GE4
                  .replace("l_stage_s_ge4", "l_stage_s_ge5")
                  .replace("tw_gen_L_base_s_ge4", "tw_gen_L_base_s_ge5")
                  .replace("tw_complete_L_ratio_s_ge4", "tw_complete_L_ratio_s_ge5")
                  .replace("bf_unit_ge4", "bf_unit_s_ge5"))


# ---------------------------------------------------------------------------
# Generic StagePlan-driven emitters: the production path.
# Established naming irregularities are encoded in the spec table
# (never reformulated): bf_unit_ge3 (s3, +3) / bf_unit_ge4 (default otf suffix,
# +4) vs stage-order bf_unit_s4 / bf_unit_s_ge5; s0-s2 shells pass stage_local
# THROUGH; s0-s2 wrappers have NO INLINE off and use "){" (no space).
# ---------------------------------------------------------------------------

def gen_bf_unit_shell(cpp_name: str, offset) -> str:
    """bf_unit shell over the SHARED bf_unit child. offset None -> passthrough stage_local
    for s0/s1/s2; otherwise emit `int global_stage = stage_local + offset;`."""
    if offset is None:
        body = "  bf_unit(stage_local, bf_id, input_stream, output_stream, tw_i);"
    else:
        body = ("  int global_stage = stage_local + %d;\n"
                "  bf_unit(global_stage, bf_id, input_stream, output_stream, tw_i);" % offset)
    return ("void %s(int stage_local, const int bf_id, tapa::istream<Data2>& input_stream, "
            "tapa::ostream<Data2>& output_stream, tapa::istream<Data>& tw_i) {\n"
            "#pragma HLS INLINE off\n%s\n}" % (cpp_name, body))


def gen_l_stage_singleton(stage: int, bf_name: str, bf_stage_arg: str,
                          inline_off: bool, space_brace: bool,
                          tw_width_expr: str = "BU") -> str:
    """Singleton precomputed-stage wrapper l_stage_s<N> (tw_gen_L_s<N> + bf shell x BU)."""
    return (
        "void l_stage_s%d(const int stage, tapa::istreams<Data2, BU>& input_stream, "
        "tapa::ostreams<Data2, BU>& output_stream)%s{\n" % (stage, " " if space_brace else "")
        + ("#pragma HLS INLINE off\n" if inline_off else "")
        + "\ttapa::streams<Data, %s, 2> tw_L(\"twL\");\n" % tw_width_expr
        + "\ttapa::task()\n"
        + "\t\t.invoke<tapa::detach>(tw_gen_L_s%d, stage, tw_L)\n" % stage
        + "\t\t.invoke<tapa::detach, BU>(%s, %s, tapa::seq(), input_stream, output_stream, tw_L);\n"
          % (bf_name, bf_stage_arg)
        + "}")


def gen_l_stage_ge(suffix: str, bf_name: str, func_name: str = None, tw_width_expr: str = "BU") -> str:
    """OTF suffix-group wrapper l_stage_<suffix> (base producer + ratio producer + bf shell x BU).

    ``func_name`` can override the emitted function name; the default is
    ``l_stage_<suffix>``."""
    name = func_name if func_name is not None else "l_stage_%s" % suffix
    return (
        "void %s(int stage, tapa::istreams<Data2, BU>& input_stream, "
        "tapa::ostreams<Data2, BU>& output_stream) {\n" % name
        + "#pragma HLS INLINE off\n"
        + "\ttapa::stream<Data, 2> base_fifo(\"base_fifo\");\n"
        + "\ttapa::streams<Data, %s, 2> tw_L(\"twL\");\n" % tw_width_expr
        + "\ttapa::task()\n"
        + "\t\t.invoke<tapa::detach>(tw_gen_L_base_%s, stage, base_fifo)\n" % suffix
        + "\t\t.invoke<tapa::detach>(tw_complete_L_ratio_%s, stage, base_fifo, tw_L)\n" % suffix
        + "\t\t.invoke<tapa::detach, BU>(%s, stage, tapa::seq(), input_stream, output_stream, tw_L);\n"
          % bf_name
        + "}")


def gen_tw_gen_L_base_ge(struct: dict, suffix: str) -> str:
    """Generic TFG-L base producer for an OTF suffix group (body = canonical ge4 string, renamed)."""
    base = _TW_GEN_L_BASE_GE4_SINGLE if struct.get("is_single", False) else _TW_GEN_L_BASE_GE4_MULTI
    return base.replace("tw_gen_L_base_s_ge4", "tw_gen_L_base_%s" % suffix)


def gen_tw_complete_L_ratio_ge(struct: dict, suffix: str, out_width_expr: str = "BU") -> str:
    """Generic TFR-L ratio producer for an OTF suffix group (body = canonical ge4 string, renamed)."""
    return emit_tw_complete_L_ratio(
        "tw_complete_L_ratio_%s" % suffix,
        is_single=struct.get("is_single", False),
        out_width_expr=out_width_expr,
    )


def _bf_shell_spec(plan, g):
    """Return the established C++ name and stage offset for a butterfly group."""
    if g.kind == "precomp":
        s = g.lo
        if s <= 2:
            return ("bf_unit_s%d" % s, None)
        if s == 3:
            return ("bf_unit_ge3", 3)
        return ("bf_unit_s%d" % s, s)
    if g.lo == 4 and g.hi == plan.num_l_stage - 1:
        return ("bf_unit_ge4", 4)                    # established default-plan OTF suffix
    return ("bf_unit_%s" % plan.group_name(g), g.lo)


def gen_bf_unit_for_group(plan, g) -> str:
    name, off = _bf_shell_spec(plan, g)
    return gen_bf_unit_shell(name, off)


def gen_l_stage_for_group(plan, g, tw_width_expr: str = "BU") -> str:
    if g.kind == "otf":
        return gen_l_stage_ge(plan.group_name(g), _bf_shell_spec(plan, g)[0], tw_width_expr=tw_width_expr)
    s = g.lo
    bf_name, _ = _bf_shell_spec(plan, g)
    if s <= 2:
        return gen_l_stage_singleton(s, bf_name, "stage", inline_off=False, space_brace=False, tw_width_expr=tw_width_expr)
    return gen_l_stage_singleton(s, bf_name, "0", inline_off=True, space_brace=True, tw_width_expr=tw_width_expr)


def region2_emission_map(plan, tw_width_expr: str = "BU") -> dict:
    """marker -> emitted C++ (PRODUCTION path). Marker SITES are the fixed template anchors; the
    s4-and-above content rides the two ge4-named sites (multi-function strings under PRECOMP=5).
    A plan with no OTF group emits the ge4 pair at those sites; for small
    ``num_l_stage`` values the invocation chain prunes those functions at compile time."""
    singles = {g.lo: g for g in plan.groups if g.kind == "precomp"}
    otf = plan.otf_groups()[0] if plan.otf_groups() else None
    m = {}
    for s in (0, 1, 2, 3):
        bf_marker = "bf_unit_s%d" % s if s <= 2 else "bf_unit_ge3"
        g = singles.get(s)
        if g is not None:
            m[bf_marker] = gen_bf_unit_for_group(plan, g)
            m["l_stage_s%d" % s] = gen_l_stage_for_group(plan, g, tw_width_expr=tw_width_expr)
        else:                                        # small N: keep fixed template anchors populated
            m[bf_marker] = _REGION2_WRAPPERS[bf_marker]
            m["l_stage_s%d" % s] = _REGION2_WRAPPERS["l_stage_s%d" % s]
    hi_singles = [singles[s] for s in sorted(singles) if s >= 4]
    if otf is None:
        bf_parts = [_BF_UNIT_GE4]
        ls_parts = [_L_STAGE_S_GE4]
    else:
        bf_parts = [gen_bf_unit_for_group(plan, g) for g in hi_singles] + [gen_bf_unit_for_group(plan, otf)]
        ls_parts = [gen_l_stage_for_group(plan, g, tw_width_expr=tw_width_expr) for g in hi_singles]
        ls_parts.append(gen_l_stage_for_group(plan, otf, tw_width_expr=tw_width_expr))
    m["bf_unit_ge4"] = "\n\n".join(bf_parts)
    m["l_stage_s_ge4"] = "\n\n".join(ls_parts)
    return m


def region1_emission_map(plan, struct: dict, tw_small_src_fn, out_width_expr: str = "BU") -> tuple:
    """((marker, emitted), ...) for the region-1 producer SITES (PRODUCTION path). Under a plan with
    precomputed stages >= 4, their tw_gen_L_s<N> producers are prepended at the base-producer site
    (declaration-before-use). tw_small_src_fn = generate_code._gen_tw_gen_L_small (passed in to keep
    this module free of a generate_code import)."""
    otf = plan.otf_groups()[0] if plan.otf_groups() else None
    hi_singles = [s for s in plan.precomp_stages() if s >= 4]
    suffix = plan.group_name(otf) if otf is not None else "s_ge4"
    base = gen_tw_gen_L_base_ge(struct, suffix)
    if hi_singles:
        base = "\n\n".join([tw_small_src_fn(struct, s, out_width_expr=out_width_expr) for s in hi_singles] + [base])
    return (("tw_gen_L_base_s_ge4", base),
            ("tw_complete_L_ratio_s_ge4", gen_tw_complete_L_ratio_ge(struct, suffix, out_width_expr=out_width_expr)))


# ---------------------------------------------------------------------------
# REFERENCE TEMPLATE ANCHORS (test oracle only; production uses the maps above).
# Marker object name -> verbatim wrapper source extracted from the template.
# ---------------------------------------------------------------------------
_REGION2_WRAPPERS = {
    "bf_unit_s0": _BF_UNIT_S0,
    "bf_unit_s1": _BF_UNIT_S1,
    "bf_unit_s2": _BF_UNIT_S2,
    "bf_unit_ge3": _BF_UNIT_GE3,
    "bf_unit_ge4": _BF_UNIT_GE4,
    "l_stage_s0": _L_STAGE_S0,
    "l_stage_s1": _L_STAGE_S1,
    "l_stage_s2": _L_STAGE_S2,
    "l_stage_s3": _L_STAGE_S3,
    "l_stage_s_ge4": _L_STAGE_S_GE4,
}

# Object-level marker names for generate_code.py to inject (single source of truth).
REGION2_WRAPPER_MARKERS = tuple(_REGION2_WRAPPERS.keys())


def gen_region2_wrapper(name: str, struct: dict) -> str:
    """Emit a region-2 L-stage wrapper (`bf_unit_s*/ge*` or `l_stage_s*`) by object name.

    These wrappers are SHAPE-INDEPENDENT: identical for single- and multi-prime
    generation. They carry only naming + stage offset + TAPA task wiring; the
    shape-dependent compute lives in the shared static `bf_unit` child (not emitted
    here). The body is a faithful verbatim extraction of the committed template
    (no behavior change). Unified C++ names are preserved (no `single_/multi_`).

    Args:
        name: object marker name (a key of ``REGION2_WRAPPER_MARKERS``).
        struct: assembly state dict; accepted for emitter-signature uniformity with
            region 1, but UNUSED here (region-2 wrappers do not branch on shape).

    Returns:
        The C++ wrapper source (no trailing newline; no marker scaffolding).
    """
    return _REGION2_WRAPPERS[name]


# Reference accessors and override map used by import-time emission checks.
def gen_tw_gen_L_base_ge5(struct: dict) -> str:
    """Reference accessor; production uses gen_tw_gen_L_base_ge(struct, 's_ge5')."""
    return gen_tw_gen_L_base_ge4(struct).replace("tw_gen_L_base_s_ge4", "tw_gen_L_base_s_ge5")


def gen_tw_complete_L_ratio_ge5(struct: dict) -> str:
    """Reference accessor; production uses gen_tw_complete_L_ratio_ge(struct, 's_ge5')."""
    return gen_tw_complete_L_ratio_ge4(struct).replace("tw_complete_L_ratio_s_ge4", "tw_complete_L_ratio_s_ge5")


B5_REGION2_OVERRIDES = {       # B=5 reference map used by the self-check.
    "bf_unit_ge4": _BF_UNIT_S4 + "\n\n" + _BF_UNIT_S_GE5,
    "l_stage_s_ge4": _L_STAGE_S4 + "\n\n" + _L_STAGE_S_GE5,
}


def _stage_emission_self_check() -> None:
    """Verify that generic StagePlan emitters reproduce the reference strings."""
    from code_generator.stage_plan import plan_from_precomp_boundary
    p4 = plan_from_precomp_boundary(6, 4)
    p5 = plan_from_precomp_boundary(6, 5)
    m4 = region2_emission_map(p4)
    for k, v in _REGION2_WRAPPERS.items():
        assert m4[k] == v, "stage emission drift (default plan): %s" % k
    m5 = region2_emission_map(p5)
    for k, v in B5_REGION2_OVERRIDES.items():
        assert m5[k] == v, "stage emission drift (PRECOMP=5 plan): %s" % k
    for st in ({}, {"is_single": True}):
        assert gen_tw_gen_L_base_ge(st, "s_ge4") == gen_tw_gen_L_base_ge4(st)
        assert gen_tw_complete_L_ratio_ge(st, "s_ge4") == gen_tw_complete_L_ratio_ge4(st)
        assert gen_tw_gen_L_base_ge(st, "s_ge5") == gen_tw_gen_L_base_ge5(st)
        assert gen_tw_complete_L_ratio_ge(st, "s_ge5") == gen_tw_complete_L_ratio_ge5(st)


_stage_emission_self_check()


# ---------------------------------------------------------------------------
# Region 3: ICBU write-limit delay micro-regions
# ---------------------------------------------------------------------------
# Generator-emitted write-limit pieces of the bf_unit child, driven by the
# internal alias struct["_alias_icbu_write_limit_delay"] (global knob; values
# {1,2,3,4}, default 1). delay=1 emits the VERBATIM current Match_Delay /
# write_limit_1d bytes (byte-identical anchor); delay>=2 emits an unconditional
# named-scalar delay chain (write_limit_1d..Nd): write_safe compares write_limit_Nd,
# the shift is deepest-first and stays before the (unchanged) write_limit update.
# This touches only three small micro-regions of the bf_unit child; it is NOT the
# storage or compute region (buffers, bind_storage, and BF_UNIT_LOOP are untouched).

# Canonical delay=1 emission.
_ICBU_DECL_D1 = "\tap_uint<logDEPTH> write_limit_1d = 0;  // write-in with a one-cycle delay"

_ICBU_WRITE_SAFE_D1 = (
    "#ifdef Match_Delay // Match the delay of different RAMs r&w (e.g., 4-URAM)\n"
    "\t\tbool write_safe = write_limit_1d != write_idx;\n"
    "#else\n"
    "\t\tbool write_safe = write_limit != write_idx;\n"
    "#endif"
)

_ICBU_SHIFT_D1 = (
    "#ifdef Match_Delay\n"
    "\t\twrite_limit_1d = write_limit;\n"
    "#endif"
)

# Object-level marker names for the three write-limit micro-regions (no WORK prefix).
ICBU_WRITE_LIMIT_MARKERS = ("icbu_write_limit_decl", "icbu_write_safe", "icbu_write_limit_shift")


def gen_icbu_write_limit(region: str, struct: dict) -> str:
    """Emit one ICBU write-limit micro-region of the bf_unit child.

    Global write-limit delay knob driven by ``struct["_alias_icbu_write_limit_delay"]``
    (depth in {1,2,3,4}, default 1). delay=1 returns the verbatim current Match_Delay /
    write_limit_1d bytes (byte-identical anchor). delay>=2 emits an unconditional
    named-scalar chain: declarations write_limit_1d..Nd, write_safe comparing
    write_limit_Nd, and a deepest-first shift update emitted in place (before the
    unchanged write_limit update). Buffers / bind_storage / BF_UNIT_LOOP are NOT
    touched (this is not the wrapper/child decomposition).

    Args:
        region: one of ``ICBU_WRITE_LIMIT_MARKERS``.
        struct: assembly state; ``struct["_alias_icbu_write_limit_delay"]`` selects depth.

    Returns:
        The C++ micro-region source (no trailing newline; no marker scaffolding).
    """
    delay = struct.get("_alias_icbu_write_limit_delay", 1)
    if region == "icbu_write_limit_decl":
        if delay == 1:
            return _ICBU_DECL_D1
        return "\n".join(
            "\tap_uint<logDEPTH> write_limit_%dd = 0;" % d for d in range(1, delay + 1)
        )
    if region == "icbu_write_safe":
        if delay == 1:
            return _ICBU_WRITE_SAFE_D1
        return "\t\tbool write_safe = write_limit_%dd != write_idx;" % delay
    if region == "icbu_write_limit_shift":
        if delay == 1:
            return _ICBU_SHIFT_D1
        lines = [
            "\t\twrite_limit_%dd = write_limit_%dd;" % (d, d - 1)
            for d in range(delay, 1, -1)
        ]
        lines.append("\t\twrite_limit_1d = write_limit;")
        return "\n".join(lines)
    raise KeyError("unknown ICBU write-limit region: %r" % region)


# ---------------------------------------------------------------------------
# Region 4: bf_unit ping-pong storage block (declarations + bind_storage)
# ---------------------------------------------------------------------------
# The generator controls mem0..3 declarations and bind_storage pragmas for the
# supported ICBU recipes. The default 2U2B recipe maps mem0/1 to URAM and mem2/3
# to BRAM, all as RAM_S2P. Not touched: the separate input_mem_stage LUTRAM storage,
# the BF_UNIT_LOOP, the dependence pragmas, the compute (butterfly/reduce), and the
# write_limit micro-regions. No array_partition pragma is emitted.
#
# ``array_partition`` is a first-class control dimension, not a simple
# string drop-in. It is coupled with the storage recipe (LUTRAM/BRAM/URAM), the mem0/1/2/3
# ping-pong bank structure, the BF_UNIT_LOOP access pattern, write_limit /
# _alias_icbu_write_limit_delay, HLS RAM inference + bind_storage compatibility, and resource
# growth / II stability. It must be designed either as part of a COMBINED storage+partition
# recipe OR as a SEPARATE _alias_bf_unit_partition_recipe with STRICT validation against the
# storage recipe. The per-bank attach points below only mark WHERE partition would go.
#
# Emitted WITH leading indentation + trailing whitespace exactly as the template and with NO
# trailing newline (injection convention; cf. gen_icbu_write_limit / gen_region2_wrapper).

_BF_UNIT_STORAGE_2U2B = (
    "\t// memory for entry with EVEN indices\n"
    "\tData mem0[DEPTH/2]; \n"
    "\tData mem1[DEPTH/2];\n"
    "#pragma HLS bind_storage variable=mem0 type=RAM_S2P impl=uram \n"
    "#pragma HLS bind_storage variable=mem1 type=RAM_S2P impl=uram \n"
    # No array_partition pragma is emitted for mem0/mem1.
    "\n"
    "\t// memory for entry with ODD indices\n"
    "\tData mem2[DEPTH/2]; \n"
    "\tData mem3[DEPTH/2];\n"
    "#pragma HLS bind_storage variable=mem2 type=RAM_S2P impl=bram \n"
    "#pragma HLS bind_storage variable=mem3 type=RAM_S2P impl=bram "
    # No array_partition pragma is emitted for mem2/mem3.
)

# Canonical storage recipe -> per-bank impl tuple (mem0,mem1,mem2,mem3). U=uram, B=bram, L=lutram.
# "2U2B" means exactly (uram, uram, bram, bram), not every possible 2U/2B permutation.
# The supported recipes do not emit array_partition pragmas.
_BF_UNIT_STORAGE_RECIPES = {
    "2U2B": ("uram", "uram", "bram", "bram"),
    "4U":   ("uram", "uram", "uram", "uram"),
    "4B":   ("bram", "bram", "bram", "bram"),
    "4L":   ("lutram", "lutram", "lutram", "lutram"),
    "3U1B": ("uram", "uram", "uram", "bram"),
}

# Valid recipe names -- single source of truth for the generate_code.py env validation.
BF_UNIT_STORAGE_RECIPES = tuple(_BF_UNIT_STORAGE_RECIPES.keys())

_BF_UNIT_STORAGE_IMPL_UC = {"uram": "URAM", "bram": "BRAM", "lutram": "LUTRAM"}


def _bf_unit_storage_recipe_comment(impls) -> str:
    """A generated comment naming the per-bank storage impl (rule N5 -- recipe info in COMMENTS, never function names)."""
    return "\t// storage recipe: mem0=%s, mem1=%s, mem2=%s, mem3=%s" % tuple(
        _BF_UNIT_STORAGE_IMPL_UC[i] for i in impls
    )


def _bf_unit_storage_decls(impls) -> str:
    """Build the bf_unit storage decl + bind_storage block for a per-bank impl tuple (mem0..3).

    Same structure/whitespace as the static template; ONLY the four `impl=<X>` tokens vary (comments, decls,
    trailing spaces, and the no-trailing-newline convention preserved). NO recipe comment, NO array_partition.
    The 2U2B tuple reproduces _BF_UNIT_STORAGE_2U2B byte-for-byte (asserted below).
    """
    _bind = "#pragma HLS bind_storage variable=mem%d type=RAM_S2P impl=%s "
    lines = [
        "\t// memory for entry with EVEN indices",
        "\tData mem0[DEPTH/2]; ",
        "\tData mem1[DEPTH/2];",
        _bind % (0, impls[0]),
        _bind % (1, impls[1]),
        "",
        "\t// memory for entry with ODD indices",
        "\tData mem2[DEPTH/2]; ",
        "\tData mem3[DEPTH/2];",
        _bind % (2, impls[2]),
        _bind % (3, impls[3]),
    ]
    return "\n".join(lines)


# Invariant: the decl builder reproduces the verbatim 2U2B literal byte-for-byte (no recipe comment), so the 2U2B
# default stays byte-identical and non-default recipes differ ONLY in impl= tokens + the prepended recipe comment.
assert _bf_unit_storage_decls(_BF_UNIT_STORAGE_RECIPES["2U2B"]) == _BF_UNIT_STORAGE_2U2B

# Object-level marker name(s) for generate_code.py to inject (single source of truth).
BF_UNIT_STORAGE_MARKERS = ("bf_unit_storage",)


def gen_bf_unit_storage(struct: dict) -> str:
    """Emit the bf_unit ping-pong storage block (mem0..3 declarations and bind_storage).

    If struct["_icbu_recipe_schedule_resolved"] is set (a uniform ICBU recipe
    schedule), emit that resolved recipe + an "// icbu recipe schedule:" comment; otherwise fall back
    to the scalar/default path. No array_partition pragma is emitted.

    Generator-owned storage shell for the bf_unit child. The per-bank storage impl is selected by
    struct["_alias_bf_unit_storage_recipe"] (default "2U2B" = mem0/1->URAM, mem2/3->BRAM, RAM_S2P). The
    2U2B/unset path returns the canonical _BF_UNIT_STORAGE_2U2B literal without a recipe
    comment. Non-default recipes prepend a per-bank recipe comment and then the declaration
    block. No array_partition pragma is emitted because partitioning is coupled with the
    storage recipe, bank structure, access pattern, RAM inference, and target II.
    Recipe names live only in this struct field and the generated comment, never in C++ function names. The separate
    input_mem_stage LUTRAM storage, the BF_UNIT_LOOP, the dependence pragmas, the compute (butterfly/reduce), and the
    write_limit micro-regions are NOT touched.

    Args:
        struct: assembly state dict; struct["_alias_bf_unit_storage_recipe"] selects the recipe (one of
            BF_UNIT_STORAGE_RECIPES; default "2U2B"). Validated upstream (generate_code.py).

    Returns:
        The C++ storage block (leading indentation + trailing whitespace preserved; no
        trailing newline; no marker scaffolding).
    """
    resolved = struct.get("_icbu_recipe_schedule_resolved")
    if resolved is not None:
        # Uniform schedule path; realization checks are done upstream in generate_code.py.
        impls = resolved.uniform_impls
        return ("\t// icbu recipe schedule: %s\n" % resolved.normalized
                + _bf_unit_storage_recipe_comment(impls) + "\n" + _bf_unit_storage_decls(impls))
    recipe = struct.get("_alias_bf_unit_storage_recipe", "2U2B")
    if recipe == "2U2B":
        return _BF_UNIT_STORAGE_2U2B  # verbatim default (no recipe comment) -> byte-identical
    impls = _BF_UNIT_STORAGE_RECIPES[recipe]
    return _bf_unit_storage_recipe_comment(impls) + "\n" + _bf_unit_storage_decls(impls)

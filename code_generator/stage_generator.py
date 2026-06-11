"""StreamNTT stage/control generator framework (Phase 3.2).

Home for Python-controlled, object-level emission of the segmented NTT
stage/topology regions. Complements:
  - generate_code.py      mainline assembly / orchestration; owns the assembly,
                          sets struct["is_single"], injects region markers, and
                          calls into here.
  - shiftadd_generator.py q-derived shiftadd_reduce / shiftadd_mul (UNCHANGED).

Planned region coverage (incremental; only region 1 is implemented in Phase 3.2):
  1. TFG-L + TFR-L              <- Phase 3.2 (this module's first target)
  2. bf_unit / ICBU + l_stage   <- Phase 3.3 (incl. _alias_icbu_write_limit_delay)
  3. TFG-X + TFR-X
  4. X-stage + x_split          <- X-stage split deferred to Phase 4
  5. ntt_core

Out of scope here (separate region / separate module): reduce / butterfly /
TFS+butterfly arithmetic. The reduce recipe family gets reduce_generator.py
later; Phase 3.2 does NOT start reduce_generator.py or a clean same-bit Barrett.

Naming contract: emitted C++ function names stay unified across shapes (no
single_/multi_ names). Single- vs multi-prime is a generation SHAPE selected by
struct["is_single"], not a name. The ge4 producers stay single C++ functions
(parameterized by the runtime `stage` arg) -- they are NOT fanned out to
per-stage names, which would change call sites and TAPA task/instance naming.
"""

__all__ = [
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
# Byte-identical extractions of the committed ntt.cpp #ifndef/#else branches
# (Phase 3.1 tree 1ef88ef): base 610-642 (multi) / 644-673 (single);
# ratio 677-706 (multi) / 708-734 (single). The emitted C++ keeps the base
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


def gen_tw_complete_L_ratio_ge4(struct: dict) -> str:
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
    if struct.get("is_single", False):
        return _TW_COMPLETE_L_RATIO_GE4_SINGLE
    return _TW_COMPLETE_L_RATIO_GE4_MULTI


# ---------------------------------------------------------------------------
# Region 2: bf_unit / l_stage wrappers (Phase 3.2.2)
# ---------------------------------------------------------------------------
# Minimal wrapper emission: the bf_unit_s*/ge* stage wrappers and the l_stage_s*
# task shells, extracted byte-identically from the committed template
# (HEAD e716480). These wrappers are SHAPE-INDEPENDENT (identical for single- and
# multi-prime: the only shape-dependence in region 2 lives in the shared `bf_unit`
# CHILD, which stays static template code and is NOT emitted here). The wrappers
# carry only naming + stage offset + TAPA task wiring.
#
# Out of scope for this step (Phase 3.3): the `bf_unit` child body, the
# write_limit / write_limit_1d delay logic, and the mem0..3 bind_storage pragmas
# all remain in the static template.

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
# Phase 3.4 D2: LEGACY BYTE-IDENTITY ANCHORS (db153f1 B=5 stage-order variants).
# Anchors / test oracle ONLY — the PRODUCTION path is the generic StagePlan-driven
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
# Phase 3.4 D2: GENERIC StagePlan-driven emitters — THE PRODUCTION PATH.
# Naming irregularities are FROZEN public interface, encoded in the spec table
# (never reformulated): bf_unit_ge3 (s3, +3) / bf_unit_ge4 (default otf suffix,
# +4) vs stage-order bf_unit_s4 / bf_unit_s_ge5; s0-s2 shells pass stage_local
# THROUGH; s0-s2 wrappers have NO INLINE off and use "){" (no space).
# ---------------------------------------------------------------------------

def gen_bf_unit_shell(cpp_name: str, offset) -> str:
    """bf_unit shell over the SHARED bf_unit child. offset None -> passthrough stage_local
    (s0/s1/s2 historical shape); else `int global_stage = stage_local + offset;`."""
    if offset is None:
        body = "  bf_unit(stage_local, bf_id, input_stream, output_stream, tw_i);"
    else:
        body = ("  int global_stage = stage_local + %d;\n"
                "  bf_unit(global_stage, bf_id, input_stream, output_stream, tw_i);" % offset)
    return ("void %s(int stage_local, const int bf_id, tapa::istream<Data2>& input_stream, "
            "tapa::ostream<Data2>& output_stream, tapa::istream<Data>& tw_i) {\n"
            "#pragma HLS INLINE off\n%s\n}" % (cpp_name, body))


def gen_l_stage_singleton(stage: int, bf_name: str, bf_stage_arg: str,
                          inline_off: bool, space_brace: bool) -> str:
    """Singleton precomputed-stage wrapper l_stage_s<N> (tw_gen_L_s<N> + bf shell x BU)."""
    return (
        "void l_stage_s%d(const int stage, tapa::istreams<Data2, BU>& input_stream, "
        "tapa::ostreams<Data2, BU>& output_stream)%s{\n" % (stage, " " if space_brace else "")
        + ("#pragma HLS INLINE off\n" if inline_off else "")
        + "\ttapa::streams<Data, BU, 2> tw_L(\"twL\");\n"
        + "\ttapa::task()\n"
        + "\t\t.invoke<tapa::detach>(tw_gen_L_s%d, stage, tw_L)\n" % stage
        + "\t\t.invoke<tapa::detach, BU>(%s, %s, tapa::seq(), input_stream, output_stream, tw_L);\n"
          % (bf_name, bf_stage_arg)
        + "}")


def gen_l_stage_ge(suffix: str, bf_name: str) -> str:
    """OTF suffix-group wrapper l_stage_<suffix> (base producer + ratio producer + bf shell x BU)."""
    return (
        "void l_stage_%s(int stage, tapa::istreams<Data2, BU>& input_stream, "
        "tapa::ostreams<Data2, BU>& output_stream) {\n" % suffix
        + "#pragma HLS INLINE off\n"
        + "\ttapa::stream<Data, 2> base_fifo(\"base_fifo\");\n"
        + "\ttapa::streams<Data, BU, 2> tw_L(\"twL\");\n"
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


def gen_tw_complete_L_ratio_ge(struct: dict, suffix: str) -> str:
    """Generic TFR-L ratio producer for an OTF suffix group (body = canonical ge4 string, renamed)."""
    base = (_TW_COMPLETE_L_RATIO_GE4_SINGLE if struct.get("is_single", False)
            else _TW_COMPLETE_L_RATIO_GE4_MULTI)
    return base.replace("tw_complete_L_ratio_s_ge4", "tw_complete_L_ratio_%s" % suffix)


def _bf_shell_spec(plan, g):
    """(cpp_name, offset) for a group's bf shell — frozen historical naming table."""
    if g.kind == "precomp":
        s = g.lo
        if s <= 2:
            return ("bf_unit_s%d" % s, None)
        if s == 3:
            return ("bf_unit_ge3", 3)
        return ("bf_unit_s%d" % s, s)
    if g.lo == 4 and g.hi == plan.num_l_stage - 1:
        return ("bf_unit_ge4", 4)                    # frozen legacy name (default-plan otf suffix)
    return ("bf_unit_%s" % plan.group_name(g), g.lo)


def gen_bf_unit_for_group(plan, g) -> str:
    name, off = _bf_shell_spec(plan, g)
    return gen_bf_unit_shell(name, off)


def gen_l_stage_for_group(plan, g) -> str:
    if g.kind == "otf":
        return gen_l_stage_ge(plan.group_name(g), _bf_shell_spec(plan, g)[0])
    s = g.lo
    bf_name, _ = _bf_shell_spec(plan, g)
    if s <= 2:
        return gen_l_stage_singleton(s, bf_name, "stage", inline_off=False, space_brace=False)
    return gen_l_stage_singleton(s, bf_name, "0", inline_off=True, space_brace=True)


def region2_emission_map(plan) -> dict:
    """marker -> emitted C++ (PRODUCTION path). Marker SITES are the fixed template anchors; the
    s4-and-above content rides the two ge4-named sites (multi-function strings under PRECOMP=5).
    A plan with NO otf group falls back to the legacy ge4 pair at those sites (historical behavior
    for small num_l_stage: the functions are emitted; the chain prunes them at C compile time)."""
    singles = {g.lo: g for g in plan.groups if g.kind == "precomp"}
    otf = plan.otf_groups()[0] if plan.otf_groups() else None
    m = {}
    for s in (0, 1, 2, 3):
        bf_marker = "bf_unit_s%d" % s if s <= 2 else "bf_unit_ge3"
        g = singles.get(s)
        if g is not None:
            m[bf_marker] = gen_bf_unit_for_group(plan, g)
            m["l_stage_s%d" % s] = gen_l_stage_for_group(plan, g)
        else:                                        # small-n: emit historical bytes regardless
            m[bf_marker] = _REGION2_WRAPPERS[bf_marker]
            m["l_stage_s%d" % s] = _REGION2_WRAPPERS["l_stage_s%d" % s]
    hi_singles = [singles[s] for s in sorted(singles) if s >= 4]
    if otf is None:
        bf_parts = [_BF_UNIT_GE4]
        ls_parts = [_L_STAGE_S_GE4]
    else:
        bf_parts = [gen_bf_unit_for_group(plan, g) for g in hi_singles] + [gen_bf_unit_for_group(plan, otf)]
        ls_parts = [gen_l_stage_for_group(plan, g) for g in hi_singles] + [gen_l_stage_for_group(plan, otf)]
    m["bf_unit_ge4"] = "\n\n".join(bf_parts)
    m["l_stage_s_ge4"] = "\n\n".join(ls_parts)
    return m


def region1_emission_map(plan, struct: dict, tw_small_src_fn) -> tuple:
    """((marker, emitted), ...) for the region-1 producer SITES (PRODUCTION path). Under a plan with
    precomputed stages >= 4, their tw_gen_L_s<N> producers are prepended at the base-producer site
    (declaration-before-use). tw_small_src_fn = generate_code._gen_tw_gen_L_small (passed in to keep
    this module free of a generate_code import)."""
    otf = plan.otf_groups()[0] if plan.otf_groups() else None
    hi_singles = [s for s in plan.precomp_stages() if s >= 4]
    suffix = plan.group_name(otf) if otf is not None else "s_ge4"
    base = gen_tw_gen_L_base_ge(struct, suffix)
    if hi_singles:
        base = "\n\n".join([tw_small_src_fn(struct, s) for s in hi_singles] + [base])
    return (("tw_gen_L_base_s_ge4", base),
            ("tw_complete_L_ratio_s_ge4", gen_tw_complete_L_ratio_ge(struct, suffix)))


# ---------------------------------------------------------------------------
# LEGACY BYTE-IDENTITY ANCHORS (test oracle ONLY — production = the maps above).
# Marker object-name -> verbatim wrapper bytes (committed-template extraction).
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


# Legacy-anchor accessors + override map (test oracle ONLY; production = the emission maps).
def gen_tw_gen_L_base_ge5(struct: dict) -> str:
    """LEGACY ANCHOR accessor (db153f1 bytes). Production path: gen_tw_gen_L_base_ge(struct, 's_ge5')."""
    return gen_tw_gen_L_base_ge4(struct).replace("tw_gen_L_base_s_ge4", "tw_gen_L_base_s_ge5")


def gen_tw_complete_L_ratio_ge5(struct: dict) -> str:
    """LEGACY ANCHOR accessor (db153f1 bytes). Production path: gen_tw_complete_L_ratio_ge(struct, 's_ge5')."""
    return gen_tw_complete_L_ratio_ge4(struct).replace("tw_complete_L_ratio_s_ge4", "tw_complete_L_ratio_s_ge5")


B5_REGION2_OVERRIDES = {       # LEGACY ANCHOR map (db153f1 bytes; test oracle ONLY)
    "bf_unit_ge4": _BF_UNIT_S4 + "\n\n" + _BF_UNIT_S_GE5,
    "l_stage_s_ge4": _L_STAGE_S4 + "\n\n" + _L_STAGE_S_GE5,
}


def _d2_anchor_self_check() -> None:
    """Import-time fail-fast: the generic StagePlan-driven emitters MUST reproduce the legacy anchor
    bytes exactly (PRECOMP=4 default plan == committed-template strings; PRECOMP=5 plan == db153f1
    B=5 strings). Any drift aborts generation immediately."""
    from code_generator.stage_plan import plan_from_precomp_boundary
    p4 = plan_from_precomp_boundary(6, 4)
    p5 = plan_from_precomp_boundary(6, 5)
    m4 = region2_emission_map(p4)
    for k, v in _REGION2_WRAPPERS.items():
        assert m4[k] == v, "D2 anchor drift (default plan): %s" % k
    m5 = region2_emission_map(p5)
    for k, v in B5_REGION2_OVERRIDES.items():
        assert m5[k] == v, "D2 anchor drift (PRECOMP=5 plan): %s" % k
    for st in ({}, {"is_single": True}):
        assert gen_tw_gen_L_base_ge(st, "s_ge4") == gen_tw_gen_L_base_ge4(st)
        assert gen_tw_complete_L_ratio_ge(st, "s_ge4") == gen_tw_complete_L_ratio_ge4(st)
        assert gen_tw_gen_L_base_ge(st, "s_ge5") == gen_tw_gen_L_base_ge5(st)
        assert gen_tw_complete_L_ratio_ge(st, "s_ge5") == gen_tw_complete_L_ratio_ge5(st)


_d2_anchor_self_check()


# ---------------------------------------------------------------------------
# Region 3 (Phase 3.3.1): ICBU write-limit delay micro-regions
# ---------------------------------------------------------------------------
# Generator-emitted write-limit pieces of the bf_unit child, driven by the
# internal alias struct["_alias_icbu_write_limit_delay"] (global knob; values
# {1,2,3,4}, default 1). delay=1 emits the VERBATIM current Match_Delay /
# write_limit_1d bytes (byte-identical anchor); delay>=2 emits an unconditional
# named-scalar delay chain (write_limit_1d..Nd): write_safe compares write_limit_Nd,
# the shift is deepest-first and stays before the (unchanged) write_limit update.
# This touches only three small micro-regions of the bf_unit child; it is NOT the
# future wrapper/child decomposition (buffers / bind_storage / BF_UNIT_LOOP untouched).

# delay=1 verbatim current bytes (must match ntt.cpp 329 / 356-360 / 376-378 @ d994859):
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
    """Emit one ICBU write-limit micro-region of the bf_unit child (Phase 3.3.1).

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
# Region 4 (Phase 3.3.3): bf_unit ping-pong storage block (decl + bind_storage)
# ---------------------------------------------------------------------------
# Brings the previously-static bf_unit-child storage block (mem0..3 declarations +
# bind_storage pragmas) under generator control so storage recipes / array_partition
# can later be varied for ICBU storage minor DSE (Phase 3.3.4). This step emits ONLY the
# default 2U2B recipe (mem0,mem1 -> URAM ; mem2,mem3 -> BRAM ; RAM_S2P), byte-identical to
# the static template storage block (ntt.cpp EVEN/ODD bf_unit-child block @ HEAD bf4043f).
# NOT touched (explicitly out of scope here): the SEPARATE input_mem_stage LUTRAM storage,
# the BF_UNIT_LOOP, the dependence pragmas, the compute (butterfly/reduce), and the
# write_limit micro-regions. No storage-recipe alias and no array_partition are emitted yet
# (Phase 3.3.4+); default emits NO partition and stays byte-identical.
#
# NOTE (Phase 3.3.4 design): array_partition is a FIRST-CLASS control dimension, NOT a simple
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
    # mem0/mem1 array_partition attach point (Phase 3.3.4 first-class recipe dim; see header NOTE) -- none now
    "\n"
    "\t// memory for entry with ODD indices\n"
    "\tData mem2[DEPTH/2]; \n"
    "\tData mem3[DEPTH/2];\n"
    "#pragma HLS bind_storage variable=mem2 type=RAM_S2P impl=bram \n"
    "#pragma HLS bind_storage variable=mem3 type=RAM_S2P impl=bram "
    # mem2/mem3 array_partition attach point (Phase 3.3.4 first-class recipe dim; see header NOTE) -- none now
)

# Canonical storage recipe -> per-bank impl tuple (mem0,mem1,mem2,mem3). U=uram, B=bram, L=lutram.
# Phase 3.3.3 Step C first whitelist (GLOBAL recipe knob). "2U2B" means EXACTLY (uram,uram,bram,bram) = the
# byte-identical default (mem0/1=URAM, mem2/3=BRAM) -- NOT all 2U/2B permutations. Same-side/cross-side 2U2B and
# 3U1L / U-L mixed are DEFERRED Phase 3.3.4 candidates, not first whitelist. (No array_partition; partition = 3.3.4.)
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
    """Emit the bf_unit ping-pong storage block (mem0..3 decl + bind_storage) -- Phase 3.3.3/3.3.4.

    Phase 3.3.4: if struct["_icbu_recipe_schedule_resolved"] is set (a uniform Layer-2 ICBU recipe
    schedule), emit that resolved recipe + an "// icbu recipe schedule:" comment; otherwise fall back
    to the Phase 3.3.3 scalar/default path (byte-identical). NO array_partition is emitted (deferred).

    Generator-owned storage shell for the bf_unit child. The per-bank storage impl is selected by
    struct["_alias_bf_unit_storage_recipe"] (default "2U2B" = mem0/1->URAM, mem2/3->BRAM, RAM_S2P). The
    2U2B/unset path returns the verbatim _BF_UNIT_STORAGE_2U2B literal (BYTE-IDENTICAL anchor; NO recipe
    comment). Non-default recipes PREPEND a per-bank recipe COMMENT (rule N5) then the decl block
    (_bf_unit_storage_decls), so they differ from 2U2B ONLY in the impl= tokens + that comment. NO
    array_partition is emitted (partition is a Phase 3.3.4 FIRST-CLASS recipe dimension -- coupled with storage
    recipe / bank structure / BF_UNIT_LOOP access / write_limit / HLS RAM inference / II -- see the module NOTE).
    Recipe names live ONLY in this struct field + the generated comment, NEVER in C++ function names. The SEPARATE
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
        # Phase 3.3.4 Layer-2 schedule path (R-uniform; realization checks done upstream in
        # generate_code.py). Recipe info -> comments only (rule N5); NO array_partition this slice.
        impls = resolved.uniform_impls
        return ("\t// icbu recipe schedule: %s\n" % resolved.normalized
                + _bf_unit_storage_recipe_comment(impls) + "\n" + _bf_unit_storage_decls(impls))
    recipe = struct.get("_alias_bf_unit_storage_recipe", "2U2B")
    if recipe == "2U2B":
        return _BF_UNIT_STORAGE_2U2B  # verbatim default (no recipe comment) -> byte-identical
    impls = _BF_UNIT_STORAGE_RECIPES[recipe]
    return _bf_unit_storage_recipe_comment(impls) + "\n" + _bf_unit_storage_decls(impls)

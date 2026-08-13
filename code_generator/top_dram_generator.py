#!/usr/bin/env python3
"""Top / DRAM / group-wrapper code generator.

Emits the complete bottom top/DRAM region (real C++ code, not markers
around template code) for the ACTIVE GROUP_CH_NUM only, to be injected between the broad
``// [top_dram_ntt:begin]`` / ``// [top_dram_ntt:end]`` markers of templates/ntt.cpp.

It OWNS and emits:
    read_dram_s / read_dram_m, read_dist_s, write_reshape_s, write_dram_s / write_dram_m,
    read_collect_{N}_core_m, read_collect_{N}_m, read_collect_m,
    write_dist_{N}_core_m, write_dist_{N}_m, write_dist_m,
    ntt_group_dram{N}, ntt(...)
It does NOT own (left in templates/ntt.cpp, outside the marker): reduce, butterfly, TFG/TFR/TFS,
bf_unit, l_stage, x_stage, ntt_core.

Only the active family is emitted: for GROUP_CH_NUM=2 only the 2-family (read_collect_2_*, write_dist_2_*,
ntt_group_dram2); for GROUP_CH_NUM=16 only the 16-family; for single-channel (GROUP_CH_NUM=1) only the
single-channel functions. No inactive static bodies are emitted. Core invoke is fenced by
``// [top_dram_core_invoke:begin]/[:end]``; the ntt() invoke by ``// [top_dram_ntt_invoke:begin]/[:end]``
and stays a single line.

Self-contained: stdlib only; does not import generate_code.
"""
from __future__ import annotations

__all__ = [
    "MCH_GROUP_CH_NUMS",
    "emit_region",
    "emit_read_dram_m",
    "emit_write_dram_m",
    "emit_read_collect_core_m",
    "emit_read_collect_n_m",
    "emit_read_collect_m",
    "emit_write_dist_core_m",
    "emit_write_dist_n_m",
    "emit_write_dist_m",
    "emit_single_channel_movers",
    "emit_ntt_group_dram_m",
    "emit_ntt_group_dram_s",
    "emit_ntt_top",
    "emit_top_dram_region",
]

# GROUP_CH_NUM values that use the multi-channel (MCH) family. GROUP_CH_NUM == 1 = single-channel.
MCH_GROUP_CH_NUMS = (2, 4, 8, 16)


def emit_region(name: str, body: str) -> str:
    """Wrap ``body`` in ``// [name:begin]`` / ``// [name:end]`` markers (caller controls indentation)."""
    return "// [%s:begin]\n%s\n// [%s:end]" % (name, body, name)


# ---------------------------------------------------------------------------
# Fixed DRAM movement bodies (same for all GROUP_CH_NUM). Functionally identical to templates/ntt.cpp;
# whitespace normalized to tabs (HLS/C++ are whitespace-insensitive; correctness gated by SW PASS).
# ---------------------------------------------------------------------------
_READ_DRAM_M = """void read_dram_m(tapa::mmap<bits<DataVec>> x, tapa::ostreams<Data, 2*BU> & dramrd_streams, int poly_num){

\tfor(int poly = 0; poly < poly_num/CH; poly++){
\t\tfor(int i = 0; i < n/DataCHLen; i++){
#pragma HLS PIPELINE II = 1
\t\t\tDataVec data = tapa::bit_cast<DataVec>( x[poly * (n/DataCHLen) + i] );
\t\t\tfor(int d = 0; d < DataCHLen; d++){
#pragma HLS UNROLL
\t\t\t\tdramrd_streams[(i % CORE_DRAM_WIDTH_RATIO)*DataCHLen+d].write( data[d] );
\t\t\t}
\t\t}
\t}
}"""

_WRITE_DRAM_M = """void write_dram_m(tapa::mmap<bits<DataVec>> y, tapa::istreams<Data, 2*BU> & dramwr_streams, int poly_num){

\tfor(int poly = 0; poly < poly_num/CH; poly++){
\t\tfor(int i = 0; i < n/DataCHLen; i++){
#pragma HLS PIPELINE II = 1
\t\t\tDataVec data;
\t\t\tfor(int d = 0; d < DataCHLen; d++){
#pragma HLS UNROLL
\t\t\t\tdata[d] = dramwr_streams[(i % CORE_DRAM_WIDTH_RATIO)*DataCHLen+d].read();
\t\t\t}
\t\t\ty[poly * (n/DataCHLen) + i] =  tapa::bit_cast<bits<DataVec>>(data);
\t\t}
\t}
}"""

_READ_DRAM_S = """void read_dram_s(tapa::mmap<bits<DataVec>> x, tapa::ostreams<DataVec, GROUP_CORE_NUM> & dramrd_streams, int poly_num){

\tfor(int poly = 0; poly < poly_num/CH; poly++){
\t\tfor(int i = 0; i < n/DataCHLen; i++){
#pragma HLS PIPELINE II = 1
\t\t\tDataVec data = tapa::bit_cast<DataVec>( x[poly * (n/DataCHLen) + i] );
\t\t\tdramrd_streams[poly%GROUP_CORE_NUM].write(data);
\t\t}
\t}
}"""

_READ_DIST_S = """void read_dist_s(tapa::istream<DataVec> & dramrd_stream, tapa::ostreams<Data2, BU> & core_istreams){

READ_DIST_S_LOOP:
\tfor(;;){
#pragma HLS PIPELINE II = DRAM_CORE_WIDTH_RATIO
\t\tif( !dramrd_stream.empty() ){
\t\t\tDataVec datavec = dramrd_stream.read();
\t\t\tfor(int core = 0; core < DRAM_CORE_WIDTH_RATIO; core++){
\t\t\t\tfor(int d = 0; d < BU; d++){
#pragma HLS UNROLL
\t\t\t\t\tData2 data = ( static_cast<Data>(datavec[core*2*BU+BU+d]), static_cast<Data>(datavec[core*2*BU+d]) );
\t\t\t\t\tcore_istreams[d].write( data );
\t\t\t\t}
\t\t\t}
\t\t}
\t}
}"""

_WRITE_RESHAPE_S = """void write_reshape_s(tapa::ostream<DataVec> & dramwr_stream, tapa::istreams<Data2, BU> & core_ostreams ){

WRITE_RESHAPE_S_LOOP:
\tfor(;;){
#pragma HLS PIPELINE II = DRAM_CORE_WIDTH_RATIO
\t\tDataVec datavec;
\t\tfor(int c = 0; c < DRAM_CORE_WIDTH_RATIO; c++){
\t\t\tfor(int d = 0; d < BU; d++){
\t\t\t\tData2 data = core_ostreams[d].read();
\t\t\t\tData data_e = data(K-1,0);
\t\t\t\tData data_o = data(2*K-1,K);
\t\t\t\tdatavec[c*2*BU+2*d]   = static_cast<HostData>(data_e);
\t\t\t\tdatavec[c*2*BU+2*d+1] = static_cast<HostData>(data_o);
\t\t\t}
\t\t}
\t\tdramwr_stream.write(datavec);
\t}
}"""

_WRITE_DRAM_S = """void write_dram_s(tapa::mmap<bits<DataVec>> y, tapa::istreams<DataVec, GROUP_CORE_NUM> & dramwr_streams, int poly_num){

\tfor(int poly = 0; poly < poly_num/CH; poly++){
\t\tfor(int i = 0; i < n/DataCHLen; i++){
#pragma HLS PIPELINE II = 1
\t\t\tDataVec data = dramwr_streams[poly%GROUP_CORE_NUM].read();
\t\t\ty[poly * (n/DataCHLen) + i] =  tapa::bit_cast<bits<DataVec>>(data);
\t\t}
\t}
}"""


def emit_read_dram_m() -> str:
    """read_dram_m: per-channel HBM read movement (DataVec -> 2*BU Data lanes)."""
    return _READ_DRAM_M


def emit_write_dram_m() -> str:
    """write_dram_m: per-channel HBM write movement (2*BU Data lanes -> DataVec)."""
    return _WRITE_DRAM_M


# ---------------------------------------------------------------------------
# Parametric multi-channel families (by N = GROUP_CH_NUM in MCH_GROUP_CH_NUMS).
# ---------------------------------------------------------------------------
def _ch_branch(k: int, n: int) -> str:
    """The ``if( ch == k )`` / ``else if`` / ``else`` head for channel k of an N-way selector."""
    if k == 0:
        return "if( ch == 0 )"
    if k == n - 1:
        return "else"
    return "else if( ch == %d )" % k


def emit_read_collect_core_m(n: int) -> str:
    """read_collect_{n}_core_m: one BU-replicated worker; N-way (by ch) read of dramrd_streams{k}_e/_o
    into a single core_istream (Data2 = (data_o, data_e))."""
    params = (["tapa::istream<Data> & dramrd_streams%d_e" % k for k in range(n)]
              + ["tapa::istream<Data> & dramrd_streams%d_o" % k for k in range(n)]
              + ["tapa::ostream<Data2> & core_istream"])
    sel = "\n".join(
        "\t\t\t\t%s{ data_e = dramrd_streams%d_e.read(); data_o = dramrd_streams%d_o.read(); }"
        % (_ch_branch(k, n), k, k) for k in range(n))
    return "\n".join([
        "void read_collect_%d_core_m(%s){" % (n, ", ".join(params)),
        "",
        "READ_COLLECT_LOOP:",
        "\tfor(;;){",
        "\t\tfor(int ch = 0; ch < GROUP_CH_NUM; ch++){",
        "\t\t\tfor(int i = 0; i < n/(2*BU); i++){",
        "#pragma HLS PIPELINE II = 1",
        "\t\t\t\tData data_e;",
        "\t\t\t\tData data_o;",
        sel,
        "\t\t\t\tData2 data = (data_o, data_e);",
        "\t\t\t\tcore_istream.write( data );",
        "\t\t\t}",
        "\t\t}",
        "\t}",
        "}",
    ])


def emit_read_collect_n_m(n: int) -> str:
    """read_collect_{n}_m: thin wrapper invoking read_collect_{n}_core_m BU-replicated, reordering the
    interleaved (per-channel e,o) inputs into all-e-then-all-o."""
    params = []
    for k in range(n):
        params.append("tapa::istreams<Data, BU> & dramrd_streams%d_e" % k)
        params.append("tapa::istreams<Data, BU> & dramrd_streams%d_o" % k)
    params.append("tapa::ostreams<Data2, BU> & core_istreams")
    args = (["dramrd_streams%d_e" % k for k in range(n)]
            + ["dramrd_streams%d_o" % k for k in range(n)] + ["core_istreams"])
    return "\n".join([
        "void read_collect_%d_m(%s ){" % (n, ", ".join(params)),
        "",
        "\ttapa::task()",
        "\t\t.invoke<tapa::detach, BU>(read_collect_%d_core_m, %s)" % (n, ", ".join(args)),
        "\t;",
        "}",
    ])


def emit_read_collect_m(n: int) -> str:
    """read_collect_m dispatcher: single active-arm invoke of read_collect_{n}_m (dramrd_streams x 2N)."""
    args = ", ".join(["dramrd_streams"] * (2 * n) + ["core_istreams"])
    return "\n".join([
        "void read_collect_m(tapa::istreams<Data, 2*BU*GROUP_CH_NUM> & dramrd_streams, tapa::ostreams<Data2, BU> & core_istreams ){",
        "",
        "\ttapa::task()",
        "\t\t.invoke<tapa::detach, 1>(read_collect_%d_m, %s)" % (n, args),
        "\t;",
        "}",
    ])


def emit_write_dist_core_m(n: int) -> str:
    """write_dist_{n}_core_m: one BU-replicated worker; reads a Data2 from core_ostream, splits e/o, and
    N-way (by ch) writes to dramwr_streams{k}[0]/[1]."""
    params = (["tapa::ostreams<Data,2> & dramwr_streams%d" % k for k in range(n)]
              + ["tapa::istream<Data2> & core_ostream"])
    sel = "\n".join(
        "\t\t\t\t%s{ dramwr_streams%d[0].write( data_e ); dramwr_streams%d[1].write( data_o ); }"
        % (_ch_branch(k, n), k, k) for k in range(n))
    return "\n".join([
        "void write_dist_%d_core_m(%s){" % (n, ", ".join(params)),
        "",
        "WRITE_DIST_LOOP:",
        "\tfor(;;){",
        "\t\tfor(int ch = 0; ch < GROUP_CH_NUM; ch++){",
        "\t\t\tfor(int i = 0; i < n/(2*BU); i++){",
        "#pragma HLS PIPELINE II = 1",
        "\t\t\t\tData2 data = core_ostream.read();",
        "\t\t\t\tData data_e = data(K-1,0);",
        "\t\t\t\tData data_o = data(2*K-1,K);",
        sel,
        "\t\t\t}",
        "\t\t}",
        "\t}",
        "}",
    ])


def emit_write_dist_n_m(n: int) -> str:
    """write_dist_{n}_m: thin wrapper invoking write_dist_{n}_core_m BU-replicated."""
    params = (["tapa::ostreams<Data, 2*BU> & dramwr_streams%d" % k for k in range(n)]
              + ["tapa::istreams<Data2, BU> & core_ostreams"])
    args = ["dramwr_streams%d" % k for k in range(n)] + ["core_ostreams"]
    return "\n".join([
        "void write_dist_%d_m(%s ){" % (n, ", ".join(params)),
        "",
        "\ttapa::task()",
        "\t\t.invoke<tapa::detach, BU>(write_dist_%d_core_m, %s)" % (n, ", ".join(args)),
        "\t;",
        "}",
    ])


def emit_write_dist_m(n: int) -> str:
    """write_dist_m dispatcher: single active-arm invoke of write_dist_{n}_m (dramwr_streams x N)."""
    args = ", ".join(["dramwr_streams"] * n + ["core_ostreams"])
    return "\n".join([
        "void write_dist_m(tapa::ostreams<Data, 2*BU*GROUP_CH_NUM> & dramwr_streams, tapa::istreams<Data2, BU> & core_ostreams ){",
        "",
        "\ttapa::task()",
        "\t\t.invoke<tapa::detach, 1>(write_dist_%d_m, %s)" % (n, args),
        "\t;",
        "}",
    ])


def _fence(name: str, line: str) -> list:
    """Fence a single task-chain invoke ``line`` (two-tab indent) in // [name:begin]/[:end] markers."""
    return ["\t\t// [%s:begin]" % name, line, "\t\t// [%s:end]" % name]


def _core_invoke(line: str) -> list:
    """Fence the ntt_core invoke in // [top_dram_core_invoke:begin]/[:end] (ntt_core stays owned elsewhere)."""
    return _fence("top_dram_core_invoke", line)


def _read_adapter(line: str) -> list:
    """Fence the read adapter (read_collect_m / read_dist_s) in // [top_dram_read_adapter:begin]/[:end]."""
    return _fence("top_dram_read_adapter", line)


def _write_adapter(line: str) -> list:
    """Fence the write adapter (write_dist_m / write_reshape_s) in // [top_dram_write_adapter:begin]/[:end]."""
    return _fence("top_dram_write_adapter", line)


def emit_ntt_group_dram_m(n: int, core_name: str = "ntt_core") -> str:
    """ntt_group_dram{n} (multi-channel): read_dram_m x N -> read_collect_m -> [ntt_core] -> write_dist_m
    -> write_dram_m x N. ntt_core is invoked (owned outside this generator) inside the core-invoke region."""
    params = (["\ttapa::mmap<bits<DataVec>> x%d," % k for k in range(n)]
              + ["\ttapa::mmap<bits<DataVec>> y%d," % k for k in range(n)] + ["\tint poly_num"])
    lines = [
        "void ntt_group_dram%d(" % n,
        "\n".join(params),
        "){",
        "",
        "\ttapa::streams<Data, 2*BU*GROUP_CH_NUM, POLY_FIFO_DEPTH_M> dramrd_streams(\"dramrd_streams\");",
        "\ttapa::streams<Data, 2*BU*GROUP_CH_NUM, POLY_FIFO_DEPTH_M> dramwr_streams(\"dramwr_streams\");",
        "\t// [top_dram_core_streams:begin]",
        "\ttapa::streams<Data2, BU, 2> core_istreams(\"core_istreams\");",
        "\ttapa::streams<Data2, BU, 2> core_ostreams(\"core_ostreams\");",
        "\t// [top_dram_core_streams:end]",
        "",
        "\ttapa::task()",
    ]
    lines += ["\t\t.invoke<tapa::join>(read_dram_m, x%d, dramrd_streams, poly_num)" % k for k in range(n)]
    lines += _read_adapter("\t\t.invoke<tapa::detach>(read_collect_m, dramrd_streams, core_istreams)")
    lines += _core_invoke("\t\t.invoke<tapa::detach>(%s, core_istreams, core_ostreams)" % core_name)
    lines += _write_adapter("\t\t.invoke<tapa::detach>(write_dist_m, dramwr_streams, core_ostreams)")
    lines += ["\t\t.invoke<tapa::join>(write_dram_m, y%d, dramwr_streams, poly_num)" % k for k in range(n)]
    lines += ["\t;", "}"]
    return "\n".join(lines)


def emit_ntt_group_dram_s(core_name: str = "ntt_core") -> str:
    """ntt_group_dram1 (single-channel): read_dram_s -> read_dist_s -> [ntt_core] -> write_reshape_s ->
    write_dram_s (all GROUP_CORE_NUM-replicated). ntt_core owned outside this generator."""
    lines = [
        "void ntt_group_dram1(tapa::mmap<bits<DataVec>> x, tapa::mmap<bits<DataVec>> y, int poly_num){",
        "",
        "\ttapa::streams<DataVec, GROUP_CORE_NUM, POLY_FIFO_DEPTH_S> dramrd_streams(\"dramrd_streams\");",
        "\ttapa::streams<DataVec, GROUP_CORE_NUM, POLY_FIFO_DEPTH_S> dramwr_streams(\"dramwr_streams\");",
        "\t// [top_dram_core_streams:begin]",
        "\ttapa::streams<Data2, BU*GROUP_CORE_NUM, 2> core_istreams(\"core_istreams\");",
        "\ttapa::streams<Data2, BU*GROUP_CORE_NUM, 2> core_ostreams(\"core_ostreams\");",
        "\t// [top_dram_core_streams:end]",
        "",
        "\ttapa::task()",
        "\t\t.invoke<tapa::join>(read_dram_s, x, dramrd_streams, poly_num)",
    ]
    lines += _read_adapter("\t\t.invoke<tapa::detach, GROUP_CORE_NUM>(read_dist_s, dramrd_streams, core_istreams)")
    lines += _core_invoke("\t\t.invoke<tapa::detach, GROUP_CORE_NUM>(%s, core_istreams, core_ostreams)" % core_name)
    lines += _write_adapter("\t\t.invoke<tapa::detach, GROUP_CORE_NUM>(write_reshape_s, dramwr_streams, core_ostreams)")
    lines += [
        "\t\t.invoke<tapa::join>(write_dram_s, y, dramwr_streams, poly_num)",
        "\t;",
        "}",
    ]
    return "\n".join(lines)


def emit_single_channel_movers() -> str:
    """The single-channel (GROUP_CH_NUM==1) DRAM movers: read_dram_s, read_dist_s, write_reshape_s,
    write_dram_s (DataVec streams; no read_collect/write_dist)."""
    return "\n\n".join([_READ_DRAM_S, _READ_DIST_S, _WRITE_RESHAPE_S, _WRITE_DRAM_S])


def emit_ntt_top(n: int) -> str:
    """ntt(): ONE one-line active invoke of ntt_group_dram{n} (hbm_ch x 2N), fenced by the
    top_dram_ntt_invoke markers. No #if GROUP_CH_NUM family."""
    hbm = ", ".join(["hbm_ch"] * (2 * n))
    return "\n".join([
        "void ntt(tapa::mmaps<bits<DataVec>, 2*CH> hbm_ch, int poly_num){",
        "",
        "\ttapa::task()",
        "\t\t// [top_dram_ntt_invoke:begin]",
        "\t\t.invoke<tapa::join, GROUP_NUM>(ntt_group_dram%d, %s, poly_num);" % (n, hbm),
        "\t\t// [top_dram_ntt_invoke:end]",
        "}",
    ])


def emit_top_dram_region(group_ch_num: int, core_name: str = "ntt_core") -> str:
    """Assemble the FULL active-GROUP_CH_NUM top/DRAM region (the body injected between
    // [top_dram_ntt:begin] and // [top_dram_ntt:end]). Multi-channel families are wrapped in #ifdef MCH,
    single-channel in #ifndef MCH; ntt() is emitted outside the guard. Only the active family appears."""
    if group_ch_num == 1:
        blocks = [
            "#ifndef MCH",
            emit_single_channel_movers(),
            emit_ntt_group_dram_s(core_name),
            "#endif // MCH",
            emit_ntt_top(1),
        ]
    elif group_ch_num in MCH_GROUP_CH_NUMS:
        n = group_ch_num
        blocks = [
            "#ifdef MCH",
            emit_read_dram_m(),
            emit_read_collect_core_m(n),
            emit_read_collect_n_m(n),
            emit_read_collect_m(n),
            emit_write_dist_core_m(n),
            emit_write_dist_n_m(n),
            emit_write_dist_m(n),
            emit_write_dram_m(),
            emit_ntt_group_dram_m(n, core_name),
            "#endif // MCH",
            emit_ntt_top(n),
        ]
    else:
        raise ValueError("emit_top_dram_region: unsupported GROUP_CH_NUM=%d (expected 1,2,4,8,16)"
                         % group_ch_num)
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------

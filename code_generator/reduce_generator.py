"""Reduction-family emission and prime-set classification.

The generator emits the active ``reduce()`` implementation through the
``// [reduce:begin]`` / ``// [reduce:end]`` seam in templates/ntt.cpp. It owns
mixed-bit two-class parameterization, normalized one-class shapes, and the
optional X-twiddle route-specialized reducer.

Layering: stdlib-only; generate_code.py imports this module (never the reverse).
An import-time self-check pins the template seam content to
``_REDUCE_BLOCK_VERBATIM`` so template drift fails before generation.
"""
import os
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
# This module lives under code_generator/ while templates live at the repository root.
_REPO_ROOT = os.path.dirname(_HERE)
_SEG_NTT_CPP_TEMPLATE = os.path.join(_REPO_ROOT, "templates", "ntt.cpp")

# Marker names consumed by generate_code.py's injection loop ("// [<name>:begin/end]").
REDUCE_MARKERS = ("reduce",)

# The segmented template reduce() block, byte-exact. NO trailing newline: the injection
# replaces "<begin-marker>...<end-marker>" INCLUSIVE, leaving the end-marker line's own
# newline in place, matching the other seam emitters.
_REDUCE_BLOCK_VERBATIM = """// Two-class Barrett-style reduce
// Selected by USE_REDUCE_SHIFTADD when the generated shift-add reducer is used

// cls == 0: 52-class / 51/52-bit family
// cls == 1: 62-class / 61/62-bit family
#ifndef SINGLE_PRIME
void reduce(Data A, Data B, Data &Z, int mod_id){
#else
void reduce(Data A, Data B, Data &Z){
#endif
#pragma HLS INLINE

	// --- Full product (wide, single path); manually tiled ---
	Data2 U = mul_full_data_nonstd(A, B);
#ifndef SINGLE_PRIME
	Data q = MODS[mod_id];
#else
	Data q = MOD;
#endif

#if defined(USE_REDUCE_SHIFTADD)
	// BU shift-add (fold-based reducer)
#ifndef SINGLE_PRIME
	Z = reduce_shiftadd(U, (ModId)mod_id, q);
#else
	Z = reduce_shiftadd(U, (ModId)0, q);   // single-prime: only recipe index 0 exists
#endif
#elif defined(USE_REDUCE_SHIFTADD_MUL)
	// BU shift-add multiplication (Barrett/Shoup-preserving)
#ifndef SINGLE_PRIME
	Z = reduce_shiftadd_mul(U, (ModId)mod_id, q);
#else
	Z = reduce_shiftadd_mul(U, (ModId)0, q);   // single-prime: only recipe index 0 exists
#endif
#else
	// Mixed K=62 main path with explicit _cls0 candidates ONLY at recipesensitive points
#ifndef SINGLE_PRIME
	const int cls = PRIME_CLASS[mod_id];
	Data2 T = BARRETT_MUS[mod_id];
#else
	const int cls = PRIME_CLASS;
	Data2 T = BARRETT_MU;
#endif

	// V shift (recipe-sensitive)
	Dataplus V_cls0 = (Dataplus)(U >> 51);
	Dataplus V = (Dataplus)(U >> (K - 1));
	Dataplus V_mux = cls ? V : V_cls0;

	// V_l, V_h: 62-compatible (R1, R2 PASS).
	Data V_l = (Data)((ap_uint<62>)V_mux);
	ap_uint<1> V_h = (ap_uint<1>)(V_mux >> 62);

	// T_l / T_h split (recipe-sensitive)
	Data T_l_cls0 = (Data)((ap_uint<52>)T);
	Data T_l = (Data)((ap_uint<K>)T);
	Data T_l_mux = cls ? T_l : T_l_cls0;

	Data T_h_cls0 = (Data)((ap_uint<52>)(T >> 52));
	Data T_h = (Data)((ap_uint<K>)(T >> K));
	Data T_h_mux = cls ? T_h : T_h_cls0;

	// Wide multiplications (single path, operands already muxed)
	Data2 W0 = mul_full_data_nonstd(V_l, T_l_mux);
	Data2 W1 = mul_w1_compiletime(V_l, T_h_mux);
	Data2 W2 = V_h ? (Data2)T_l_mux : (Data2)0;

	// W3: 62-compatible (R3 PASS); use 62-style with T_h_mux.
	Data2 W3 = V_h ? ((Data2)T_h_mux << 61) : (Data2)0;

	// W_buffer shift/reduction (recipe-sensitive)
	Data2 W12 = (Data2)(W1 + W2);

	ap_uint<2*K+1> W_pre_cls0 =
	    (ap_uint<2*K+1>)W0 + ((ap_uint<2*K+1>)W12 << 52);
	Data2 W_shifted_cls0 = (Data2)(W_pre_cls0 >> 53);

	ap_uint<2*K+1> W_pre =
	    (ap_uint<2*K+1>)W0 + ((ap_uint<2*K+1>)W12 << K);
	Data2 W_shifted = (Data2)(W_pre >> (K + 1));

	Data2 W_shifted_mux = cls ? W_shifted : W_shifted_cls0;
	Data2 W = W_shifted_mux + W3;

	// W_l, W_h: 62-compatible (R4, R5 PASS).
	Data W_l = (Data)((ap_uint<62>)W);
	ap_uint<1> W_h = (ap_uint<1>)(W >> 62);

	// IntMult3 (single path)
	Data2 X0 = mul_low_data_data_kplus2_nonstd(W_l, q);

	// X1: 62-compatible (R6 PASS).
	Data2 X1 = W_h ? ((Data2)q << 62) : (Data2)0;

	// mask / ring (recipe-sensitive)
	Data2 mask_cls0 = ((Data2)1 << 54) - 1;
	Data2 mask = ((Data2)1 << (K + 2)) - 1;
	Data2 mask_mux = cls ? mask : mask_cls0;

	Data2 ring_cls0 = ((Data2)1 << 54);
	Data2 ring = ((Data2)1 << (K + 2));
	Data2 ring_mux = cls ? ring : ring_cls0;

	Data2 X = (X0 + X1) & mask_mux;
	Data2 Y = U & mask_mux;

	Dataplus2 Z0 = (Dataplus2)((Y + ring_mux - X) & mask_mux);
	Dataplus2 Z1 = Z0;
	Dataplus2 two_q = (Dataplus2)q << 1;
	Dataplus2 Z2 = Z1 - (Dataplus2)q;
	Dataplus2 Z3 = Z1 - two_q;
	Dataplus2 Z_buffer = (Z1 >= two_q) ? Z3
	                  : ((Z1 >= (Dataplus2)q) ? Z2 : Z1);
	Z = static_cast<Data>(Z_buffer);
#endif  // USE_REDUCE_SHIFTADD
}"""


# ---------------------------------------------------------------------------
# Mixed-bit two-class parameterization
# ---------------------------------------------------------------------------
# The block template is DERIVED AT IMPORT TIME from _REDUCE_BLOCK_VERBATIM (the
# retained byte-identity anchor / test oracle) via the context-anchored
# substitution table below — there is no second stored literal to drift.
# Tokens (@..@ + str.replace; no str.format brace hazard with the C code):
#   @KS@ = k_small (smaller K_group class PRESENT; cls0 side)   @KSm1@/@KSp1@/@KSp2@
#   @KL@ = k_large (larger class PRESENT; shared-path widths)    @KLm1@
# The cls1 dual-candidate side stays SYMBOLIC env-K text (final_K) — byte-identity
# requires it, and gen_reduce_block fail-fasts if mixed k_large != final_K.
# Selection machinery (PRIME_CLASS read, the six muxes, dual-candidate order,
# SINGLE_PRIME swap, USE_REDUCE_SHIFTADD[_MUL]/Barrett #if chain) is NOT
# parameterized and stays verbatim.

_TEMPLATE_SUBSTITUTIONS = (
    # Generated comment literals.
    ("// cls == 0: 52-class / 51/52-bit family",
     "// cls == 0: @KS@-class / @KSm1@/@KS@-bit family"),
    ("// cls == 1: 62-class / 61/62-bit family",
     "// cls == 1: @KL@-class / @KLm1@/@KL@-bit family"),
    ("// Mixed K=62 main path with explicit _cls0 candidates ONLY at recipesensitive points",
     "// Mixed K=@KL@ main path with explicit _cls0 candidates ONLY at recipesensitive points"),
    ("// V_l, V_h: 62-compatible (R1, R2 PASS).",
     "// V_l, V_h: @KL@-compatible (R1, R2 PASS)."),
    ("// W3: 62-compatible (R3 PASS); use 62-style with T_h_mux.",
     "// W3: @KL@-compatible (R3 PASS); use @KL@-style with T_h_mux."),
    ("// W_l, W_h: 62-compatible (R4, R5 PASS).",
     "// W_l, W_h: @KL@-compatible (R4, R5 PASS)."),
    ("// X1: 62-compatible (R6 PASS).",
     "// X1: @KL@-compatible (R6 PASS)."),
    # family A: cls0 dual-candidate literals (k_small-1 .. k_small+2)
    ("Dataplus V_cls0 = (Dataplus)(U >> 51);",
     "Dataplus V_cls0 = (Dataplus)(U >> @KSm1@);"),
    ("Data T_l_cls0 = (Data)((ap_uint<52>)T);",
     "Data T_l_cls0 = (Data)((ap_uint<@KS@>)T);"),
    ("Data T_h_cls0 = (Data)((ap_uint<52>)(T >> 52));",
     "Data T_h_cls0 = (Data)((ap_uint<@KS@>)(T >> @KS@));"),
    ("(ap_uint<2*K+1>)W0 + ((ap_uint<2*K+1>)W12 << 52);",
     "(ap_uint<2*K+1>)W0 + ((ap_uint<2*K+1>)W12 << @KS@);"),
    ("Data2 W_shifted_cls0 = (Data2)(W_pre_cls0 >> 53);",
     "Data2 W_shifted_cls0 = (Data2)(W_pre_cls0 >> @KSp1@);"),
    ("Data2 mask_cls0 = ((Data2)1 << 54) - 1;",
     "Data2 mask_cls0 = ((Data2)1 << @KSp2@) - 1;"),
    ("Data2 ring_cls0 = ((Data2)1 << 54);",
     "Data2 ring_cls0 = ((Data2)1 << @KSp2@);"),
    # family C: SHARED-PATH fixed-width literals (k_large-1 / k_large)
    ("Data V_l = (Data)((ap_uint<62>)V_mux);",
     "Data V_l = (Data)((ap_uint<@KL@>)V_mux);"),
    ("ap_uint<1> V_h = (ap_uint<1>)(V_mux >> 62);",
     "ap_uint<1> V_h = (ap_uint<1>)(V_mux >> @KL@);"),
    ("Data2 W3 = V_h ? ((Data2)T_h_mux << 61) : (Data2)0;",
     "Data2 W3 = V_h ? ((Data2)T_h_mux << @KLm1@) : (Data2)0;"),
    ("Data W_l = (Data)((ap_uint<62>)W);",
     "Data W_l = (Data)((ap_uint<@KL@>)W);"),
    ("ap_uint<1> W_h = (ap_uint<1>)(W >> 62);",
     "ap_uint<1> W_h = (ap_uint<1>)(W >> @KL@);"),
    ("Data2 X1 = W_h ? ((Data2)q << 62) : (Data2)0;",
     "Data2 X1 = W_h ? ((Data2)q << @KL@) : (Data2)0;"),
)

_CLASS_DIGITS = ("51", "52", "53", "54", "61", "62")


def _derive_template() -> str:
    """Build the parameterized template from the verbatim anchor (import time).

    Fail-fast on any anchor that does not hit EXACTLY once and on any residual
    class numeral after substitution — a derivation drift aborts generation.
    """
    t = _REDUCE_BLOCK_VERBATIM
    for old, new in _TEMPLATE_SUBSTITUTIONS:
        n = t.count(old)
        if n != 1:
            raise RuntimeError(
                "reduce_generator template derivation: anchor hit %d times "
                "(expected 1): %r" % (n, old))
        t = t.replace(old, new)
    for d in _CLASS_DIGITS:
        if d in t:
            raise RuntimeError(
                "reduce_generator template derivation: residual class "
                "numeral %r left unparameterized." % d)
    return t


_REDUCE_BLOCK_TEMPLATE = _derive_template()


def _render_reduce_block(k_small: int, k_large: int) -> str:
    """Pure, permissive renderer that substitutes class tokens.

    Domain enforcement (today: only (52, 62) is reachable) lives in
    gen_reduce_block, not here — tests exercise other values freely.
    """
    t = _REDUCE_BLOCK_TEMPLATE
    for tok, val in (("@KSm1@", k_small - 1), ("@KSp1@", k_small + 1),
                     ("@KSp2@", k_small + 2), ("@KS@", k_small),
                     ("@KLm1@", k_large - 1), ("@KL@", k_large)):
        t = t.replace(tok, str(val))
    if "@K" in t:
        raise RuntimeError("reduce_generator render: residual @K token.")
    return t


def _reduce_params_for_struct(struct: dict) -> Tuple[int, int]:
    """Return the mixed-class template parameters under a strict domain.

    mixed_bit -> (class_map[0], class_map[1]) from reduction metadata, with
    impossible-today fail-fasts (k_large must equal final_K and K_group; the
    only legal mixed map today is (52, 62)). same_bit / single / absent
    metadata -> the legacy default (52, 62). No guessing or silent rewrites.
    """
    mode = struct.get("_reduce_class_mode")
    if mode == "mixed_bit":
        cmap = struct.get("_reduce_class_map")
        if cmap is None or len(cmap) != 2:
            raise RuntimeError(
                "reduce_generator: mixed_bit set without a 2-class "
                "class_map (got %r)." % (cmap,))
        k_small, k_large = int(cmap[0]), int(cmap[1])
        final_k = struct.get("_reduce_final_k")
        k_group = struct.get("_reduce_k_group")
        if final_k != k_large:
            raise RuntimeError(
                "reduce_generator: mixed_bit k_large=%d != final_K=%r — "
                "the cls1 side is SYMBOLIC env-K text and requires k_large == "
                "final_K; refusing to render." % (k_large, final_k))
        if k_group != k_large:
            raise RuntimeError(
                "reduce_generator: mixed_bit K_group=%r != k_large=%d — "
                "inconsistent metadata; refusing to render." % (k_group, k_large))
        if (k_small, k_large) != (52, 62):
            raise RuntimeError(
                "reduce_generator: mixed class map (%d, %d) is outside the "
                "supported domain {(52, 62)}; refusing to render." % (k_small, k_large))
        return k_small, k_large
    # same_bit / single / metadata absent: use the legacy dual-candidate text.
    return 52, 62


# ---------------------------------------------------------------------------
# Normalized one-class shapes (mode-gated; legacy default)
# ---------------------------------------------------------------------------
# One-class emission performs generation-time mux resolution of
# the legacy datapath — delete the cls read and the unselected candidates,
# resolve each "X_mux = cls ? X : X_cls0;" to the side legacy would select,
# keep EVERYTHING else verbatim (shared-path @KL@/@KLm1@ "62-compatible"
# superset widths, dual-candidate ordering of survivors, the SINGLE_PRIME
# skeleton, the route #if chain). Resolved assignments KEEP the *_mux names
# so all downstream text stays stable. No formulas are re-derived.
# This applies only to shiftadd == "none"; shift-add routes and mixed
# sets emit the legacy two-class text in both modes.

_ONE_CLASS_COMMON_SUBS = (
    ("// Two-class Barrett-style reduce",
     "// One-class Barrett-style reduce (normalized mode)"),
    ("// cls == 0: @KS@-class / @KSm1@/@KS@-bit family\n"
     "// cls == 1: @KL@-class / @KLm1@/@KL@-bit family",
     "// class: @CC@-class / @CCm1@/@CC@-bit family (cls muxes resolved at generation time)"),
    ("\t// Mixed K=@KL@ main path with explicit _cls0 candidates ONLY at recipesensitive points",
     "\t// One-class main path: class selection resolved at generation time (no cls muxes)"),
    ("#ifndef SINGLE_PRIME\n\tconst int cls = PRIME_CLASS[mod_id];\n"
     "\tData2 T = BARRETT_MUS[mod_id];\n#else\n\tconst int cls = PRIME_CLASS;\n"
     "\tData2 T = BARRETT_MU;\n#endif",
     "#ifndef SINGLE_PRIME\n\tData2 T = BARRETT_MUS[mod_id];\n#else\n"
     "\tData2 T = BARRETT_MU;\n#endif"),
)

_ONE_CLASS_LARGE_SUBS = (
    ("\t// V shift (recipe-sensitive)\n\tDataplus V_cls0 = (Dataplus)(U >> @KSm1@);\n"
     "\tDataplus V = (Dataplus)(U >> (K - 1));\n\tDataplus V_mux = cls ? V : V_cls0;\n",
     "\t// V shift (recipe-sensitive)\n\tDataplus V_mux = (Dataplus)(U >> (K - 1));\n"),
    ("\t// T_l / T_h split (recipe-sensitive)\n\tData T_l_cls0 = (Data)((ap_uint<@KS@>)T);\n"
     "\tData T_l = (Data)((ap_uint<K>)T);\n\tData T_l_mux = cls ? T_l : T_l_cls0;\n\n"
     "\tData T_h_cls0 = (Data)((ap_uint<@KS@>)(T >> @KS@));\n"
     "\tData T_h = (Data)((ap_uint<K>)(T >> K));\n\tData T_h_mux = cls ? T_h : T_h_cls0;\n",
     "\t// T_l / T_h split (recipe-sensitive)\n\tData T_l_mux = (Data)((ap_uint<K>)T);\n\n"
     "\tData T_h_mux = (Data)((ap_uint<K>)(T >> K));\n"),
    ("\tap_uint<2*K+1> W_pre_cls0 =\n"
     "\t    (ap_uint<2*K+1>)W0 + ((ap_uint<2*K+1>)W12 << @KS@);\n"
     "\tData2 W_shifted_cls0 = (Data2)(W_pre_cls0 >> @KSp1@);\n\n"
     "\tap_uint<2*K+1> W_pre =\n"
     "\t    (ap_uint<2*K+1>)W0 + ((ap_uint<2*K+1>)W12 << K);\n"
     "\tData2 W_shifted = (Data2)(W_pre >> (K + 1));\n\n"
     "\tData2 W_shifted_mux = cls ? W_shifted : W_shifted_cls0;\n",
     "\tap_uint<2*K+1> W_pre =\n"
     "\t    (ap_uint<2*K+1>)W0 + ((ap_uint<2*K+1>)W12 << K);\n"
     "\tData2 W_shifted_mux = (Data2)(W_pre >> (K + 1));\n"),
    ("\t// mask / ring (recipe-sensitive)\n\tData2 mask_cls0 = ((Data2)1 << @KSp2@) - 1;\n"
     "\tData2 mask = ((Data2)1 << (K + 2)) - 1;\n\tData2 mask_mux = cls ? mask : mask_cls0;\n\n"
     "\tData2 ring_cls0 = ((Data2)1 << @KSp2@);\n\tData2 ring = ((Data2)1 << (K + 2));\n"
     "\tData2 ring_mux = cls ? ring : ring_cls0;\n",
     "\t// mask / ring (recipe-sensitive)\n\tData2 mask_mux = ((Data2)1 << (K + 2)) - 1;\n\n"
     "\tData2 ring_mux = ((Data2)1 << (K + 2));\n"),
)

_ONE_CLASS_SMALL_SUBS = (
    ("\t// V shift (recipe-sensitive)\n\tDataplus V_cls0 = (Dataplus)(U >> @KSm1@);\n"
     "\tDataplus V = (Dataplus)(U >> (K - 1));\n\tDataplus V_mux = cls ? V : V_cls0;\n",
     "\t// V shift (recipe-sensitive)\n\tDataplus V_mux = (Dataplus)(U >> @KSm1@);\n"),
    ("\t// T_l / T_h split (recipe-sensitive)\n\tData T_l_cls0 = (Data)((ap_uint<@KS@>)T);\n"
     "\tData T_l = (Data)((ap_uint<K>)T);\n\tData T_l_mux = cls ? T_l : T_l_cls0;\n\n"
     "\tData T_h_cls0 = (Data)((ap_uint<@KS@>)(T >> @KS@));\n"
     "\tData T_h = (Data)((ap_uint<K>)(T >> K));\n\tData T_h_mux = cls ? T_h : T_h_cls0;\n",
     "\t// T_l / T_h split (recipe-sensitive)\n\tData T_l_mux = (Data)((ap_uint<@KS@>)T);\n\n"
     "\tData T_h_mux = (Data)((ap_uint<@KS@>)(T >> @KS@));\n"),
    ("\tap_uint<2*K+1> W_pre_cls0 =\n"
     "\t    (ap_uint<2*K+1>)W0 + ((ap_uint<2*K+1>)W12 << @KS@);\n"
     "\tData2 W_shifted_cls0 = (Data2)(W_pre_cls0 >> @KSp1@);\n\n"
     "\tap_uint<2*K+1> W_pre =\n"
     "\t    (ap_uint<2*K+1>)W0 + ((ap_uint<2*K+1>)W12 << K);\n"
     "\tData2 W_shifted = (Data2)(W_pre >> (K + 1));\n\n"
     "\tData2 W_shifted_mux = cls ? W_shifted : W_shifted_cls0;\n",
     "\tap_uint<2*K+1> W_pre =\n"
     "\t    (ap_uint<2*K+1>)W0 + ((ap_uint<2*K+1>)W12 << @KS@);\n"
     "\tData2 W_shifted_mux = (Data2)(W_pre >> @KSp1@);\n"),
    ("\t// mask / ring (recipe-sensitive)\n\tData2 mask_cls0 = ((Data2)1 << @KSp2@) - 1;\n"
     "\tData2 mask = ((Data2)1 << (K + 2)) - 1;\n\tData2 mask_mux = cls ? mask : mask_cls0;\n\n"
     "\tData2 ring_cls0 = ((Data2)1 << @KSp2@);\n\tData2 ring = ((Data2)1 << (K + 2));\n"
     "\tData2 ring_mux = cls ? ring : ring_cls0;\n",
     "\t// mask / ring (recipe-sensitive)\n\tData2 mask_mux = ((Data2)1 << @KSp2@) - 1;\n\n"
     "\tData2 ring_mux = ((Data2)1 << @KSp2@);\n"),
)


def _derive_one_class(side: str) -> str:
    """Derive a one-class shape from the template (import time; fail-fast)."""
    subs = _ONE_CLASS_COMMON_SUBS + (
        _ONE_CLASS_LARGE_SUBS if side == "large" else _ONE_CLASS_SMALL_SUBS)
    t = _REDUCE_BLOCK_TEMPLATE
    for old, new in subs:
        n = t.count(old)
        if n != 1:
            raise RuntimeError(
                "reduce_generator one-class(%s) derivation: anchor hit %d "
                "times (expected 1): %r..." % (side, n, old[:60]))
    # apply only after all anchors verified (order-independent sites)
    for old, new in subs:
        t = t.replace(old, new)
    for residue in ("const int cls", "cls ?", "_cls0", "PRIME_CLASS"):
        if residue in t:
            raise RuntimeError(
                "reduce_generator one-class(%s): residue %r survived "
                "resolution." % (side, residue))
    if side == "large" and "@KS" in t:
        raise RuntimeError(
            "reduce_generator one-class(large): unexpected @KS token.")
    return t


_REDUCE_ONE_CLASS_LARGE = _derive_one_class("large")
_REDUCE_ONE_CLASS_SMALL = _derive_one_class("small")


def _render_one_class(shape: str, k_class: int) -> str:
    """Render a one-class shape. Class constants come from k_class (= K_group,
    NEVER blindly final_K); the shared-path superset widths stay fixed at
    62/61 exactly as the canonical two-class text uses them."""
    t = shape
    for tok, val in (("@KSm1@", k_class - 1), ("@KSp1@", k_class + 1),
                     ("@KSp2@", k_class + 2), ("@KS@", k_class),
                     ("@CCm1@", k_class - 1), ("@CC@", k_class),
                     ("@KLm1@", 61), ("@KL@", 62)):
        t = t.replace(tok, str(val))
    if "@K" in t or "@C" in t:
        raise RuntimeError("reduce_generator one-class render: residual token.")
    return t


# ---------------------------------------------------------------------------
# X-twiddle route-specialized reducer
# ---------------------------------------------------------------------------
# reduce_tw_x is a GENERATOR-SELECTED, SINGLE-ROUTE reducer used ONLY at selected X-twiddle call
# sites (TFG-X / TFR-X / TFS-X). It is emitted ONLY when an X-twiddle region uses a non-default
# (shiftadd) route (struct["_xtw_needs_shiftadd"]); otherwise gen_reduce_block returns the existing
# block byte-identically. The global reduce() above is UNCHANGED and stays Barrett.
#
# HARD RULE (user): reduce_tw_x must contain ONLY the selected route body -- NO
# #if USE_REDUCE_SHIFTADD / #elif USE_REDUCE_SHIFTADD_MUL / #else route ladder. The #ifndef
# SINGLE_PRIME guard below is the multi/single SHAPE selector (mod_id presence), NOT a route ladder.
# reduce_shiftadd is available because ntt_shiftadd.h is force-included for the hybrid build state
# (generate_seg_ntt_header) WITHOUT defining USE_REDUCE_SHIFTADD (so reduce() stays Barrett).

def _gen_reduce_tw_x_block(struct: dict) -> str:
    """Emit the reduce_tw_x() definition, or "" when no X-twiddle region needs shiftadd (byte-identical).

    Only "shiftadd_reduce" is supported. The emitted function contains one route and no route ladder.
    """
    if not struct.get("_xtw_needs_shiftadd"):
        return ""
    route = struct.get("_xtw_reduce_route", "shiftadd_reduce")
    if route != "shiftadd_reduce":
        raise RuntimeError(
            "reduce_tw_x: only 'shiftadd_reduce' is supported (got %r)." % (route,))
    return (
        "\n\n"
        "// X-twiddle route-specialized reduction.\n"
        "// Generator-selected SINGLE route (shiftadd_reduce); NO USE_REDUCE_SHIFTADD/_MUL #if ladder.\n"
        "// Global reduce() above stays Barrett; reduce_tw_x is called ONLY at selected X-twiddle sites\n"
        "// (TFG-X / TFR-X / TFS-X). reduce_shiftadd is provided by the force-included ntt_shiftadd.h.\n"
        "#ifndef SINGLE_PRIME\n"
        "void reduce_tw_x(Data A, Data B, Data &Z, int mod_id){\n"
        "#else\n"
        "void reduce_tw_x(Data A, Data B, Data &Z){\n"
        "#endif\n"
        "#pragma HLS INLINE\n"
        "\tData2 U = mul_full_data_nonstd(A, B);\n"
        "#ifndef SINGLE_PRIME\n"
        "\tData q = MODS[mod_id];\n"
        "\tZ = reduce_shiftadd(U, (ModId)mod_id, q);\n"
        "#else\n"
        "\tData q = MOD;\n"
        "\tZ = reduce_shiftadd(U, (ModId)0, q);   // single-prime: only recipe index 0 exists\n"
        "#endif\n"
        "}"
    )


# Wide-K Barrett reduce (K > 62 / anchor-63 K_group=64). The 52/62 template hardcodes 62-bit
# V/W/X splits (ap_uint<62>, ap_uint<1>) that TRUNCATE the high product bits for K>62 (proven: for
# (q-1)*(q-1), V = U>>(K-1) is K bits, > 62+1). This is a clean K-PARAMETERIZED textbook Barrett with no
# hardcoded 62/61 constants: mu=floor(2^2K/q) (up to K+1 bits), q2=q1*mu (2K+2 bits), 2 corrections.
_WIDE_K_REDUCE_BLOCK = """// Wide-K Barrett reduce (K > 62): K-parameterized textbook Barrett (no hardcoded 62/61 splits).
#ifndef SINGLE_PRIME
void reduce(Data A, Data B, Data &Z, int mod_id){
#else
void reduce(Data A, Data B, Data &Z){
#endif
#pragma HLS INLINE
\tData2 U = (Data2)A * (Data2)B;                 // A,B < 2^K  ->  U = A*B < 2^(2K)
#ifndef SINGLE_PRIME
\tData q = MODS[mod_id];
\tData2 mu = BARRETT_MUS[mod_id];
#else
\tData q = MOD;
\tData2 mu = BARRETT_MU;                          // floor(2^(2K)/q); up to K+1 bits (full-width literal)
#endif
\tap_uint<K+1>   q1 = (ap_uint<K+1>)(U >> (K - 1));
\tap_uint<2*K+2> q2 = (ap_uint<2*K+2>)q1 * (ap_uint<2*K+2>)mu;
\tap_uint<K+1>   q3 = (ap_uint<K+1>)(q2 >> (K + 1));
\tap_uint<2*K+2> qq = (ap_uint<2*K+2>)q3 * (ap_uint<2*K+2>)q;
\tData2 r = (Data2)(((ap_uint<2*K+2>)U) - qq);    // U >= q3*q (Barrett); r < 3q fits Data2
\tif (r >= (Data2)q) r -= (Data2)q;
\tif (r >= (Data2)q) r -= (Data2)q;
\tZ = (Data)r;
}"""


def _render_wide_k_reduce_block() -> str:
    """The K-parameterized wide-K (K>62) textbook Barrett reduce() block. Uses the K macro (= final_K);
    no hardcoded 62/61 constants, so it is correct for K in {63, 64}. Single- and multi-prime arms both
    present. Mixed-K K>62 batches are not supported."""
    return _WIDE_K_REDUCE_BLOCK


def gen_reduce_block(struct: dict) -> str:
    """Return the reduce() block for injection at the [reduce] seam.

    Legacy mode (default/unset): mixed parameterized, same-bit/
    single legacy (52, 62) dual-candidate text.
    normalized mode: same-bit / single-prime Barrett (shiftadd == "none" ONLY)
    emit the one-class resolved shape for the set's K_group class; mixed-bit
    and the shiftadd routes stay byte-identical to legacy.
    """
    mode = struct.get("_alias_reduce_cls_mode", "legacy")
    if mode not in ("legacy", "normalized"):
        raise RuntimeError(
            "reduce_generator: invalid _alias_reduce_cls_mode=%r (use "
            "'legacy' or 'normalized')." % (mode,))
    route = struct.get("_reduce_shiftadd_mode", "none")
    cmode = struct.get("_reduce_class_mode")
    if mode == "normalized" and route == "none" and cmode in ("single", "same_bit"):
        cmap = struct.get("_reduce_class_map")
        if not cmap or len(cmap) != 1:
            raise RuntimeError(
                "reduce_generator: %s set without a 1-class class_map "
                "(got %r)." % (cmode, cmap))
        k_class = int(cmap[0])
        if k_class not in (52, 62):
            raise RuntimeError(
                "reduce_generator: class %d outside the supported domain "
                "{52, 62}; refusing to render." % k_class)
        if struct.get("_reduce_k_group") != k_class:
            raise RuntimeError(
                "reduce_generator: K_group=%r != class %d — inconsistent "
                "metadata; refusing to render." % (struct.get("_reduce_k_group"), k_class))
        if k_class == 62 and struct.get("_reduce_final_k") != 62:
            raise RuntimeError(
                "reduce_generator: one-class-62 requires final_K == 62 "
                "(symbolic-K side); got final_K=%r." % (struct.get("_reduce_final_k"),))
        shape = _REDUCE_ONE_CLASS_LARGE if k_class == 62 else _REDUCE_ONE_CLASS_SMALL
        return _render_one_class(shape, k_class) + _gen_reduce_tw_x_block(struct)
    # Wide-K anchor-63 / K_group=64: the 52/62 template hardcodes 62-bit splits that truncate
    # for K>62. Emit a clean K-parameterized textbook Barrett instead (the 52/62 path is untouched).
    final_k = struct.get("_reduce_final_k")
    if route == "none" and final_k is not None and int(final_k) > 62:
        return _render_wide_k_reduce_block() + _gen_reduce_tw_x_block(struct)
    # Legacy mode, mixed-bit sets, and shift-add routes use the two-class path.
    k_small, k_large = _reduce_params_for_struct(struct)
    return _render_reduce_block(k_small, k_large) + _gen_reduce_tw_x_block(struct)


def _reduce_anchor_self_check(template_path: str = _SEG_NTT_CPP_TEMPLATE) -> None:
    """Read-only fail-fast guard: the template's [reduce] seam content must equal
    the verbatim constant exactly; raises RuntimeError on drift (aborts generation
    at import). ``template_path`` is parameterized for tests ONLY."""
    with open(template_path, "r") as f:
        src = f.read()
    gb, ge = "// [reduce:begin]", "// [reduce:end]"
    try:
        i = src.index(gb)
        j = src.index(ge) + len(ge)
    except ValueError as exc:
        raise RuntimeError(
            "reduce_generator anchor self-check: [reduce] markers missing in %s (%s)"
            % (template_path, exc))
    expected = gb + "\n" + _REDUCE_BLOCK_VERBATIM + "\n" + ge
    if src[i:j] != expected:
        raise RuntimeError(
            "reduce_generator anchor self-check FAILED: template [reduce] seam "
            "diverged from _REDUCE_BLOCK_VERBATIM in %s -- refusing to generate."
            % template_path)


_reduce_anchor_self_check()

# The parameterized renderer must reproduce the
# verbatim anchor exactly at the legacy point — render(52, 62) == anchor ==
# template seam. Any drift aborts generation.
if _render_reduce_block(52, 62) != _REDUCE_BLOCK_VERBATIM:
    raise RuntimeError(
        "reduce_generator parameterization self-check FAILED: _render_reduce_block(52, 62) "
        "!= _REDUCE_BLOCK_VERBATIM — refusing to generate.")


# ---------------------------------------------------------------------------
# final_K inference and reduce-set classification (metadata only)
# ---------------------------------------------------------------------------
# Set-relative class semantics (binding user correction): cls0 = the SMALLER
# K_group class PRESENT in the current prime set; cls1 = the LARGER class
# PRESENT. Never "cls0=51" (51/61 are sparse ANCHOR exponents, not classes) and
# never the legacy absolute 0/1 emitted values. Classification does not change emission:
# the emitted PRIME_CLASS keeps the legacy absolute rule and the reduce() block
# stays the canonical constant above.


@dataclass(frozen=True)
class AnchorFamily:
    """A sparse-prime anchor family. anchor_B is the sparse anchor exponent
    (q ~ 2^anchor_B +/- small terms); legal_K_candidates are the K widths that can contain such a prime
    (bit_length B or B+1); default_K_group is the grouping envelope (largest legal K). K_group is a
    CONSEQUENCE of the anchor, never the source of truth, and anchor_B is NOT defined as K_group-1."""
    anchor_B: int
    legal_K_candidates: Tuple[int, ...]
    default_K_group: int


# Known anchor families (derived/validated). The default K_group SET {32,52,62,64} is the CONSEQUENCE
# of anchors {31,51,61,63}, not a hardcoded design constraint.
ANCHOR_FAMILIES: Tuple[AnchorFamily, ...] = (
    AnchorFamily(anchor_B=31, legal_K_candidates=(31, 32), default_K_group=32),
    AnchorFamily(anchor_B=51, legal_K_candidates=(51, 52), default_K_group=52),
    AnchorFamily(anchor_B=61, legal_K_candidates=(61, 62), default_K_group=62),
    AnchorFamily(anchor_B=63, legal_K_candidates=(63, 64), default_K_group=64),
)
_ANCHOR_BY_B = {f.anchor_B: f for f in ANCHOR_FAMILIES}


def _anchor_b_for(q: int) -> int:
    """Discover the sparse anchor exponent B of q (q ~ 2^B +/- small terms): the nearest power-of-2
    exponent. ANCHOR-FIRST -- the anchor is read from the prime's sparse magnitude, NOT from a hardcoded
    K_group->anchor map. q closer to 2^bl (q = 2^B + small, bit_length B+1) -> B = bit_length;
    q closer to 2^(bl-1) (q = 2^B - small, bit_length B) -> B = bit_length-1."""
    bl = q.bit_length()
    return bl if ((1 << bl) - q) < (q - (1 << (bl - 1))) else (bl - 1)


def anchor_family_for(q: int) -> AnchorFamily:
    """The AnchorFamily for q, derived anchor-first. Known anchors use the table; an unlisted anchor B
    derives legal_K_candidates {B, B+1} with default_K_group=B+1 (same rule, not a hardcoded special case)."""
    B = _anchor_b_for(q)
    fam = _ANCHOR_BY_B.get(B)
    if fam is not None:
        return fam
    if B > 64:
        raise RuntimeError("Unsupported prime anchor_B=%d for q=%d (>64-bit out of envelope)." % (B, q))
    return AnchorFamily(anchor_B=B, legal_K_candidates=(B, B + 1), default_K_group=B + 1)


def _k_arith_for(q: int) -> int:
    """Per-prime arithmetic class width (K_group), derived anchor-first:
    recognize the prime's anchor_B from its sparse form, then take that family's default K_group (the
    largest legal K candidate). K_group is the grouping CONSEQUENCE of the anchor, never the source of
    truth. The known set {32,52,62,64} follows from anchors {31,51,61,63}; existing 51/61-anchor primes
    keep K_group 52/62 (byte-identity preserved)."""
    return anchor_family_for(q).default_K_group


@dataclass(frozen=True)
class ReduceClassInfo:
    """Pure reduction metadata for one prime set."""
    config_k: Optional[int]            # verbatim cfg value; None when omitted
    k_group: int                       # max per-prime K_arith
    final_k: int                       # max(config_k, k_group); k_group when omitted
    provenance: str                    # "inferred" | "preserved" | "lifted"
    mode: str                          # "single" | "same_bit" | "mixed_bit"
    class_map: Tuple[int, ...]         # K classes PRESENT, ascending (cls0=small, cls1=large)
    per_prime: Tuple[Tuple[str, int, int, int], ...]  # (alias, q, K_arith, relative_class)


def classify_reduce_set(aliases_q: Sequence[Tuple[str, int]],
                        config_k: Optional[int] = None) -> ReduceClassInfo:
    """Classify a resolved prime set and derive final_K.

    final_K rule: K omitted -> final_K = K_group; K present -> final_K =
    max(config_K, K_group); config_K < K_group -> LIFTED (caller reports the
    warning); config_K >= K_group -> PRESERVED exactly. Raises (via
    _k_arith_for) on primes wider than 62 bits.
    """
    if not aliases_q:
        raise ValueError("classify_reduce_set: empty prime set.")
    ks = [(alias, q, _k_arith_for(q)) for alias, q in aliases_q]
    class_map = tuple(sorted({k for _alias, _q, k in ks}))
    rel = {c: i for i, c in enumerate(class_map)}
    per_prime = tuple((alias, q, k, rel[k]) for alias, q, k in ks)
    k_group = class_map[-1]
    if len(aliases_q) == 1:
        mode = "single"
    elif len(class_map) == 1:
        mode = "same_bit"
    else:
        mode = "mixed_bit"
    if config_k is None:
        final_k, provenance = k_group, "inferred"
    elif config_k < k_group:
        final_k, provenance = k_group, "lifted"
    else:
        final_k, provenance = config_k, "preserved"
    return ReduceClassInfo(config_k=config_k, k_group=k_group, final_k=final_k,
                           provenance=provenance, mode=mode, class_map=class_map,
                           per_prime=per_prime)


__all__ = ["gen_reduce_block", "REDUCE_MARKERS", "classify_reduce_set", "ReduceClassInfo"]

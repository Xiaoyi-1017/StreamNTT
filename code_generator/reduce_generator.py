"""reduce_generator.py -- reduce-family emission owner (Phase 3.5).

S1a-0 scope: emit the segmented template's reduce() block VERBATIM through the
// [reduce:begin] / // [reduce:end] seam in templates/ntt.cpp (flat layout 2026-06-11).
The struct argument is ACCEPTED BUT UNUSED in this slice (reserved for the
single-prime / same-bit / mixed-bit classification of later Phase 3.5 slices);
gen_reduce_block must stay byte-identical regardless of struct content.

Layering: stdlib-only; generate_code.py imports this module (never the reverse).
The import-time self-check (read-only, fail-fast) pins the template seam content
to _REDUCE_BLOCK_VERBATIM -- any drift aborts generation (D2 anchor discipline).
"""
import os
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
# Repo cleanup (2026-06-10): this module lives in code_generator/; the template tree stays
# at the REPO ROOT -> resolve one level up. Template flatten (2026-06-11): the active
# segmented template is templates/ntt.cpp (flat layout; old subtree removed/archived).
_REPO_ROOT = os.path.dirname(_HERE)
_SEG_NTT_CPP_TEMPLATE = os.path.join(_REPO_ROOT, "templates", "ntt.cpp")

# Marker names consumed by generate_code.py's injection loop ("// [<name>:begin/end]").
REDUCE_MARKERS = ("reduce",)

# The segmented template reduce() block, byte-exact. NO trailing newline: the injection
# replaces "<begin-marker>...<end-marker>" INCLUSIVE, leaving the end-marker line's own
# newline in place -- the same arithmetic as every existing seam emitter (d2c3599).
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
# Phase 3.5 S1c — mixed-bit two-class PARAMETERIZATION (byte-identical slice)
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
    # family D: comment literals (OD-s1c-1)
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
                "reduce_generator S1c template derivation: anchor hit %d times "
                "(expected 1): %r" % (n, old))
        t = t.replace(old, new)
    for d in _CLASS_DIGITS:
        if d in t:
            raise RuntimeError(
                "reduce_generator S1c template derivation: residual class "
                "numeral %r left unparameterized." % d)
    return t


_REDUCE_BLOCK_TEMPLATE = _derive_template()


def _render_reduce_block(k_small: int, k_large: int) -> str:
    """Pure, PERMISSIVE renderer (OD-s1c-5): substitute the class tokens.

    Domain enforcement (today: only (52, 62) is reachable) lives in
    gen_reduce_block, not here — tests exercise other values freely.
    """
    t = _REDUCE_BLOCK_TEMPLATE
    for tok, val in (("@KSm1@", k_small - 1), ("@KSp1@", k_small + 1),
                     ("@KSp2@", k_small + 2), ("@KS@", k_small),
                     ("@KLm1@", k_large - 1), ("@KL@", k_large)):
        t = t.replace(tok, str(val))
    if "@K" in t:
        raise RuntimeError("reduce_generator S1c render: residual @K token.")
    return t


def _reduce_params_for_struct(struct: dict) -> Tuple[int, int]:
    """Parameter source (STRICT domain; OD-s1c-2/3/5).

    mixed_bit -> (class_map[0], class_map[1]) from the S1b metadata, with
    impossible-today fail-fasts (k_large must equal final_K and K_group; the
    only legal mixed map today is (52, 62)). same_bit / single / absent
    metadata -> the LEGACY DEFAULT (52, 62), preserving current output until
    S1d. No guessing, no silent rewrites.
    """
    mode = struct.get("_reduce_class_mode")
    if mode == "mixed_bit":
        cmap = struct.get("_reduce_class_map")
        if cmap is None or len(cmap) != 2:
            raise RuntimeError(
                "reduce_generator S1c: mixed_bit set without a 2-class "
                "class_map (got %r)." % (cmap,))
        k_small, k_large = int(cmap[0]), int(cmap[1])
        final_k = struct.get("_reduce_final_k")
        k_group = struct.get("_reduce_k_group")
        if final_k != k_large:
            raise RuntimeError(
                "reduce_generator S1c: mixed_bit k_large=%d != final_K=%r — "
                "the cls1 side is SYMBOLIC env-K text and requires k_large == "
                "final_K; refusing to render." % (k_large, final_k))
        if k_group != k_large:
            raise RuntimeError(
                "reduce_generator S1c: mixed_bit K_group=%r != k_large=%d — "
                "inconsistent S1b metadata; refusing to render." % (k_group, k_large))
        if (k_small, k_large) != (52, 62):
            raise RuntimeError(
                "reduce_generator S1c: mixed class map (%d, %d) is outside the "
                "supported domain {(52, 62)}; refusing to render." % (k_small, k_large))
        return k_small, k_large
    # same_bit / single / metadata absent: legacy dual-candidate text until S1d.
    return 52, 62


# ---------------------------------------------------------------------------
# Phase 3.5 S1d — normalized one-class shapes (mode-gated; legacy default)
# ---------------------------------------------------------------------------
# PRINCIPLE (OD-s1d-2): one-class emission = GENERATOR-TIME MUX RESOLUTION of
# the legacy datapath — delete the cls read and the unselected candidates,
# resolve each "X_mux = cls ? X : X_cls0;" to the side legacy would select,
# keep EVERYTHING else verbatim (shared-path @KL@/@KLm1@ "62-compatible"
# superset widths, dual-candidate ordering of survivors, the SINGLE_PRIME
# skeleton, the route #if chain). Resolved assignments KEEP the *_mux names
# (OD-s1d-3) so all downstream text stays verbatim. No re-derived formulas.
# Applies ONLY to shiftadd == "none" (OD-s1d-4); shiftadd routes and mixed
# sets emit the legacy/S1c text byte-identically in BOTH modes.

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
                "reduce_generator S1d one-class(%s) derivation: anchor hit %d "
                "times (expected 1): %r..." % (side, n, old[:60]))
    # apply only after all anchors verified (order-independent sites)
    for old, new in subs:
        t = t.replace(old, new)
    for residue in ("const int cls", "cls ?", "_cls0", "PRIME_CLASS"):
        if residue in t:
            raise RuntimeError(
                "reduce_generator S1d one-class(%s): residue %r survived "
                "resolution." % (side, residue))
    if side == "large" and "@KS" in t:
        raise RuntimeError(
            "reduce_generator S1d one-class(large): unexpected @KS token.")
    return t


_REDUCE_ONE_CLASS_LARGE = _derive_one_class("large")
_REDUCE_ONE_CLASS_SMALL = _derive_one_class("small")


def _render_one_class(shape: str, k_class: int) -> str:
    """Render a one-class shape. Class constants come from k_class (= K_group,
    NEVER blindly final_K); the shared-path superset widths stay fixed at
    62/61 exactly as the legacy text uses them (OD-s1d-2)."""
    t = shape
    for tok, val in (("@KSm1@", k_class - 1), ("@KSp1@", k_class + 1),
                     ("@KSp2@", k_class + 2), ("@KS@", k_class),
                     ("@CCm1@", k_class - 1), ("@CC@", k_class),
                     ("@KLm1@", 61), ("@KL@", 62)):
        t = t.replace(tok, str(val))
    if "@K" in t or "@C" in t:
        raise RuntimeError("reduce_generator S1d render: residual token.")
    return t


def gen_reduce_block(struct: dict) -> str:
    """Return the reduce() block for injection at the [reduce] seam.

    legacy mode (default/unset): S1c behavior — mixed parameterized, same-bit/
    single legacy (52, 62) dual-candidate text; byte-identical to 9813216.
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
                "reduce_generator S1d: %s set without a 1-class class_map "
                "(got %r)." % (cmode, cmap))
        k_class = int(cmap[0])
        if k_class not in (52, 62):
            raise RuntimeError(
                "reduce_generator S1d: class %d outside the supported domain "
                "{52, 62}; refusing to render." % k_class)
        if struct.get("_reduce_k_group") != k_class:
            raise RuntimeError(
                "reduce_generator S1d: K_group=%r != class %d — inconsistent "
                "metadata; refusing to render." % (struct.get("_reduce_k_group"), k_class))
        if k_class == 62 and struct.get("_reduce_final_k") != 62:
            raise RuntimeError(
                "reduce_generator S1d: one-class-62 requires final_K == 62 "
                "(symbolic-K side); got final_K=%r." % (struct.get("_reduce_final_k"),))
        shape = _REDUCE_ONE_CLASS_LARGE if k_class == 62 else _REDUCE_ONE_CLASS_SMALL
        return _render_one_class(shape, k_class)
    # legacy mode / mixed-bit / shiftadd routes: S1c path (byte-identical today).
    k_small, k_large = _reduce_params_for_struct(struct)
    return _render_reduce_block(k_small, k_large)


def _s1a_anchor_self_check(template_path: str = _SEG_NTT_CPP_TEMPLATE) -> None:
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
            "reduce_generator S1a anchor self-check: [reduce] markers missing in %s (%s)"
            % (template_path, exc))
    expected = gb + "\n" + _REDUCE_BLOCK_VERBATIM + "\n" + ge
    if src[i:j] != expected:
        raise RuntimeError(
            "reduce_generator S1a anchor self-check FAILED: template [reduce] seam "
            "diverged from _REDUCE_BLOCK_VERBATIM in %s -- refusing to generate."
            % template_path)


_s1a_anchor_self_check()

# Phase 3.5 S1c import-time leg: the parameterized renderer must reproduce the
# verbatim anchor exactly at the legacy point — render(52, 62) == anchor ==
# template seam (the S1a check above). Any drift aborts generation.
if _render_reduce_block(52, 62) != _REDUCE_BLOCK_VERBATIM:
    raise RuntimeError(
        "reduce_generator S1c self-check FAILED: _render_reduce_block(52, 62) "
        "!= _REDUCE_BLOCK_VERBATIM — refusing to generate.")


# ---------------------------------------------------------------------------
# Phase 3.5 S1b — final_K inference + reduce-set classification (METADATA ONLY)
# ---------------------------------------------------------------------------
# Set-relative class semantics (binding user correction): cls0 = the SMALLER
# K_group class PRESENT in the current prime set; cls1 = the LARGER class
# PRESENT. Never "cls0=51" (51/61 are sparse ANCHOR exponents, not classes) and
# never the legacy absolute 0/1 emitted values. This slice changes NO emission:
# the emitted PRIME_CLASS keeps the legacy absolute rule and the reduce() block
# stays the S1a-0 verbatim constant above.


def _k_arith_for(q: int) -> int:
    """Per-prime arithmetic class width (moved VERBATIM from generate_code.py,
    Phase 3.5 S1b; generate_code imports it back for compute_all_params)."""
    bl = q.bit_length()
    if bl <= 52:
        return 52
    elif bl <= 62:
        return 62
    else:
        raise RuntimeError(f"Unsupported prime bit_length={bl} for q={q}.")


@dataclass(frozen=True)
class ReduceClassInfo:
    """Pure metadata for one prime set (struct + stdout carriage ONLY in S1b)."""
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

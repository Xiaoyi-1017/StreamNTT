"""ICBU bind_storage resource-balancing DSE.

Estimate ICBU ping-pong storage pressure (LUTRAM/BRAM/URAM) for a configured
architecture and compare it with a configurable routability proxy guard.  The
module reports storage-recipe recommendations; selection and application are
owned by ``generate_code.py``.

This model covers ``bind_storage`` choices only.  It does not model array
partitioning, implementation optimization, arithmetic routes, X-stage
partitioning, or heterogeneous butterfly-unit realizations.  The default 0.80
guard is a planning proxy rather than a physical-routing guarantee.

Model layers (self-validating against measured csynth, N1024/BU8): (1) closed-form arch quantities; (2) deterministic
primitive packing; (3) empirical calibration (CALIB + optional background from csynth reports). Uncertain background
terms (shiftadd mux-tree LUT, X-stage DSP, TAPA FIFO, HBM/DRAM, reorder/twiddle memory) are NAMED background, never
folded into the ICBU storage levers.

Stdlib-only except `from code_generator.icbu_generator import parse_storage_count_form` and the
stdlib-only leaf `dse.device_profiles` (neither imports this module back)."""
from __future__ import annotations

import json
import math
import os
from typing import NamedTuple, Optional

from code_generator.icbu_generator import parse_storage_count_form
from dse.device_profiles import get_device_profile

__all__ = [
    "BOARD_U55C", "DEFAULT_GUARD", "CALIBRATION_ROOT", "CANDIDATE_RECIPES", "RECIPE_COUNTFORM", "RECIPE_CLASS", "CALIB",
    "BRAM_BANK_CALIB", "BackgroundKey", "RecipeKey",
    "Arch", "arch_quantities", "uram_per_bank", "bram18k_per_bank", "bram18k_per_bank_calibrated",
    "estimate_icbu", "guard_status", "Recommendation", "recommend", "load_calibration",
    "lookup_calibration", "lookup_background_keyed", "lookup_recipe_keyed", "format_report",
]

# U55C available-resource profile from the shared device-profile definition.
BOARD_U55C = get_device_profile("u55c").as_board()
DEFAULT_GUARD = 0.80                      # PROXY routability guard; NOT the RapidStream ~86% slot threshold.
CALIBRATION_ROOT = os.path.join(os.path.dirname(__file__), "calibration")

RES_KEYS = ("URAM", "BRAM_18K", "DSP", "LUT", "FF")
MEM_RES = ("URAM", "BRAM_18K")           # the bind_storage-balancing-relevant resources

# recipe name -> count form (L<l>B<b>U<u>); count form is fed to icbu_generator.parse_storage_count_form.
RECIPE_COUNTFORM = {
    "2U2B": "L0B2U2", "4U": "L0B0U4", "4B": "L0B4U0",
    "4L": "L4B0U0", "3U1B": "L0B1U3", "3U1L": "L1B0U3",
}
CANDIDATE_RECIPES = ("2U2B", "4U", "4B", "4L", "3U1B")   # 3U1L optional; excluded from the default candidate set.

# Only calibrated recipes may become strong recommendations.  Closed-form and
# exploratory recipes remain informational until measured LUT/FF data exists.
RECIPE_CLASS = {"2U2B": "calibrated", "3U1B": "calibrated", "4U": "closed-form", "4B": "closed-form",
                "4L": "exploratory", "3U1L": "exploratory"}

# Empirical per-bf_unit module totals (LUT/FF compute-dominated; DSP recipe-invariant).
# 3U1B/4L LUT/FF = None until a csynth build calibrates them. URAM/BRAM_18K come from Layer-2 packing, not CALIB.
CALIB = {
    "4U":   {"LUT": 7355, "FF": 5549, "DSP": 30},
    "2U2B": {"LUT": 7355, "FF": 5678, "DSP": 30},
    "4B":   {"LUT": 7346, "FF": 7656, "DSP": 30},
    "3U1B": {"LUT": None, "FF": None, "DSP": 30},
    "4L":   {"LUT": None, "FF": None, "DSP": 30},
    "3U1L": {"LUT": None, "FF": None, "DSP": 30},
}


# ---------------------------------------------------------------------------
# Layer 1: closed-form architecture quantities (exact)
# ---------------------------------------------------------------------------
class Arch(NamedTuple):
    N: int
    BU: int
    CH: int
    K: int
    NUM_CORE: int
    num_l_stage: int
    bf_unit_instances: int
    bank_depth: int
    icbu_bit_volume: int


def arch_quantities(N: int, BU: int, CH: int, K: int, NUM_CORE: int) -> Arch:
    """Closed-form ICBU architecture quantities. bf_unit_instances = num_l_stage*BU*NUM_CORE (validated 48 @
    N1024/BU8/NUM_CORE1; the NUM_CORE multiplier is confirmed only at NUM_CORE=1 -> calibrate for >1). bank_depth =
    N/(4*BU) (= DEPTH/2); word width = K bits (Data = ap_uint<K>)."""
    logN = int(math.log2(N))
    logBU = int(math.log2(BU))
    num_l_stage = logN - logBU - 1
    instances = num_l_stage * BU * NUM_CORE
    bank_depth = N // (4 * BU)
    return Arch(N, BU, CH, K, NUM_CORE, num_l_stage, instances, bank_depth,
                instances * 4 * bank_depth * K)


# ---------------------------------------------------------------------------
# Layer 2: deterministic primitive packing (per bank; width K, depth bank_depth)
# ---------------------------------------------------------------------------
def uram_per_bank(K: int, depth: int) -> int:
    """URAM primitives for one bank: ceil(K/72) wide x ceil(depth/4096) deep (=1 for K<=72, depth<=4096)."""
    return math.ceil(K / 72) * math.ceil(depth / 4096)


def bram18k_per_bank(K: int, depth: int) -> int:
    """BRAM_18K primitives for one bank: ceil(K/36) wide x ceil(depth/512) deep (=2 for K in [37,72], depth<=512)."""
    return math.ceil(K / 36) * math.ceil(depth / 512)


# Calibrated BRAM_18K per B bank from RapidStream vertex measurements.
# Domain HARD-LIMITED: K=62, NUM_CORE=1, depth in {4096,2048,1024} (per-instance 28/14/8 over 2 B banks; 3U1B
# 14/inst over 1 B bank cross-checks 14/bank @4096). csynth bf_unit BRAM is volatile -> NEVER calibrate from it.
# Do NOT generalize to other K/QK; outside domain -> conservative closed-form + "uncalibrated/conservative".
BRAM_BANK_CALIB = {(62, 4096): 14, (62, 2048): 7, (62, 1024): 4}


def bram18k_per_bank_calibrated(K: int, depth: int, NUM_CORE: int = 1):
    """(value, 'calibrated') inside the RapidStream-calibrated domain; else (closed-form, 'uncalibrated/conservative')."""
    if NUM_CORE == 1 and (K, depth) in BRAM_BANK_CALIB:
        return BRAM_BANK_CALIB[(K, depth)], "calibrated"
    return bram18k_per_bank(K, depth), "uncalibrated/conservative"


def estimate_icbu(arch: Arch, recipe: str) -> dict:
    """ICBU (bf_unit) totals for a recipe. URAM/BRAM_18K/DSP from Layer-2 packing * instances; LUT/FF from CALIB
    (per recipe) * instances (None if not calibrated). LUTRAM banks reported separately (LUT-based; approximate)."""
    impls = parse_storage_count_form(RECIPE_COUNTFORM[recipe])   # 4-tuple mem0..3
    upb = uram_per_bank(arch.K, arch.bank_depth)
    bpb, packing = bram18k_per_bank_calibrated(arch.K, arch.bank_depth, arch.NUM_CORE)
    n = arch.bf_unit_instances
    uram = sum(upb for i in impls if i == "uram") * n
    bram = sum(bpb for i in impls if i == "bram") * n
    lutram_banks = sum(1 for i in impls if i == "lutram") * n
    cal = CALIB.get(recipe, {})
    lut = None if cal.get("LUT") is None else cal["LUT"] * n
    ff = None if cal.get("FF") is None else cal["FF"] * n
    return {
        "URAM": uram, "BRAM_18K": bram, "DSP": cal.get("DSP", 0) * n,
        "LUT": lut, "FF": ff,
        "bit_volume": arch.icbu_bit_volume,
        "primitive_count": uram + bram,
        "lutram_banks": lutram_banks,
        "packing": packing,                       # 'calibrated' | 'uncalibrated/conservative' (BRAM bank source)
    }


# ---------------------------------------------------------------------------
# Guard + recommender
# ---------------------------------------------------------------------------
def guard_status(total: int, res: str, board: dict, guard: float) -> str:
    """OVER (>= guard*board), NEAR (>= 0.95*guard*board), else UNDER."""
    g = guard * board[res]
    if total >= g:
        return "OVER"
    if total >= 0.95 * g:
        return "NEAR"
    return "UNDER"


class Recommendation(NamedTuple):
    summary: str
    rows: list
    calibrated: bool
    # Machine-readable decision fields. `best` is set only by a strong D2
    # recommendation: calibrated background plus a calibrated candidate under
    # both memory-resource guards.
    best: Optional[str] = None
    decision: str = ""
    reason: str = ""


def _totals(arch: Arch, recipe: str, background: Optional[dict]):
    icbu = estimate_icbu(arch, recipe)
    tot = {}
    for res in RES_KEYS:
        if icbu[res] is None:
            tot[res] = None
        else:
            tot[res] = icbu[res] + (0 if background is None else background.get(res, 0))
    return icbu, tot


def _max_mem_util(arch: Arch, recipe: str, background: Optional[dict], board: dict) -> float:
    _, tot = _totals(arch, recipe, background)
    return max(tot[r] / board[r] for r in MEM_RES if tot[r] is not None)


def _delay_risk(recipe: str) -> str:
    impls = parse_storage_count_form(RECIPE_COUNTFORM[recipe])
    return "URAM-present(write_limit>=1)" if "uram" in impls else "none"


def _row(arch: Arch, recipe: str, background: Optional[dict], board: dict, guard: float) -> dict:
    icbu, tot = _totals(arch, recipe, background)
    status = {r: (guard_status(tot[r], r, board, guard) if tot[r] is not None else "n/a") for r in RES_KEYS}
    return {
        "recipe": recipe,
        "schedule": "0-%d:%s:none" % (arch.num_l_stage - 1, RECIPE_COUNTFORM[recipe]),
        "LUT": icbu["LUT"], "FF": icbu["FF"], "BRAM_18K": icbu["BRAM_18K"],
        "URAM": icbu["URAM"], "DSP": icbu["DSP"],
        "bit_volume": icbu["bit_volume"], "primitive_count": icbu["primitive_count"],
        "status": status, "delay_risk": _delay_risk(recipe), "applyable": "yes(R-uniform)",
        "class": RECIPE_CLASS[recipe],
    }


def _info_candidates(arch: Arch, background, board, n: int = 2) -> str:
    order = sorted(CANDIDATE_RECIPES, key=lambda r: _max_mem_util(arch, r, background, board))
    return ", ".join("%s(%s)" % (r, RECIPE_CLASS[r]) for r in order[:n])


def recommend(arch: Arch, background: Optional[dict], board: dict, guard: float,
              configured: str = "2U2B") -> Recommendation:
    """D0-D4 recommendation policy. Report-only; never auto-applies. Strong recommendations
    require calibrated background + a `calibrated`-class candidate under guard on BOTH mem resources; otherwise
    output is INFORMATIONAL. Background alone over guard -> not solvable by ICBU bind_storage (suppress)."""
    calibrated = background is not None
    rows = [_row(arch, r, background, board, guard) for r in CANDIDATE_RECIPES]
    cfg_max = _max_mem_util(arch, configured, background, board)
    if not calibrated:
        # D0/D1-informational: ICBU-only LOWER BOUND; never strong.
        if cfg_max < guard:
            return Recommendation(
                "ICBU-only lower bound under guard for %s (max mem util %.1f%%); background not calibrated; "
                "informational only" % (configured, 100 * cfg_max), rows, False,
                best=None, decision="D0",
                reason="background not calibrated; ICBU-only lower bound under guard; informational only")
        return Recommendation(
            "informational (background NOT calibrated -> ICBU-only LOWER BOUND): configured %s lower bound "
            "over guard; candidates by lower-bound util: %s; no strong recommendation"
            % (configured, _info_candidates(arch, None, board)), rows, False,
            best=None, decision="D0",
            reason="background not calibrated; ICBU-only lower bound over guard; no strong recommendation")
    # D4: background alone over guard on any mem resource -> ICBU storage cannot fix it.
    bg_over = [r for r in MEM_RES if background.get(r, 0) >= guard * board[r]]
    if bg_over:
        return Recommendation(
            "not solvable by ICBU bind_storage: background alone over guard on %s; recipe recommendation "
            "suppressed; other architecture or configuration changes are outside this storage model"
            % "/".join(bg_over),
            rows, True,
            best=None, decision="D4",
            reason="background alone over guard on %s; not solvable by ICBU bind_storage" % "/".join(bg_over))
    # D1: calibrated total under guard.
    if cfg_max < guard:
        return Recommendation(
            "no action: configured recipe %s within guard (calibrated max mem util %.1f%%)"
            % (configured, 100 * cfg_max), rows, True,
            best=None, decision="D1",
            reason="configured recipe %s within guard (max mem util %.1f%%)" % (configured, 100 * cfg_max))
    # D2: a calibrated-class candidate fits BOTH mem guards -> strong recommendation.
    strong = [r for r in CANDIDATE_RECIPES if RECIPE_CLASS[r] == "calibrated"
              and _max_mem_util(arch, r, background, board) < guard]
    if strong:
        best = min(strong, key=lambda r: _max_mem_util(arch, r, background, board))
        over = [r for r in MEM_RES if (estimate_icbu(arch, configured)[r] or 0) + background.get(r, 0)
                >= guard * board[r]]
        return Recommendation(
            "rebalance: configured %s over guard on %s -> RECOMMEND %s (class calibrated, confidence calibrated, "
            "max mem util %.1f%%, delay_risk %s)"
            % (configured, "/".join(over), best, 100 * _max_mem_util(arch, best, background, board),
               _delay_risk(best)), rows, True,
            best=best, decision="D2",
            reason="configured %s over guard on %s; calibrated candidate %s fits both mem guards"
                   % (configured, "/".join(over), best))
    # D3: no calibrated candidate fits (e.g. BRAM and URAM both over) -> no strong recommendation.
    return Recommendation(
        "no single calibrated bind_storage recipe can be strongly recommended (mem guard not satisfiable by "
        "calibrated candidates); exploratory recipes (4L/3U1L) require measured validation", rows, True,
        best=None, decision="D3",
        reason="configured over guard; no calibrated candidate fits both mem guards")


# ---------------------------------------------------------------------------
# Calibration: manifest + csynth background (large-N evidence; may not exist)
# ---------------------------------------------------------------------------
_REQUIRED_MANIFEST = ("N", "BU", "CH", "K", "prime_set", "shiftadd",
                      "target_clock", "source_phase", "included_reports")


class BackgroundKey(NamedTuple):
    N: int
    BU: int
    CH: int
    K: int
    source_phase_tag: str
    prime_mode: str            # 'single' | 'multi'
    prime_count: int
    shiftadd_mode: str


class RecipeKey(NamedTuple):
    background: BackgroundKey
    storage_recipe: str        # missing/null in manifest -> '2U2B' DEFAULT ONLY; non-default must be explicit


def _manifest_keys(man: dict):
    """Derive (BackgroundKey, RecipeKey) from a manifest, or None if unkeyable.

    source_phase_tag must be an EXPLICIT manifest field (no substring inference from free-text source_phase).
    prime_mode from prime_set.mode first token ('single'|'multi'); prime_count from prime_set.count.
    storage_recipe missing/null -> default '2U2B' ONLY (never infer non-default from absence)."""
    tag = man.get("source_phase_tag")
    ps = man.get("prime_set") or {}
    mode = str(ps.get("mode", "")).split("-")[0]
    if not tag or mode not in ("single", "multi") or "count" not in ps:
        return None
    bg = BackgroundKey(man["N"], man["BU"], man["CH"], man["K"], tag, mode, ps["count"], man["shiftadd"])
    return bg, RecipeKey(bg, man.get("storage_recipe") or "2U2B")


def _parse_csynth_total(rpt_path: str) -> Optional[dict]:
    """Parse the 'Total' row of a Vitis csynth.rpt Utilization Estimates table.
    Columns: | Name | BRAM_18K | DSP | FF | LUT | URAM |."""
    try:
        with open(rpt_path) as f:
            for line in f:
                if line.lstrip().startswith("|Total"):
                    c = [x.strip() for x in line.split("|")]
                    if len(c) >= 7:
                        return {"BRAM_18K": int(c[2]), "DSP": int(c[3]), "FF": int(c[4]),
                                "LUT": int(c[5]), "URAM": int(c[6])}
    except (OSError, ValueError):
        return None
    return None


def load_calibration(case_dir: str) -> dict:
    """Read <case_dir>/manifest.json (+ top csynth.rpt) -> {'manifest', 'background'}.

    Missing manifest / required field -> ValueError (explains the reason). background = design_total - icbu_model
    (MEM_RES) when a top work.out/report/csynth.rpt + a known storage_recipe are present; otherwise None (ICBU-only
    caveat, K3). NUM_CORE taken from manifest['NUM_CORE'] if present, else 1 (flag for >1)."""
    mpath = os.path.join(case_dir, "manifest.json")
    if not os.path.exists(mpath):
        raise ValueError("calibration manifest missing: %s (required fields: %s)"
                         % (mpath, ", ".join(_REQUIRED_MANIFEST)))
    with open(mpath) as f:
        man = json.load(f)
    miss = [k for k in _REQUIRED_MANIFEST if k not in man]
    if miss:
        raise ValueError("calibration manifest %s missing required field(s): %s"
                         % (mpath, ", ".join(miss)))
    background = None
    pre = man.get("background_design_minus_icbu")
    if isinstance(pre, dict) and all(k in pre for k in RES_KEYS):
        background = {k: pre[k] for k in RES_KEYS}      # measured (RapidStream) precomputed background, preferred
    recipe = man.get("storage_recipe")
    top = os.path.join(case_dir, "work.out", "report", "csynth.rpt")
    if background is None and recipe in RECIPE_COUNTFORM and os.path.exists(top):
        dt = _parse_csynth_total(top)
        if dt is not None:
            arch = arch_quantities(man["N"], man["BU"], man["CH"], man["K"], man.get("NUM_CORE", 1))
            icbu = estimate_icbu(arch, recipe)
            background = {r: max(0, dt[r] - (icbu[r] or 0)) for r in MEM_RES}
            for r in ("DSP", "LUT", "FF"):
                background[r] = max(0, dt[r] - (icbu[r] or 0))
    keys = _manifest_keys(man)
    return {"manifest": man, "background": background,
            "background_key": keys[0] if keys else None,
            "recipe_key": keys[1] if keys else None,
            "case_dir": case_dir}


def _scan_calibration(root: str):
    if not os.path.isdir(root):
        return
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        if not os.path.exists(os.path.join(d, "manifest.json")):
            continue
        try:
            yield load_calibration(d)
        except ValueError:
            continue


def _coarse_matches(arch: Arch, root: str = CALIBRATION_ROOT) -> list:
    return [rec for rec in _scan_calibration(root)
            if (rec["manifest"].get("N"), rec["manifest"].get("BU"),
                rec["manifest"].get("CH"), rec["manifest"].get("K")) == (arch.N, arch.BU, arch.CH, arch.K)]


def lookup_calibration(arch: Arch, root: str = CALIBRATION_ROOT) -> Optional[dict]:
    """COARSE lookup by (N,BU,CH,K). Exactly one match -> record; zero OR AMBIGUOUS (>1) -> None (no silent
    first-match; format_report renders ambiguity candidates). Keyed lookups: lookup_background_keyed /
    lookup_recipe_keyed (fail-fast on ambiguity)."""
    m = _coarse_matches(arch, root)
    return m[0] if len(m) == 1 else None


def _keyed_lookup(matches: list, key, kind: str) -> Optional[dict]:
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError("ambiguous %s lookup for %s: candidates %s"
                         % (kind, key, ", ".join(r["case_dir"] for r in matches)))
    return matches[0]


def lookup_background_keyed(key: BackgroundKey, root: str = CALIBRATION_ROOT) -> Optional[dict]:
    """Exactly-one record per full BackgroundKey (never mixes source_phase_tag); >1 -> ValueError listing dirs."""
    return _keyed_lookup([r for r in _scan_calibration(root) if r["background_key"] == key], key, "background")


def lookup_recipe_keyed(key: RecipeKey, root: str = CALIBRATION_ROOT) -> Optional[dict]:
    """Exactly-one record per RecipeKey (BackgroundKey + storage_recipe); >1 -> ValueError listing dirs."""
    return _keyed_lookup([r for r in _scan_calibration(root) if r["recipe_key"] == key], key, "recipe")


# ---------------------------------------------------------------------------
# Report formatter (stdout; report mode)
# ---------------------------------------------------------------------------
def format_report(arch: Arch, calib_record: Optional[dict], board: dict, guard: float,
                  configured: str = "2U2B", device: str = "u55c") -> str:
    """Render the ICBU bind_storage DSE table + recommendation. Pure string; the caller (generate_code.py report
    mode) only prints it -> generated artifacts are unchanged.

    `device` (default "u55c") labels the selected capacity profile. For a non-u55c device the caller MUST pass
    that device's capacity `board` AND calib_record=None
    (calibration manifests are u55c-flow evidence; never reuse them cross-device) -- the report then adds a
    CAPACITY-ONLY status line and relabels u55c BRAM packing as an explicit fallback."""
    background = calib_record["background"] if calib_record else None
    rec = recommend(arch, background, board, guard, configured)
    packing = bram18k_per_bank_calibrated(arch.K, arch.bank_depth, arch.NUM_CORE)[1]
    _dev = device.strip().lower()
    packing_display = (packing if (_dev == "u55c" or packing != "calibrated")
                       else "calibrated(u55c-fallback)")
    lines = ["=== ICBU bind_storage resource DSE ==="]
    lines.append("arch: N=%d BU=%d CH=%d K=%d NUM_CORE=%d | num_l_stage=%d bf_unit_instances=%d bank_depth=%d bitvol=%d"
                 % (arch.N, arch.BU, arch.CH, arch.K, arch.NUM_CORE, arch.num_l_stage,
                    arch.bf_unit_instances, arch.bank_depth, arch.icbu_bit_volume))
    lines.append("board=%s guard=%.2f (PROXY routability; NOT the RapidStream ~86%% slot threshold) | background=%s"
                 % (_dev.upper(), guard,
                    "calibrated" if rec.calibrated else "NOT calibrated (ICBU-only LOWER BOUND)"))
    if _dev != "u55c":
        lines.append("device %s: CAPACITY-ONLY profile, UNCALIBRATED for this device (calibration manifests, "
                     "BRAM bank packing and CALIB LUT/FF are u55c-flow evidence -> background NOT used; "
                     "u55c-derived packing figures are FALLBACK estimates; recommendations stay informational)"
                     % _dev)
    lines.append("BRAM bank packing: %s | confidence: %s"
                 % (packing_display, "calibrated" if (rec.calibrated and packing == "calibrated")
                    else ("calibrated packing / uncalibrated background" if packing == "calibrated"
                          else "uncalibrated conservative")))
    if calib_record is None:
        amb = _coarse_matches(arch)
        if len(amb) > 1:
            lines.append("coarse (N,BU,CH,K) lookup AMBIGUOUS -> no background used; candidates: "
                         + ", ".join(r["case_dir"] for r in amb)
                         + " (use lookup_background_keyed / lookup_recipe_keyed)")
    elif calib_record.get("manifest", {}).get("functional_status_source") == "user_asserted":
        lines.append("calibration functional status = user_asserted (confidence lower_than_log_verified)")
    lines.append("%-6s %-12s %-15s %9s %9s %8s %5s %6s %6s %-13s %-26s" % (
        "recipe", "class", "schedule", "ICBU_LUT", "ICBU_FF", "BRAM18K", "URAM", "DSP", "prim",
        "guard U/B", "delay_risk"))
    for r in rec.rows:
        lines.append("%-6s %-12s %-15s %9s %9s %8d %5d %6d %6d %-13s %-26s" % (
            r["recipe"], r["class"], r["schedule"],
            "n/a" if r["LUT"] is None else r["LUT"], "n/a" if r["FF"] is None else r["FF"],
            r["BRAM_18K"], r["URAM"], r["DSP"], r["primitive_count"],
            "%s/%s" % (r["status"]["URAM"], r["status"]["BRAM_18K"]), r["delay_risk"]))
    lines.append("recommendation: " + rec.summary)
    return "\n".join(lines)

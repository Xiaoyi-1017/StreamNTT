"""Shift-add header and arithmetic generation for StreamNTT."""

import os


def _shiftadd_naf(n):
    """Non-adjacent form of an integer; same algorithm as
    shiftadd_profile_report.py."""
    if n == 0:
        return []
    s = 1 if n > 0 else -1
    n = abs(n)
    out = []
    i = 0
    while n != 0:
        if n & 1:
            zi = 2 - (n & 3)
            n -= zi
            out.append((s * zi, i))
        n //= 2
        i += 1
    return sorted(out, key=lambda x: x[1])

def _shiftadd_ell(q):
    bl = q.bit_length()
    if bl in (31, 32):        # anchor-31 family (K_group 32)
        return 31
    if bl in (51, 52):
        return 51
    if bl in (61, 62):
        return 61
    if bl in (63, 64):        # anchor-63 family (K_group 64)
        return 63
    raise ValueError(f"shift-add: unsupported bit_length {bl} for q={q}.")

def _shiftadd_ceil_log2(n):
    if n <= 1:
        return 0
    return (n - 1).bit_length()

def _shiftadd_signed_width(a, b):
    """Smallest w such that -2^(w-1) <= a and b <= 2^(w-1) - 1."""
    pos_need = _shiftadd_ceil_log2(b + 1) if b > 0 else 0
    neg_need = _shiftadd_ceil_log2(-a) if a < 0 else 0
    return max(1, max(pos_need, neg_need) + 1)

def _shiftadd_fold_step(a, b, ell, alpha):
    M = 1 << ell
    h_min = a // M
    h_max = b // M
    if alpha >= 0:
        ha_lo = h_min * alpha
        ha_hi = h_max * alpha
    else:
        ha_lo = h_max * alpha
        ha_hi = h_min * alpha
    return ha_lo, (M - 1) + ha_hi

def _shiftadd_q_anchor(aliases_q):
    """q-derived anchor for the shift-add batch. Returns (B, K_group, K_spread).

    K_i = q_i.bit_length(); K_group = max(K_i); K_spread = max - min. B (the sparse anchor
    exponent / shared fold shift M = 2**B) is selected from the recognized anchor family,
    not from min(K_i). B is not K_env (the HLS Data2 container width) or K_group.
    Only K_spread < 3 (compatible-width) groups are accepted; wider spreads raise.
    """
    # B is chosen from recognized sparse-prime anchor families rather than
    # min(q.bit_length()). For q = 2**e +/- small terms, q.bit_length() can be
    # e+1, so min(bit_length) can select the wrong fold anchor. Examples:
    #     TI_9  = 2**61 + 2**23 + 2**21 + 1  -> bit_length 62, but the anchor B should be 61.
    #     TII_33                             -> bit_length 52, but the anchor B should be 51.
    # _shiftadd_ell is the single anchor-family recognizer for this arithmetic path.
    Ks = [q.bit_length() for _alias, q in aliases_q]
    K_group, K_min = max(Ks), min(Ks)
    K_spread = K_group - K_min
    if K_spread >= 3:
        raise ValueError(
            f"shift-add: unsupported mixed-width group (K_spread={K_spread} >= 3); "
            f"bit_lengths={sorted(Ks)}.")
    # Anchor-first per-prime selection; B is the minimum anchor over the batch.
    B = min(_shiftadd_ell(q) for _alias, q in aliases_q)
    return B, K_group, K_spread


def _resolve_anchor(aliases_q, recipe=None):
    """Resolve the batch anchor and check recipe/helper agreement.

    Recipe-governed path (``recipe`` present AND ``recipe.batch_legal_shiftadd``):
    the shift-add batch anchor comes FROM the recipe; ``_shiftadd_q_anchor`` is
    invoked only to assert agreement (the recipe consistency guard pins
    ``(b_anchor, anchor_k_group, k_spread) == _shiftadd_q_anchor(...)``, so this
    never trips for a legal known/in-family set). Fallback (``recipe`` is None,
    or NOT batch-legal -> custom / k_spread>=3): ``_shiftadd_q_anchor`` is the
    decision source EXACTLY as before, including its k_spread>=3 ValueError. The
    returned ``(B, K_group, K_spread)`` is byte-identical to the legacy helper
    either way -- the recipe supplies the SAME integers, never new ones.

    The recipe is consumed by DUCK-TYPING (attribute reads); shiftadd_generator
    stays a leaf and never imports reduce_recipe (no import cycle)."""
    if recipe is not None and getattr(recipe, "batch_legal_shiftadd", False):
        got = _shiftadd_q_anchor(aliases_q)
        want = (recipe.b_anchor, recipe.anchor_k_group, recipe.k_spread)
        if got != want:
            raise RuntimeError(
                "shiftadd anchor drift: _shiftadd_q_anchor %r != recipe "
                "(b_anchor, anchor_k_group, k_spread) %r" % (got, want))
        return want
    return _shiftadd_q_anchor(aliases_q)


def _shiftadd_envelope_widths(aliases_q, K_env, fold_depth=2, *, recipe=None):
    """Compute (WFold1, WFold2, WFold3, WFinal, ell, all_FN_OK, per_prime_info).

    q-derived: anchor ell = B = min(K_i); each prime is validated over its own
    product range X in [0, (q_i-1)^2]; emitted widths are the max over primes.
    fold_depth selects the per-prime gate (2 -> F2_OK, 3 -> F3_OK). K_env is the
    HLS Data2 container width only and does NOT gate feasibility.
    """
    if fold_depth not in (2, 3):
        raise RuntimeError(
            f"_shiftadd_envelope_widths: fold_depth must be 2 or 3 "
            f"(got {fold_depth!r}).")
    B, K_group, K_spread = _resolve_anchor(aliases_q, recipe)
    ell = B  # sparse anchor exponent = min(K_i); shared fold shift M = 2**B
    f1_widths, f2_widths, f3_widths = [], [], []
    all_FN = True
    info = []
    for alias, q in aliases_q:
        # Per-prime product range: A,B in [0,q) => X in [0,(q-1)^2].
        # K_env is the Data2 container width only and must not gate feasibility.
        Xprod_lo, Xprod_hi = 0, (q - 1) ** 2
        alpha = (1 << ell) - q
        f1_lo, f1_hi = _shiftadd_fold_step(Xprod_lo, Xprod_hi, ell, alpha)
        f2_lo, f2_hi = _shiftadd_fold_step(f1_lo, f1_hi, ell, alpha)
        WF1 = _shiftadd_signed_width(f1_lo, f1_hi)
        WF2 = _shiftadd_signed_width(f2_lo, f2_hi)
        F2_OK = (f2_lo >= -q) and (f2_hi < 2 * q)
        d = {
            "alias": alias, "q": q, "alpha": alpha,
            "alpha_naf": _shiftadd_naf(alpha),
            "WFold1": WF1, "WFold2": WF2,
            "f2_lo": f2_lo, "f2_hi": f2_hi, "F2_OK": F2_OK,
        }
        f1_widths.append(WF1)
        f2_widths.append(WF2)
        if fold_depth == 3:
            f3_lo, f3_hi = _shiftadd_fold_step(f2_lo, f2_hi, ell, alpha)
            WF3 = _shiftadd_signed_width(f3_lo, f3_hi)
            F3_OK = (f3_lo >= -q) and (f3_hi < 2 * q)
            d.update({"WFold3": WF3, "f3_lo": f3_lo, "f3_hi": f3_hi,
                      "F3_OK": F3_OK})
            f3_widths.append(WF3)
            if not F3_OK:
                all_FN = False
        else:
            if not F2_OK:
                all_FN = False
        info.append(d)
    WFold1 = max(f1_widths)
    WFold2 = max(f2_widths)
    if fold_depth == 3:
        WFold3 = max(f3_widths)
        WFinal = WFold3
    else:
        WFold3 = WFold2  # harmless fill when fold-3 is unused
        WFinal = WFold2
    return WFold1, WFold2, WFold3, WFinal, ell, all_FN, info


def _shiftadd_per_prime_F2_pos_only(p):
    return (p["f2_lo"] >= 0) and (p["f2_hi"] < 2 * p["q"])

def _shiftadd_per_prime_F3_pos_only(p):
    return (p.get("f3_lo", 1) >= 0) and (p.get("f3_hi", 0) < 2 * p["q"])


def _shiftadd_resolve_fold_and_correction(aliases_q, K_env,
                                           fold_depth_req,
                                           correction_req,
                                           *, recipe=None):
    """Decision rules per fold/correction resolver spec:

    Returns (fold_depth_int, correction_mode_str, report_lines).

    fold_depth_req in {2, 3, "auto"}; correction_req in
    {"positive_only", "signed", "auto"}.

    Refuses with RuntimeError on unsatisfiable combinations rather than
    silently downgrading.
    """
    report = []
    # Compute F2 envelope first (needed by every branch).
    res_f2 = _shiftadd_envelope_widths(aliases_q, K_env, fold_depth=2, recipe=recipe)
    info_f2 = res_f2[6]
    all_F2 = res_f2[5]
    all_F2_pos_only = all(_shiftadd_per_prime_F2_pos_only(p) for p in info_f2)

    report.append(f"per-prime F2_OK: " + ", ".join(
        f"{p['alias']}={p['F2_OK']}" for p in info_f2))
    report.append(f"all F2_OK: {all_F2}")

    def _need_f3():
        res_f3 = _shiftadd_envelope_widths(aliases_q, K_env, fold_depth=3, recipe=recipe)
        info_f3 = res_f3[6]
        all_F3 = res_f3[5]
        all_F3_pos_only = all(
            _shiftadd_per_prime_F3_pos_only(p) for p in info_f3)
        report.append("per-prime F3_OK: " + ", ".join(
            f"{p['alias']}={p['F3_OK']}" for p in info_f3))
        report.append(f"all F3_OK: {all_F3}")
        report.append("per-prime F3_pos_only_OK: " + ", ".join(
            f"{p['alias']}={_shiftadd_per_prime_F3_pos_only(p)}"
            for p in info_f3))
        report.append(f"all F3_pos_only_OK: {all_F3_pos_only}")
        return all_F3, all_F3_pos_only, info_f3

    if fold_depth_req == 2:
        if not all_F2:
            blockers = [p["alias"] for p in info_f2 if not p["F2_OK"]]
            raise RuntimeError(
                f"fold/correction resolver: shiftadd_fold_depth=2 but not all primes "
                f"are F2_OK. Blockers: {blockers}.")
        if correction_req == "auto":
            chosen_corr = "positive_only"
        elif correction_req == "signed":
            raise RuntimeError(
                "fold/correction resolver: shiftadd_correction='signed' is only "
                "implemented under fold_depth=3. Use fold_depth='auto' or "
                "fold_depth=3, or change correction to 'positive_only'.")
        else:
            chosen_corr = "positive_only"
        report.append(f"chosen fold_depth=2, correction={chosen_corr}")
        return 2, chosen_corr, report

    if fold_depth_req == 3:
        all_F3, all_F3_pos_only, info_f3 = _need_f3()
        if not all_F3:
            blockers = [p["alias"] for p in info_f3 if not p["F3_OK"]]
            raise RuntimeError(
                f"fold/correction resolver: shiftadd_fold_depth=3 but not all primes "
                f"are F3_OK. Blockers: {blockers}.")
        if correction_req == "positive_only":
            if not all_F3_pos_only:
                blockers = [p["alias"] for p in info_f3
                            if not _shiftadd_per_prime_F3_pos_only(p)]
                raise RuntimeError(
                    f"fold/correction resolver: shiftadd_correction='positive_only' is "
                    f"insufficient under fold_depth=3 for primes "
                    f"{blockers}. Need 'signed' or 'auto'.")
            chosen_corr = "positive_only"
        elif correction_req == "signed":
            chosen_corr = "signed"
        else:  # auto
            chosen_corr = "positive_only" if all_F3_pos_only else "signed"
        report.append(f"chosen fold_depth=3, correction={chosen_corr}")
        return 3, chosen_corr, report

    # fold_depth_req == "auto"
    if all_F2:
        # Prefer F2 path when sufficient.
        if correction_req == "positive_only" or correction_req == "auto":
            chosen_corr = "positive_only"
        elif correction_req == "signed":
            raise RuntimeError(
                "fold/correction resolver: shiftadd_correction='signed' explicitly "
                "requested but the batch is F2_OK and signed correction "
                "is only meaningful under fold_depth=3. Use "
                "shiftadd_fold_depth=3 explicitly to override.")
        report.append(f"chosen fold_depth=2, correction={chosen_corr}")
        return 2, chosen_corr, report

    # F2 fails -> try F3
    all_F3, all_F3_pos_only, info_f3 = _need_f3()
    if not all_F3:
        blockers_f2 = [p["alias"] for p in info_f2 if not p["F2_OK"]]
        blockers_f3 = [p["alias"] for p in info_f3 if not p["F3_OK"]]
        raise RuntimeError(
            f"fold/correction resolver: shiftadd_fold_depth='auto': neither F2_OK nor "
            f"F3_OK holds for the whole batch. F2 blockers: {blockers_f2}. "
            f"F3 blockers: {blockers_f3}.")
    if correction_req == "positive_only":
        if not all_F3_pos_only:
            blockers = [p["alias"] for p in info_f3
                        if not _shiftadd_per_prime_F3_pos_only(p)]
            raise RuntimeError(
                f"fold/correction resolver: shiftadd_correction='positive_only' is "
                f"insufficient for these fold-3 primes: {blockers}. Use "
                f"'signed' or 'auto'.")
        chosen_corr = "positive_only"
    elif correction_req == "signed":
        chosen_corr = "signed"
    else:  # auto
        chosen_corr = "positive_only" if all_F3_pos_only else "signed"
    report.append(f"chosen fold_depth=3, correction={chosen_corr}")
    return 3, chosen_corr, report

def _shiftadd_simulate_reduce_shiftadd(X, q, ell, fold_depth, correction):
    """Pure-Python simulation of the generated reduce_shiftadd math.
    Mirrors the kernel's signed-int arithmetic and the unsigned cast at
    the end. Returns the would-be (Data) output (i.e. the value after
    cast to ap_uint<K>).
    """
    M = 1 << ell
    alpha = (1 << ell) - q
    cur = X
    for _ in range(fold_depth):
        # Bit-extract low ell bits and arithmetic-shift right by ell.
        # In Python, `& (M-1)` and `>> ell` reproduce the kernel's
        # `range(ell-1,0)` low-bit extract and signed `>> ell` semantics
        # (Python int has arbitrary precision; this matches the bit pattern
        # of ap_int<W>).
        cur_l = cur & (M - 1)
        cur_h = cur >> ell
        cur = cur_l + cur_h * alpha
    # Correction.
    if correction == "positive_only":
        if cur >= q:
            cur = cur - q
    elif correction == "signed":
        if cur < 0:
            cur = cur + q
        if cur >= q:
            cur = cur - q
    else:
        raise ValueError(f"unknown correction {correction!r}")
    # Final cast to unsigned-K (we use a wide envelope; for the kernel,
    # K is per-case but always >= q.bit_length()). Emulate ap_uint
    # truncation by masking to K bits. We choose K = q.bit_length()+1
    # (loose upper bound) to detect "wraps past q" cleanly.
    K_cast = q.bit_length() + 1  # >= ell+1
    return cur & ((1 << K_cast) - 1)

def _shiftadd_math_self_check(aliases_q, K_env, fold_depth, correction,
                               num_random=2000, log_lines=None, *, recipe=None):
    """Returns (ok, log_lines). ok=True iff every prime passes for every
    sampled X with cur output == X mod q.

    Each prime is sampled over its own q-derived product range X in
    [0, (q-1)^2]; K_env is only the Data2 container width.
    """
    import random
    rng = random.Random(0xC0FFEEC0FFEE)
    if log_lines is None:
        log_lines = []
    log_lines.append(
        f"=== math_self_check fold_depth={fold_depth} "
        f"correction={correction} K_env={K_env} ===")
    all_ok = True
    B_anchor, _kg, _ks = _resolve_anchor(aliases_q, recipe)
    for alias, q in aliases_q:
        ell = B_anchor  # shared fold shift = min(K_i)
        # Per-prime product range (q-derived); K_env is container width only.
        Xprod_max = (q - 1) ** 2
        edges = [0, 1, q - 1, q, q + 1, 2 * q,
                 Xprod_max - 1, Xprod_max,
                 (q - 1) * (q - 1), q * q]
        edges = [x for x in edges if 0 <= x <= Xprod_max]
        # Random product samples and high-edge product samples.
        rand = [rng.randint(0, Xprod_max) for _ in range(num_random)]
        prods = [(rng.randint(0, q - 1) * rng.randint(0, q - 1)) % (Xprod_max + 1)
                 for _ in range(num_random // 4)]
        high_edge = [(q - 1) * rng.randint(q // 2, q - 1)
                     for _ in range(num_random // 4)]
        high_edge = [x for x in high_edge if 0 <= x <= Xprod_max]
        inputs = edges + rand + prods + high_edge
        fails = 0
        first_fail = None
        for X in inputs:
            got = _shiftadd_simulate_reduce_shiftadd(
                X, q, ell, fold_depth, correction)
            expected = X % q
            if got != expected:
                fails += 1
                if first_fail is None:
                    first_fail = (X, got, expected)
        if fails == 0:
            log_lines.append(f"  {alias:<8s} q={q} tested={len(inputs)} OK")
        else:
            all_ok = False
            x, g, e = first_fail
            log_lines.append(
                f"  {alias:<8s} q={q} tested={len(inputs)} FAIL ({fails}); "
                f"first X={x} got={g} expected={e}")
    log_lines.append(f"=== overall: {'PASS' if all_ok else 'FAIL'} ===")
    return all_ok, log_lines

def _shiftadd_slot_assignment(info, max_slots=2, heuristic="asc_exp"):
    """For each prime, partition its non-zero NAF terms into Slot A / B (/ C).

    max_slots=2 uses the original ascending exponent order. max_slots=3
    fills Slot C with the third sparse term and uses descending exponent
    order in the retained emitters.

    Returns:
      slotA, slotB, slotC: lists of (sign, exp) or None per prime. slotC
      is always returned (all-None when max_slots=2 or when no prime has
      a third term); old callers can simply ignore it.
    """
    if max_slots not in (2, 3):
        raise RuntimeError(
            f"_shiftadd_slot_assignment: max_slots must be 2 or 3 "
            f"(got {max_slots!r}).")
    if heuristic not in ("asc_exp", "desc_exp"):
        raise RuntimeError(
            f"_shiftadd_slot_assignment: heuristic must be 'asc_exp' or "
            f"'desc_exp' (got {heuristic!r}).")
    slotA, slotB, slotC = [], [], []
    for inf in info:
        nz = [(s, e) for (s, e) in inf["alpha_naf"] if e != 0]
        if heuristic == "desc_exp":
            nz_sorted = sorted(nz, key=lambda x: -x[1])
        else:
            nz_sorted = sorted(nz, key=lambda x: x[1])
        if len(nz_sorted) > max_slots:
            raise RuntimeError(
                f"Slot allocator: prime {inf['alias']} has {len(nz_sorted)} "
                f"sparse terms but shiftadd_max_slots={max_slots}. "
                f"Bump max_slots or use a different style."
            )
        slotA.append(nz_sorted[0] if len(nz_sorted) >= 1 else None)
        slotB.append(nz_sorted[1] if len(nz_sorted) >= 2 else None)
        slotC.append(nz_sorted[2] if len(nz_sorted) >= 3 else None)
    return slotA, slotB, slotC

def _emit_shiftadd_slot_direct(out_path, K_env, info, header_top,
                               WFold1, WFold2, WFinal, ell,
                               balanced_selector_scope="none",
                               fold_depth=2, WFold3=None,
                               correction="positive_only",
                               max_slots=2,
                               single_prime_normalized=False):
    """Emit reduce_shiftadd(...) for the stable slot-direct path: per-prime
    ``is_<alias>`` predicates derived directly from ``mod_id``, then a
    chained ternary on those predicates picks the active signed candidate
    per slot. No CODE_A / CODE_B lookup tables, no codeA / codeB variables;
    the only per-prime state on the data path is the predicate set.

    Stays a slot-level mux (not a result-level mux): each slot collapses to
    one signed shifted candidate via mod_id-driven gating, then
    ``v = base + stage_a + stage_b`` is shared across primes. One correction
    layer; literal shifts only; shift 0 collapsed into ``base = X_l - X_h``.
    """
    NP = len(info)
    K_high_bits = 2 * K_env - ell

    _slot_heuristic = "desc_exp" if max_slots == 3 else "asc_exp"
    slotA, slotB, slotC = _shiftadd_slot_assignment(
        info, max_slots=max_slots, heuristic=_slot_heuristic)
    has_slot_c = any(t is not None for t in slotC)

    # Per-prime predicate alias: strip non-alphanumerics so e.g. "TI_2" -> "TI2"
    # to match the shape `is_TI2` from the controlling task spec.
    def _sanitize(a):
        return "".join(ch for ch in a if ch.isalnum())

    aliases_clean = [_sanitize(p["alias"]) for p in info]

    def _term_name(c, fold_tag):
        sign, exp = c
        prefix = "p" if sign > 0 else "n"
        return f"{prefix}{exp}_{fold_tag}"

    def _term_expr(c, source_var):
        sign, exp = c
        if sign > 0:
            return f"({source_var} << {exp})"
        return f"-({source_var} << {exp})"

    def _fmt_signed(t):
        if t is None:
            return "0"
        sign, exp = t
        if exp == 0:
            return "0"
        return ("+" if sign > 0 else "-") + f"2^{exp}"

    # Balanced ternary mux tree for the slot selector.
    # Active count <= 3   -> keep the chained ternary priority chain
    #                        (preserves N3/P3-style baseline output).
    # Active count >= 4   -> emit a balanced ternary mux tree IF the
    #                        current slot is in the configured scope:
    #                          * "none"          -> never balance.
    #                          * "all"           -> every slot.
    #                        Pure mux/select; no add tree, no switch/case,
    #                        no one-hot sum.
    # Default fall-through (no predicate matches) is 0 in both forms.
    BALANCED_FANIN_THRESHOLD = 4

    # Map scope -> set of (stage_var, fold_tag) tuples that get balanced.
    # The closure below uses this to decide per call site.
    _BALANCE_SCOPE_TABLE = {
        "none":          set(),
        "all":           {("stage_a_1", "1"), ("stage_b_1", "1"),
                          ("stage_c_1", "1"),
                          ("stage_a_2", "2"), ("stage_b_2", "2"),
                          ("stage_c_2", "2"),
                          ("stage_a_3", "3"), ("stage_b_3", "3"),
                          ("stage_c_3", "3")},
    }
    if balanced_selector_scope not in _BALANCE_SCOPE_TABLE:
        raise RuntimeError(
            f"Unknown balanced_selector_scope "
            f"{balanced_selector_scope!r}; expected one of "
            f"{sorted(_BALANCE_SCOPE_TABLE)}."
        )
    _BALANCED_SLOTS = _BALANCE_SCOPE_TABLE[balanced_selector_scope]

    def _slot_chain(stage_var, slot_per_prime, fold_tag, type_macro, indent="    "):
        """Render the per-prime predicate selector for one slot.

        Selector form is data-dependent on active fan-in AND the configured
        balanced_selector_scope:
          * fan-in <= 3                              -> chained ternary.
          * fan-in >= 4 AND slot in scope            -> balanced mux tree.
          * fan-in >= 4 AND slot NOT in scope        -> chained ternary.
        Default fall-through (no predicate matches) is 0 in both forms.
        """
        # Collect (predicate, candidate-name) for primes with an active term in this slot.
        active = []
        for i, t in enumerate(slot_per_prime):
            if t is None:
                continue
            active.append((f"is_{aliases_clean[i]}", _term_name(t, fold_tag)))

        if not active:
            return [f"{indent}ap_int<{type_macro}> {stage_var} = (ap_int<{type_macro}>)0;"]

        if single_prime_normalized:
            # S4: NUM_PRIMES==1 -> exactly one active term; its is_<alias>
            # predicate is always true -> resolve the mux to the bare candidate.
            _p, c = active[0]
            return [f"{indent}ap_int<{type_macro}> {stage_var} = {c};"]

        slot_in_scope = (stage_var, fold_tag) in _BALANCED_SLOTS
        if (len(active) < BALANCED_FANIN_THRESHOLD) or (not slot_in_scope):
            # ---- Chained ternary (priority chain) ----
            L = [f"{indent}ap_int<{type_macro}> {stage_var} ="]
            for p, c in active:
                L.append(f"{indent}    {p} ? {c} :")
            L.append(f"{indent}                 (ap_int<{type_macro}>)0;")
            return L

        # ---- Balanced ternary mux tree for fan-in >= 4 (slot in scope) ----
        helper_lines = []
        counter = [0]

        def fresh():
            n = counter[0]
            counter[0] += 1
            return f"{stage_var}_n{n}"

        def build(pairs):
            """Emit a balanced sub-mux for ``pairs``; return (var_name, predicate_list)."""
            if len(pairs) == 1:
                p, c = pairs[0]
                name = fresh()
                helper_lines.append(
                    f"{indent}ap_int<{type_macro}> {name} = "
                    f"{p} ? {c} : (ap_int<{type_macro}>)0;"
                )
                return name, [p]
            if len(pairs) == 2:
                (p0, c0), (p1, c1) = pairs
                name = fresh()
                helper_lines.append(
                    f"{indent}ap_int<{type_macro}> {name} = "
                    f"{p0} ? {c0} : ({p1} ? {c1} : (ap_int<{type_macro}>)0);"
                )
                return name, [p0, p1]
            # Recursive split: left-biased halves.
            mid = (len(pairs) + 1) // 2
            left_name, left_preds = build(pairs[:mid])
            right_name, right_preds = build(pairs[mid:])
            or_left = " || ".join(left_preds)
            name = fresh()
            helper_lines.append(
                f"{indent}ap_int<{type_macro}> {name} = "
                f"({or_left}) ? {left_name} : {right_name};"
            )
            return name, left_preds + right_preds

        root, _ = build(active)
        L = list(helper_lines)
        L.append(f"{indent}ap_int<{type_macro}> {stage_var} = {root};")
        return L

    # Unique signed (sign, exp) used across all slots; sorted deterministically.
    union = []
    for c in [t for t in (slotA + slotB + slotC) if t is not None]:
        if c not in union:
            union.append(c)
    union.sort(key=lambda x: (x[1], -x[0]))

    lines = list(header_top)
    lines.append("// Generated shift-add reducer for this prime batch.")
    lines.append(f"// Reduction base: B = 2^{ell}.")
    lines.append("// Shift-add step: W' = W_low + alpha * W_high, where alpha = B - q.")
    lines.append("")
    lines.append(f"constexpr int WFold1 = {WFold1};")
    lines.append(f"constexpr int WFold2 = {WFold2};")
    if fold_depth == 3:
        lines.append(f"constexpr int WFold3 = {WFold3};")
    lines.append(f"constexpr int WFinal = {WFinal};")
    lines.append(f"constexpr int W_v1_h = WFold1 - {ell - 1};")
    if fold_depth == 3:
        lines.append(f"constexpr int W_v2_h = WFold2 - {ell - 1};")
    lines.append("")
    # Helper that emits the fold-1 body lines (predicates, base, candidates,
    # slot mux, v_1) at the requested indent.
    def _emit_fold1_body(indent="    "):
        L = []
        if not single_prime_normalized:
            for i in range(NP):
                L.append(f"{indent}const bool is_{aliases_clean[i]} = (mod_id == (ModId){i});")
            L.append("")
        L.append(f"{indent}ap_uint<{ell}> X_l_u = (ap_uint<{ell}>)X.range({ell - 1}, 0);")
        L.append(
            f"{indent}ap_uint<{K_high_bits}> X_h_u = (ap_uint<{K_high_bits}>)X.range({2 * K_env - 1}, {ell});"
        )
        L.append(f"{indent}ap_int<WFold1> X_l_s = (ap_int<WFold1>)X_l_u;")
        L.append(f"{indent}ap_int<WFold1> X_h_s = (ap_int<WFold1>)X_h_u;")
        L.append(f"{indent}ap_int<WFold1> base_1 = X_l_s - X_h_s;")
        for c in union:
            L.append(f"{indent}ap_int<WFold1> {_term_name(c, '1')} = {_term_expr(c, 'X_h_s')};")
        # Pairwise mux tree for the selected shifted term.
        L.extend(_slot_chain("stage_a_1", slotA, "1", "WFold1", indent=indent))
        L.extend(_slot_chain("stage_b_1", slotB, "1", "WFold1", indent=indent))
        if has_slot_c:
            L.extend(_slot_chain("stage_c_1", slotC, "1", "WFold1", indent=indent))
            L.append(f"{indent}ap_int<WFold1> v_1 = base_1 + stage_a_1 + stage_b_1 + stage_c_1;")
        else:
            L.append(f"{indent}ap_int<WFold1> v_1 = base_1 + stage_a_1 + stage_b_1;")
        return L

    lines.extend([
        "#define USE_REDUCE_SHIFTADD 1",
        "",
        "using ShiftaddPre = ap_int<WFinal + 1>;",
        "",
        "inline Data reduce_shiftadd(Data2 X, ModId mod_id, Data q_cur) {",
        "#pragma HLS INLINE",
        "",
    ])
    lines.extend(_emit_fold1_body(indent="    "))
    lines.append("")

    # Fold 2
    lines.append(f"    ap_uint<{ell}> v_1_l_u = (ap_uint<{ell}>)v_1.range({ell - 1}, 0);")
    lines.append(f"    ap_int<W_v1_h> v_1_h_tight = (ap_int<W_v1_h>)(v_1 >> {ell});")
    lines.append("    ap_int<WFold2> v_1_l_s = (ap_int<WFold2>)v_1_l_u;")
    lines.append("    ap_int<WFold2> v_1_h_s = (ap_int<WFold2>)v_1_h_tight;")
    lines.append("    ap_int<WFold2> base_2 = v_1_l_s - v_1_h_s;")
    for c in union:
        lines.append(f"    ap_int<WFold2> {_term_name(c, '2')} = {_term_expr(c, 'v_1_h_s')};")
    lines.extend(_slot_chain("stage_a_2", slotA, "2", "WFold2"))
    lines.extend(_slot_chain("stage_b_2", slotB, "2", "WFold2"))
    if has_slot_c:
        lines.extend(_slot_chain("stage_c_2", slotC, "2", "WFold2"))
        lines.append("    ap_int<WFold2> v_2 = base_2 + stage_a_2 + stage_b_2 + stage_c_2;")
    else:
        lines.append("    ap_int<WFold2> v_2 = base_2 + stage_a_2 + stage_b_2;")
    lines.append("")

    # Optional fold 3
    if fold_depth == 3:
        lines.append(f"    ap_uint<{ell}> v_2_l_u = (ap_uint<{ell}>)v_2.range({ell - 1}, 0);")
        lines.append(f"    ap_int<W_v2_h> v_2_h_tight = (ap_int<W_v2_h>)(v_2 >> {ell});")
        lines.append("    ap_int<WFold3> v_2_l_s = (ap_int<WFold3>)v_2_l_u;")
        lines.append("    ap_int<WFold3> v_2_h_s = (ap_int<WFold3>)v_2_h_tight;")
        lines.append("    ap_int<WFold3> base_3 = v_2_l_s - v_2_h_s;")
        for c in union:
            lines.append(f"    ap_int<WFold3> {_term_name(c, '3')} = {_term_expr(c, 'v_2_h_s')};")
        lines.extend(_slot_chain("stage_a_3", slotA, "3", "WFold3"))
        lines.extend(_slot_chain("stage_b_3", slotB, "3", "WFold3"))
        if has_slot_c:
            lines.extend(_slot_chain("stage_c_3", slotC, "3", "WFold3"))
            lines.append("    ap_int<WFold3> v_3 = base_3 + stage_a_3 + stage_b_3 + stage_c_3;")
        else:
            lines.append("    ap_int<WFold3> v_3 = base_3 + stage_a_3 + stage_b_3;")
        lines.append("")
        final_var = "v_3"
    else:
        final_var = "v_2"

    lines.append("    // Correction")
    if correction == "signed":
        lines.append(f"    ShiftaddPre r = (ShiftaddPre){final_var};")
        lines.append("    ShiftaddPre q_ext = (ShiftaddPre)q_cur;")
        lines.append("    if (r < 0) {")
        lines.append("        r += q_ext;")
        lines.append("    }")
        lines.append("    if (r >= q_ext) {")
        lines.append("        r -= q_ext;")
        lines.append("    }")
        lines.append("    return (Data)r;")
    else:
        lines.append(f"    ShiftaddPre r = (ShiftaddPre){final_var};")
        lines.append("    ShiftaddPre q_ext = (ShiftaddPre)q_cur;")
        lines.append("    if (r >= q_ext) {")
        lines.append("        r -= q_ext;")
        lines.append("    }")
        lines.append("    return (Data)r;")
    lines.append("}")
    lines.append("")
    lines.append("#endif  // NTT_SHIFTADD_H")
    lines.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Recipe-class mixed-prime emission
# ---------------------------------------------------------------------------
# A mixed shift-add reducer is emitted only when the ReduceRecipe supplies more
# than one incompatible shift-add class. Each class gets an independent helper
# with its own anchor, fold/correction resolver, widths, and global mod_id
# predicates; the wrapper calls every helper and selects only the final Data.
# Single-family batches never enter this path, preserving the stable emitter.

def _shiftadd_anchor_family_of(q):
    """Known anchor family of one prime: 51 (52-class) | 61 (62-class) |
    None (bit_length outside the known families -> custom; legacy path)."""
    bl = q.bit_length()
    if bl in (51, 52):
        return 51
    if bl in (61, 62):
        return 61
    return None


def _shiftadd_known_anchor_families(aliases_q):
    """Sorted known anchor families of a batch ([51] | [61] | [51, 61]) or
    None when ANY prime has an unknown bit length (custom -> legacy path)."""
    fams = set()
    for _alias, q in aliases_q:
        fam = _shiftadd_anchor_family_of(q)
        if fam is None:
            return None
        fams.add(fam)
    return sorted(fams)


def _shiftadd_class_sub_batches(aliases_q):
    """Split a known-family batch into per-class sub-batches in cls order
    (cls0 = 52-class / B=51 FIRST, cls1 = 62-class / B=61). Original prime
    order is preserved inside each class; predicate names stay GLOBAL
    (is_<alias> from the full-batch mod_id index). Returns
    [(k_class, B, sub_aliases_q), ...] with empty classes omitted."""
    subs = []
    for B, k_class in ((51, 52), (61, 62)):
        sub = [(alias, q) for alias, q in aliases_q
               if _shiftadd_anchor_family_of(q) == B]
        if sub:
            subs.append((k_class, B, sub))
    return subs


def _shiftadd_mixed_guard(aliases_q):
    """Class split for the mixed emitters with the {52, 62} domain pinned."""
    subs = _shiftadd_class_sub_batches(aliases_q)
    if [(k, b) for k, b, _s in subs] != [(52, 51), (62, 61)]:
        raise RuntimeError(
            "mixed shiftadd emitter requires exactly the {52, 62} two-family "
            "batch (got classes %r)." % ([(k, b) for k, b, _s in subs],))
    return subs


def _shiftadd_recipe_class_sub_batches(aliases_q, recipe):
    """Return recipe-discovered class sub-batches.

    The ReduceRecipe is the semantic source for mixed mode: class order,
    K_group, and anchor come from recipe.cls_recipes. The sub-batch keeps the
    original prime order, and the emitted predicates keep original mod_id
    values.
    """
    cls_recipes = getattr(recipe, "cls_recipes", None) if recipe is not None else None
    if not cls_recipes:
        return []
    out = []
    for class_id, cr in enumerate(cls_recipes):
        alias_set = set(cr.aliases)
        sub = [(alias, q) for alias, q in aliases_q if alias in alias_set]
        if len(sub) != len(cr.aliases):
            raise RuntimeError(
                "shiftadd mixed recipe class %d (%d-class) aliases %r do not "
                "match the resolved prime batch %r." % (
                    class_id, cr.k_class, cr.aliases, [a for a, _q in aliases_q]))
        out.append((class_id, cr.k_class, cr.b_anchor, sub))
    return out


def _shiftadd_mixed_reduce_sub_batches(aliases_q, recipe):
    """Sub-batches for the experimental mixed reduce path.

    The production path uses recipe.cls_recipes. A direct-helper fallback is
    retained for low-level callers that invoke this module without a recipe.
    """
    subs = _shiftadd_recipe_class_sub_batches(aliases_q, recipe)
    if subs:
        if len(subs) != 2:
            raise RuntimeError(
                "mixed shiftadd reduce currently supports exactly two recipe "
                "classes; got %d classes: %r." % (
                    len(subs), [(k, b) for _cid, k, b, _sub in subs]))
        return subs
    if recipe is not None:
        return []
    legacy_subs = _shiftadd_class_sub_batches(aliases_q)
    if len(legacy_subs) <= 1:
        return []
    if len(legacy_subs) != 2:
        raise RuntimeError(
            "mixed shiftadd reduce direct-call fallback currently supports "
            "exactly two classes; got %r." % (
                [(k, b) for k, b, _sub in legacy_subs],))
    return [(class_id, k_class, b_anchor, sub)
            for class_id, (k_class, b_anchor, sub) in enumerate(legacy_subs)]


def _emit_shiftadd_reduce_header_mixed(out_path, K_env, aliases_q, header_top,
                                       balanced_selector_scope="none",
                                       math_self_check_log_path=None,
                                       *, recipe=None):
    """Emit reduce_shiftadd(...) for a recipe-class-mixed known batch.

    Every per-class helper reuses the SINGLE-FAMILY machinery on its class
    sub-batch (auto fold/correction resolver, q-derived envelope widths, math
    self-check, slot assignment). Per-prime ``is_<alias>`` predicates keep
    GLOBAL mod_id indices, so each class's slot muxes select per-prime
    candidates exactly as in the single-family emitter.

    When ``recipe.cls_recipes`` is
    present, the per-class resolved (fold, correction) come FROM the recipe
    and the live resolution is the assert shim; ``recipe=None`` resolves live
    and emits byte-identically.
    """
    if balanced_selector_scope not in ("none", "all"):
        raise RuntimeError(
            f"Unknown balanced_selector_scope "
            f"{balanced_selector_scope!r}; expected one of "
            f"['all', 'none'].")
    subs = _shiftadd_mixed_reduce_sub_batches(aliases_q, recipe)
    if len(subs) != 2:
        raise RuntimeError(
            "mixed shiftadd reduce requires exactly two discovered classes; "
            "got %r." % ([(k, b) for _cid, k, b, _sub in subs],))

    cls_recipes = (getattr(recipe, "cls_recipes", None)
                   if recipe is not None else None)
    recipe_by_class = ({cr.k_class: cr for cr in cls_recipes}
                       if cls_recipes else {})

    combined_log = []
    classes = []
    for helper_idx, (class_id, k_class, B, sub) in enumerate(subs):
        fold_c, corr_c, report_c = _shiftadd_resolve_fold_and_correction(
            sub, K_env, fold_depth_req="auto", correction_req="auto")
        cr = recipe_by_class.get(k_class)
        if cr is not None:
            want = tuple(cr.reduce_resolver_params)
            if (cr.b_anchor != B) or ((fold_c, corr_c) != want):
                raise RuntimeError(
                    "shiftadd mixed recipe drift (cls%d %d-class): live "
                    "(B, fold, correction) %r != recipe %r" % (
                        helper_idx, k_class, (B, fold_c, corr_c),
                        (cr.b_anchor,) + want))
            fold_c, corr_c = want
        print("[shiftadd auto-analysis cls%d %d-class B=%d]"
              % (helper_idx, k_class, B))
        for line in report_c:
            print("  " + line)
        WFold1, WFold2, WFold3, WFinal, ell, all_FN, info = \
            _shiftadd_envelope_widths(sub, K_env, fold_depth=fold_c)
        if ell != B:
            raise RuntimeError(
                "mixed shiftadd internal: class %d envelope anchor %d != %d."
                % (k_class, ell, B))
        if not all_FN:
            gate = "F3_OK" if fold_c == 3 else "F2_OK"
            raise RuntimeError(
                f"reduce_shiftadd (mixed cls{helper_idx} {k_class}-class): not "
                f"all primes are {gate} under the envelope bound "
                f"K_env={K_env}, fold_depth={fold_c}.")
        ok_math, math_log = _shiftadd_math_self_check(
            sub, K_env, fold_depth=fold_c, correction=corr_c)
        for line in math_log:
            print("  " + line)
        combined_log.extend(math_log)
        if not ok_math:
            raise RuntimeError(
                f"math self-check FAILED for mixed cls{helper_idx} "
                f"{[a for a, _ in sub]}; refusing to emit reducer.")
        required_slots = max(
            (sum(1 for (_s, e) in p["alpha_naf"] if e != 0) for p in info),
            default=2)
        max_slots = max(2, required_slots)
        heuristic = "desc_exp" if max_slots == 3 else "asc_exp"
        slotA, slotB, slotC = _shiftadd_slot_assignment(
            info, max_slots=max_slots, heuristic=heuristic)
        global_indices = []
        for alias, q in sub:
            for gi, (ga, gq) in enumerate(aliases_q):
                if ga == alias and gq == q:
                    global_indices.append(gi)
                    break
            else:
                raise RuntimeError(
                    "mixed shiftadd internal: class prime %s q=%d not found "
                    "in global batch." % (alias, q))
        classes.append(dict(
            helper_idx=helper_idx, class_id=class_id,
            k_class=k_class, B=B, fold=fold_c, corr=corr_c,
            WFold1=WFold1, WFold2=WFold2, WFold3=WFold3, WFinal=WFinal,
            info=info, slotA=slotA, slotB=slotB, slotC=slotC,
            global_indices=global_indices,
            has_slot_c=any(t is not None for t in slotC)))

    if math_self_check_log_path is not None:
        try:
            os.makedirs(os.path.dirname(math_self_check_log_path),
                        exist_ok=True)
        except (OSError, ValueError):
            pass
        with open(math_self_check_log_path, "w") as f:
            f.write("\n".join(combined_log) + "\n")

    def _sanitize(a):
        return "".join(ch for ch in a if ch.isalnum())

    def _term_name(c, fold_tag, tag):
        sign, exp = c
        prefix = "p" if sign > 0 else "n"
        return f"{prefix}{exp}_{fold_tag}{tag}"

    def _term_expr(c, source_var):
        sign, exp = c
        if sign > 0:
            return f"({source_var} << {exp})"
        return f"-({source_var} << {exp})"

    BALANCED_FANIN_THRESHOLD = 4
    balance_slots = (balanced_selector_scope == "all")

    def _slot_chain(stage_var, slot_per_prime, fold_tag, type_macro, info_c,
                    tag, indent="    "):
        """Per-class slot selector; same forms as the single-family emitter
        (chained ternary below fan-in 4; balanced mux tree at >= 4 when the
        scope balances). Other-class mod_ids match no predicate -> 0
        fall-through -> the pipeline value is discarded by the cls mux."""
        active = []
        for i, t in enumerate(slot_per_prime):
            if t is None:
                continue
            active.append((f"is_{_sanitize(info_c[i]['alias'])}",
                           _term_name(t, fold_tag, tag)))
        if not active:
            return [f"{indent}ap_int<{type_macro}> {stage_var} = "
                    f"(ap_int<{type_macro}>)0;"]
        if (len(active) < BALANCED_FANIN_THRESHOLD) or (not balance_slots):
            L = [f"{indent}ap_int<{type_macro}> {stage_var} ="]
            for p, c in active:
                L.append(f"{indent}    {p} ? {c} :")
            L.append(f"{indent}                 (ap_int<{type_macro}>)0;")
            return L
        helper_lines = []
        counter = [0]

        def fresh():
            n = counter[0]
            counter[0] += 1
            return f"{stage_var}_n{n}"

        def build(pairs):
            if len(pairs) == 1:
                p, c = pairs[0]
                name = fresh()
                helper_lines.append(
                    f"{indent}ap_int<{type_macro}> {name} = "
                    f"{p} ? {c} : (ap_int<{type_macro}>)0;")
                return name, [p]
            if len(pairs) == 2:
                (p0, c0), (p1, c1) = pairs
                name = fresh()
                helper_lines.append(
                    f"{indent}ap_int<{type_macro}> {name} = "
                    f"{p0} ? {c0} : ({p1} ? {c1} : (ap_int<{type_macro}>)0);")
                return name, [p0, p1]
            mid = (len(pairs) + 1) // 2
            left_name, left_preds = build(pairs[:mid])
            right_name, right_preds = build(pairs[mid:])
            or_left = " || ".join(left_preds)
            name = fresh()
            helper_lines.append(
                f"{indent}ap_int<{type_macro}> {name} = "
                f"({or_left}) ? {left_name} : {right_name};")
            return name, left_preds + right_preds

        root, _ = build(active)
        L = list(helper_lines)
        L.append(f"{indent}ap_int<{type_macro}> {stage_var} = {root};")
        return L

    def _class_helper(c):
        helper_idx = c["helper_idx"]
        tag = "_c%d" % helper_idx
        ell = c["B"]
        fold = c["fold"]
        corr = c["corr"]
        info_c = c["info"]
        slotA, slotB, slotC = c["slotA"], c["slotB"], c["slotC"]
        has_slot_c = c["has_slot_c"]
        K_high_bits = 2 * K_env - ell
        W1, W2, W3 = (f"WFold1{tag}", f"WFold2{tag}", f"WFold3{tag}")

        union = []
        for t in [t for t in (slotA + slotB + slotC) if t is not None]:
            if t not in union:
                union.append(t)
        union.sort(key=lambda x: (x[1], -x[0]))

        L = []
        L.append(
            f"inline Data reduce_shiftadd_cls{helper_idx}(Data2 X, ModId mod_id, Data q_cur) {{")
        L.append("#pragma HLS INLINE")
        L.append("")
        for i, p in enumerate(info_c):
            L.append(f"    const bool is_{_sanitize(p['alias'])} = "
                     f"(mod_id == (ModId){c['global_indices'][i]});")
        L.append("")
        L.append(f"    // Class {c['class_id']}: {c['k_class']}-class path, fold base B = 2^{ell}.")
        L.append(f"    ap_uint<{ell}> X_l_u{tag} = "
                 f"(ap_uint<{ell}>)X.range({ell - 1}, 0);")
        L.append(f"    ap_uint<{K_high_bits}> X_h_u{tag} = "
                 f"(ap_uint<{K_high_bits}>)X.range({2 * K_env - 1}, {ell});")
        L.append(f"    ap_int<{W1}> X_l_s{tag} = (ap_int<{W1}>)X_l_u{tag};")
        L.append(f"    ap_int<{W1}> X_h_s{tag} = (ap_int<{W1}>)X_h_u{tag};")
        L.append(f"    ap_int<{W1}> base_1{tag} = X_l_s{tag} - X_h_s{tag};")
        for t in union:
            L.append(f"    ap_int<{W1}> {_term_name(t, '1', tag)} = "
                     f"{_term_expr(t, 'X_h_s' + tag)};")
        L.extend(_slot_chain(f"stage_a_1{tag}", slotA, "1", W1, info_c, tag))
        L.extend(_slot_chain(f"stage_b_1{tag}", slotB, "1", W1, info_c, tag))
        if has_slot_c:
            L.extend(_slot_chain(f"stage_c_1{tag}", slotC, "1", W1, info_c, tag))
            L.append(f"    ap_int<{W1}> v_1{tag} = base_1{tag} + "
                     f"stage_a_1{tag} + stage_b_1{tag} + stage_c_1{tag};")
        else:
            L.append(f"    ap_int<{W1}> v_1{tag} = base_1{tag} + "
                     f"stage_a_1{tag} + stage_b_1{tag};")
        L.append("")
        # Fold 2
        L.append(f"    ap_uint<{ell}> v_1_l_u{tag} = "
                 f"(ap_uint<{ell}>)v_1{tag}.range({ell - 1}, 0);")
        L.append(f"    ap_int<W_v1_h{tag}> v_1_h_tight{tag} = "
                 f"(ap_int<W_v1_h{tag}>)(v_1{tag} >> {ell});")
        L.append(f"    ap_int<{W2}> v_1_l_s{tag} = (ap_int<{W2}>)v_1_l_u{tag};")
        L.append(f"    ap_int<{W2}> v_1_h_s{tag} = "
                 f"(ap_int<{W2}>)v_1_h_tight{tag};")
        L.append(f"    ap_int<{W2}> base_2{tag} = v_1_l_s{tag} - v_1_h_s{tag};")
        for t in union:
            L.append(f"    ap_int<{W2}> {_term_name(t, '2', tag)} = "
                     f"{_term_expr(t, 'v_1_h_s' + tag)};")
        L.extend(_slot_chain(f"stage_a_2{tag}", slotA, "2", W2, info_c, tag))
        L.extend(_slot_chain(f"stage_b_2{tag}", slotB, "2", W2, info_c, tag))
        if has_slot_c:
            L.extend(_slot_chain(f"stage_c_2{tag}", slotC, "2", W2, info_c, tag))
            L.append(f"    ap_int<{W2}> v_2{tag} = base_2{tag} + "
                     f"stage_a_2{tag} + stage_b_2{tag} + stage_c_2{tag};")
        else:
            L.append(f"    ap_int<{W2}> v_2{tag} = base_2{tag} + "
                     f"stage_a_2{tag} + stage_b_2{tag};")
        L.append("")
        if fold == 3:
            L.append(f"    ap_uint<{ell}> v_2_l_u{tag} = "
                     f"(ap_uint<{ell}>)v_2{tag}.range({ell - 1}, 0);")
            L.append(f"    ap_int<W_v2_h{tag}> v_2_h_tight{tag} = "
                     f"(ap_int<W_v2_h{tag}>)(v_2{tag} >> {ell});")
            L.append(f"    ap_int<{W3}> v_2_l_s{tag} = "
                     f"(ap_int<{W3}>)v_2_l_u{tag};")
            L.append(f"    ap_int<{W3}> v_2_h_s{tag} = "
                     f"(ap_int<{W3}>)v_2_h_tight{tag};")
            L.append(f"    ap_int<{W3}> base_3{tag} = "
                     f"v_2_l_s{tag} - v_2_h_s{tag};")
            for t in union:
                L.append(f"    ap_int<{W3}> {_term_name(t, '3', tag)} = "
                         f"{_term_expr(t, 'v_2_h_s' + tag)};")
            L.extend(_slot_chain(f"stage_a_3{tag}", slotA, "3", W3, info_c, tag))
            L.extend(_slot_chain(f"stage_b_3{tag}", slotB, "3", W3, info_c, tag))
            if has_slot_c:
                L.extend(_slot_chain(f"stage_c_3{tag}", slotC, "3", W3,
                                     info_c, tag))
                L.append(f"    ap_int<{W3}> v_3{tag} = base_3{tag} + "
                         f"stage_a_3{tag} + stage_b_3{tag} + stage_c_3{tag};")
            else:
                L.append(f"    ap_int<{W3}> v_3{tag} = base_3{tag} + "
                         f"stage_a_3{tag} + stage_b_3{tag};")
            L.append("")
            final_var = f"v_3{tag}"
        else:
            final_var = f"v_2{tag}"
        # Per-class correction (the class's resolved correction mode).
        pre = "ShiftaddPreC%d" % helper_idx
        L.append(f"    // Class {c['class_id']} correction ({corr}).")
        L.append(f"    {pre} r{tag} = ({pre}){final_var};")
        L.append(f"    {pre} q_ext{tag} = ({pre})q_cur;")
        if corr == "signed":
            L.append(f"    if (r{tag} < 0) {{")
            L.append(f"        r{tag} += q_ext{tag};")
            L.append("    }")
        L.append(f"    if (r{tag} >= q_ext{tag}) {{")
        L.append(f"        r{tag} -= q_ext{tag};")
        L.append("    }")
        L.append(f"    return (Data)r{tag};")
        L.append("}")
        return L

    lines = list(header_top)
    lines.append("// Generated shift-add reducer for this recipe-class-mixed prime batch.")
    lines.append("// Each class has an independent inline helper; the wrapper selects only the final Data result.")
    lines.append("// Discovered class mapping:")
    for c in classes:
        lines.append(
            f"//   PRIME_CLASS == {c['class_id']}: {c['k_class']}-class, fold base B = 2^{c['B']}.")
    lines.append("// Shift-add step per class: W' = W_low + alpha * W_high, where alpha = B - q.")
    lines.append("")
    for c in classes:
        tag = "_c%d" % c["helper_idx"]
        ell = c["B"]
        lines.append(f"constexpr int WFold1{tag} = {c['WFold1']};")
        lines.append(f"constexpr int WFold2{tag} = {c['WFold2']};")
        if c["fold"] == 3:
            lines.append(f"constexpr int WFold3{tag} = {c['WFold3']};")
        lines.append(f"constexpr int WFinal{tag} = {c['WFinal']};")
        lines.append(f"constexpr int W_v1_h{tag} = WFold1{tag} - {ell - 1};")
        if c["fold"] == 3:
            lines.append(f"constexpr int W_v2_h{tag} = WFold2{tag} - {ell - 1};")
        lines.append("")
    lines.extend([
        "#define USE_REDUCE_SHIFTADD 1",
        "",
        "using ShiftaddPreC0 = ap_int<WFinal_c0 + 1>;",
        "using ShiftaddPreC1 = ap_int<WFinal_c1 + 1>;",
        "",
    ])
    for c in classes:
        lines.extend(_class_helper(c))
        lines.append("")
    lines.extend([
        "inline Data reduce_shiftadd(Data2 X, ModId mod_id, Data q_cur) {",
        "#pragma HLS INLINE",
        "    Data r_cls0 = reduce_shiftadd_cls0(X, mod_id, q_cur);",
        "    Data r_cls1 = reduce_shiftadd_cls1(X, mod_id, q_cur);",
        "    const int cls = PRIME_CLASS[mod_id];",
        f"    return (cls == {classes[0]['class_id']}) ? r_cls0 : r_cls1;",
        "}",
        "",
        "#endif  // NTT_SHIFTADD_H",
        "",
    ])

    with open(out_path, "w") as f:
        f.write("\n".join(lines))


def _shiftadd_mul_self_check(aliases_q, base, num_random=4000):
    """Envelope-aware self-check for the shiftadd_mul reducer.

    For each prime in the batch:
      - verify T_user == floor(2^(2*base)/q) (i.e., the formula
        T_user = 2^base - s - 1 matches the canonical Barrett constant)
      - reconstruct q and T_user from the NAF of s and verify exact
        integer match (the emitter materializes shift-add terms from
        the same NAF, so this verifies generated q2 == q1*T_user and
        generated p == q3*q hold by construction)
      - sample edge inputs and 4000 random (A, B) in [0, q)^2,
        compute r_pre = X - q3*q with X = A*B, q1 = X >> base,
        q2 = q1 * T_user, q3 = q2 >> base, p = q3 * q
      - track rmax / q and the minimum number of `if (r >= q) r -= q;`
        steps needed to bring r into [0, q)

    Returns:
      info: list of dicts (one per prime) with all diagnostic fields.
      batch_corrections: int in {1, 2}; max required corrections over
                         all primes (clamped to >=1).
      unsupported: list of (alias, q, reasons_dict). Empty if all primes
                   can be served by at most two corrections.
      log_lines: list of strings for human-readable reporting.

    Hard rules:
      - r_pre observed in [2q, 3q): prime requires 2 corrections (still
        supported).
      - r_pre observed >= 3q OR r_pre < 0 OR T_user != T_floor OR
        NAF reconstruction mismatch: prime is UNSUPPORTED_OR_CODEGEN_BUG.
    """
    import random
    rng = random.Random(0xBA77E77BA77E77)
    log = []
    log.append(f"=== shiftadd_mul self-check (envelope-aware) base={base} ===")
    info = []
    unsupported = []
    batch_corrections = 1  # we always emit at least one correction step

    for alias, q in aliases_q:
        s = q - (1 << base) - 1
        T_user = (1 << base) - s - 1
        T_floor = (1 << (2 * base)) // q
        T_match = (T_user == T_floor)

        # Reconstruct q and T from NAF(s) — what the emitter materializes.
        s_naf = _shiftadd_naf(s)
        s_from_naf = sum(sign * (1 << exp) for sign, exp in s_naf)
        q_from_naf = (1 << base) + s_from_naf + 1
        T_from_naf = (1 << base) - s_from_naf - 1
        q_naf_match = (q_from_naf == q)
        T_naf_match = (T_from_naf == T_user)

        # Integer simulation of generated shift-add path.
        rmax = 0
        max_corr_seen = 0
        bad_negative = False
        bad_too_large = False
        first_two_corr_sample = None
        first_unsupported_sample = None
        edge_pairs = [(0, 0), (0, 1), (1, 1), (q - 1, 1),
                      (q - 1, q - 1), (q - 2, q - 1)]
        rand_pairs = [(rng.randint(0, q - 1), rng.randint(0, q - 1))
                      for _ in range(num_random)]
        for A, B in edge_pairs + rand_pairs:
            X = A * B
            q1 = X >> base
            q2 = q1 * T_floor   # effective reciprocal = exact T_floor (emitter adds q1*delta)
            q3 = q2 >> base
            p = q3 * q
            r_pre = X - p
            if r_pre < 0:
                bad_negative = True
                if first_unsupported_sample is None:
                    first_unsupported_sample = ("r_pre<0", A, B, r_pre)
                continue
            if r_pre > rmax:
                rmax = r_pre
            if r_pre >= 3 * q:
                bad_too_large = True
                if first_unsupported_sample is None:
                    first_unsupported_sample = (
                        "r_pre>=3q", A, B, r_pre)
                continue
            corr = 0
            cur = r_pre
            while cur >= q:
                cur -= q
                corr += 1
                if corr > 2:
                    break
            if corr > max_corr_seen:
                max_corr_seen = corr
            if corr == 2 and first_two_corr_sample is None:
                first_two_corr_sample = (A, B, X, r_pre)

        rmax_per_q = (rmax / q) if q else float("nan")
        reasons = {}
        # T_user != T_floor is SUPPORTED via additive delta correction (effective
        # reciprocal = T_floor); it is no longer an unsupported reason.
        if not q_naf_match:
            reasons["q_from_naf_mismatch"] = (q_from_naf, q)
        if not T_naf_match:
            reasons["T_from_naf_mismatch"] = (T_from_naf, T_user)
        if bad_negative:
            reasons["r_pre_negative"] = first_unsupported_sample
        if bad_too_large:
            reasons["r_pre_ge_3q"] = first_unsupported_sample

        if reasons:
            req_corr = None
            unsupported.append((alias, q, reasons))
        else:
            req_corr = max(1, max_corr_seen)
            if req_corr > batch_corrections:
                batch_corrections = req_corr

        info.append({
            "alias": alias, "q": q, "base": base,
            "s": s, "s_naf": s_naf,
            "T_user": T_user, "T_floor": T_floor, "T_match": T_match,
            "q_from_naf_match": q_naf_match,
            "T_from_naf_match": T_naf_match,
            "rmax": rmax, "rmax_per_q": rmax_per_q,
            "max_corr_seen": max_corr_seen,
            "bad_negative": bad_negative,
            "bad_too_large": bad_too_large,
            "first_two_corr_sample": first_two_corr_sample,
            "required_corrections": req_corr,
        })
        log.append(
            f"  {alias:<10s} q={q}  T==Tfloor={T_match}  "
            f"naf:q={q_naf_match} T={T_naf_match}  "
            f"rmax/q={rmax_per_q:.4f}  max_corr={max_corr_seen}  "
            f"req_corr={req_corr}"
        )
        if first_two_corr_sample is not None:
            A_, B_, X_, r_pre_ = first_two_corr_sample
            log.append(
                f"      first sample needing 2 corrections: A={A_} B={B_} "
                f"r_pre={r_pre_} (r_pre/q={r_pre_/q:.4f})"
            )

    if unsupported:
        log.append("=== UNSUPPORTED primes (cannot be served with <=2 corrections) ===")
        for alias, q, reasons in unsupported:
            log.append(f"  {alias} q={q}: {reasons}")
    log.append(f"=== batch SHIFTADD_MUL_CORRECTIONS = {batch_corrections} ===")
    return info, batch_corrections, unsupported, log

def _decompose_signed_pow2(n):
    """Decompose an integer n into a sorted list of (sign, exp) such that
    n = sum sign * 2**exp. Uses the canonical NAF (non-adjacent form) so
    weight is minimised; 2^0 terms are allowed. Returns [] for n == 0.
    """
    if n == 0:
        return []
    sign_outer = 1
    if n < 0:
        n = -n
        sign_outer = -1
    out = []
    e = 0
    while n > 0:
        if n & 1:
            if (n & 3) == 3:
                out.append((-1, e))
                n += 1
            else:
                out.append((1, e))
                n -= 1
        n >>= 1
        e += 1
    return [(sign_outer * s, exp) for s, exp in out]



def _emit_shiftadd_mul_header(out_path, K_env, aliases_q, header_top,
                               variant="balanced", *, recipe=None,
                               single_prime_normalized=False):
    """Emit src/ntt_shiftadd.h with the reduce_shiftadd_mul (BU side) helper.

    q-derived balanced path. anchor base = B = min(K_i) via _shiftadd_q_anchor
    (only K_spread < 3 compatible-width groups are accepted). The effective
    Barrett reciprocal is the exact T_floor = floor(2^(2B)/q), materialized as
    q1*T_user (sparse, from NAF(s)) plus an additive, mod_id-gated q1*delta
    correction (delta = T_floor - T_user). Selection is a per-prime mod_id
    predicate mux over shared per-exponent shifted candidates on a shared base.
    (Only the balanced emitter is supported; `variant` is retained internally
    for the emitted provenance comment.)

    The Barrett correction count is selected by self_check (1 or 2);
    a guarded second correction is emitted under ``SHIFTADD_MUL_CORRECTIONS
    >= 2``.
    """
    if variant != "balanced":
        raise RuntimeError(
            f"_emit_shiftadd_mul_header: only the q-derived balanced emitter "
            f"is supported (got {variant!r}).")

    # q-derived anchor: base = B = min(K_i); accept only K_spread < 3 (a wider
    # group is rejected by the K_spread<3 gate in _shiftadd_q_anchor).
    # On the recipe-governed path base = recipe.b_anchor and helper agreement is asserted;
    # mul_corrections is NOT governed here (the recipe value is solo-aggregated;
    # the live batch self-check below stays the decision source -- see plan Q4).
    base, _K_group, _K_spread = _resolve_anchor(aliases_q, recipe)  # base = min(K_i)

    # Per-prime info: T_i, s_i, NAF(s_i). Validation gates (formula gate +
    # math self-check) are NOTE-only per current instruction; never raise.
    info = []
    for alias, q in aliases_q:
        s = q - (1 << base) - 1
        T_user = (1 << base) - s - 1  # = 2^(base+1) - q (sparse candidate)
        T_floor = (1 << (2 * base)) // q  # exact reciprocal (effective T)
        delta = T_floor - T_user
        s_naf = _shiftadd_naf(s)
        for _sign, _exp in s_naf:
            if _exp == 0:
                raise RuntimeError(
                    f"shiftadd_mul: prime {alias}: s NAF contains a 2^0 "
                    f"term, which collides with the constant tail. Not "
                    f"supported.")
        # Exact-reciprocal support: the effective T is T_floor = T_user + delta.
        # q1*delta is emitted additively on the T multiply (delta may be odd, so
        # its decomposition can include a 2^0 term -- kept as an explicit additive
        # term, distinct from the +/- constant tail). No T_user!=T_floor reject.
        delta_naf = _decompose_signed_pow2(delta)
        info.append({
            "alias": alias, "q": q, "T": T_floor, "T_user": T_user,
            "s": s, "s_naf": s_naf, "delta": delta, "delta_naf": delta_naf,
        })

    # Envelope-aware math self-check. Determines per-prime required
    # corrections (1 or 2). Hard-gates on UNSUPPORTED primes (r_pre out of
    # [0, 3q), T_user != T_floor, or NAF reconstruction mismatch).
    sc_info, batch_corrections, sc_unsupported, sc_log = (
        _shiftadd_mul_self_check(aliases_q, base))
    print("[shiftadd_mul self-check]")
    for line in sc_log:
        print("  " + line)
    if sc_unsupported:
        details = "; ".join(
            f"{alias} q={q} reasons={reasons}"
            for alias, q, reasons in sc_unsupported)
        raise RuntimeError(
            f"shiftadd_mul self-check: UNSUPPORTED_OR_CODEGEN_BUG. "
            f"Cannot emit a correct reducer for: {details}. "
            "This is either a Barrett envelope blow-up beyond two "
            "corrections, a generator NAF mismatch, or T_user != "
            "T_floor.")
    aliases_two_corr = [
        i["alias"] for i in sc_info if i["required_corrections"] == 2]

    NP = len(info)

    def _sanitize(a):
        return "".join(ch for ch in a if ch.isalnum())

    aliases_clean = [_sanitize(p["alias"]) for p in info]

    W_acc = 2 * K_env + 4
    W_q1 = 2 * K_env - base + 1

    # Union of NAF tail exponents across all primes — used for shared-term
    # candidates in the balanced variant.
    union_exps = set()
    for p in info:
        for _sign, exp in p["s_naf"]:
            union_exps.add(exp)
    union_exps = sorted(union_exps)

    # Slot allocation for the balanced variant.
    # Switch path keeps the legacy combined slot table (sorted asc).
    slot_count = max((len(p["s_naf"]) for p in info), default=0)
    slot_terms = [[None] * NP for _ in range(slot_count)]
    for i, p in enumerate(info):
        sorted_terms = sorted(p["s_naf"], key=lambda x: x[1])
        for j, term in enumerate(sorted_terms):
            slot_terms[j][i] = term

    # Balanced path uses pos/neg-separated slot tables. Descending exponent
    # sort tries to land popular high exponents in the same slot index across
    # primes, which lowers per-slot exponent diversity (Optimization 4).
    pos_slot_count = max((sum(1 for s, _ in p["s_naf"] if s > 0) for p in info),
                          default=0)
    neg_slot_count = max((sum(1 for s, _ in p["s_naf"] if s < 0) for p in info),
                          default=0)
    pos_slot_terms = [[None] * NP for _ in range(pos_slot_count)]
    neg_slot_terms = [[None] * NP for _ in range(neg_slot_count)]
    for i, p in enumerate(info):
        pos_terms = sorted([(s, e) for s, e in p["s_naf"] if s > 0],
                            key=lambda x: -x[1])
        neg_terms = sorted([(s, e) for s, e in p["s_naf"] if s < 0],
                            key=lambda x: -x[1])
        for j, term in enumerate(pos_terms):
            pos_slot_terms[j][i] = term
        for j, term in enumerate(neg_terms):
            neg_slot_terms[j][i] = term

    L = list(header_top)
    L.append("// Auto-generated lightweight-prime shift-add multiplication helpers")
    L.append("// Contract: keeps Barrett/Shoup structure; replaces selected")
    L.append("// constant multiplications with shift-add expressions.")
    L.append(f"// base = {base}, K = {K_env}, NUM_PRIMES = {NP}, variant = {variant}")
    L.append("//")
    L.append("// Per-prime data:")
    for i, p in enumerate(info):
        naf_str = ", ".join(
            f"{'+' if s > 0 else '-'}2^{e}" for s, e in p["s_naf"])
        L.append(f"//   [{i}] {p['alias']:<8s} q={p['q']} T={p['T']} s_naf={naf_str}")
    L.append("")
    L.append("#define USE_REDUCE_SHIFTADD_MUL 1")
    L.append("")
    L.append(f"constexpr int SHIFTADD_MUL_BASE = {base};")
    L.append(f"constexpr int SHIFTADD_MUL_W_ACC = {W_acc};")
    L.append("using ShiftaddMulAcc = ap_int<SHIFTADD_MUL_W_ACC>;")
    L.append("")
    L.append(
        f"// shiftadd_mul self_check selected {batch_corrections} "
        f"correction step(s)")
    if aliases_two_corr:
        L.append(
            f"// Reason: {', '.join(aliases_two_corr)} require r in "
            f"[2q, 3q) under one-correction; two corrections selected "
            f"for the whole batch (max-policy)")
    else:
        L.append(
            "// Reason: all primes in this batch keep r in [0, 2q); "
            "one correction is sufficient")
    L.append(f"#define SHIFTADD_MUL_CORRECTIONS {batch_corrections}")
    L.append("")

    # ----- Helpers: emit the per-mod_id mul body -----
    # signed_factor = +1  -> factor expression for q3 * Q_i  (or qh * Q_i)
    #                          Q_i = 2^base + s_i + 1
    # signed_factor = -1  -> factor expression for q1 * T_i
    #                          T_i = 2^base - s_i - 1

    def _balanced_mux_tree(items, slot_var, type_macro, indent):
        """Emit a real balanced mux tree assigning the result to slot_var.

        Naming uses path-based suffixes ``_l`` / ``_h`` (left / right
        children of each internal node):

            <slot>_l = pred0 ? expr0 : expr1;
            <slot>_h = pred2 ? expr2 : expr3;
            <slot>   = (pred0 || pred1) ? <slot>_l : <slot>_h;

        Leaves:
          * size 2 (no 0 fallback — the outer mux already restricts
            mod_id to one of the pair members):
              <name> = pred_first ? expr_first : expr_second;
          * size 1 (used only when an inner subtree count is odd):
              <name> = pred ? expr : (type_macro)0;
        Internal nodes:
              <name> = (or_of_left_preds) ? left_var : right_var;
        """
        lines = []

        def build(items_list, path, target_var=None):
            name = target_var if target_var else f"{slot_var}{path}"
            if len(items_list) == 1:
                pred, expr = items_list[0]
                lines.append(
                    f"{indent}{type_macro} {name} = "
                    f"{pred} ? {expr} : ({type_macro})0;")
                return name, [pred]
            if len(items_list) == 2:
                (p0, e0), (p1, e1) = items_list
                lines.append(
                    f"{indent}{type_macro} {name} = "
                    f"{p0} ? {e0} : {e1};")
                return name, [p0, p1]
            mid = (len(items_list) + 1) // 2
            left_name, left_preds = build(items_list[:mid], path + "_l")
            right_name, right_preds = build(items_list[mid:], path + "_h")
            or_left = " || ".join(left_preds)
            lines.append(
                f"{indent}{type_macro} {name} = "
                f"({or_left}) ? {left_name} : {right_name};")
            return name, left_preds + right_preds

        if not items:
            lines.append(
                f"{indent}{type_macro} {slot_var} = ({type_macro})0;")
            return lines
        build(items, "", target_var=slot_var)
        return lines

    def _emit_pn_slots(out, kind, slot_terms_list, var_prefix, type_macro,
                        indent):
        """Emit pos_slot_<i> or neg_slot_<i> variables using the balanced
        mux-tree skeleton. ``kind`` is "pos" or "neg".

        For each slot:
          * 0 contributors → skip (no add into the sum).
          * 1 contributor → ``slot = pred ? cand : 0;`` (single mux,
                              not a chained priority chain).
          * NP contributors → balanced mux tree without an outer gate.
          * partial (>1, <NP) → balanced mux over real contributors,
                                 gated by an OR of contributor predicates.

        Returns the list of slot-var names that should be added into the
        per-sign sum.
        """
        sum_terms = []
        for slot_idx, slot_row in enumerate(slot_terms_list):
            slot_var = f"{var_prefix}_{kind}_slot_{slot_idx}"
            contributors = []
            for i in range(NP):
                t = slot_row[i]
                if t is None:
                    continue
                _sign, exp = t
                contributors.append(
                    (f"is_{aliases_clean[i]}", f"{var_prefix}_sh_{exp}"))
            if not contributors:
                continue
            if single_prime_normalized:
                # S4: NUM_PRIMES==1 -> one contributor; is_<alias> always true
                # -> resolve the mux to the bare shifted candidate.
                _pred, cand = contributors[0]
                out.append(f"{indent}{type_macro} {slot_var} = {cand};")
                sum_terms.append(slot_var)
                continue
            if len(contributors) == NP:
                out.extend(
                    _balanced_mux_tree(contributors, slot_var, type_macro,
                                        indent))
            elif len(contributors) == 1:
                pred, cand = contributors[0]
                out.append(
                    f"{indent}{type_macro} {slot_var} = "
                    f"{pred} ? {cand} : ({type_macro})0;")
            else:
                inner = f"{slot_var}_inner"
                out.extend(
                    _balanced_mux_tree(contributors, inner, type_macro,
                                        indent))
                preds_or = " || ".join(p for p, _ in contributors)
                out.append(
                    f"{indent}{type_macro} {slot_var} = "
                    f"({preds_or}) ? {inner} : ({type_macro})0;")
            sum_terms.append(slot_var)
        return sum_terms

    def _emit_mul_balanced(operand, dst, q_or_t, var_prefix, type_macro,
                           indent="    "):
        """Emit pos/neg-separated balanced shift-add multiplication.

        For Q multiplication (q_or_t == "Q"):
            dst = (operand << base) + operand + pos_sum - neg_sum
        For T multiplication (q_or_t == "T"):
            dst = (operand << base) - operand - pos_sum + neg_sum

        where pos_sum (resp. neg_sum) is the sum over slots of mod_id-keyed
        positive-tail (resp. negative-tail) shifted-operand selections.
        """
        if q_or_t == "Q":
            const_op, pos_op, neg_op = "+", "+", "-"
        elif q_or_t == "T":
            const_op, pos_op, neg_op = "-", "-", "+"
        else:
            raise RuntimeError(
                f"_emit_mul_balanced: q_or_t must be 'Q' or 'T' "
                f"(got {q_or_t!r}).")
        out = []
        # Shared base: (operand << base) ± operand
        out.append(
            f"{indent}{type_macro} {var_prefix}_base = "
            f"({operand} << {base}) {const_op} {operand};")
        out.append("")
        # Shared shifted candidates (one per unique exponent).
        for exp in union_exps:
            out.append(
                f"{indent}{type_macro} {var_prefix}_sh_{exp} = "
                f"({operand} << {exp});")
        out.append("")
        pos_terms = _emit_pn_slots(out, "pos", pos_slot_terms, var_prefix,
                                    type_macro, indent)
        neg_terms = _emit_pn_slots(out, "neg", neg_slot_terms, var_prefix,
                                    type_macro, indent)
        sum_str = f"{var_prefix}_base"
        for term in pos_terms:
            sum_str += f" {pos_op} {term}"
        for term in neg_terms:
            sum_str += f" {neg_op} {term}"
        # Exact-reciprocal correction (T multiply only): add a per-prime, mod_id-gated
        # q1*delta_i term so the effective reciprocal is the exact T_floor. delta_i may
        # be 0 (no contribution) or include a 2^0 term (emitted as an explicit shift).
        if q_or_t == "T":
            delta_items = []
            for i, p in enumerate(info):
                dn = p.get("delta_naf", [])
                if not dn:
                    continue
                parts = [f"{'+' if ds > 0 else '-'} ({operand} << {de})"
                         for ds, de in dn]
                expr = " ".join(parts)
                if expr.startswith("+ "):
                    expr = expr[2:]
                # Cast each branch to the slot type so the mux-tree '?:' operands
                # have matching ap_int widths (q1*delta fits: < 2^82 << acc width).
                delta_items.append((f"is_{aliases_clean[i]}", f"({type_macro})({expr})"))
            if delta_items:
                dvar = f"{var_prefix}_delta"
                if single_prime_normalized:
                    # S4: single prime -> one delta candidate, no mod_id mux.
                    _pred, cand = delta_items[0]
                    out.append(f"{indent}{type_macro} {dvar} = {cand};")
                elif len(delta_items) == NP:
                    out.extend(_balanced_mux_tree(delta_items, dvar, type_macro, indent))
                elif len(delta_items) == 1:
                    pred, cand = delta_items[0]
                    out.append(
                        f"{indent}{type_macro} {dvar} = {pred} ? {cand} : ({type_macro})0;")
                else:
                    inner = f"{dvar}_inner"
                    out.extend(_balanced_mux_tree(delta_items, inner, type_macro, indent))
                    preds_or = " || ".join(pp for pp, _ in delta_items)
                    out.append(
                        f"{indent}{type_macro} {dvar} = "
                        f"({preds_or}) ? {inner} : ({type_macro})0;")
                sum_str += f" + {dvar}"
        out.append(f"{indent}{type_macro} {dst} = {sum_str};")
        return out

    def _emit_mul_body(operand, dst, q_or_t, var_prefix, type_macro,
                       indent="    "):
        return _emit_mul_balanced(operand, dst, q_or_t, var_prefix,
                                   type_macro, indent=indent)

    # ----- BU helper: reduce_shiftadd_mul -----
    L.append("// BU side: full Barrett-style modular reduction with shift-add T*q1 and Q*q3.")
    L.append("//   x  = X (= mul_full_data_nonstd(A, B), preserved upstream)")
    L.append(f"//   q1 = x >> {base}")
    L.append("//   q2 = q1 * T_i  (shift-add)")
    L.append(f"//   q3 = q2 >> {base}")
    L.append("//   p  = q3 * Q_i  (shift-add)")
    L.append("//   r  = X - p; correction step(s) per SHIFTADD_MUL_CORRECTIONS")
    L.append("inline Data reduce_shiftadd_mul(Data2 X, ModId mod_id, Data q_cur) {")
    L.append("#pragma HLS INLINE")
    L.append("")
    if variant == "balanced" and not single_prime_normalized:
        for i in range(NP):
            L.append(
                f"    const bool is_{aliases_clean[i]} = (mod_id == (ModId){i});")
        L.append("")
    L.append(f"    ap_uint<{W_q1}> q1_u = (ap_uint<{W_q1}>)(X >> {base});")
    L.append("    ShiftaddMulAcc q1 = (ShiftaddMulAcc)q1_u;")
    L.append("")
    L.extend(_emit_mul_body("q1", "q2", "T", "q2", "ShiftaddMulAcc"))
    L.append("")
    L.append(f"    ShiftaddMulAcc q3 = (q2 >> {base});")
    L.append("")
    L.extend(_emit_mul_body("q3", "p", "Q", "p", "ShiftaddMulAcc"))
    L.append("")
    L.append("    ShiftaddMulAcc r = (ShiftaddMulAcc)X - p;")
    L.append("    ShiftaddMulAcc qq = (ShiftaddMulAcc)q_cur;")
    L.append("    if (r >= qq) r -= qq;")
    L.append("#if SHIFTADD_MUL_CORRECTIONS >= 2")
    L.append("    if (r >= qq) r -= qq;")
    L.append("#endif")
    L.append("    return (Data)r;")
    L.append("}")
    L.append("")

    L.append("#endif  // NTT_SHIFTADD_H")
    L.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(L))

def _emit_shiftadd_mul_header_mixed(out_path, K_env, aliases_q, header_top,
                                    *, recipe=None):
    """Emit reduce_shiftadd_mul(...) for a K_group-mixed {52, 62} known batch.

    Per class: the prime's own Barrett-style base (51 for the 52-class, 61
    for the 62-class), the exact-reciprocal T_floor materialized as q1*T_user
    (sparse NAF(s)) plus the mod_id-gated q1*delta correction, and the class
    correction count from the per-class envelope-aware self-check. The two
    class pipelines (``q1_c0/q2_c0/q3_c0/p_c0/r_c0`` and ``..._c1``) share the
    wide accumulator; the result is ``cls ? r_c1 : r_c0`` with
    cls = PRIME_CLASS[mod_id]. mul corrections stay self-check-governed (the
    recipe is informational here, matching the single-family emitter choice).
    """
    subs = _shiftadd_mixed_guard(aliases_q)
    NP = len(aliases_q)

    def _sanitize(a):
        return "".join(ch for ch in a if ch.isalnum())

    aliases_clean_all = [_sanitize(a) for a, _q in aliases_q]
    W_acc = 2 * K_env + 4

    classes = []
    for cls_idx, (k_class, base, sub) in enumerate(subs):
        info = []
        for alias, q in sub:
            s = q - (1 << base) - 1
            T_user = (1 << base) - s - 1
            T_floor = (1 << (2 * base)) // q
            delta = T_floor - T_user
            s_naf = _shiftadd_naf(s)
            for _sign, _exp in s_naf:
                if _exp == 0:
                    raise RuntimeError(
                        f"shiftadd_mul (mixed cls{cls_idx}): prime {alias}: "
                        f"s NAF contains a 2^0 term, which collides with the "
                        f"constant tail. Not supported.")
            delta_naf = _decompose_signed_pow2(delta)
            info.append({
                "alias": alias, "q": q, "T": T_floor, "T_user": T_user,
                "s": s, "s_naf": s_naf, "delta": delta,
                "delta_naf": delta_naf,
            })
        sc_info, batch_corrections, sc_unsupported, sc_log = (
            _shiftadd_mul_self_check(sub, base))
        print("[shiftadd_mul self-check cls%d %d-class base=%d]"
              % (cls_idx, k_class, base))
        for line in sc_log:
            print("  " + line)
        if sc_unsupported:
            details = "; ".join(
                f"{alias} q={q} reasons={reasons}"
                for alias, q, reasons in sc_unsupported)
            raise RuntimeError(
                f"shiftadd_mul self-check (mixed cls{cls_idx} {k_class}-"
                f"class): UNSUPPORTED_OR_CODEGEN_BUG. Cannot emit a correct "
                f"reducer for: {details}.")
        aliases_two_corr = [
            i["alias"] for i in sc_info if i["required_corrections"] == 2]

        NPc = len(info)
        union_exps = set()
        for p in info:
            for _sign, exp in p["s_naf"]:
                union_exps.add(exp)
        union_exps = sorted(union_exps)
        pos_slot_count = max(
            (sum(1 for s2, _ in p["s_naf"] if s2 > 0) for p in info), default=0)
        neg_slot_count = max(
            (sum(1 for s2, _ in p["s_naf"] if s2 < 0) for p in info), default=0)
        pos_slot_terms = [[None] * NPc for _ in range(pos_slot_count)]
        neg_slot_terms = [[None] * NPc for _ in range(neg_slot_count)]
        for i, p in enumerate(info):
            pos_terms = sorted([(s2, e) for s2, e in p["s_naf"] if s2 > 0],
                               key=lambda x: -x[1])
            neg_terms = sorted([(s2, e) for s2, e in p["s_naf"] if s2 < 0],
                               key=lambda x: -x[1])
            for j, term in enumerate(pos_terms):
                pos_slot_terms[j][i] = term
            for j, term in enumerate(neg_terms):
                neg_slot_terms[j][i] = term
        classes.append(dict(
            cls_idx=cls_idx, k_class=k_class, base=base, info=info,
            corrections=batch_corrections, two_corr=aliases_two_corr,
            union_exps=union_exps, pos_slot_terms=pos_slot_terms,
            neg_slot_terms=neg_slot_terms))

    # ---- emission ----
    def _balanced_mux_tree(items, slot_var, type_macro, indent):
        lines = []

        def build(items_list, path, target_var=None):
            name = target_var if target_var else f"{slot_var}{path}"
            if len(items_list) == 1:
                pred, expr = items_list[0]
                lines.append(
                    f"{indent}{type_macro} {name} = "
                    f"{pred} ? {expr} : ({type_macro})0;")
                return name, [pred]
            if len(items_list) == 2:
                (p0, e0), (p1, e1) = items_list
                lines.append(
                    f"{indent}{type_macro} {name} = "
                    f"{p0} ? {e0} : {e1};")
                return name, [p0, p1]
            mid = (len(items_list) + 1) // 2
            left_name, left_preds = build(items_list[:mid], path + "_l")
            right_name, right_preds = build(items_list[mid:], path + "_h")
            or_left = " || ".join(left_preds)
            lines.append(
                f"{indent}{type_macro} {name} = "
                f"({or_left}) ? {left_name} : {right_name};")
            return name, left_preds + right_preds

        if not items:
            lines.append(
                f"{indent}{type_macro} {slot_var} = ({type_macro})0;")
            return lines
        build(items, "", target_var=slot_var)
        return lines

    def _emit_pn_slots(out, kind, slot_terms_list, var_prefix, type_macro,
                       indent, info_c):
        NPc = len(info_c)
        sum_terms = []
        for slot_idx, slot_row in enumerate(slot_terms_list):
            slot_var = f"{var_prefix}_{kind}_slot_{slot_idx}"
            contributors = []
            for i in range(NPc):
                t = slot_row[i]
                if t is None:
                    continue
                _sign, exp = t
                contributors.append(
                    (f"is_{_sanitize(info_c[i]['alias'])}",
                     f"{var_prefix}_sh_{exp}"))
            if not contributors:
                continue
            if len(contributors) == NPc:
                out.extend(
                    _balanced_mux_tree(contributors, slot_var, type_macro,
                                       indent))
            elif len(contributors) == 1:
                pred, cand = contributors[0]
                out.append(
                    f"{indent}{type_macro} {slot_var} = "
                    f"{pred} ? {cand} : ({type_macro})0;")
            else:
                inner = f"{slot_var}_inner"
                out.extend(
                    _balanced_mux_tree(contributors, inner, type_macro,
                                       indent))
                preds_or = " || ".join(p for p, _ in contributors)
                out.append(
                    f"{indent}{type_macro} {slot_var} = "
                    f"({preds_or}) ? {inner} : ({type_macro})0;")
            sum_terms.append(slot_var)
        return sum_terms

    def _emit_mul_balanced(c, operand, dst, q_or_t, var_prefix, type_macro,
                           indent="    "):
        base = c["base"]
        info_c = c["info"]
        if q_or_t == "Q":
            const_op, pos_op, neg_op = "+", "+", "-"
        elif q_or_t == "T":
            const_op, pos_op, neg_op = "-", "-", "+"
        else:
            raise RuntimeError(
                f"_emit_mul_balanced: q_or_t must be 'Q' or 'T' "
                f"(got {q_or_t!r}).")
        out = []
        out.append(
            f"{indent}{type_macro} {var_prefix}_base = "
            f"({operand} << {base}) {const_op} {operand};")
        out.append("")
        for exp in c["union_exps"]:
            out.append(
                f"{indent}{type_macro} {var_prefix}_sh_{exp} = "
                f"({operand} << {exp});")
        out.append("")
        pos_terms = _emit_pn_slots(out, "pos", c["pos_slot_terms"], var_prefix,
                                   type_macro, indent, info_c)
        neg_terms = _emit_pn_slots(out, "neg", c["neg_slot_terms"], var_prefix,
                                   type_macro, indent, info_c)
        sum_str = f"{var_prefix}_base"
        for term in pos_terms:
            sum_str += f" {pos_op} {term}"
        for term in neg_terms:
            sum_str += f" {neg_op} {term}"
        if q_or_t == "T":
            delta_items = []
            for i, p in enumerate(info_c):
                dn = p.get("delta_naf", [])
                if not dn:
                    continue
                parts = [f"{'+' if ds > 0 else '-'} ({operand} << {de})"
                         for ds, de in dn]
                expr = " ".join(parts)
                if expr.startswith("+ "):
                    expr = expr[2:]
                delta_items.append(
                    (f"is_{_sanitize(p['alias'])}",
                     f"({type_macro})({expr})"))
            if delta_items:
                dvar = f"{var_prefix}_delta"
                if len(delta_items) == len(info_c):
                    out.extend(_balanced_mux_tree(delta_items, dvar,
                                                  type_macro, indent))
                elif len(delta_items) == 1:
                    pred, cand = delta_items[0]
                    out.append(
                        f"{indent}{type_macro} {dvar} = "
                        f"{pred} ? {cand} : ({type_macro})0;")
                else:
                    inner = f"{dvar}_inner"
                    out.extend(_balanced_mux_tree(delta_items, inner,
                                                  type_macro, indent))
                    preds_or = " || ".join(pp for pp, _ in delta_items)
                    out.append(
                        f"{indent}{type_macro} {dvar} = "
                        f"({preds_or}) ? {inner} : ({type_macro})0;")
                sum_str += f" + {dvar}"
        out.append(f"{indent}{type_macro} {dst} = {sum_str};")
        return out

    L = list(header_top)
    L.append("// Auto-generated lightweight-prime shift-add multiplication helpers")
    L.append("// (K_group-mixed batch: per-class recipes selected by cls = PRIME_CLASS).")
    L.append("// Contract: keeps Barrett/Shoup structure; replaces selected")
    L.append("// constant multiplications with shift-add expressions.")
    L.append(f"// cls == 0: 52-class, base = 51 ; cls == 1: 62-class, base = 61 ; "
             f"K = {K_env}, NUM_PRIMES = {NP}")
    L.append("//")
    L.append("// Per-prime data:")
    for c in classes:
        for p in c["info"]:
            naf_str = ", ".join(
                f"{'+' if s2 > 0 else '-'}2^{e}" for s2, e in p["s_naf"])
            gi = aliases_clean_all.index(_sanitize(p["alias"]))
            L.append(f"//   [{gi}] cls{c['cls_idx']} {p['alias']:<8s} "
                     f"q={p['q']} T={p['T']} s_naf={naf_str}")
    L.append("")
    L.append("#define USE_REDUCE_SHIFTADD_MUL 1")
    L.append("")
    L.append("constexpr int SHIFTADD_MUL_BASE_C0 = 51;")
    L.append("constexpr int SHIFTADD_MUL_BASE_C1 = 61;")
    L.append(f"constexpr int SHIFTADD_MUL_W_ACC = {W_acc};")
    L.append("using ShiftaddMulAcc = ap_int<SHIFTADD_MUL_W_ACC>;")
    L.append("")
    for c in classes:
        tag = "C%d" % c["cls_idx"]
        if c["two_corr"]:
            L.append(
                f"// cls{c['cls_idx']}: {', '.join(c['two_corr'])} require r in "
                f"[2q, 3q) under one-correction; two corrections selected "
                f"for the class (max-policy)")
        else:
            L.append(
                f"// cls{c['cls_idx']}: all primes keep r in [0, 2q); "
                f"one correction is sufficient")
        L.append(f"#define SHIFTADD_MUL_CORRECTIONS_{tag} {c['corrections']}")
    L.append("")
    L.append("// BU side: full Barrett-style modular reduction with shift-add T*q1 and Q*q3,")
    L.append("// one pipeline per K_group class; result selected by cls.")
    L.append("inline Data reduce_shiftadd_mul(Data2 X, ModId mod_id, Data q_cur) {")
    L.append("#pragma HLS INLINE")
    L.append("")
    for i in range(NP):
        L.append(
            f"    const bool is_{aliases_clean_all[i]} = (mod_id == (ModId){i});")
    L.append("")
    L.append("    // Per-class recipe select: the absolute PRIME_CLASS rule")
    L.append("    // (0 = 52-class, 1 = 62-class) equals the set-relative cls")
    L.append("    // order for the only legal mixed map {52, 62}.")
    L.append("    const int cls = PRIME_CLASS[mod_id];")
    L.append("    ShiftaddMulAcc qq = (ShiftaddMulAcc)q_cur;")
    L.append("")
    for c in classes:
        cls_idx = c["cls_idx"]
        base = c["base"]
        tag = "_c%d" % cls_idx
        W_q1 = 2 * K_env - base + 1
        L.append(f"    // ---- cls{cls_idx}: {c['k_class']}-class pipeline "
                 f"(base = {base}) ----")
        L.append(f"    ap_uint<{W_q1}> q1_u{tag} = "
                 f"(ap_uint<{W_q1}>)(X >> {base});")
        L.append(f"    ShiftaddMulAcc q1{tag} = (ShiftaddMulAcc)q1_u{tag};")
        L.append("")
        L.extend(_emit_mul_balanced(c, f"q1{tag}", f"q2{tag}", "T",
                                    f"q2{tag}", "ShiftaddMulAcc"))
        L.append("")
        L.append(f"    ShiftaddMulAcc q3{tag} = (q2{tag} >> {base});")
        L.append("")
        L.extend(_emit_mul_balanced(c, f"q3{tag}", f"p{tag}", "Q",
                                    f"p{tag}", "ShiftaddMulAcc"))
        L.append("")
        L.append(f"    ShiftaddMulAcc r{tag} = (ShiftaddMulAcc)X - p{tag};")
        L.append(f"    if (r{tag} >= qq) r{tag} -= qq;")
        L.append(f"#if SHIFTADD_MUL_CORRECTIONS_C{cls_idx} >= 2")
        L.append(f"    if (r{tag} >= qq) r{tag} -= qq;")
        L.append("#endif")
        L.append("")
    L.append("    return (Data)(cls ? r_c1 : r_c0);")
    L.append("}")
    L.append("")
    L.append("#endif  // NTT_SHIFTADD_H")
    L.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(L))


def emit_shiftadd_header(case_dir, K_env, aliases_q, enabled,
                         style="slot_direct",
                         balanced_selector_scope="none",
                         shiftadd_fold_depth=2,
                         shiftadd_correction="positive_only",
                         shiftadd_max_slots=2,
                         shiftadd_mul=False,
                         math_self_check_log_path=None,
                         *, recipe=None, single_prime_mode="legacy"):
    """Emit ``src/ntt_shiftadd.h`` for a multi-prime case.

    When ``shiftadd_mul`` is true, the file emits the Barrett/Shoup-preserving
    shift-add multiplication helper ``reduce_shiftadd_mul`` (BU side) and
    defines ``USE_REDUCE_SHIFTADD_MUL``.

    When ``shiftadd_mul`` is false and ``enabled`` is true, the legacy
    ``reduce_shiftadd`` fold-based reducer is emitted (defines
    ``USE_REDUCE_SHIFTADD``).

    ``single_prime_mode`` accepts "legacy" (default) or "normalized".
    ``normalized`` applies ONLY when NUM_PRIMES==1 -- it resolves the redundant
    single-prime ``is_<alias>`` / ``mod_id`` recipe muxes at generation time
    (NP==1 -> the predicate is always true) and emits the selected candidate
    directly. The function SIGNATURE keeps ``ModId mod_id`` (retained, unused)
    so the ntt.cpp call site stays byte-identical. ``legacy`` and every
    multi-prime case (NP>=2) emit the unchanged byte-identical path.
    """
    if single_prime_mode not in ("legacy", "normalized"):
        raise ValueError(
            "single_prime_mode=%r is not valid. Use 'legacy' (default) or "
            "'normalized'." % single_prime_mode)
    # S4 normalized resolution is single-prime ONLY; NP>=2 stays legacy.
    sp_norm = (single_prime_mode == "normalized" and len(aliases_q) == 1)
    src_dir = os.path.join(case_dir, "src")
    out_path = os.path.join(src_dir, "ntt_shiftadd.h")

    header_top = [
        "#ifndef NTT_SHIFTADD_H",
        "#define NTT_SHIFTADD_H",
        "",
    ]

    if not enabled:
        with open(out_path, "w") as f:
            f.write("\n".join(header_top + [
                "// shiftadd disabled: stub header.",
                "// Neither USE_REDUCE_SHIFTADD nor USE_REDUCE_SHIFTADD_MUL is defined.",
                "",
                "#endif  // NTT_SHIFTADD_H",
                "",
            ]))
        return

    # Recipe-class mixed dispatch. The active generation path passes a
    # ReduceRecipe; direct low-level callers without a recipe get only the
    # compatibility fallback in _shiftadd_mixed_reduce_sub_batches.
    _mixed_subs = _shiftadd_mixed_reduce_sub_batches(aliases_q, recipe)
    _mixed_reduce = len(_mixed_subs) > 1

    if shiftadd_mul:
        if _mixed_reduce:
            return _emit_shiftadd_mul_header_mixed(
                out_path,
                K_env,
                aliases_q,
                header_top,
                recipe=recipe,
            )
        return _emit_shiftadd_mul_header(
            out_path, K_env, aliases_q, header_top, recipe=recipe,
            single_prime_normalized=sp_norm)

    if style != "slot_direct":
        raise RuntimeError(
            f"Unsupported shiftadd_style {style!r}; only 'slot_direct' is "
            f"accepted in the stable generator path.")

    if _mixed_reduce:
        return _emit_shiftadd_reduce_header_mixed(
            out_path, K_env, aliases_q, header_top,
            balanced_selector_scope=balanced_selector_scope,
            math_self_check_log_path=math_self_check_log_path,
            recipe=recipe)

    def _write_math_log(lines):
        if math_self_check_log_path is None:
            return
        try:
            os.makedirs(os.path.dirname(math_self_check_log_path),
                        exist_ok=True)
        except (OSError, ValueError):
            pass
        with open(math_self_check_log_path, "w") as f:
            f.write("\n".join(lines) + "\n")



    # q-derived reduce: auto-select fold_depth (2 or 3) and correction internally.
    # The segmented schema does not expose these; the B=min(K_i) anchor + per-prime
    # (q_i-1)^2 bounds drive the choice (TI -> fold 2; TII 51/52 -> fold 3).
    # The resolver is deterministic and recipe.reduce_resolver_params is the
    # SAME batch call -> on the recipe-governed path consume (fold, correction)
    # FROM the recipe and keep the live resolver as an assert-shim (byte-identical).
    (resolved_fold_depth,
     resolved_correction,
     auto_report) = _shiftadd_resolve_fold_and_correction(
        aliases_q, K_env,
        fold_depth_req="auto",
        correction_req="auto",
        recipe=recipe)
    if recipe is not None and getattr(recipe, "batch_legal_shiftadd", False):
        want = tuple(recipe.reduce_resolver_params)
        if (resolved_fold_depth, resolved_correction) != want:
            raise RuntimeError(
                "shiftadd resolver drift: live (fold, correction) %r != recipe "
                "reduce_resolver_params %r" % (
                    (resolved_fold_depth, resolved_correction), want))
        resolved_fold_depth, resolved_correction = want
    shiftadd_fold_depth = resolved_fold_depth
    shiftadd_correction = resolved_correction
    print("[shiftadd auto-analysis]")
    for line in auto_report:
        print("  " + line)

    WFold1, WFold2, WFold3, WFinal, ell, all_FN, info = _shiftadd_envelope_widths(
        aliases_q, K_env, fold_depth=shiftadd_fold_depth, recipe=recipe)

    if not all_FN:
        gate = "F3_OK" if shiftadd_fold_depth == 3 else "F2_OK"
        with open(out_path, "w") as f:
            f.write("\n".join(header_top + [
                f"#error \"reduce_shiftadd: at least one prime is not {gate}\"",
                "#error \"  under the envelope bound. Stop and report.\"",
                "",
                "#endif  // NTT_SHIFTADD_H",
                "",
            ]))
        raise RuntimeError(
            f"reduce_shiftadd: not all primes are {gate} under the "
            f"envelope bound K_env={K_env}, fold_depth={shiftadd_fold_depth}.\n"
            f"  Per-prime info: " + ", ".join(
                f"{p['alias']}({gate}={p.get('F3_OK', p['F2_OK'])})"
                for p in info))

    ok_math, math_log = _shiftadd_math_self_check(
        aliases_q, K_env,
        fold_depth=shiftadd_fold_depth,
        correction=shiftadd_correction,
        recipe=recipe)
    for line in math_log:
        print("  " + line)
    _write_math_log(math_log)
    if not ok_math:
        raise RuntimeError(
            f"math self-check FAILED for {[a for a, _ in aliases_q]}; "
            f"refusing to emit reducer.")

    # q-derived emit slot budget: required slots = max over primes of the count of
    # non-zero-exponent NAF terms of alpha (the exact descriptors the slot allocator
    # packs, see _shiftadd_slot_assignment). TI -> 2; TII 51/52 -> 3. Derived from the
    # batch, not hard-coded, not JSON-exposed; a batch needing <= 2 keeps max_slots = 2.
    required_slots = max(
        (sum(1 for (_s, e) in p["alpha_naf"] if e != 0) for p in info),
        default=2)
    derived_max_slots = max(2, required_slots)
    return _emit_shiftadd_slot_direct(
        out_path, K_env, info, header_top,
        WFold1, WFold2, WFinal, ell,
        balanced_selector_scope=balanced_selector_scope,
        fold_depth=shiftadd_fold_depth,
        WFold3=WFold3,
        correction=shiftadd_correction,
        max_slots=derived_max_slots,
        single_prime_normalized=sp_norm)

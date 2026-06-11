#!/usr/bin/env python3
"""Phase 3.5 S3b tests for reduce_recipe.py (V1-V5 of the S3b plan, as amended
by the 2026-06-07 two-layer field-semantics decision).

Oracles:
  (a) the LIVE pure helpers (_shiftadd_ell/_q_anchor/_naf/envelope/resolver/
      self-checks, _k_arith_for, classify_reduce_set) -- equivalence-by-
      construction (R1);
  (b) the S3a inventory table (checkpoint 20260607_2054) re-embedded below --
      every recorded per-alias observation must be reproduced exactly;
  (c) the S3b live-probe refinement of S3a F1 (2026-06-07): at the FIXED
      default point (fd2, positive_only) the large family passes 28/28 and the
      small family is MIXED (21 pass incl. TII_33; 20 fail hard); the LIVE
      auto-resolver emits every known-family set probed (all solos, the named
      pairs, and both whole families -- small via (3, signed)).
NOTE: anchor_exp == k_arith - 1 is asserted ONLY as the recorded S3a
OBSERVATION over the current 69 known aliases -- it is NOT a derivation rule
and the constructor must not (and does not) derive anchor_exp from k_arith.
Run (no pytest needed):  ~/autontt-venv/bin/python3 tests/test_reduce_recipe.py"""
import ast
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from code_generator import reduce_recipe as rr
from code_generator import reduce_generator as rg
from code_generator import shiftadd_generator as sg
import generate_code as gc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PT = rr.REDUCE_SELF_CHECK_DEFAULT_PARAMS  # (2, "positive_only")


def _pairs(*names):
    return tuple((n, gc.KNOWN_PRIMES[n]) for n in names)


def _solo(name):
    return rr.build_reduce_recipe(_pairs(name)).primes[0]


# S3a 69-alias inventory pin (checkpoint phase3_5_s3a_reduce_recipe_inventory_
# checkpoint_20260607_2054.md, section A): alias | q(hex) | bl | k_arith |
# anchorB | |alpha_naf| | T_user==T_floor.
S3A_TABLE = (
    ("TI_1", "1FFFFFFFFC000001", 61, 62, 61, 2, True),
    ("TI_2", "1FFFFFFFFEF00001", 61, 62, 61, 3, True),
    ("TI_3", "1FFFFFFFFF000001", 61, 62, 61, 2, True),
    ("TI_4", "1FFFFFFFFFC80001", 61, 62, 61, 3, True),
    ("TI_5", "1FFFFFFFFFE00001", 61, 62, 61, 2, True),
    ("TI_6", "1FFFFFFFFFE10001", 61, 62, 61, 3, True),
    ("TI_7", "2000000000500001", 62, 62, 61, 3, True),
    ("TI_8", "20000000007C0001", 62, 62, 61, 3, True),
    ("TI_9", "2000000000A00001", 62, 62, 61, 3, True),
    ("TI_10", "2000000000F80001", 62, 62, 61, 3, True),
    ("TI_11", "2000000002800001", 62, 62, 61, 3, True),
    ("TI_12", "2000000004010001", 62, 62, 61, 3, True),
    ("TII_1", "0007FFFFDFF80001", 51, 52, 51, 3, False),
    ("TII_2", "0007FFFFEF880001", 51, 52, 51, 4, False),
    ("TII_3", "0007FFFFEFC00001", 51, 52, 51, 3, False),
    ("TII_4", "0007FFFFF4100001", 51, 52, 51, 4, False),
    ("TII_5", "0007FFFFF7400001", 51, 52, 51, 4, False),
    ("TII_6", "0007FFFFF7F80001", 51, 52, 51, 3, False),
    ("TII_7", "0007FFFFF9000001", 51, 52, 51, 3, False),
    ("TII_8", "0007FFFFFBB80001", 51, 52, 51, 4, False),
    ("TII_9", "0007FFFFFC900001", 51, 52, 51, 4, False),
    ("TII_10", "0007FFFFFCFC0001", 51, 52, 51, 4, False),
    ("TII_11", "0007FFFFFD880001", 51, 52, 51, 4, True),
    ("TII_12", "0007FFFFFDB00001", 51, 52, 51, 4, True),
    ("TII_13", "0007FFFFFDC40001", 51, 52, 51, 4, True),
    ("TII_14", "0007FFFFFDDC0001", 51, 52, 51, 4, True),
    ("TII_15", "0007FFFFFE0C0001", 51, 52, 51, 4, True),
    ("TII_16", "0007FFFFFE780001", 51, 52, 51, 4, True),
    ("TII_17", "0007FFFFFE900001", 51, 52, 51, 4, True),
    ("TII_18", "0007FFFFFF080001", 51, 52, 51, 3, True),
    ("TII_19", "0007FFFFBFF00001", 51, 52, 51, 3, False),
    ("TII_20", "0007FFFFDFC40001", 51, 52, 51, 4, False),
    ("TII_21", "0007FFFFFF240001", 51, 52, 51, 4, True),
    ("TII_22", "0007FFFFFF900001", 51, 52, 51, 3, True),
    ("TII_23", "0007FFFFFF9C0001", 51, 52, 51, 4, True),
    ("TII_24", "00080000001C0001", 52, 52, 51, 3, True),
    ("TII_25", "00080000002C0001", 52, 52, 51, 4, True),
    ("TII_26", "0008000000500001", 52, 52, 51, 3, True),
    ("TII_27", "0008000000940001", 52, 52, 51, 4, True),
    ("TII_28", "0008000001600001", 52, 52, 51, 4, True),
    ("TII_29", "0008000001D00001", 52, 52, 51, 4, True),
    ("TII_30", "0008000002000001", 52, 52, 51, 2, True),
    ("TII_31", "0008000002080001", 52, 52, 51, 3, True),
    ("TII_32", "0008000002200001", 52, 52, 51, 3, True),
    ("TII_33", "0008000002480001", 52, 52, 51, 4, True),
    ("TII_34", "0008000003EC0001", 52, 52, 51, 4, False),
    ("TII_35", "0008000004240001", 52, 52, 51, 4, False),
    ("TII_36", "0008000006E00001", 52, 52, 51, 4, False),
    ("TII_37", "0008000010040001", 52, 52, 51, 3, False),
    ("TII_38", "0008000010100001", 52, 52, 51, 3, False),
    ("TII_39", "0008000020000001", 52, 52, 51, 2, False),
    ("TII_40", "0007FFFFC0080001", 51, 52, 51, 3, False),
    ("TII_41", "0008000020240001", 52, 52, 51, 4, False),
    ("TIII_1", "1FFFFFFFFF000001", 61, 62, 61, 2, True),
    ("TIII_2", "1FFFFFFFFFE00001", 61, 62, 61, 2, True),
    ("TIII_3", "20000000007C0001", 62, 62, 61, 3, True),
    ("TIII_4", "2000000000F80001", 62, 62, 61, 3, True),
    ("TIII_5", "2000000008200001", 62, 62, 61, 3, True),
    ("TIII_6", "200000001FC00001", 62, 62, 61, 3, True),
    ("TIII_7", "2000000040040001", 62, 62, 61, 3, True),
    ("TIII_8", "2000000044000001", 62, 62, 61, 3, True),
    ("TIII_9", "1FFFFFFFFFC80001", 61, 62, 61, 3, True),
    ("TIII_10", "2000000000500001", 62, 62, 61, 3, True),
    ("TIII_11", "2000000000A00001", 62, 62, 61, 3, True),
    ("TIII_12", "2000000002800001", 62, 62, 61, 3, True),
    ("TIII_13", "200000000E000001", 62, 62, 61, 3, True),
    ("TIII_14", "2000000020080001", 62, 62, 61, 3, True),
    ("TIII_15", "2000000040100001", 62, 62, 61, 3, True),
    ("TIII_16", "2000000050000001", 62, 62, 61, 3, True),
)

# S3b live-probe pin (2026-06-07): the 20 small-family aliases whose FIXED-
# POINT (fd2, positive_only) self-check fails HARD (the other 21 pass). All
# 41 remain emittable by the live resolver. OBSERVED to coincide with the
# small-family t_match=False rows (both track |alpha| magnitude) -- a
# recorded coincidence of the current table, NOT a rule.
SMALL_FIXED_POINT_FAILERS = frozenset(
    ["TII_%d" % i for i in range(1, 11)] + ["TII_19", "TII_20"]
    + ["TII_%d" % i for i in range(34, 42)])


def test_v1_table_covers_known_primes_exactly():
    assert len(S3A_TABLE) == 69, len(S3A_TABLE)
    assert {r[0] for r in S3A_TABLE} == set(gc.KNOWN_PRIMES), "alias set drift"
    for alias, qhex, _bl, _ka, _b, _w, _t in S3A_TABLE:
        assert int(qhex, 16) == gc.KNOWN_PRIMES[alias], "q drift for %s" % alias


def test_v1_sixty_nine_alias_equivalence_sweep():
    """V1: per-alias recipe facts == live helpers == the S3a recorded rows."""
    for alias, qhex, bl, ka, anchor, w, tmatch in S3A_TABLE:
        q = int(qhex, 16)
        f = _solo(alias)
        assert (f.q, f.k_bits, f.k_arith) == (q, bl, ka), alias
        # anchor_exp: stored independently; EQUAL to the current heuristic
        # (equality-to-heuristic, NOT a derivation rule -- C3).
        assert f.anchor_exp == anchor == sg._shiftadd_ell(q), alias
        assert f.anchor_exp == f.k_arith - 1, \
            "%s: recorded S3a OBSERVATION violated (current families only)" % alias
        # reduce-route working forms == the exact live helpers.
        assert f.alpha == (1 << anchor) - q, alias
        assert f.alpha_naf == tuple(sg._shiftadd_naf(f.alpha)), alias
        assert f.alpha_naf_weight == len(f.alpha_naf) == w, alias
        assert sum(s * (1 << e) for s, e in f.alpha_naf) == f.alpha, alias
        # mul-route working forms == the exact live mul math.
        assert f.s == q - (1 << anchor) - 1, alias
        assert f.s_naf == tuple(sg._shiftadd_naf(f.s)), alias
        assert f.t_user == (1 << anchor) - f.s - 1, alias
        assert f.t_floor == (1 << (2 * anchor)) // q, alias
        assert f.t_match == (f.t_user == f.t_floor) == tmatch, alias
        assert f.delta == f.t_floor - f.t_user, alias
        assert f.delta_naf == tuple(sg._decompose_signed_pow2(f.delta)), alias
        if f.t_match:
            assert f.delta == 0 and f.delta_naf == (), alias
        else:
            assert f.delta != 0, alias
        assert f.t_user + f.delta == f.t_floor, alias  # T-exactness domain rule
        # identity / recording discipline.
        assert f.provenance == "known_table", alias
        assert f.rel_class == 0, alias  # solo set: single class present
        assert f.reduce_default_self_check_params == DEFAULT_PT, alias
        assert f.wfold1 and f.wfold2 and f.wfold3, alias
        assert f.f2_ok is not None and f.f3_ok is not None, alias


def test_v1_family_counts_and_distributions():
    """V1/S3a summary pins: 41@51/28@61; NAF weights {2:7,3:37,4:25}; T 49/20."""
    anchors = [r[4] for r in S3A_TABLE]
    assert anchors.count(51) == 41 and anchors.count(61) == 28
    weights = [r[5] for r in S3A_TABLE]
    assert {wv: weights.count(wv) for wv in set(weights)} == {2: 7, 3: 37, 4: 25}
    tm = [r[6] for r in S3A_TABLE]
    assert tm.count(True) == 49 and tm.count(False) == 20
    assert [r[3] for r in S3A_TABLE].count(52) == 41  # k_arith mirrors families
    assert [r[3] for r in S3A_TABLE].count(62) == 28


def test_v1_route_status_two_layers_by_family():
    """F1 as REFINED by the S3b live probes (two never-conflated layers):
    FIXED POINT (fd2, positive_only): large 28/28 PASS; small MIXED -- 21
    pass (incl. solo TII_33), the pinned 20 fail hard.
    LIVE RESOLVER: every known alias is solo-emittable (69/69) -- small-
    family reduce is a parameterization/resolver matter, NOT unsupported.
    Mul: both families served (0 unsupported, <= 2 corrections)."""
    resolver_dist = {}
    for alias, qhex, _bl, _ka, anchor, _w, _t in S3A_TABLE:
        f = _solo(alias)
        # fixed-point layer (recorded data).
        expect_default = (anchor == 61) or (alias not in SMALL_FIXED_POINT_FAILERS)
        assert f.reduce_default_self_check_ok is expect_default, alias
        assert f.reduce_default_self_check_params == DEFAULT_PT, alias
        # live-resolver layer (practical capability): all 69 solo-emittable.
        assert f.reduce_resolver_emit_ok is True, alias
        assert f.reduce_resolver_reason is None, alias
        assert f.reduce_unsupported_reason is None, alias  # NEVER set by a fixed-point FAIL
        rfd, rcorr = f.reduce_resolver_params
        if expect_default:
            assert (rfd, rcorr) == DEFAULT_PT, alias  # envelope-safe at fold 2
            assert f.f2_ok is True, alias
        else:
            assert rfd == 3 and rcorr in ("positive_only", "signed"), alias
            assert f.f2_ok is False, alias  # fd2 envelope fails -> resolver went fd3
        resolver_dist[(anchor, f.reduce_default_self_check_ok,
                       f.reduce_resolver_params)] = resolver_dist.get(
            (anchor, f.reduce_default_self_check_ok, f.reduce_resolver_params), 0) + 1
        # OBSERVED coincidence pin (current table only; NOT a rule).
        if anchor == 51:
            assert (alias in SMALL_FIXED_POINT_FAILERS) == (not f.t_match), alias
        # mul route: BOTH families served by the current emitter.
        assert f.mul_unsupported_reason is None, alias
        assert f.mul_corrections in (1, 2), alias
        if not f.t_match:  # supported-with-correction, NOT unsupported (C5)
            assert f.mul_unsupported_reason is None and f.delta != 0, alias
    # live-probed distribution pin (2026-06-07).
    assert resolver_dist == {
        (61, True, (2, "positive_only")): 28,
        (51, True, (2, "positive_only")): 21,
        (51, False, (3, "positive_only")): 13,
        (51, False, (3, "signed")): 7,
    }, resolver_dist


def test_v1_f1_refinement_family_batches():
    """The whole-family batches that grounded S3a's family-level F1 claim:
    small family fails AS A BATCH at the fixed point (20 hard failers) yet the
    live resolver EMITS it via (3, signed); large family passes at the fixed
    point and emits via (2, positive_only)."""
    small = tuple(p for p in
                  ((a, gc.KNOWN_PRIMES[a]) for a, *_ in S3A_TABLE)
                  if p[1].bit_length() <= 52)
    large = tuple(p for p in
                  ((a, gc.KNOWN_PRIMES[a]) for a, *_ in S3A_TABLE)
                  if p[1].bit_length() > 52)
    rs = rr.build_reduce_recipe(small)
    assert len(rs.primes) == 41
    assert rs.reduce_default_self_check_ok is False      # family-level F1 stands
    assert rs.reduce_resolver_emit_ok is True            # ...but the emit path serves it
    assert rs.reduce_resolver_params == (3, "signed")
    assert rs.can_emit_shiftadd_reduce and rs.can_emit_shiftadd_mul
    rl = rr.build_reduce_recipe(large)
    assert len(rl.primes) == 28
    assert rl.reduce_default_self_check_ok is True
    assert rl.reduce_resolver_emit_ok is True
    assert rl.reduce_resolver_params == DEFAULT_PT
    assert rl.can_emit_shiftadd_reduce and rl.can_emit_shiftadd_mul


def test_v2_same_family_batch_equivalence_sweep():
    """V2: every same-family 2-prime combination reproduces
    _shiftadd_q_anchor (B, K_group, K_spread) exactly, is batch-legal, keeps
    the two layers consistent, and is live-resolver emittable."""
    small = [r[0] for r in S3A_TABLE if r[4] == 51]
    large = [r[0] for r in S3A_TABLE if r[4] == 61]
    n = 0
    dist = {}
    for fam, fam_anchor in ((small, 51), (large, 61)):
        for a, b in itertools.combinations(fam, 2):
            pair = _pairs(a, b)
            r = rr.build_reduce_recipe(pair)
            heur = sg._shiftadd_q_anchor(pair)
            assert (r.b_anchor, r.anchor_k_group, r.k_spread) == heur, (a, b)
            assert r.b_anchor == r.anchor_family == fam_anchor, (a, b)
            assert r.k_spread in (0, 1), (a, b)
            assert r.batch_legal_shiftadd and not r.fail_fast_reasons, (a, b)
            # fixed-point batch layer = AND over member solo verdicts.
            assert r.reduce_default_self_check_ok == (
                r.primes[0].reduce_default_self_check_ok
                and r.primes[1].reduce_default_self_check_ok), (a, b)
            # live-resolver batch layer = the practical capability.
            assert r.reduce_resolver_emit_ok is True, (a, b)
            assert r.can_emit_shiftadd_reduce is r.reduce_resolver_emit_ok, (a, b)
            assert r.reduce_resolver_reason is None, (a, b)
            assert r.reduce_resolver_params[0] in (2, 3), (a, b)
            assert r.can_emit_shiftadd_mul, (a, b)
            assert r.provenance == "known_table" and r.barrett_always, (a, b)
            dist[r.reduce_resolver_params] = dist.get(r.reduce_resolver_params, 0) + 1
            n += 1
    assert n == 820 + 378, n  # C(41,2) + C(28,2)
    print("  (v2: %d same-family pairs swept; resolver params dist: %s)"
          % (n, sorted(dist.items())))


def test_v2_mixed_pairs_kgroup_legal_sweep():
    """V2 (updated for K_group mixed support): every known cross-family pair
    is a LEGAL two-family batch with per-class cls_recipes (cls0 = 52-class
    solo, cls1 = 62-class solo; the 1-member class verdict == the cached solo
    resolver facts). The raw single-anchor heuristic STILL raises for the
    full pair (it is unchanged; the mixed emitters never call it batch-wide)."""
    small = [r[0] for r in S3A_TABLE if r[4] == 51]
    large = [r[0] for r in S3A_TABLE if r[4] == 61]
    n = 0
    for a in small:
        for b in large:
            pair = _pairs(a, b)
            r = rr.build_reduce_recipe(pair)
            assert r.batch_legal_shiftadd, (a, b)
            assert r.fail_fast_reasons == (), (a, b)
            assert r.can_emit_shiftadd_reduce and r.can_emit_shiftadd_mul, (a, b)
            assert r.reduce_resolver_emit_ok is True, (a, b)
            assert r.reduce_resolver_params is None, (a, b)  # params live per class
            assert r.reduce_resolver_reason is None, (a, b)
            assert r.barrett_always, (a, b)
            assert r.k_spread >= 3 and r.class_map == (52, 62), (a, b)
            assert r.cls_recipes is not None and len(r.cls_recipes) == 2, (a, b)
            c0, c1 = r.cls_recipes
            assert (c0.k_class, c0.b_anchor, c0.aliases) == (52, 51, (a,)), (a, b)
            assert (c1.k_class, c1.b_anchor, c1.aliases) == (62, 61, (b,)), (a, b)
            by_alias = {f.alias: f for f in r.primes}
            assert c0.reduce_resolver_params == \
                by_alias[a].reduce_resolver_params, (a, b)
            assert c1.reduce_resolver_params == \
                by_alias[b].reduce_resolver_params, (a, b)
            try:
                sg._shiftadd_q_anchor(pair)
            except ValueError as exc:
                assert "K_spread=%d >= 3" % r.k_spread in str(exc), (a, b)
            else:
                raise AssertionError("heuristic accepted mixed pair %r" % ((a, b),))
            n += 1
    assert n == 41 * 28, n
    print("  (v2: %d cross-family pairs swept)" % n)


def test_v3_named_set_route_status_table():
    """V3: the 5 named sets, route status at BOTH layers (live facts; the
    [TII_33] fixed-point row REFINES the S3a section-C table per the S3b
    live probes -- solo TII_33 passes the fixed point and would emit)."""
    # (names, b_anchor, family, anchor_k_group, k_spread, class_map,
    #  default_ok, resolver_params, can_emit_reduce, can_mul, legal,
    #  mul_corr, max_slots, rel_classes)
    table = (
        (("TI_9",), 61, 61, 62, 0, (62,),
         True, (2, "positive_only"), True, True, True, 1, 2, (0,)),
        (("TII_33",), 51, 51, 52, 0, (52,),
         True, (2, "positive_only"), True, True, True, 2, 3, (0,)),
        (("TI_2", "TI_9"), 61, 61, 62, 1, (62,),
         True, (2, "positive_only"), True, True, True, 1, 2, (0, 0)),
        (("TII_2", "TII_33"), 51, 51, 52, 1, (52,),
         False, (3, "signed"), True, True, True, 2, 3, (0, 0)),
        # K_group-mixed two-family pair: LEGAL (per-class recipes); the batch
        # fixed-point verdict is the evaluable AND (both solos pass here) and
        # the batch resolver params live per class (None at batch level).
        (("TI_9", "TII_33"), 51, None, 62, 10, (52, 62),
         True, None, True, True, True, 2, 3, (1, 0)),
    )
    for (names, b, fam, kg, ks, cmap, dok, rparams, can_r, can_m, legal,
         mc, slots, rels) in table:
        r = rr.build_reduce_recipe(_pairs(*names))
        ctx = names
        assert r.aliases == names, ctx
        assert (r.b_anchor, r.anchor_family) == (b, fam), ctx
        assert (r.anchor_k_group, r.k_spread, r.class_map) == (kg, ks, cmap), ctx
        # fixed-point layer (data) vs live-resolver layer (capability).
        assert r.reduce_default_self_check_ok is dok, ctx
        assert r.reduce_default_self_check_params == DEFAULT_PT, ctx
        assert r.reduce_resolver_params == rparams, ctx
        assert r.reduce_resolver_emit_ok == can_r, ctx
        assert r.can_emit_shiftadd_reduce == can_r, ctx
        assert r.can_emit_shiftadd_mul == can_m, ctx
        assert r.batch_legal_shiftadd == legal, ctx
        assert r.mul_corrections == mc, ctx
        assert (r.max_slots, r.slot_heuristic) == \
            (slots, "desc_exp" if slots == 3 else "asc_exp"), ctx
        assert r.barrett_always and r.provenance == "known_table", ctx
        assert r.domain_tags == rr.DOMAIN_TAGS and len(r.domain_tags) == 4, ctx
        assert tuple(f.rel_class for f in r.primes) == rels, ctx
        if legal:
            assert not r.fail_fast_reasons, ctx
        else:
            assert any("k_spread>=3" in x for x in r.fail_fast_reasons), ctx
        # cls semantics guard: mixed -> cls0 = SMALLER class PRESENT.
        if len(cmap) == 2:
            by_alias = {f.alias: f for f in r.primes}
            assert by_alias["TII_33"].rel_class == 0, "TII_33 must be cls0"
            assert by_alias["TI_9"].rel_class == 1, "TI_9 must be cls1"
            assert r.cls_recipes is not None and len(r.cls_recipes) == 2, ctx
            assert [(c.k_class, c.b_anchor) for c in r.cls_recipes] == \
                [(52, 51), (62, 61)], ctx
        else:
            assert r.cls_recipes is None, ctx
    # [TII_2, TII_33] per-member fixed-point split (user-required detail):
    # the batch fixed-point verdict is False through TII_2 alone.
    r = rr.build_reduce_recipe(_pairs("TII_2", "TII_33"))
    by_alias = {f.alias: f for f in r.primes}
    assert by_alias["TII_2"].reduce_default_self_check_ok is False
    assert by_alias["TII_33"].reduce_default_self_check_ok is True
    assert by_alias["TII_2"].reduce_resolver_emit_ok is True   # solo (3, positive_only)
    assert by_alias["TII_2"].reduce_resolver_params == (3, "positive_only")
    assert by_alias["TII_33"].reduce_resolver_params == (2, "positive_only")


def test_two_layer_non_conflation():
    """The semantics decision's core rule: a fixed-point FAIL must NOT mark
    the route unsupported when the live resolver can emit (TII_2); capability
    fields must equal the resolver layer, never the fixed-point layer."""
    f = _solo("TII_2")
    assert f.reduce_default_self_check_ok is False           # fixed point: FAIL
    assert f.reduce_resolver_emit_ok is True                 # live emit path: serves it
    assert f.reduce_resolver_params == (3, "positive_only")
    assert f.reduce_unsupported_reason is None               # NOT unsupported
    r2 = rr.build_reduce_recipe(_pairs("TII_2"))
    assert r2.can_emit_shiftadd_reduce is True               # capability = resolver layer
    assert r2.reduce_default_self_check_ok is False          # data = fixed point


def test_v4_int_custom_recorded_not_raised():
    """V4 (C5): INT/custom -> provenance 'unsupported_custom' RECORDED, no
    raise in S3b; Barrett stays recipe-free/unaffected (barrett_always)."""
    # (a) in-family custom (62-bit value): current emitters would process it.
    q62 = gc.KNOWN_PRIMES["TI_9"]
    r = rr.build_reduce_recipe((("INT_%d" % q62, q62),))
    f = r.primes[0]
    assert f.provenance == r.provenance == "unsupported_custom"
    assert f.anchor_exp == 61 and r.batch_legal_shiftadd and r.barrett_always
    assert f.reduce_resolver_emit_ok is True and r.can_emit_shiftadd_reduce
    # (b) out-of-family bit length (40-bit): anchor unavailable, recorded.
    q40 = (1 << 39) + 7
    r40 = rr.build_reduce_recipe((("INT_%d" % q40, q40),))
    f40 = r40.primes[0]
    assert f40.provenance == "unsupported_custom" and f40.k_bits == 40
    assert f40.anchor_exp is None and f40.k_arith == 52  # Barrett still classifies
    assert "outside {51, 52, 61, 62}" in f40.reduce_unsupported_reason
    assert "outside {51, 52, 61, 62}" in f40.mul_unsupported_reason
    assert f40.alpha is None and f40.s is None
    assert f40.reduce_default_self_check_ok is None
    assert f40.reduce_resolver_emit_ok is None and f40.reduce_resolver_params is None
    assert not r40.batch_legal_shiftadd
    assert not r40.can_emit_shiftadd_reduce and not r40.can_emit_shiftadd_mul
    assert r40.reduce_resolver_emit_ok is False
    assert any("outside {51, 52, 61, 62}" in x for x in r40.fail_fast_reasons)
    assert r40.b_anchor is None and r40.anchor_family is None
    assert r40.barrett_always, "Barrett serves bl<=62 -- unaffected"
    # (c) >62-bit custom: even classification is unavailable -- still recorded.
    q63 = (1 << 63) + 29
    r63 = rr.build_reduce_recipe((("INT_%d" % q63, q63),))
    f63 = r63.primes[0]
    assert f63.k_arith is None and f63.rel_class is None and r63.class_map == ()
    assert not r63.barrett_always
    assert any("classification unavailable" in x for x in r63.fail_fast_reasons)
    # (d) known + custom mix: worst provenance dominates. Both bit lengths
    # are RECOGNIZED, so the recipe records emit-level two-family legality
    # (same posture as the in-family custom in (a)); the S3d generate-level
    # custom gate is what blocks it on shiftadd routes, by provenance.
    rmix = rr.build_reduce_recipe((("TI_9", q62),
                                   ("INT_%d" % gc.KNOWN_PRIMES["TII_33"],
                                    gc.KNOWN_PRIMES["TII_33"])))
    assert rmix.provenance == "unsupported_custom"
    assert rmix.batch_legal_shiftadd
    assert rmix.cls_recipes is not None and len(rmix.cls_recipes) == 2


def test_v4_int_custom_value_equivalence():
    """An INT alias carrying a KNOWN value must reproduce the known facts
    exactly (the current emitters key on q, not on the alias) -- only alias /
    provenance differ. Audited current behavior (S3a section D), not changed."""
    import dataclasses
    q = gc.KNOWN_PRIMES["TII_33"]
    known = _solo("TII_33")
    custom = rr.build_reduce_recipe((("INT_%d" % q, q),)).primes[0]
    assert dataclasses.replace(custom, alias=known.alias,
                               provenance=known.provenance) == known


def _module_imports(mod):
    """Imported-module names from a source file, normalized for the 2026-06-10 package
    split: every dotted module registers BOTH its full form and its final component
    (`from code_generator.reduce_generator import X` -> {"code_generator.reduce_generator",
    "reduce_generator"}), and `from code_generator/dse import <submodule>` registers the
    submodule. Audits target the MOVED sources (root shims are trivial aliases)."""
    with open(os.path.join(ROOT, mod), "r") as fh:
        tree = ast.parse(fh.read())
    imported = set()

    def _add(dotted):
        imported.add(dotted)
        imported.add(dotted.split(".")[-1])

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                _add(a.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            _add(node.module)
            if node.module in ("code_generator", "dse"):
                for a in node.names:
                    _add("%s.%s" % (node.module, a.name))
    return imported


def test_v5_layering_leaf_emitters_clean():
    """V5 structural (UPDATED for S3c, 2026-06-07; paths updated for the 2026-06-10
    package split): the LEAF emitters must NOT import reduce_recipe -- they consume a
    recipe INSTANCE by duck-typing only. generate_code IS the sanctioned constructor
    site (S3c) -> it imports build_reduce_recipe; that is now REQUIRED, not forbidden.
    reduce_recipe still imports DOWN only (never generate_code). Import statements are
    checked via ast (not raw text) so docstrings mentioning the rule do not trip it."""
    # Leaf emitters stay clean (no up/sideways import of the composition layer).
    for mod in ("code_generator/reduce_generator.py", "code_generator/shiftadd_generator.py",
                "code_generator/stage_generator.py", "code_generator/icbu_generator.py",
                "dse/icbu_dse.py", "code_generator/stage_plan.py",
                "code_generator/twiddle_generator.py"):
        assert "reduce_recipe" not in _module_imports(mod), \
            "%s must not import reduce_recipe (leaf; duck-typing only)" % mod
    # generate_code is the constructor site (S3c): it MUST import it now.
    assert "reduce_recipe" in _module_imports("generate_code.py"), \
        "generate_code should import reduce_recipe (S3c constructor site)"
    # reduce_recipe imports DOWN only.
    imported = _module_imports("code_generator/reduce_recipe.py")
    assert "generate_code" not in imported, \
        "layering violation: reduce_recipe must not import generate_code"
    assert {"reduce_generator", "shiftadd_generator"} <= imported, imported
    non_std = ({n for n in imported if "." not in n}
               - {"dataclasses", "functools", "typing", "code_generator", "dse"})
    assert non_std == {"reduce_generator", "shiftadd_generator"}, non_std


def test_dataclasses_frozen_pure():
    import dataclasses
    f = _solo("TI_9")
    r = rr.build_reduce_recipe(_pairs("TI_9"))
    for obj, field, val in ((f, "alias", "X"), (r, "anchor_k_group", 0)):
        try:
            object.__getattribute__(obj, field)  # field exists
            setattr(obj, field, val)
        except dataclasses.FrozenInstanceError:
            pass
        else:
            raise AssertionError("%s not frozen" % type(obj).__name__)
    # construction is deterministic / value-equal across calls.
    assert r == rr.build_reduce_recipe(_pairs("TI_9"))


def test_override_hook_empty_and_guarded():
    assert rr._RECIPE_OVERRIDES == {}, "S3b requires an EMPTY override hook"
    rr._RECIPE_OVERRIDES["x"] = 1
    try:
        rr.build_reduce_recipe(_pairs("TI_9"))
    except NotImplementedError:
        pass
    else:
        raise AssertionError("populated override hook must fail-fast in S3b")
    finally:
        rr._RECIPE_OVERRIDES.clear()


def test_classify_info_passthrough_and_mismatch():
    pair = _pairs("TI_9", "TII_33")
    info = rg.classify_reduce_set(pair, None)
    assert rr.build_reduce_recipe(pair, classify_info=info) == \
        rr.build_reduce_recipe(pair), "provided classify_info must be equivalent"
    other = rg.classify_reduce_set(_pairs("TI_2", "TI_9"), None)
    try:
        rr.build_reduce_recipe(pair, classify_info=other)
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched classify_info must be rejected")


def test_k_env_is_container_only():
    """K_env is the Data2 container width only: it must not change any fact."""
    pair = _pairs("TII_2", "TII_33")
    assert rr.build_reduce_recipe(pair) == \
        rr.build_reduce_recipe(pair, k_env=62), "k_env leaked into the facts"


def test_input_validation():
    for bad in ((), (("TI_9",),), ((123, 5),), (("TI_9", "5"),),
                (("TI_9", True),), (("TI_9", 0),), (("TI_9", -7),)):
        try:
            rr.build_reduce_recipe(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted invalid input %r" % (bad,))


def main():
    tests = [
        test_v1_table_covers_known_primes_exactly,
        test_v1_sixty_nine_alias_equivalence_sweep,
        test_v1_family_counts_and_distributions,
        test_v1_route_status_two_layers_by_family,
        test_v1_f1_refinement_family_batches,
        test_v2_same_family_batch_equivalence_sweep,
        test_v2_mixed_pairs_kgroup_legal_sweep,
        test_v3_named_set_route_status_table,
        test_two_layer_non_conflation,
        test_v4_int_custom_recorded_not_raised,
        test_v4_int_custom_value_equivalence,
        test_v5_layering_leaf_emitters_clean,
        test_dataclasses_frozen_pure,
        test_override_hook_empty_and_guarded,
        test_classify_info_passthrough_and_mismatch,
        test_k_env_is_container_only,
        test_input_validation,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print("PASS", t.__name__)
        except AssertionError as e:
            failed += 1
            print("FAIL", t.__name__, "--", e)
        except Exception as e:
            failed += 1
            print("ERROR", t.__name__, "--", type(e).__name__, e)
    print("\n%d/%d passed" % (len(tests) - failed, len(tests)))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

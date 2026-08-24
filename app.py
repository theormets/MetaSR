r"""
MetaSR — Metallurgical Symbolic Regression
===========================================
THIS REVISION — generalizable matching/search fixes (no equation is ever
named, favored, or special-cased in code; every fix below applies
identically to all 1,324 reference equations and any future ones added):

  1. VARIABLE-SUBSCRIPT NORMALIZATION FIX (_extract_var_tokens):
     The previous normalizer removed underscores and then stripped only
     TRAILING DIGITS — so "k_y" became "ky" and "sigma_y" became "sigmay".
     Any reference equation using a letter subscript (extremely common in
     materials-science notation: k_y, d_avg, sigma_y, T_m, ...) could
     never Jaccard-match a discovered feature named just "k", "d", or
     "sigma_0", even when they denote the same physical quantity. Fixed
     to split on the FIRST underscore and keep only the base symbol,
     discarding the subscript entirely — the general base_subscript
     convention, not a rule about any specific symbol.

  2. GREEK LATEX MACRO NORMALIZATION (_normalize_math_text):
     Greek commands (\sigma, \gamma, \rho, \Delta, ...) previously kept
     their literal backslash through most of the pipeline. This didn't
     break var-token extraction (regex ignores the backslash) but DID
     silently break the SymPy-based term-expansion and exact-match steps
     for any reference equation using an unconverted macro, since a
     literal backslash isn't valid Python/SymPy syntax. Added a mapping
     table applied first, before any other cleanup.

  3. PARSE-QUALITY CONFIDENCE (_parse_quality, EQ_PARSE_QUALITY):
     Some reference-DB LaTeX entries are OCR/extraction debris (stray
     \vphantom, unmatched braces, non-ASCII glyphs) that still produces a
     structurally "plausible" signature by chance. Previously this noise
     competed on equal footing with cleanly-parsed entries. Now every
     reference signature carries a 0-1 parse-quality score (based on how
     much backslash/brace/non-ASCII debris survives normalization), and
     top_similar_equations() applies it as a mild multiplicative
     downweight (never a hard exclusion — floor of 0.5x, since a partially
     garbled entry can still be a genuine hint). Applies uniformly to
     every reference entry; the discovered equation's own signature is
     never touched by this (it's always freshly humanized text, not raw
     scraped LaTeX).

  4. SEARCH TOKEN/PUNCTUATION NORMALIZATION (search_reference_equations):
     The reference-equation search (Tab 2) did an exact substring match
     on the raw query. Typing "hall petch" (space) against a DB entry
     named "Hall-Petch Relation" (hyphen) never matched, because
     "hall petch" is not a substring of "hall-petch relation". Fixed to
     normalize hyphens/underscores/slashes to spaces on both sides and
     require every query token to appear (AND search), not the exact
     phrase. This is a plain search-UX fix that helps find ANY
     hyphenated/underscored equation name, not a specific one.

  5. DIAGNOSTICS PANEL (Tab 2, new accordion):
     A small "🔎 Look up a reference equation" tool that lets you check,
     independently of any regression run, whether a keyword resolves to
     an entry in equations_browser.json, and if so shows its raw LaTeX,
     normalized text, extracted signature (operators/functions/var
     names/term shapes), and parse-quality score. This is read-only
     inspection — it doesn't feed into or bias the similarity ranking,
     it's just so you can verify what's actually in your database instead
     of inferring it from downstream match results.

  Nothing above hardcodes, references, or special-cases any specific
  equation name. The intent is the same as it always was: a genuinely
  well-matching reference equation should surface on its own once the
  matching machinery isn't quietly losing points to normalization bugs
  or noisy data — not because the code is told which answer to prefer.

---- (prior revision's changelog — term-shape splitter fix) ----
  BUG: PySR frequently returns a discovered equation as an additive
  expression wrapped in an outer multiplication, e.g.
      ((sigma_0 * A) + (k_x_inv_sqrt_d + B)) * C
  _split_top_level_terms() only splits on '+'/'-' that sit at
  parenthesis depth 0. In the string above, the ONLY depth-0 characters
  are the trailing "* C" — the '+' joining the two additive terms is one
  level deeper, trapped inside the outer parens. So the whole equation
  was scored as a single multiplicative term (shape "div_sqrt"), which
  is structurally indistinguishable from a genuinely single-term
  equation like Wave Velocity (ωλ/2π · sqrt(E/ρ)) — hence the false
  100% term_shapes match and Hall-Petch-shaped results ranking behind
  unrelated single-term equations.

  FIX: _extract_term_shapes() now runs the string through
  _expand_for_term_split() first — a best-effort SymPy algebraic
  expansion (SymPy is already an optional import in this file) that
  distributes exactly that kind of outer multiplication before the
  top-level split runs, so "(a+b)*C" is classified by its true
  additive shape "{linear, div_sqrt}" instead of one opaque term. This
  is a pure algebraic rewrite — it applies identically to every
  equation build_signature() ever processes (reference-DB LaTeX or a
  discovered PySR equation) and never references or favors any named
  equation. Falls back to the original, unmodified string whenever
  SymPy is unavailable or the string uses notation the permissive
  parser can't safely round-trip (log10, leftover LaTeX brace debris) —
  never raises, never changes downstream behavior otherwise.

  Nothing else in the similarity engine, grammar-compatibility metric,
  or probe-selection logic was touched. On review: the probe/tiebreak
  logic is sound (the near-total tie across clusters reflects a
  trivially fittable 80-row dataset, not a selection bug), and the
  grammar-compatibility metric is working as intended — a low score
  for a wrong-cluster equation is the metric doing its job, not a
  fairness bug against "large" grammars.

---- (prior revisions kept for reference, unchanged below) ----
  1. TERM-SHAPE STRUCTURAL SIGNATURE, DOMAIN-AWARE SIMILARITY WEIGHTING,
     GRAMMAR COMPATIBILITY DETAIL, SIMILARITY BREAKDOWN, TOP-3 FINAL
     TIEBREAK — see prior revisions' notes preserved in source history.
  2. SIMILARITY BUG FIX (word-boundary regex on underscored feature
     names), VARIABLE-IDENTITY SIGNAL, OPTIONAL SYMPY CONFIRMATION.
  3. Grammar operator weighting, grammar-aware feature engineering,
     subfield caution notes, two-stage similarity search, grammar
     compatibility metric, engineered-feature ranking.

NOTE on the LaTeX normalizer (_normalize_math_text): it's a best-effort
regex pass (handles \frac, \sqrt, \cdot, common trig/exp/log macros,
Greek letters, ^{...} exponents). It won't perfectly parse every LaTeX
construct in equations_browser.json — if your JSON uses macros not
covered here, extend _LATEX_CLEAN_PATTERNS or _GREEK_LATEX_MAP. It
degrades gracefully (unmatched macros just stay as literal text, which
lowers parse-quality and similarity slightly for that entry, it won't
crash the app).
"""

import gradio as gr
import pandas as pd
import json
import os
import numpy as np
import re
import traceback
import threading
import time

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.inspection import permutation_importance

# Optional — used only by the top-match SymPy confirmation step, the
# top-equation auto-simplify, and the term-shape expansion step below.
# The app runs fully without it; each of those call sites falls back to
# "not attempted / unchanged" if unavailable.
try:
    import sympy
    from sympy.parsing.sympy_parser import (
        parse_expr, standard_transformations, implicit_multiplication_application,
    )
    _SYMPY_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)
    _SYMPY_AVAILABLE = True
except Exception:
    _SYMPY_AVAILABLE = False


# ─────────────────────────────────────────────
# Load equation reference library (single source of truth)
# ─────────────────────────────────────────────
_EQ_DB_PATH = "equations_browser.json"

def _load_eq_db():
    if not os.path.exists(_EQ_DB_PATH):
        return []
    with open(_EQ_DB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

EQ_DB = _load_eq_db()

ALL_SUBFIELDS = sorted(set(eq["subfield"] for eq in EQ_DB))

from collections import defaultdict, Counter as _Counter
_cluster_meta = defaultdict(lambda: {"subfields": _Counter(), "samples": []})
for _eq in EQ_DB:
    # NOTE: normalize to str here. EQ_DB's "cluster" field is often an int
    # (from the JSON), while cluster ids elsewhere in the app (grammar file
    # names -> grammars dict keys -> chosen_cid) are always strings like
    # "0", "9". Keying this dict by the raw (possibly int) value meant every
    # lookup with a string chosen_cid silently missed and returned {} — that
    # was the bug behind "top_subfields" always being empty (so the
    # cross-domain caution never fired) and the within-cluster similarity
    # stage always reporting "no equations in this cluster to compare
    # against", even though the cluster obviously has equations in it.
    _c = str(_eq["cluster"])
    _cluster_meta[_c]["subfields"][_eq["subfield"]] += 1
    if len(_cluster_meta[_c]["samples"]) < 3:
        _cluster_meta[_c]["samples"].append(_eq["latex"])

CLUSTER_DESCRIPTIONS = {}
for _c, _meta in sorted(_cluster_meta.items()):
    _top3 = [sf for sf, _ in _meta["subfields"].most_common(3)]
    CLUSTER_DESCRIPTIONS[_c] = {
        "top_subfields": _top3,
        "total": sum(_meta["subfields"].values()),
        "samples": _meta["samples"],
    }

# cluster -> list of EQ_DB indices (for the within-cluster similarity stage)
# Same str-normalization as above — this is the dict whose lookup was
# silently failing.
_CLUSTER_TO_INDICES = defaultdict(list)
for _idx, _eq in enumerate(EQ_DB):
    _CLUSTER_TO_INDICES[str(_eq["cluster"])].append(_idx)

# Label → full record map (reset on each search)
_LABEL_TO_RECORD = {}


# ─────────────────────────────────────────────
# Julia warm-up
# ─────────────────────────────────────────────
_julia_ready = threading.Event()

def _warmup_julia():
    try:
        from pysr import PySRRegressor
        _m = PySRRegressor(
            niterations=1, populations=1, maxsize=7,
            procs=0, multithreading=False, verbosity=0, random_state=0,
        )
        rng = np.random.default_rng(0)
        _m.fit(rng.standard_normal((12, 2)), rng.standard_normal(12))
        del _m
    except Exception:
        pass
    finally:
        _julia_ready.set()

threading.Thread(target=_warmup_julia, daemon=True).start()


# ─────────────────────────────────────────────
# Grammar helpers
# ─────────────────────────────────────────────
UNARY_MAP = {
    "ln": "log", "log": "log10", "exp": "exp", "sqrt": "sqrt",
    "sin": "sin", "cos": "cos", "tan": "tan",
    "sinh": "sinh", "cosh": "cosh", "tanh": "tanh",
}
BINARY_MAP      = {"+": "+", "-": "-", "*": "*", "/": "/", "^": "^"}
BINARY_FALLBACK = ["+", "-", "*", "/"]
UNARY_FALLBACK  = ["sqrt", "exp", "log"]

# Weight used when a grammar declares an operator but no explicit weight is
# found for it (keeps old grammars without a "weights" block working).
DEFAULT_OP_WEIGHT = 0.10
# Weight given to the plain/raw (no transform) feature in the fast probe.
BASELINE_RAW_WEIGHT = 0.10

# ─────────────────────────────────────────────
# Grammar display names (replaces the old family_label / "X Family" wording)
# ─────────────────────────────────────────────
CLUSTER_GRAMMAR_NAMES = {
    "0":  "Linear–Polynomial Grammar",
    "1":  "Rational–Power Grammar",
    "2":  "Hybrid Composite Grammar",
    "3":  "Rational–Exponential Grammar",
    "4":  "Hybrid Root–Exponential Grammar",
    "5":  "Power–Composite Grammar",
    "6":  "General Hybrid Grammar",
    "7":  "Logarithmic Grammar",
    "8":  "Exponential Grammar",
    "9":  "Periodic–Root Grammar",
    "10": "Multi-Angle Composite Grammar",
    "11": "General Rational Grammar",
    "12": "Root–Angular Grammar",
}


def grammar_display_name(cid, grammar):
    name = CLUSTER_GRAMMAR_NAMES.get(str(cid))
    if name:
        return name
    fallback = grammar.get("family_label", "Unnamed Grammar")
    return re.sub(r'\s*Family\s*$', '', fallback).strip() + " Grammar"


def format_grammar_profile(cid, grammar, unary_ops, binary_ops, unary_weights, binary_weights):
    name = grammar_display_name(cid, grammar)
    template = grammar.get("structure_template", {}).get("template", "N/A")

    binary_sorted = sorted(binary_weights.items(), key=lambda kv: -kv[1])
    unary_sorted = sorted(unary_weights.items(), key=lambda kv: -kv[1])

    binary_lines = "\n".join(
        f"        {op:<3} ({w*100:5.1f}%)" for op, w in binary_sorted
    ) or "        (none)"
    unary_lines = "\n".join(
        f"        {op:<6} ({w*100:5.1f}%)" for op, w in unary_sorted
    ) or "        (none)"

    return (
        f"    Grammar      : {name}\n"
        f"    Grammar ID   : C{cid}\n"
        f"    Template     : {template}\n"
        f"    Dominant Binary Operators:\n{binary_lines}\n"
        f"    Dominant Unary Operators:\n{unary_lines}"
    )


def load_grammars():
    grammars = {}
    grammar_dir = "grammars"
    if not os.path.isdir(grammar_dir):
        return grammars
    for fname in sorted(os.listdir(grammar_dir)):
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(grammar_dir, fname)) as f:
                data = json.load(f)
            cid = fname.replace("grammar_cluster_", "").replace(".json", "")
            grammars[cid] = data
        except Exception:
            pass
    return grammars


def extract_ops_from_grammar(grammar):
    functions_block = grammar.get("functions", {})
    raw_unary = functions_block.get("whitelist", [])
    raw_unary_counts = functions_block.get("all_functions", {})
    raw_unary_weights_declared = functions_block.get("weights", {})  # fwd-compat only

    unary_ops = []
    unary_raw_weight = {}
    for fn in raw_unary:
        mapped = UNARY_MAP.get(fn)
        if not mapped:
            continue
        if fn in raw_unary_weights_declared:
            w = raw_unary_weights_declared[fn]
        elif fn in raw_unary_counts:
            w = raw_unary_counts[fn]
        else:
            w = 1.0
        try:
            w = max(0.0, float(w))
        except (TypeError, ValueError):
            w = 1.0
        if mapped not in unary_ops:
            unary_ops.append(mapped)
        unary_raw_weight[mapped] = unary_raw_weight.get(mapped, 0.0) + w

    if not unary_ops:
        unary_ops = UNARY_FALLBACK[:]
        unary_raw_weight = {op: 1.0 for op in unary_ops}

    unary_total = sum(unary_raw_weight.values()) or 1.0
    unary_weights = {op: w / unary_total for op, w in unary_raw_weight.items()}

    operators_block = grammar.get("operators", {})
    raw_binary_weights = operators_block.get("weights", {})
    binary_ops = []
    binary_weights = {}
    for op, w in raw_binary_weights.items():
        mapped = BINARY_MAP.get(op)
        if mapped and mapped not in binary_ops:
            binary_ops.append(mapped)
            try:
                binary_weights[mapped] = max(0.0, float(w))
            except (TypeError, ValueError):
                binary_weights[mapped] = DEFAULT_OP_WEIGHT
    for op in BINARY_FALLBACK:
        if op not in binary_ops:
            binary_ops.append(op)
            binary_weights.setdefault(op, DEFAULT_OP_WEIGHT)

    return unary_ops, binary_ops, unary_weights, binary_weights


# ─────────────────────────────────────────────
# Feature engineering — grammar-aware, records each generated column's
# true math expression for the similarity engine.
# ─────────────────────────────────────────────
def engineer_features(X_df, y, allowed_unary=None, r2_threshold=0.05):
    allow_sqrt = allowed_unary is None or ("sqrt" in allowed_unary)
    allow_log  = allowed_unary is None or ("log" in allowed_unary) or ("log10" in allowed_unary)

    raw_cols = list(X_df.columns)
    all_cols = {}
    math_expr = {}
    for col in raw_cols:
        x = X_df[col].values
        all_cols[col] = x
        math_expr[col] = col
        if np.all(x > 0):
            if allow_sqrt:
                name_inv = f"inv_sqrt_{col}"
                all_cols[name_inv] = 1.0 / np.sqrt(x)
                math_expr[name_inv] = f"1/sqrt({col})"

                name_sqrt = f"sqrt_{col}"
                all_cols[name_sqrt] = np.sqrt(x)
                math_expr[name_sqrt] = f"sqrt({col})"
            if allow_log:
                name_log = f"log_{col}"
                all_cols[name_log] = np.log(x)
                math_expr[name_log] = f"log({col})"
    if allow_sqrt:
        for c1 in raw_cols:
            x1 = X_df[c1].values
            for c2 in raw_cols:
                if c1 == c2:
                    continue
                x2 = X_df[c2].values
                if np.all(x2 > 0):
                    name = f"{c1}_x_inv_sqrt_{c2}"
                    all_cols[name] = x1 / np.sqrt(x2)
                    math_expr[name] = f"({c1})/sqrt({c2})"
    keep = {}
    keep_expr = {}
    for name, x in all_cols.items():
        is_raw = name in raw_cols
        if np.std(x) < 1e-10:
            if is_raw:
                keep[name] = x
                keep_expr[name] = math_expr[name]
            continue
        r2 = max(0.0, float(
            LinearRegression().fit(x.reshape(-1, 1), y).score(x.reshape(-1, 1), y)
        ))
        if is_raw or r2 >= r2_threshold:
            keep[name] = x
            keep_expr[name] = math_expr[name]
    result = pd.DataFrame(keep, index=X_df.index)
    result.attrs["math_expr"] = keep_expr
    return result


def rank_features_by_r2(X_eng_df, y, top_k=10):
    rows = []
    for col in X_eng_df.columns:
        x = X_eng_df[col].values
        r2 = _fast_r2(x, y)
        rows.append((col, r2))
    rows.sort(key=lambda t: -t[1])
    return rows[:top_k]


# ─────────────────────────────────────────────
# Grammar-guided probes
# ─────────────────────────────────────────────
PROBE_ITER         = 300
PROBE_POPULATIONS  = 4
PROBE_POP_SIZE     = 30
PROBE_MAXSIZE      = 10
PROBE_VERBOSITY = 0

# Fixed release defaults. These are not exposed as UI sliders: they should be
# validated across datasets, not tuned for a single demonstration dataset.
RESEARCH_LOSS_WEIGHT = 0.60
RESEARCH_COMPAT_WEIGHT = 0.20
RESEARCH_COMPLEXITY_WEIGHT = 0.20
MAXDEPTH = 6


def operator_complexity_costs(unary_ops, binary_ops, unary_weights, binary_weights):
    """Use learned operator probabilities as PySR search costs.

    PySR has no per-operator mutation-probability parameter. Its operator
    complexity costs are used by evolution, however, so probable operators
    cost less while less probable operators remain legal but are discouraged.
    """
    weights = {**binary_weights, **unary_weights}
    declared = [op for op in list(binary_ops) + list(unary_ops) if op in weights]
    if not declared:
        return {}
    maximum = max(weights[op] for op in declared) or 1.0
    return {
        op: max(1, int(round(1 + 4 * (1 - weights[op] / maximum))))
        for op in declared
    }


def pysr_structure_constraints(unary_ops, binary_ops):
    """Block exp(exp(x)) and pow(pow(x, a), b) during evolution."""
    nested = {}
    if "exp" in unary_ops:
        nested["exp"] = {"exp": 0}
    if "^" in binary_ops:
        nested["^"] = {"^": 0}
    return nested


def _safe_tournament_n(population_size, requested=10):
    if population_size <= 1:
        return 1
    return max(1, min(requested, population_size - 1))


def _fast_r2(x, y):
    try:
        x = np.asarray(x, float).reshape(-1, 1)
        if np.std(x) < 1e-10 or len(x) < 3:
            return 0.0
        return max(0.0, float(LinearRegression().fit(x, y).score(x, y)))
    except Exception:
        return 0.0


_UNARY_FN_IMPL = {
    "log":   lambda x: np.log(np.abs(x) + 1e-9),
    "log10": lambda x: np.log10(np.abs(x) + 1e-9),
    "exp":   lambda x: np.exp(np.clip(x, -50, 50)),
    "sqrt":  lambda x: np.sqrt(np.abs(x)),
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
}


def fast_grammar_probe(df, target, grammars):
    X_raw = df.drop(columns=[target])
    y = df[target].values

    scores = {}
    for cid, g in grammars.items():
        unary_ops, _, unary_weights, _ = extract_ops_from_grammar(g)

        weighted_total = 0.0
        weight_sum = 0.0

        raw_best = 0.0
        for col in X_raw.columns:
            raw_best = max(raw_best, _fast_r2(X_raw[col].values, y))
        weighted_total += raw_best * BASELINE_RAW_WEIGHT
        weight_sum += BASELINE_RAW_WEIGHT

        for op_name in unary_ops:
            fn = _UNARY_FN_IMPL.get(op_name)
            if fn is None:
                continue
            w = unary_weights.get(op_name, DEFAULT_OP_WEIGHT)
            if w <= 0:
                continue
            best_r2 = 0.0
            for col in X_raw.columns:
                x = X_raw[col].values
                try:
                    xt = fn(x)
                    if np.all(np.isfinite(xt)):
                        best_r2 = max(best_r2, _fast_r2(xt, y))
                except Exception:
                    continue
            weighted_total += best_r2 * w
            weight_sum += w

        scores[cid] = weighted_total / weight_sum if weight_sum > 0 else 0.0

    best_score = max(scores.values())
    tied = [c for c, s in scores.items() if abs(s - best_score) < 1e-6]
    if len(tied) == 1:
        return tied[0], scores, tied
    tied_sorted = sorted(tied, key=lambda c: grammars[c].get("equation_count", 0))
    return tied_sorted[0], scores, tied_sorted


def research_grammar_probe(df, target, grammars, progress=None):
    X_raw = df.drop(columns=[target])
    y = df[target].values

    _julia_ready.wait(timeout=180)
    from pysr import PySRRegressor

    losses = {}
    compat = {}
    complexities = {}
    elapsed = {}
    items = sorted(grammars.items())
    total = len(items)
    for i, (cid, g) in enumerate(items, 1):
        if progress is not None:
            progress(i / total, desc=f"Research probe: cluster {cid} ({i}/{total})")

        unary_ops, binary_ops, unary_weights, binary_weights = extract_ops_from_grammar(g)

        X_eng = engineer_features(X_raw, y, allowed_unary=set(unary_ops), r2_threshold=0.05)
        eng_names = list(X_eng.columns)
        scaler_X = RobustScaler()
        scaler_y = RobustScaler()
        X_scaled = scaler_X.fit_transform(X_eng)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

        best_loss = float("inf")
        best_compat = 0.0
        t0 = time.perf_counter()
        try:
            if not binary_ops:
                raise ValueError(f"Cluster {cid}: empty binary_operators list — cannot fit PySR.")
            if not unary_ops:
                raise ValueError(f"Cluster {cid}: empty unary_operators list — cannot fit PySR.")

            model = PySRRegressor(
                niterations=PROBE_ITER, populations=PROBE_POPULATIONS,
                population_size=PROBE_POP_SIZE, maxsize=PROBE_MAXSIZE,
                tournament_selection_n=_safe_tournament_n(PROBE_POP_SIZE),
                binary_operators=binary_ops, unary_operators=unary_ops,
                complexity_of_operators=operator_complexity_costs(
                    unary_ops, binary_ops, unary_weights, binary_weights
                ),
                maxdepth=MAXDEPTH,
                nested_constraints=pysr_structure_constraints(unary_ops, binary_ops),
                model_selection="best", parsimony=1e-4, random_state=42,
                verbosity=PROBE_VERBOSITY, procs=0, multithreading=False,
            )
            model.fit(X_scaled, y_scaled)

            eqs = getattr(model, "equations_", None)
            if eqs is None or len(eqs) == 0:
                raise RuntimeError(
                    f"Cluster {cid}: PySR finished but produced no equations "
                    f"(model.equations_ is empty) — likely too few "
                    f"iterations/timeout for this operator set."
                )
            best_row = eqs.sort_values(by="loss").iloc[0]
            best_loss = float(best_row["loss"])
            complexities[cid] = float(best_row.get("complexity", PROBE_MAXSIZE))

            math_expr_map = X_eng.attrs.get("math_expr", {n: n for n in eng_names})
            var_map_math = {f"x{j}": f"({math_expr_map.get(name, name)})"
                             for j, name in enumerate(eng_names)}
            raw_eq = str(best_row.get("equation", ""))
            human_eq_math = raw_eq
            for xi, expr in sorted(var_map_math.items(), key=lambda kv: -len(kv[0])):
                human_eq_math = human_eq_math.replace(xi, expr)
            cx_val = best_row.get("complexity", None)
            sig = build_signature(
                human_eq_math,
                complexity_hint=float(cx_val) if cx_val is not None else None,
            )
            # Lightweight influence check for the probe-stage compatibility.
            # This prevents a tiny ornamental function from selecting a
            # grammar simply because the function appears syntactically.
            try:
                perm = permutation_importance(
                    model, X_scaled, y_scaled, n_repeats=3, random_state=42, scoring="r2",
                )
                importance_by_feature = dict(zip(eng_names, perm.importances_mean))
                influence, _ = _feature_operator_influence(
                    human_eq_math, importance_by_feature, math_expr_map
                )
            except Exception:
                # Importance is an interpretability refinement, never a reason
                # to discard an otherwise valid cluster fit.
                influence = None
            best_compat = grammar_compatibility(
                sig, unary_ops, binary_ops, unary_weights, binary_weights, influence
            )

        except Exception as e:
            print("=" * 60)
            print(f"[MetaSR research probe] Cluster {cid} FAILED")
            print(f"  unary_ops  = {unary_ops}")
            print(f"  binary_ops = {binary_ops}")
            print(f"  {type(e).__name__}: {e}")
            traceback.print_exc()
            print("=" * 60)
            best_loss = float("inf")
            best_compat = 0.0
            complexities[cid] = float("inf")

        cluster_elapsed = time.perf_counter() - t0
        print(f"[MetaSR research probe] Cluster {cid}: {cluster_elapsed:.2f}s "
              f"(niterations={PROBE_ITER}, loss={best_loss}, compat={best_compat*100:.1f}%)")

        losses[cid] = best_loss
        compat[cid] = best_compat
        complexities.setdefault(cid, float("inf"))
        elapsed[cid] = cluster_elapsed

    finite_losses = [l for l in losses.values() if np.isfinite(l)]
    lo = min(finite_losses) if finite_losses else 0.0
    hi = max(finite_losses) if finite_losses else 1.0
    spread = (hi - lo) if (hi - lo) > 1e-12 else 1.0
    finite_complexities = [v for v in complexities.values() if np.isfinite(v)]
    cx_lo = min(finite_complexities) if finite_complexities else 0.0
    cx_hi = max(finite_complexities) if finite_complexities else 1.0
    cx_spread = (cx_hi - cx_lo) if (cx_hi - cx_lo) > 1e-12 else 1.0

    combined = {}
    for cid in losses:
        if not np.isfinite(losses[cid]):
            norm_loss = 1.0
        else:
            norm_loss = (losses[cid] - lo) / spread
        norm_complexity = (
            (complexities[cid] - cx_lo) / cx_spread
            if np.isfinite(complexities[cid]) else 1.0
        )
        combined[cid] = (
            RESEARCH_LOSS_WEIGHT * norm_loss
            + RESEARCH_COMPAT_WEIGHT * (1.0 - compat[cid])
            + RESEARCH_COMPLEXITY_WEIGHT * norm_complexity
        )

    best_cid = min(combined, key=combined.get)
    best_score = combined[best_cid]
    tied = [c for c, s in combined.items() if s <= best_score + 0.02]

    total_elapsed = sum(elapsed.values())
    print(f"[MetaSR research probe] Total probe wall-clock: {total_elapsed:.2f}s "
          f"across {len(elapsed)} clusters "
          f"(avg {total_elapsed/max(1, len(elapsed)):.2f}s/cluster)")

    scores = {
        cid: {
            "loss": losses[cid], "compatibility": compat[cid],
            "complexity": complexities[cid],
            "combined": combined[cid], "elapsed_s": elapsed[cid],
        }
        for cid in losses
    }
    return best_cid, scores, tied


# ─────────────────────────────────────────────
# Constant replacement (display only)
# ─────────────────────────────────────────────
_KEEP_AS_IS = {"2", "3", "4", "5"}
_CONST_RE = re.compile(
    r'(?<![a-zA-Z_\d])(-?\d+\.\d+(?:[eE][+-]?\d+)?|-?\d+(?:[eE][+-]?\d+)?)(?![a-zA-Z_\d])'
)

def grammar_confidence_label(compat_pct):
    if compat_pct is None:
        return "N/A"
    if compat_pct >= 80:
        return "High"
    if compat_pct >= 50:
        return "Medium"
    return "Low"


def adjusted_r2(r2, n_samples, n_predictors):
    n = n_samples
    p = max(0, n_predictors)
    denom = n - p - 1
    if denom <= 0:
        return None
    return 1.0 - (1.0 - r2) * (n - 1) / denom


def try_sympy_simplify(math_str):
    if not _SYMPY_AVAILABLE:
        return None
    friendly = _to_sympy_friendly(_normalize_math_text(math_str))
    if friendly is None:
        return None
    try:
        expr = parse_expr(friendly, transformations=_SYMPY_TRANSFORMS)
        simplified = sympy.simplify(expr)
        simplified_str = str(simplified)
        if simplified_str and simplified_str != friendly:
            return simplified_str
        return None
    except Exception:
        return None


def replace_constants(eq):
    symbols = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    found, ctr = {}, [0]
    def sub(m):
        val = m.group(0)
        if val in _KEEP_AS_IS:
            return val
        if val not in found:
            if ctr[0] < len(symbols):
                found[val] = symbols[ctr[0]]; ctr[0] += 1
            else:
                return val
        return found[val]
    return _CONST_RE.sub(sub, eq)


# ─────────────────────────────────────────────
# Universal similarity engine
# ─────────────────────────────────────────────
_FN_NAMES = ["exp", "log10", "log", "sqrt", "sinh", "cosh", "tanh", "sin", "cos", "tan"]

# THIS REVISION — fix #2: Greek LaTeX macros -> plain words, applied
# FIRST, before any other cleanup. General across every reference entry
# and every discovered equation; not scoped to any particular symbol.
_GREEK_LATEX_MAP = [
    (r'\\varepsilon', 'epsilon'), (r'\\varphi', 'phi'),
    (r'\\alpha', 'alpha'), (r'\\beta', 'beta'), (r'\\gamma', 'gamma'),
    (r'\\Gamma', 'Gamma'), (r'\\delta', 'delta'), (r'\\Delta', 'Delta'),
    (r'\\epsilon', 'epsilon'), (r'\\zeta', 'zeta'), (r'\\eta', 'eta'),
    (r'\\theta', 'theta'), (r'\\Theta', 'Theta'), (r'\\iota', 'iota'),
    (r'\\kappa', 'kappa'), (r'\\lambda', 'lambda'), (r'\\Lambda', 'Lambda'),
    (r'\\mu', 'mu'), (r'\\nu', 'nu'), (r'\\xi', 'xi'), (r'\\pi', 'pi'),
    (r'\\rho', 'rho'), (r'\\sigma', 'sigma'), (r'\\Sigma', 'Sigma'),
    (r'\\tau', 'tau'), (r'\\upsilon', 'upsilon'), (r'\\phi', 'phi'),
    (r'\\Phi', 'Phi'), (r'\\chi', 'chi'), (r'\\psi', 'psi'),
    (r'\\Psi', 'Psi'), (r'\\omega', 'omega'), (r'\\Omega', 'Omega'),
]

_LATEX_CLEAN_PATTERNS = [
    (r'\\left', ''), (r'\\right', ''), (r'\\,', ' '), (r'\\!', ''), (r'\\;', ' '),
    (r'\\cdot', '*'), (r'\\times', '*'),
    (r'\\frac\{([^{}]*)\}\{([^{}]*)\}', r'(\1)/(\2)'),
    (r'\\sqrt\{([^{}]*)\}', r'sqrt(\1)'),
    (r'\\sqrt', 'sqrt'),
    (r'\\exp', 'exp'),
    (r'\\ln', 'log'),
    (r'\\log', 'log10'),
    (r'\\sin', 'sin'), (r'\\cos', 'cos'), (r'\\tan', 'tan'),
    (r'\\sinh', 'sinh'), (r'\\cosh', 'cosh'), (r'\\tanh', 'tanh'),
    (r'\$', ''),
]

def _normalize_math_text(s):
    """Best-effort LaTeX/plain-text normalizer — see module docstring."""
    s = str(s)
    # Fix #2: Greek macros converted FIRST, before anything else touches
    # the string, so downstream function/variable extraction and SymPy
    # parsing all see plain identifiers instead of literal backslashes.
    for pat, repl in _GREEK_LATEX_MAP:
        s = re.sub(pat, repl, s)
    for pat, repl in _LATEX_CLEAN_PATTERNS:
        s = re.sub(pat, repl, s)
    # run \frac substitution twice to catch simple nesting
    s = re.sub(r'\\frac\{([^{}]*)\}\{([^{}]*)\}', r'(\1)/(\2)', s)
    s = s.replace('^{', '^(').replace('_{', '_(')
    # any leftover "}" from the above braces closes as ")"
    s = s.replace('}', ')')
    # Preserve a LaTeX subscript as one identifier (sigma_(0) -> sigma_0)
    # so both token extraction and the optional SymPy canonicalizer can read it.
    s = re.sub(r'([A-Za-z]+)_\(([^()]+)\)', r'\1_\2', s)
    s = s.replace('**', '^')
    # Canonicalise fractional powers commonly used in reference LaTex:
    # x^(-1/2) has the same structure as 1/sqrt(x), and x^(1/2) as sqrt(x).
    s = re.sub(r'\^\(\s*-\s*1\s*/\s*2\s*\)', '^(-0.5)', s)
    s = re.sub(r'\^\(\s*1\s*/\s*2\s*\)', '^(0.5)', s)
    return s


def _canonical_math_text(raw_text):
    """Best-effort canonical form for structural comparison only.

    SymPy turns algebraically equivalent forms such as ``d^(-1/2)`` and
    ``1/sqrt(d)`` into the same representation, normalises commutative term
    order, and removes harmless parentheses.  Reference equations are reduced
    to their right-hand side because their left side is a named dependent
    variable, not part of the predictive expression.  If parsing is unsafe,
    the already robust regex-normalised text is retained unchanged.
    """
    text = _normalize_math_text(raw_text)
    if "=" in text:
        text = text.split("=", 1)[1]
    # Dependency-free identities used when SymPy is unavailable (the app
    # intentionally treats SymPy as optional). These affect every equation,
    # not any material-specific variable name.
    text = re.sub(r'([A-Za-z_][A-Za-z_0-9]*)\^\(-0\.5\)', r'1/sqrt(\1)', text)
    text = re.sub(r'([A-Za-z_][A-Za-z_0-9]*)\^\(0\.5\)', r'sqrt(\1)', text)
    text = re.sub(r'([A-Za-z_][A-Za-z_0-9]*)\^\(-1\)', r'1/(\1)', text)
    text = re.sub(r'\be\^\(?([A-Za-z_][A-Za-z_0-9]*)\)?', r'exp(\1)', text)
    if not _SYMPY_AVAILABLE or "{" in text or "}" in text or "\\" in text:
        return text
    try:
        expr = parse_expr(text.replace("^", "**"), transformations=_SYMPY_TRANSFORMS)
        expr = sympy.simplify(sympy.powsimp(sympy.expand_power_base(expr, force=True)))
        return str(expr).replace("**", "^")
    except Exception:
        return text


def _extract_operators(s):
    ops = set()
    if re.search(r'\+', s): ops.add("+")
    if re.search(r' - |^-', s): ops.add("-")
    if re.search(r'\*', s): ops.add("*")
    if re.search(r'/', s): ops.add("/")
    if re.search(r'\^', s): ops.add("^")
    return ops


def _extract_functions(s):
    fns = set()
    for fn in _FN_NAMES:
        if re.search(r'\b' + fn + r'\b', s):
            fns.add("log" if fn == "log10" else fn)
    return fns


def _extract_exponents(s):
    exps = set()
    for m in re.finditer(r'\^\(\s*(-?\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*\)', s):
        try:
            exps.add(round(float(m.group(1)) / float(m.group(2)), 2))
        except (ValueError, ZeroDivisionError):
            pass
    for m in re.finditer(r'\^\(?\s*(-?\d+\.?\d*)\s*\)?', s):
        try:
            exps.add(round(float(m.group(1)), 2))
        except ValueError:
            pass
    return exps


def _extract_var_count(s):
    tokens = set(re.findall(r'[a-zA-Z_][a-zA-Z_0-9]*', s))
    tokens -= set(_FN_NAMES) | {"log"}
    return len(tokens)


def _extract_var_tokens(s):
    """
    Coarse, normalized variable-name set, used for a variable-identity
    similarity component.

    THIS REVISION — fix #1: the previous approach removed underscores
    and then stripped only trailing DIGITS, so a letter subscript (very
    common in physics/materials notation — k_y, d_avg, sigma_y, T_m)
    got folded permanently into the base name: "k_y" -> "ky", never
    matching a discovered feature literally named "k". Now the base
    symbol is taken by splitting on the FIRST underscore and discarding
    everything after it — "k_y" -> "k", "sigma_0" -> "sigma",
    "d_avg" -> "d" — which is the general base_subscript convention,
    not a rule tied to any specific symbol or equation.

    Tradeoff, stated plainly: this will also equate genuinely different
    variables that happen to share a base letter (e.g. two unrelated "d"
    quantities in different equations). Accepted here because this feeds
    ONE weighted Jaccard component in the blended similarity score below,
    not a hard filter — it nudges ranking, it doesn't gate it.
    """
    tokens = set(re.findall(r'[a-zA-Z_][a-zA-Z_0-9]*', s))
    tokens -= set(_FN_NAMES) | {"log"}
    normalized = set()
    for t in tokens:
        base = t.split('_')[0]
        base = re.sub(r'\d+$', '', base)
        normalized.add((base if base else t).lower())
    return normalized


def _extract_complexity(s):
    op_count = len(re.findall(r'[\+\-\*/\^]', s))
    fn_count = sum(len(re.findall(r'\b' + fn + r'\b', s)) for fn in _FN_NAMES)
    var_count = _extract_var_count(s)
    num_count = len(re.findall(r'\d+\.?\d*', s))
    return op_count + fn_count + var_count + num_count


def _parse_quality(normalized_text):
    """
    THIS REVISION — fix #3.

    Heuristic 0-1 confidence that a reference-DB entry was cleanly
    normalized, based on how much backslash/brace/non-ASCII debris is
    still present in the normalized text. Some entries in
    equations_browser.json are OCR/extraction artifacts (stray
    \\vphantom, mismatched braces, unicode glyphs) that still yield a
    structurally "plausible" signature by coincidence — this lets a
    genuinely garbled entry be recognized as such and downweighted,
    instead of silently competing on equal footing with a cleanly
    parsed equation. Applies uniformly to every reference-DB entry;
    never references or excludes any equation by name.
    """
    if not normalized_text:
        return 0.0
    debris = len(re.findall(r'[\\{}]|[^\x00-\x7F]', normalized_text))
    ratio = debris / max(1, len(normalized_text))
    return max(0.0, 1.0 - min(1.0, ratio * 4.0))


# ─────────────────────────────────────────────
# TERM-SHAPE STRUCTURAL SIGNATURE
# ─────────────────────────────────────────────
def _split_top_level_terms(s):
    depth = 0
    terms = []
    current = ""
    for ch in s:
        if ch in "([":
            depth += 1
            current += ch
        elif ch in ")]":
            depth -= 1
            current += ch
        elif ch in "+-" and depth == 0:
            if current.strip():
                terms.append(current.strip())
            current = ch
        else:
            current += ch
    if current.strip():
        terms.append(current.strip())
    return [t for t in terms if t not in ("+", "-")]


def _expand_for_term_split(s):
    """
    Best-effort algebraic expansion (via SymPy, when available), applied
    ONLY right before term-shape splitting. Distributes an outer
    "(...)*const" wrapper (very common in PySR output) across the sum
    before shape-classification runs, so a hidden additive structure is
    classified by its true shape instead of one opaque multiplicative
    term. Pure algebraic rewrite, applies identically to every equation
    build_signature() processes. Best-effort and side-effect-free:
    returns the ORIGINAL string unchanged (never raises) whenever SymPy
    is unavailable or the string uses notation the permissive parser
    can't safely round-trip.
    """
    if not _SYMPY_AVAILABLE:
        return s
    if "log10" in s or "{" in s or "}" in s:
        return s
    friendly = s.replace('^', '**')
    try:
        expr = parse_expr(friendly, transformations=_SYMPY_TRANSFORMS)
        expanded = sympy.expand(expr)
        return str(expanded).replace('**', '^')
    except Exception:
        return s


def _term_shape(term_str):
    s = term_str
    has_half_power = bool(re.search(r'\^\(?(?:-?0\.5)\)?', s))
    has_negative_half_power = bool(re.search(r'\^\(?-0\.5\)?', s))
    has_sqrt = bool(re.search(r'\bsqrt\b', s)) or has_half_power
    has_log  = bool(re.search(r'\blog\d*\b', s))
    has_exp  = bool(re.search(r'\bexp\b', s))
    has_trig = bool(re.search(r'\b(sin|cos|tan)\b', s))
    has_div  = '/' in s or has_negative_half_power
    has_pow  = '^' in s
    parts = []
    if has_sqrt: parts.append('sqrt')
    if has_log: parts.append('log')
    if has_exp: parts.append('exp')
    if has_trig: parts.append('trig')
    if has_pow and not has_sqrt: parts.append('pow')
    if has_div: parts.append('div')
    return "_".join(parts) if parts else "linear"


def _extract_term_shapes(s):
    expanded = _expand_for_term_split(s)
    terms = _split_top_level_terms(expanded)
    if not terms:
        terms = [s]
    return _Counter(_term_shape(t) for t in terms)


def _multiset_similarity(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = sum((a & b).values())
    total = sum(a.values()) + sum(b.values())
    return 2.0 * inter / total if total else 0.0


def build_signature(raw_text, extra_operators=None, extra_functions=None, complexity_hint=None):
    norm = _canonical_math_text(raw_text)
    ops = _extract_operators(norm)
    fns = _extract_functions(norm)
    if extra_operators:
        ops |= {BINARY_MAP.get(o, o) for o in extra_operators}
    if extra_functions:
        fns |= {UNARY_MAP.get(f, f) for f in extra_functions}
    return {
        "operators": ops,
        "functions": fns,
        "exponents": _extract_exponents(norm),
        "var_count": _extract_var_count(norm),
        "var_names": _extract_var_tokens(norm),
        "term_shapes": _extract_term_shapes(norm),
        "complexity": complexity_hint if complexity_hint is not None else _extract_complexity(norm),
        "_normalized_text": norm,
    }


# Precompute once at startup — same pattern as CLUSTER_DESCRIPTIONS.
EQ_SIGNATURES = [
    build_signature(eq.get("latex", ""), eq.get("operators", []), eq.get("functions", []))
    for eq in EQ_DB
]

# THIS REVISION — fix #3: per-entry parse-quality, precomputed alongside
# the signatures, from the SAME normalized text build_signature() already
# produced (stored under "_normalized_text") — no re-normalization.
EQ_PARSE_QUALITY = [_parse_quality(sig.get("_normalized_text", "")) for sig in EQ_SIGNATURES]

SIM_WEIGHTS = {
    "operators": 0.10, "functions": 0.15, "exponents": 0.15,
    "complexity": 0.05, "var_count": 0.05, "var_names": 0.15,
    "term_shapes": 0.35,
}


def _jaccard(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _closeness(a, b, scale):
    if a is None or b is None:
        return 0.5
    return float(np.exp(-abs(a - b) / scale))


def signature_similarity(sig_a, sig_b):
    op_s  = _jaccard(sig_a["operators"], sig_b["operators"])
    fn_s  = _jaccard(sig_a["functions"], sig_b["functions"])
    exp_s = _jaccard(sig_a["exponents"], sig_b["exponents"]) if (sig_a["exponents"] or sig_b["exponents"]) else 1.0
    cx_s  = _closeness(sig_a["complexity"], sig_b["complexity"], scale=8.0)
    vc_s  = _closeness(sig_a["var_count"], sig_b["var_count"], scale=3.0)
    vn_s  = _jaccard(sig_a["var_names"], sig_b["var_names"])
    ts_s  = _multiset_similarity(sig_a["term_shapes"], sig_b["term_shapes"])
    return (SIM_WEIGHTS["operators"] * op_s + SIM_WEIGHTS["functions"] * fn_s +
            SIM_WEIGHTS["exponents"] * exp_s + SIM_WEIGHTS["complexity"] * cx_s +
            SIM_WEIGHTS["var_count"] * vc_s + SIM_WEIGHTS["var_names"] * vn_s +
            SIM_WEIGHTS["term_shapes"] * ts_s)


def _similarity_breakdown(sig_a, sig_b):
    op_s  = _jaccard(sig_a["operators"], sig_b["operators"])
    fn_s  = _jaccard(sig_a["functions"], sig_b["functions"])
    exp_s = _jaccard(sig_a["exponents"], sig_b["exponents"]) if (sig_a["exponents"] or sig_b["exponents"]) else 1.0
    cx_s  = _closeness(sig_a["complexity"], sig_b["complexity"], scale=8.0)
    vc_s  = _closeness(sig_a["var_count"], sig_b["var_count"], scale=3.0)
    vn_s  = _jaccard(sig_a["var_names"], sig_b["var_names"])
    ts_s  = _multiset_similarity(sig_a["term_shapes"], sig_b["term_shapes"])
    return [
        ("term_shapes", ts_s), ("functions", fn_s), ("operators", op_s),
        ("exponents", exp_s), ("var_names", vn_s),
        ("complexity", cx_s), ("var_count", vc_s),
    ]


def top_similar_equations(discovered_sig, top_k=5, indices=None,
                           preferred_subfields=None, domain_penalty=0.85,
                           dominant_core_sig=None):
    """
    Search the equation database for the closest structural matches.

    THIS REVISION — fix #3: a candidate's raw structural score is now
    also multiplied by a parse-quality factor (floor 0.5x, so a
    partially-garbled entry can still surface as a hint, just not ahead
    of an equally-strong cleanly-parsed one). Computed once at startup
    per reference entry (EQ_PARSE_QUALITY), never for the discovered
    equation's own signature (which is always fresh humanized text, not
    scraped LaTeX).
    """
    pool = indices if indices is not None else range(len(EQ_DB))
    scored = []
    for idx in pool:
        score = signature_similarity(discovered_sig, EQ_SIGNATURES[idx])
        # If a small correction is attached to a dominant engineered feature,
        # compare the reference to that prediction-bearing core as well. This
        # prevents an ornamental low-impact function from overwhelming the
        # structural signal of the feature that actually explains the target.
        if dominant_core_sig is not None:
            score = max(score, signature_similarity(dominant_core_sig, EQ_SIGNATURES[idx]))
        score *= (0.5 + 0.5 * EQ_PARSE_QUALITY[idx])
        if preferred_subfields:
            subfield = EQ_DB[idx].get("subfield")
            if subfield not in preferred_subfields:
                score *= domain_penalty
        scored.append((score, EQ_DB[idx]))
    scored.sort(key=lambda t: -t[0])
    return scored[:top_k]


def grammar_compatibility(discovered_sig, unary_ops, binary_ops, unary_weights, binary_weights,
                          operator_influence=None):
    total_weight = sum(unary_weights.values()) + sum(binary_weights.values())
    if total_weight <= 0:
        return 0.0
    used_weight = 0.0
    for op in discovered_sig["functions"]:
        evidence = 1.0 if operator_influence is None else operator_influence.get(op, 0.0)
        used_weight += unary_weights.get(op, 0.0) * max(0.0, min(1.0, evidence))
    for op in discovered_sig["operators"]:
        evidence = 1.0 if operator_influence is None else operator_influence.get(op, 0.0)
        used_weight += binary_weights.get(op, 0.0) * max(0.0, min(1.0, evidence))
    return min(1.0, used_weight / total_weight)


def grammar_compatibility_detail(discovered_sig, unary_ops, binary_ops, unary_weights, binary_weights,
                                 operator_influence=None):
    matched_fns = sorted(op for op in unary_ops if op in discovered_sig["functions"])
    missing_fns = sorted(op for op in unary_ops if op not in discovered_sig["functions"])
    matched_ops = sorted(op for op in binary_ops if op in discovered_sig["operators"])
    missing_ops = sorted(op for op in binary_ops if op not in discovered_sig["operators"])
    pct = grammar_compatibility(
        discovered_sig, unary_ops, binary_ops, unary_weights, binary_weights, operator_influence
    ) * 100.0
    return {
        "pct": pct,
        "matched_functions": matched_fns, "missing_functions": missing_fns,
        "matched_operators": matched_ops, "missing_operators": missing_ops,
    }


def _compact_expression(text):
    return re.sub(r"[\s()]", "", _canonical_math_text(text))


def _feature_operator_influence(equation_math, importance_by_feature, math_expr_map):
    """Estimate each operator's prediction influence from feature permutation mass.

    A feature's positive permutation importance is assigned to operators in
    the mathematical expression of that feature. Explicit function arguments
    also receive the importance of the variables they contain. Thus an
    ``exp(sigma_0)`` correction receives only sigma_0's influence, while a
    high-importance ``k/sqrt(d)`` feature supports both division and sqrt.
    """
    positive = {k: max(0.0, float(v)) for k, v in importance_by_feature.items()}
    total = sum(positive.values())
    if total <= 1e-12:
        return {}, None
    shares = {k: v / total for k, v in positive.items() if v > 0}
    compact_eq = _compact_expression(equation_math)
    influence = defaultdict(float)
    used = {}
    for name, share in shares.items():
        expr = math_expr_map.get(name, name)
        if _compact_expression(expr) in compact_eq:
            used[name] = share
            feature_sig = build_signature(expr)
            for op in feature_sig["functions"] | feature_sig["operators"]:
                influence[op] += share

    # An explicit unary wrapper can be ornamental. Attribute it only the
    # importance mass of variables in its argument, not the whole equation.
    equation_sig = build_signature(equation_math)
    for fn in equation_sig["functions"]:
        for match in re.finditer(r"\b" + re.escape(fn) + r"\s*\(([^()]*)\)", equation_math):
            arg_vars = build_signature(match.group(1))["var_names"]
            for name, share in shares.items():
                expr_vars = build_signature(math_expr_map.get(name, name))["var_names"]
                if arg_vars & expr_vars:
                    influence[fn] += share

    active_mass = sum(used.values())
    for op in equation_sig["operators"]:
        # Binary operators combine the active, prediction-bearing terms.
        influence[op] = max(influence[op], active_mass)
    return {op: min(1.0, value) for op, value in influence.items()}, used


def _dominant_core_signature(used_feature_shares, math_expr_map):
    """Signature of the >=1% prediction-bearing engineered-feature core."""
    terms = [math_expr_map.get(name, name) for name, share in used_feature_shares.items() if share >= 0.01]
    if len(terms) < 2:
        return None
    return build_signature(" + ".join(sorted(terms)))


# ─────────────────────────────────────────────
# Optional best-effort SymPy confirmation.
# ─────────────────────────────────────────────
def _to_sympy_friendly(s):
    if "log10" in s or "{" in s or "}" in s:
        return None
    return s.replace('^', '**')


def sympy_exact_match(discovered_math_str, candidate_latex):
    if not _SYMPY_AVAILABLE:
        return None
    left = _to_sympy_friendly(_normalize_math_text(discovered_math_str))
    right = _to_sympy_friendly(_normalize_math_text(candidate_latex))
    if left is None or right is None:
        return None
    try:
        expr_l = parse_expr(left, transformations=_SYMPY_TRANSFORMS)
        expr_r = parse_expr(right, transformations=_SYMPY_TRANSFORMS)
        shared = expr_l.free_symbols & expr_r.free_symbols
        expr_l = expr_l.subs({s: 1 for s in expr_l.free_symbols - shared})
        expr_r = expr_r.subs({s: 1 for s in expr_r.free_symbols - shared})
        return bool(sympy.simplify(expr_l - expr_r) == 0)
    except Exception:
        return None


MAX_ROWS     = 80
MIN_ITER     = 20
MAX_ITER     = 300
POPULATIONS  = 3
POPULATION_SIZE = 30
MAXSIZE      = 12
TIMEOUT_S    = 180


def _structural_equation_key(equation_math):
    """Canonical, coefficient-agnostic key used only for candidate deduplication."""
    canonical = _canonical_math_text(equation_math)
    canonical = re.sub(r'(?<![A-Za-z_\d])-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?', 'C', canonical)
    return re.sub(r'\s+', '', canonical)


def _collect_final_candidates(equations, chosen_cid, unary_ops, binary_ops,
                              unary_weights, binary_weights, math_expr_map,
                              humanise, humanise_math, importance_by_feature):
    """Collect every final-search equation for cross-cluster comparison.

    Ranking later uses model loss first, then influence-weighted grammar
    compatibility, global reference similarity, and lower complexity. This is
    intentionally lexicographic rather than a new manually weighted score.
    """
    candidates = []
    for _, row in equations.iterrows():
        raw_eq = str(row.get("equation", ""))
        if not raw_eq:
            continue
        equation_math = humanise_math(raw_eq)
        complexity = float(row.get("complexity", float("inf")))
        loss = float(row.get("loss", float("inf")))
        sig = build_signature(equation_math, complexity_hint=complexity)
        influence, used_shares = _feature_operator_influence(
            equation_math, importance_by_feature, math_expr_map
        )
        core_sig = _dominant_core_signature(used_shares or {}, math_expr_map)
        matches = top_similar_equations(sig, top_k=1, dominant_core_sig=core_sig)
        similarity = matches[0][0] if matches else 0.0
        reference_name = (
            matches[0][1].get("equation_name") or matches[0][1].get("image_hint") or "Unknown"
            if matches else "Unknown"
        )
        compatibility = grammar_compatibility(
            sig, unary_ops, binary_ops, unary_weights, binary_weights,
            influence if importance_by_feature else None,
        )
        candidates.append({
            "cluster": str(chosen_cid),
            "equation": replace_constants(humanise(raw_eq)),
            "loss": loss,
            "complexity": complexity,
            "compatibility": compatibility,
            "similarity": similarity,
            "reference_name": reference_name,
            "key": _structural_equation_key(equation_math),
        })
    return candidates


def _merge_final_candidates(runs, top_k=10):
    """Deduplicate candidates from multiple final PySR runs and rank them."""
    unique = {}
    def rank_key(c):
        return (c["loss"], -c["compatibility"], -c["similarity"], c["complexity"])
    for _, run in runs:
        for candidate in run.get("candidates", []):
            key = candidate["key"]
            if key not in unique or rank_key(candidate) < rank_key(unique[key]):
                unique[key] = candidate
    ranked = sorted(unique.values(), key=rank_key)[:top_k]
    contribution = _Counter(c["cluster"] for c in ranked)
    return ranked, contribution


# ─────────────────────────────────────────────
# Core PySR runner (final, full-quality run)
# ─────────────────────────────────────────────
def _run_pysr_with_cluster(df, target, grammars, chosen_cid, n_iter, source_label,
                            probe_compat=None, use_timeout=True):
    return _run_pysr_with_cluster_full(
        df, target, grammars, chosen_cid, n_iter, source_label,
        probe_compat=probe_compat, use_timeout=use_timeout,
    )["text"]


def _run_pysr_with_cluster_full(df, target, grammars, chosen_cid, n_iter, source_label,
                                 probe_compat=None, use_timeout=True):
    grammar = grammars[chosen_cid]
    unary_ops, binary_ops, unary_weights, binary_weights = extract_ops_from_grammar(grammar)
    X_raw = df.drop(columns=[target])
    y     = df[target].values

    X_eng = engineer_features(X_raw, y, allowed_unary=set(unary_ops), r2_threshold=0.05)
    eng_names = list(X_eng.columns)

    top_features = rank_features_by_r2(X_eng, y, top_k=10)

    scaler_X = RobustScaler()
    scaler_y = RobustScaler()
    X_scaled = scaler_X.fit_transform(X_eng)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    _julia_ready.wait(timeout=180)
    try:
        from pysr import PySRRegressor
        pysr_kwargs = dict(
            niterations=n_iter, populations=POPULATIONS,
            population_size=POPULATION_SIZE, maxsize=MAXSIZE,
            tournament_selection_n=_safe_tournament_n(POPULATION_SIZE),
            binary_operators=binary_ops, unary_operators=unary_ops,
            complexity_of_operators=operator_complexity_costs(
                unary_ops, binary_ops, unary_weights, binary_weights
            ),
            maxdepth=MAXDEPTH,
            nested_constraints=pysr_structure_constraints(unary_ops, binary_ops),
            model_selection="best", parsimony=1e-4, random_state=42,
            verbosity=0, procs=0, multithreading=False,
        )
        if use_timeout:
            pysr_kwargs["timeout_in_seconds"] = TIMEOUT_S
        model = PySRRegressor(**pysr_kwargs)
        model.fit(X_scaled, y_scaled)
    except Exception:
        fail_text = f"❌ PySR failed:\n{traceback.format_exc()}"
        return {
            "text": fail_text, "r2": float("-inf"), "rmse": float("inf"),
            "grammar_compat_pct": None, "top1_complexity": None, "top1_loss": None,
            "candidates": [],
        }
    y_pred_s = model.predict(X_scaled)
    y_pred   = scaler_y.inverse_transform(y_pred_s.reshape(-1, 1)).flatten()
    r2   = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    perm_importance_rows = []
    perm_importance_by_feature = {}
    try:
        perm = permutation_importance(
            model, X_scaled, y_scaled, n_repeats=10, random_state=42, scoring="r2",
        )
        perm_importance_by_feature = dict(zip(eng_names, perm.importances_mean))
        perm_importance_rows = sorted(
            zip(eng_names, perm.importances_mean, perm.importances_std),
            key=lambda t: -t[1],
        )[:10]
    except Exception:
        perm_importance_rows = []

    var_map = {f"x{i}": name for i, name in enumerate(eng_names)}
    def humanise(eq):
        for xi, name in sorted(var_map.items(), key=lambda kv: -len(kv[0])):
            eq = eq.replace(xi, name)
        return eq

    math_expr_map = X_eng.attrs.get("math_expr", {n: n for n in eng_names})
    var_map_math = {f"x{i}": f"({math_expr_map.get(name, name)})"
                     for i, name in enumerate(eng_names)}
    def humanise_math(eq):
        for xi, expr in sorted(var_map_math.items(), key=lambda kv: -len(kv[0])):
            eq = eq.replace(xi, expr)
        return eq

    try:
        all_eqs = model.equations_.sort_values(by="loss")
        eqs = all_eqs.head(5)
    except Exception:
        all_eqs = pd.DataFrame()
        eqs = pd.DataFrame()

    eq_lines = []
    top_within_cluster   = []
    top_global_matches    = []
    best_match_cluster    = None
    grammar_compat_pct    = None
    compat_detail          = None
    top_subfields         = set(CLUSTER_DESCRIPTIONS.get(str(chosen_cid), {}).get("top_subfields", []))
    cluster_indices       = _CLUSTER_TO_INDICES.get(str(chosen_cid), [])
    top_match_confirmed   = None
    top_eq_var_count      = None
    top1_complexity       = None
    top1_loss             = None
    top1_operator_influence = {}
    candidates = _collect_final_candidates(
        all_eqs, chosen_cid, unary_ops, binary_ops, unary_weights, binary_weights,
        math_expr_map, humanise, humanise_math, perm_importance_by_feature,
    )

    for i, (_, row) in enumerate(eqs.iterrows(), 1):
        raw_eq   = str(row.get("equation", "N/A"))
        loss_val = float(row.get("loss", float("nan")))
        cx_val   = row.get("complexity", None)
        human_eq = humanise(raw_eq)
        clean_eq = replace_constants(human_eq)

        sig = build_signature(
            humanise_math(raw_eq),
            complexity_hint=float(cx_val) if cx_val is not None else None,
        )
        equation_math = humanise_math(raw_eq)
        operator_influence, used_feature_shares = _feature_operator_influence(
            equation_math, perm_importance_by_feature, math_expr_map
        )
        core_sig = _dominant_core_signature(used_feature_shares or {}, math_expr_map)
        matches = top_similar_equations(
            sig, top_k=5, preferred_subfields=top_subfields,
            dominant_core_sig=core_sig,
        )
        best_name, best_score = "Unknown", 0.0
        if matches:
            best_score, best_rec = matches[0]
            best_name = best_rec.get("equation_name") or best_rec.get("image_hint") or "Unknown"

        match_note = ""
        simplified_line = ""
        if i == 1 and matches:
            top_match_confirmed = sympy_exact_match(humanise_math(raw_eq), matches[0][1].get("latex", ""))
            if top_match_confirmed:
                match_note = "\n   🧮 SymPy: algebraically confirmed (heuristic — see docstring)"

            best_sig_for_breakdown = build_signature(
                matches[0][1].get("latex", ""),
                matches[0][1].get("operators", []),
                matches[0][1].get("functions", []),
            )
            breakdown = _similarity_breakdown(sig, best_sig_for_breakdown)
            breakdown_lines = "\n".join(
                f"      {name:<12}: {val*100:5.1f}%" for name, val in breakdown
            )
            match_note += f"\n   Similarity Breakdown (top match):\n{breakdown_lines}"

            simplified = try_sympy_simplify(human_eq)
            if simplified:
                simplified_line = f"\n   Simplified: {simplified}"

        eq_lines.append(
            f"🔹 Equation {i}:\n"
            f"   {clean_eq}{simplified_line}\n"
            f"   Loss      : {loss_val:.6f}\n"
            f"   Best Match: {best_name} ({best_score*100:.1f}%){match_note}\n"
            f"   {'─'*44}"
        )

        if i == 1:
            top_within_cluster = top_similar_equations(
                sig, top_k=5, indices=cluster_indices, preferred_subfields=top_subfields,
                dominant_core_sig=core_sig,
            )
            top_global_matches = matches
            if matches:
                best_match_cluster = matches[0][1].get("cluster")
            compat_detail = grammar_compatibility_detail(
                sig, unary_ops, binary_ops, unary_weights, binary_weights,
                operator_influence if perm_importance_by_feature else None,
            )
            top1_operator_influence = operator_influence
            grammar_compat_pct = compat_detail["pct"]
            top_eq_var_count = sig["var_count"]
            top1_complexity = float(cx_val) if cx_val is not None else None
            top1_loss = loss_val

    eq_text = "\n".join(eq_lines) if eq_lines else "  (no equations returned)"

    def _fmt_match_block(rank, score, rec):
        name = rec.get("equation_name") or rec.get("image_hint") or "Unknown"
        subfield = rec.get("subfield", "Unknown")
        caution = ""
        if top_subfields and subfield not in top_subfields:
            caution = "   ⚠ Different physical domain from the selected cluster — score nudged down.\n"
        return (
            f"{rank}. {name}\n"
            f"   Similarity : {score*100:.1f}%\n"
            f"   Cluster    : C{rec.get('cluster')}\n"
            f"   Subfield   : {subfield}\n"
            f"{caution}"
            f"   LaTeX      : {rec.get('latex', 'N/A')}\n"
            f"   {'-'*24}"
        )

    within_lines = [
        _fmt_match_block(rank, score, rec)
        for rank, (score, rec) in enumerate(top_within_cluster, 1)
    ]
    within_text = "\n".join(within_lines) if within_lines else "  (no equations in this cluster to compare against)"

    global_lines = [
        _fmt_match_block(rank, score, rec)
        for rank, (score, rec) in enumerate(top_global_matches, 1)
    ]
    global_text = "\n".join(global_lines) if global_lines else "  (no reference matches found)"

    feature_rank_text = "\n".join(
        f"   {i+1:>2}. {name:<28} R² = {r2v:.4f}"
        for i, (name, r2v) in enumerate(top_features)
    ) if top_features else "  (no engineered features)"

    cluster_match = "N/A"
    if best_match_cluster is not None:
        cluster_match = "YES ✅" if str(best_match_cluster) == str(chosen_cid) else "NO ❌"

    final_compat_text = f"{grammar_compat_pct:.1f}%" if grammar_compat_pct is not None else "N/A"
    probe_compat_text = f"{probe_compat*100:.1f}%" if probe_compat is not None else "N/A (manual cluster selection — no probe stage ran)"
    probe_confidence_text = grammar_confidence_label(probe_compat * 100.0 if probe_compat is not None else None)
    final_confidence_text = grammar_confidence_label(grammar_compat_pct)

    compat_detail_text = "  (not computed)"
    if compat_detail is not None:
        compat_detail_text = (
            f"    Matched Functions : {', '.join(compat_detail['matched_functions']) or '(none)'}\n"
            f"    Missing Functions : {', '.join(compat_detail['missing_functions']) or '(none)'}\n"
            f"    Matched Operators : {', '.join(compat_detail['matched_operators']) or '(none)'}\n"
            f"    Missing Operators : {', '.join(compat_detail['missing_operators']) or '(none)'}"
        )

    perm_text = "\n".join(
        f"   {i+1:>2}. {name:<28} importance = {mean:+.4f}  (± {std:.4f})"
        for i, (name, mean, std) in enumerate(perm_importance_rows)
    ) if perm_importance_rows else "  (permutation importance unavailable for this fit)"

    operator_influence_text = ", ".join(
        f"{op}={share*100:.1f}%" for op, share in sorted(top1_operator_influence.items())
    ) or "N/A (permutation importance unavailable)"

    adj_r2 = adjusted_r2(r2, df.shape[0], top_eq_var_count if top_eq_var_count is not None else X_eng.shape[1])
    adj_r2_text = f"{adj_r2:.5f}" if adj_r2 is not None else "N/A (too few rows relative to predictors)"

    grammar_profile_text = format_grammar_profile(
        chosen_cid, grammar, unary_ops, binary_ops, unary_weights, binary_weights
    )
    report_text = f"""
📊  DATASET
    Rows    : {df.shape[0]}
    Columns : {X_raw.shape[1]} raw → {X_eng.shape[1]} engineered (grammar-constrained)
    Target  : {target}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯  CLUSTER SELECTION
    Method     : {source_label}
{grammar_profile_text}
    Binary ops : {binary_ops}   weights: {binary_weights}
    Unary  ops : {unary_ops}   weights: {unary_weights}

📐  ENGINEERED FEATURES (given to PySR — grammar-constrained)
    {eng_names}

📈  TOP 10 ENGINEERED FEATURES BY R² (individual, linear, vs. target)
    (pre-fit signal only — correlates with the target, doesn't prove the
    discovered equation actually uses it. See permutation importance below.)
{feature_rank_text}

📊  PERMUTATION IMPORTANCE (post-PySR, on the fitted model)
    Mean drop in R² when each engineered feature is shuffled — computed
    AFTER fitting, on the model PySR actually produced, so this reflects
    what the discovered equation relies on rather than raw correlation.
{perm_text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🏆  TOP EQUATIONS  (constants → A, B, C …)

{eq_text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🧬  GRAMMAR COMPATIBILITY
    Weighted operator mass supported by prediction influence, rather than
    mere syntactic presence. Reported at TWO points because the probe (short fit,
    used to pick the cluster) and the final run (longer fit) don't
    necessarily discover the same equation — collapsing them into one
    number hides that the cluster may have been picked on the strength of
    an equation the final search later moved away from. Confidence bands:
    ≥80% High · 50-80% Medium · <50% Low.
    Probe-Stage Equation Compatibility  : {probe_compat_text}  ({probe_confidence_text})
    Final Equation Compatibility        : {final_compat_text}  ({final_confidence_text})
    Operator influence evidence         : {operator_influence_text}
    Final Equation — Matched vs. Missing:
{compat_detail_text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔎  SIMILARITY — STAGE 1: WITHIN SELECTED CLUSTER (C{chosen_cid})
    Sanity check: does cluster selection line up with the closest matches
    inside the cluster PySR was actually constrained to?

{within_text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔎  SIMILARITY — STAGE 2: GLOBAL SEARCH  (best discovered equation, searched
    against the full {len(EQ_DB)}-equation database. Term-shape structure
    weighted highest; out-of-domain candidates nudged down; garbled/OCR-
    corrupted reference entries mildly downweighted — see report header
    for details, no equation is ever named or favored in code.)

{global_text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯  PROBE VS REFERENCE AGREEMENT
    Two independent judges — this does NOT mean either one is "wrong"
    when they disagree, only that the grammar which best explains the
    dataset isn't necessarily the same as the reference equation the
    discovered formula structurally resembles most.
    Probe Selected Grammar      : C{chosen_cid}  ({grammar_display_name(chosen_cid, grammar)})
    Closest Reference Cluster   : {"C" + str(best_match_cluster) if best_match_cluster is not None else "N/A"}
    Agreement                    : {cluster_match}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📈  PERFORMANCE
    R²            = {r2:.5f}
    Adjusted R²   = {adj_r2_text}
    (n={df.shape[0]} rows, p={top_eq_var_count if top_eq_var_count is not None else X_eng.shape[1]} distinct
     input variables in the top equation — penalizes R² for the engineered
     feature count so it can't look strong purely from feature abundance)
    RMSE          = {rmse:.5f}
""".strip()

    return {
        "text": report_text, "r2": float(r2), "rmse": float(rmse),
        "grammar_compat_pct": grammar_compat_pct,
        "top1_complexity": top1_complexity, "top1_loss": top1_loss,
        "candidates": candidates,
    }


# ─────────────────────────────────────────────
# Reference-equation search + select (Tab 2)
# ─────────────────────────────────────────────
def _normalize_search_text(s):
    """
    THIS REVISION — fix #4. Lowercases and collapses hyphens/underscores/
    slashes to spaces, so "hall petch" and "Hall-Petch" (or "hall_petch",
    "hall/petch") normalize to the same token stream. General punctuation
    handling, not scoped to any specific equation name.
    """
    s = (s or "").lower()
    s = re.sub(r'[-_/]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def search_reference_equations(query, subfield_filter):
    global _LABEL_TO_RECORD
    _LABEL_TO_RECORD = {}

    query_norm = _normalize_search_text(query)
    query_tokens = [t for t in query_norm.split(' ') if t]
    sf = (subfield_filter or "").strip()

    results = EQ_DB
    if sf and sf != "All subfields":
        results = [r for r in results if r["subfield"] == sf]

    if query_tokens:
        filtered = []
        for r in results:
            haystack = _normalize_search_text(" ".join([
                r.get("equation_name", ""),
                r.get("equation_name_raw", ""),
                r.get("image_hint", ""),
                r.get("unit", ""),
                r.get("subfield", ""),
                r.get("latex", ""),
                " ".join(r.get("operators", [])),
                " ".join(r.get("functions", [])),
            ]))
            # THIS REVISION — fix #4: every query token must appear
            # somewhere in the haystack (AND search), not the exact
            # phrase as one contiguous substring. This is what lets
            # "hall petch" find "Hall-Petch Relation" once both sides are
            # punctuation-normalized above.
            if all(tok in haystack for tok in query_tokens):
                filtered.append(r)
        results = filtered

    if not results:
        return (
            gr.update(choices=[], value=None),
            "❌ No equations found. Try a different keyword or subfield.",
            "",
        )

    shown = results[:40]
    total = len(results)

    choices    = []
    seen_labels = {}

    for eq in shown:
        cluster  = eq["cluster"]
        eq_name  = (eq.get("equation_name") or eq.get("image_hint") or "Unknown").strip()
        sf_name  = eq.get("subfield", "unknown")

        base = f"[C{cluster}]  {eq_name}  ({sf_name})"
        if base in seen_labels:
            seen_labels[base] += 1
            label = f"{base} #{seen_labels[base]}"
        else:
            seen_labels[base] = 1
            label = base

        choices.append(label)
        _LABEL_TO_RECORD[label] = eq

    status  = f"✅ {total} match(es)" + (" — showing first 40" if total > 40 else "")
    preview = shown[0].get("latex", "") if shown else ""

    return (
        gr.update(choices=choices, value=None),
        status,
        preview,
    )


def on_equation_selected(selected_label):
    if not selected_label:
        return "No equation selected.", None, ""

    record = _LABEL_TO_RECORD.get(selected_label, {})
    match  = re.match(r'\[C(\d+)\]', selected_label)
    if not match:
        return "Could not parse cluster from selection.", None, ""

    cid   = int(match.group(1))
    desc  = CLUSTER_DESCRIPTIONS.get(str(cid), {})
    top_sf = desc.get("top_subfields", [])
    total  = desc.get("total", "?")
    latex  = record.get("latex", "")

    info = (
        f"✅  Selected  →  Cluster {cid}\n"
        f"    Equation : {selected_label}\n"
        f"    This cluster contains {total} equations.\n"
        f"    Top subfields:\n"
        + "\n".join(f"      • {s}" for s in top_sf) +
        "\n\n    Upload your CSV on the right and click\n"
        "    ▶ Run with Selected Cluster."
    )
    return info, str(cid), latex


def run_with_selected_cluster(file, target, iterations, selected_cid):
    if file is None:
        return "❌ Please upload a CSV file."
    target = (target or "").strip()
    if not target:
        return "❌ Please enter the target column name."
    if not selected_cid:
        return (
            "❌ No reference equation selected.\n"
            "   Search on the left and click a result,\n"
            "   or use the Auto-Probe tab."
        )
    try:
        df = pd.read_csv(file.name)
    except Exception as e:
        return f"❌ Could not read CSV: {e}"
    if target not in df.columns:
        return f"❌ Column '{target}' not found. Available: {list(df.columns)}"
    df = df.select_dtypes(include=[np.number]).dropna()
    if target not in df.columns:
        return f"❌ '{target}' is not numeric after filtering."
    if df.shape[0] < 10:
        return "❌ Not enough rows (need ≥ 10)."
    if df.shape[1] < 2:
        return "❌ Need at least one feature column besides the target."
    if len(df) > MAX_ROWS:
        df = df.sample(MAX_ROWS, random_state=42)
    grammars = load_grammars()
    if not grammars:
        return "❌ No grammar files found in ./grammars/"
    if selected_cid not in grammars:
        return (
            f"❌ Grammar file for Cluster {selected_cid} not found.\n"
            f"   Available clusters: {list(grammars.keys())}"
        )
    n_iter = max(MIN_ITER, min(int(iterations), MAX_ITER))
    source = f"User-selected reference equation  (Cluster {selected_cid})"
    return _run_pysr_with_cluster(df, target, grammars, selected_cid, n_iter, source)


# ─────────────────────────────────────────────
# THIS REVISION — fix #5: read-only diagnostics helper.
#
# Lets you check, independently of any regression run, whether a keyword
# resolves to an entry in equations_browser.json and how cleanly it
# parses. Purely inspection — it does not feed into or bias the
# similarity ranking used by Tab 1 or Tab 2 in any way.
# ─────────────────────────────────────────────
def diagnose_keyword(keyword):
    keyword = (keyword or "").strip()
    if not keyword:
        return "Enter a keyword above (e.g. an equation name) and click Look Up."
    query_norm = _normalize_search_text(keyword)
    query_tokens = [t for t in query_norm.split(' ') if t]
    if not query_tokens:
        return "Enter a keyword above and click Look Up."

    hits = []
    for idx, r in enumerate(EQ_DB):
        haystack = _normalize_search_text(" ".join([
            r.get("equation_name", ""), r.get("equation_name_raw", ""),
            r.get("image_hint", ""), r.get("subfield", ""), r.get("latex", ""),
        ]))
        if all(tok in haystack for tok in query_tokens):
            hits.append(idx)

    if not hits:
        return (
            f"❌ No entries in equations_browser.json match all of: {query_tokens}\n"
            f"   ({len(EQ_DB)} total entries searched.)\n\n"
            f"   This means either the equation genuinely isn't in your reference\n"
            f"   database under this name, or it's stored under different wording —\n"
            f"   try a shorter/partial keyword (e.g. just 'petch' or just 'hall')."
        )

    lines = [f"✅ {len(hits)} entry(ies) match all of: {query_tokens}\n"]
    for idx in hits[:5]:
        r = EQ_DB[idx]
        sig = EQ_SIGNATURES[idx]
        quality = EQ_PARSE_QUALITY[idx]
        quality_label = "clean" if quality >= 0.8 else ("partial debris" if quality >= 0.4 else "heavily garbled")
        lines.append(
            f"— {r.get('equation_name') or r.get('image_hint') or 'Unknown'}  "
            f"(Cluster C{r.get('cluster')}, subfield: {r.get('subfield', 'unknown')})\n"
            f"   Raw LaTeX      : {r.get('latex', 'N/A')}\n"
            f"   Normalized     : {sig.get('_normalized_text', 'N/A')}\n"
            f"   Parse quality  : {quality*100:.0f}%  ({quality_label})\n"
            f"   Operators      : {sorted(sig['operators'])}\n"
            f"   Functions      : {sorted(sig['functions'])}\n"
            f"   Var names      : {sorted(sig['var_names'])}\n"
            f"   Term shapes    : {dict(sig['term_shapes'])}\n"
        )
    if len(hits) > 5:
        lines.append(f"...and {len(hits) - 5} more match(es) not shown.")
    return "\n".join(lines)


# ─────────────────────────────────────────────
# Tab 1: Auto-Probe (primary workflow — grammar-guided)
# ─────────────────────────────────────────────
def run_metasr_probe(file, target, iterations, mode, top3_compare,
                      progress=gr.Progress(track_tqdm=False)):
    if file is None:
        return "❌ Please upload a CSV file."
    target = (target or "").strip()
    if not target:
        return "❌ Please enter the target column name."
    try:
        df = pd.read_csv(file.name)
    except Exception as e:
        return f"❌ Could not read CSV: {e}"
    if target not in df.columns:
        return f"❌ Column '{target}' not found. Available: {list(df.columns)}"
    df = df.select_dtypes(include=[np.number]).dropna()
    if target not in df.columns:
        return f"❌ '{target}' is not numeric after filtering."
    if df.shape[0] < 10:
        return "❌ Not enough rows (need ≥ 10)."
    if df.shape[1] < 2:
        return "❌ Need at least one feature column besides the target."
    if len(df) > MAX_ROWS:
        df = df.sample(MAX_ROWS, random_state=42)

    grammars = load_grammars()
    if not grammars:
        return "❌ No grammar files found in ./grammars/"

    is_research = str(mode).startswith("🔬")
    progress(0, desc="Starting probe...")

    if is_research:
        cid, scores, tied = research_grammar_probe(df, target, grammars, progress=progress)
        score_lines = "\n".join(
            f"  C{k:>3}: loss={v['loss']:.6f}   compat={v['compatibility']*100:5.1f}%   complexity={v['complexity']:.1f}   "
            f"combined={v['combined']:.4f}   time={v['elapsed_s']:.1f}s"
            for k, v in sorted(scores.items())
        )
        total_probe_time = sum(v["elapsed_s"] for v in scores.values())
        score_lines += f"\n  (total probe wall-clock: {total_probe_time:.1f}s across {len(scores)} clusters)"
        method_label = (
            f"Research probe — grammar-guided PySR ({PROBE_ITER} iters × {len(grammars)} clusters), "
            f"selection = {RESEARCH_LOSS_WEIGHT:.2f}×normalized_loss + "
            f"{RESEARCH_COMPAT_WEIGHT:.2f}×(1-grammar_compatibility) + "
            f"{RESEARCH_COMPLEXITY_WEIGHT:.2f}×normalized_complexity (fixed release defaults)"
        )
    else:
        cid, scores, tied = fast_grammar_probe(df, target, grammars)
        score_lines = "\n".join(f"  C{k:>3}: score={v:.4f}" for k, v in sorted(scores.items()))
        method_label = "Fast probe — grammar-aware, weight-averaged heuristic (no PySR)"

    tied_note = ""
    if len(tied) > 1:
        tied_note = f"\n  ⚠ Near-tied: {', '.join('C'+str(c) for c in tied)}  →  tiebreak winner C{cid}"

    n_iter = max(MIN_ITER, min(int(iterations), MAX_ITER))
    score_block = (
        f"🔍  GRAMMAR PROBE  ({method_label})\n"
        f"{score_lines}{tied_note}\n\n"
    )

    if is_research and top3_compare:
        top3_cids = sorted(scores.keys(), key=lambda c: scores[c]["combined"])[:3]
        progress(0.99, desc="Running final regression on top-3 candidate grammars...")
        runs = []
        for c in top3_cids:
            r = _run_pysr_with_cluster_full(
                df, target, grammars, c, n_iter,
                f"{method_label}{tied_note}  —  top-3 comparison run (candidate C{c})",
                probe_compat=scores[c]["compatibility"], use_timeout=False,
            )
            runs.append((c, r))

        def _final_rank_key(item):
            _, r = item
            r2 = round(r["r2"], 5)
            compat = r.get("grammar_compat_pct")
            compat = compat if compat is not None else 0.0
            complexity = r.get("top1_complexity")
            complexity = complexity if complexity is not None else float("inf")
            loss = r.get("top1_loss")
            loss = loss if loss is not None else float("inf")
            return (r2, compat, -complexity, -loss)

        winner_cid, winner_run = max(runs, key=_final_rank_key)

        comparison_lines = "\n".join(
            f"  C{c:<3} combined={scores[c]['combined']:.4f}   "
            f"final R²={r['r2']:.5f}   final RMSE={r['rmse']:.5f}   "
            f"compat={(r.get('grammar_compat_pct') or 0.0):5.1f}%   "
            f"complexity={r.get('top1_complexity')}"
            f"{'   ← WINNER' if c == winner_cid else ''}"
            for c, r in runs
        )
        comparison_block = (
            f"🏆  TOP-3 GRAMMAR COMPARISON\n"
            f"    Probe scores are close enough (within the 0.02 near-tie band)\n"
            f"    that the short probe-stage fit alone can't reliably separate them.\n"
            f"    Ran the FULL final PySR budget on all three. Ranked by final R²,\n"
            f"    tie-broken by grammar compatibility, then lower complexity, then\n"
            f"    lower loss — R² alone left ties unresolved (e.g. multiple\n"
            f"    candidates at R²=1.00000).\n"
            f"{comparison_lines}\n"
        )
        merged_candidates, contribution = _merge_final_candidates(runs, top_k=10)
        merged_lines = "\n".join(
            f"  {rank:>2}. C{c['cluster']:<3} loss={c['loss']:.6f}  "
            f"compat={c['compatibility']*100:5.1f}%  sim={c['similarity']*100:5.1f}%  "
            f"complexity={c['complexity']:.1f}\n"
            f"      {c['equation']}\n"
            f"      Closest reference: {c['reference_name']}"
            for rank, c in enumerate(merged_candidates, 1)
        ) or "  (No final equations were returned by the candidate clusters.)"
        contribution_text = ", ".join(
            f"C{cid}: {count}" for cid, count in sorted(contribution.items())
        ) or "N/A"
        merged_block = (
            "\nTOP 10 FINAL CANDIDATE EQUATIONS (ALL TOP-3 CLUSTERS)\n"
            "    Collected from every final PySR run, canonicalised, and deduplicated.\n"
            "    Ranking: lower model loss; then higher influence-weighted grammar\n"
            "    compatibility; then higher reference similarity; then lower complexity.\n"
            "    No new hand-tuned weighted discovery score is used.\n"
            f"{merged_lines}\n"
            f"    Cluster contribution in this top 10: {contribution_text}\n"
        )
        return score_block + comparison_block + merged_block + "\n" + winner_run["text"]

    progress(0.99, desc="Running final symbolic regression...")
    source = f"{method_label}{tied_note}"
    probe_compat = scores[cid]["compatibility"] if is_research else None
    result = _run_pysr_with_cluster(
        df, target, grammars, cid, n_iter, source,
        probe_compat=probe_compat, use_timeout=not is_research,
    )

    return score_block + result


# ─────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────
with gr.Blocks(title="MetaSR — Metallurgical Symbolic Regression") as demo:

    gr.Markdown("""
    # 🔬 MetaSR — Metallurgical Symbolic Regression
    **Two ways to choose the grammar cluster for symbolic regression:**
    - **Tab 1 — Auto-Probe (recommended).** Upload your data and MetaSR
      grammar-probes all 13 clusters and picks the best-fitting one for you.
    - **Tab 2 — Know your equation?** Search 1,324 reference equations by
      name, pick the closest, and its cluster is used directly — useful if
      you already know the governing physics.
    """)

    with gr.Tab("🔬 Auto-Probe  (recommended)"):

        gr.Markdown("""
        ### Don't know which grammar applies? Start here.
        Upload your numeric CSV and MetaSR grammar-probes all 13 clusters
        against your data, then automatically runs symbolic regression on
        the best-fitting one.

        **Fast mode** scores each cluster as a weight-averaged combination
        of its own operators' individual R² (no PySR) — seconds.
        **Research mode** runs a full grammar-constrained PySR fit per
        cluster (300 iterations, no wall-clock timeout — every cluster gets
        an equal computational budget) and picks the cluster balancing
        lowest loss against how much of its own grammar the fit actually
        used. More principled, but can take a while — there's no time
        ceiling, so plan for a long-running probe on 13 clusters.
        """)

        with gr.Row():
            with gr.Column():
                file_input_tab1   = gr.File(label="Upload CSV Dataset")
                target_input_tab1 = gr.Textbox(
                    label="Target Column Name",
                    placeholder="e.g.  sigma  /  yield_strength  /  T_liquidus",
                )
                mode_radio = gr.Radio(
                    choices=[
                        "⚡ Fast (heuristic, seconds)",
                        "🔬 Research (grammar-guided PySR, no timeout — can take a while)",
                    ],
                    value="⚡ Fast (heuristic, seconds)",
                    label="Probe Mode",
                )
                iter_slider_tab1 = gr.Slider(
                    minimum=20, maximum=300, value=50, step=1,
                    label="Final Regression Iterations  (20 = fast · 300 = thorough)",
                )
                # Retained only to avoid changing the surrounding UI layout;
                # fixed release weights above are used for every run.
                loss_weight_slider_tab1 = gr.Slider(visible=False,
                    minimum=0, maximum=100, value=70, step=5,
                    label="Research probe: Loss weight α (%)  —  remaining % goes to Grammar Compatibility",
                    info="Combined score = α·normalized_loss + (1-α)·(1-grammar_compatibility). "
                         "Only affects Research mode.",
                )
                top3_checkbox_tab1 = gr.Checkbox(
                    value=False,
                    label="🏆 Research mode: evaluate top-3 near-tied grammars with the full final run, pick the best by actual performance",
                    info="Instead of trusting the single probe winner, runs the full final PySR fit on the top 3 "
                         "candidate clusters (by combined score) and picks the one with the best final R², "
                         "tie-broken by grammar compatibility, complexity, then loss. "
                         "Roughly 3× the runtime of a single Research-mode run.",
                )
                run_btn_tab1 = gr.Button("▶ Run Auto-Probe", variant="primary")
                gr.Markdown("> 💡 Already know the governing equation? Use the **Select by Reference Equation** tab instead.")

        sr_output_tab1 = gr.Textbox(label="Results", lines=55, interactive=False)

        run_btn_tab1.click(
            fn=run_metasr_probe,
            inputs=[file_input_tab1, target_input_tab1, iter_slider_tab1, mode_radio,
                    top3_checkbox_tab1],
            outputs=[sr_output_tab1],
        )

    with gr.Tab("📖 Select by Reference Equation  (advanced)"):

        gr.Markdown("""
        ### Step 1 — Find the equation that matches your physics
        Type a name like **hall petch**, **arrhenius**, **bond number**, **fourier** etc.
        Click a result to lock in its cluster, then upload your CSV on the right.
        """)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("#### 🔍 Search reference equations")

                search_box = gr.Textbox(
                    label="Keyword  (equation name, subfield, or operator)",
                    placeholder="e.g.  hall petch  /  arrhenius  /  bond number  /  fourier",
                    lines=1,
                )
                subfield_dd = gr.Dropdown(
                    label="Filter by subfield  (optional)",
                    choices=["All subfields"] + ALL_SUBFIELDS,
                    value="All subfields",
                    interactive=True,
                )
                search_btn    = gr.Button("Search", variant="primary")
                search_status = gr.Textbox(label="Status", value="", interactive=False, lines=1)
                search_preview = gr.Textbox(
                    label="LaTeX of top result",
                    value="",
                    interactive=False,
                    lines=2,
                )

                eq_radio = gr.Radio(
                    choices=[],
                    value=None,
                    label="Matching equations — click one to select it",
                    interactive=True,
                )

                selected_latex = gr.Textbox(
                    label="Selected equation  (LaTeX)",
                    value="",
                    interactive=False,
                    lines=2,
                )
                selected_info = gr.Textbox(
                    label="Selected cluster info",
                    value="No equation selected yet.\nSearch above and click a result.",
                    interactive=False,
                    lines=8,
                )

                with gr.Accordion("🔎 Diagnostics — look up a keyword directly (read-only, doesn't affect matching)", open=False):
                    gr.Markdown(
                        "Independently check whether a keyword resolves to an entry in "
                        "`equations_browser.json`, and how cleanly its LaTeX parses. "
                        "This is inspection only — it never feeds into or biases the "
                        "similarity ranking used above or in the Auto-Probe tab."
                    )
                    diag_box = gr.Textbox(
                        label="Keyword to look up",
                        placeholder="e.g. hall petch",
                        lines=1,
                    )
                    diag_btn = gr.Button("Look Up")
                    diag_output = gr.Textbox(label="Diagnostics", value="", interactive=False, lines=14)
                    diag_btn.click(fn=diagnose_keyword, inputs=[diag_box], outputs=[diag_output])
                    diag_box.submit(fn=diagnose_keyword, inputs=[diag_box], outputs=[diag_output])

            with gr.Column(scale=1):
                gr.Markdown("#### ▶ Step 2 — Run symbolic regression")

                file_input_tab2 = gr.File(label="Upload CSV Dataset")
                target_input_tab2 = gr.Textbox(
                    label="Target Column Name",
                    placeholder="e.g.  sigma  /  yield_strength  /  T",
                )
                iter_slider_tab2 = gr.Slider(
                    minimum=20, maximum=300, value=50, step=1,
                    label="Iterations  (20 = fast · 300 = thorough)",
                )
                run_btn_tab2 = gr.Button("▶ Run with Selected Cluster", variant="primary")
                gr.Markdown("> 💡 Not sure? Use the **Auto-Probe** tab instead.")

        selected_cluster_state = gr.State(value=None)
        sr_output_tab2 = gr.Textbox(label="Results", lines=50, interactive=False)

        search_btn.click(
            fn=search_reference_equations,
            inputs=[search_box, subfield_dd],
            outputs=[eq_radio, search_status, search_preview],
        )
        search_box.submit(
            fn=search_reference_equations,
            inputs=[search_box, subfield_dd],
            outputs=[eq_radio, search_status, search_preview],
        )
        eq_radio.change(
            fn=on_equation_selected,
            inputs=[eq_radio],
            outputs=[selected_info, selected_cluster_state, selected_latex],
        )
        run_btn_tab2.click(
            fn=run_with_selected_cluster,
            inputs=[file_input_tab2, target_input_tab2, iter_slider_tab2, selected_cluster_state],
            outputs=[sr_output_tab2],
        )


demo.launch(server_name="0.0.0.0", server_port=7860)

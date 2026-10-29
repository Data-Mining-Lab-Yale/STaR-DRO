"""Ontology-driven Code, Sub-code and Span scoring for structured extraction.

No label inventory is built into this module. The Code and Sub-code inventories
come from the user's ontology JSON ({Code: [Sub-code, ...]}) or, when no ontology
is supplied, from the gold annotations being scored.

Metric definitions (manuscript Supplementary Section S8):
  * Code and Sub-code: micro precision/recall/F1 over per-record label *sets*,
    accumulated across records.
  * Span: relaxed matching. A predicted span matches an unmatched gold span when
    either span's lowercase whitespace-token set contains the other's, or their
    token Jaccard similarity is >= the threshold (0.6 by default). Predictions are
    matched in output order, each gold span at most once.

Label matching normalizes surface variation (Unicode NFKC, case, spacing,
underscores, separators around "/") and, optionally, maps near-miss spellings to
the closest inventory label with conservative fuzzy cutoffs (Codes 95, Sub-codes
92 on a 0-100 scale). The RapidFuzz QRatio scorer is used when installed;
otherwise difflib provides the fallback. The backend is reported with results.
"""
import ast
import html
import json
import re
import unicodedata

try:
    from rapidfuzz import fuzz, process
    _HAS_RAPIDFUZZ = True
except ImportError:  # pragma: no cover - exercised when rapidfuzz is absent
    import difflib
    _HAS_RAPIDFUZZ = False

CODE_FUZZY_CUTOFF = 95
SUBCODE_FUZZY_CUTOFF = 92
UNKNOWN_POLICIES = ("ignore", "count")
_SALVAGE = re.compile(r'\{\s*"Code"\s*:\s*"[^"]*"\s*,\s*"Sub-code"\s*:\s*"[^"]*"\s*,\s*"Span"\s*:\s*"[^"]*"\s*\}')


def backend():
    return "rapidfuzz" if _HAS_RAPIDFUZZ else "difflib"


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------
def parse_output(text):
    """Parse model output that is JSON or nearly JSON into a list of objects.

    Accepts {"results": [...]}, a bare list, Python-literal quoting, HTML
    entities and, as a last resort, salvages complete Code/Sub-code/Span objects
    from otherwise malformed text (for example truncated or fenced output).
    """
    if isinstance(text, dict):
        value = text.get("results")
    elif isinstance(text, list):
        value = text
    elif text is None:
        value = []
    else:
        value = None
        s = str(text).strip()
        for loader in (json.loads, ast.literal_eval):
            try:
                obj = loader(s)
            except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
                continue
            if isinstance(obj, dict) and "results" in obj:
                value = obj["results"]
                break
            if isinstance(obj, list):
                value = obj
                break
        if value is None:
            value = []
            for match in _SALVAGE.findall(html.unescape(s)):
                try:
                    value.append(json.loads(match))
                except ValueError:
                    continue
    return [item for item in (value or []) if isinstance(item, dict)]


# ---------------------------------------------------------------------------
# Label inventory and normalization
# ---------------------------------------------------------------------------
def canonical_label(value):
    """Canonical key used only for label matching (never for spans)."""
    if value is None:
        return ""
    x = unicodedata.normalize("NFKC", str(value)).strip().strip('"').strip("'").strip()
    x = re.sub(r"\s*/\s*", "/", x)
    x = re.sub(r"[\s_]+", " ", x).strip().casefold()
    # Keep letters/digits from any script and "/"; drop spaces and punctuation.
    return re.sub(r"[^\w/]|_", "", x)


def _index(labels, kind):
    index = {}
    for label in labels:
        key = canonical_label(label)
        if not key:
            raise ValueError(f"{kind} label {label!r} has no matchable characters")
        for variant in {key, key.replace("/", "")}:
            if variant in index and index[variant] != label:
                raise ValueError(f"{kind} labels {index[variant]!r} and {label!r} are indistinguishable after "
                                 "normalization; rename one or use exact label matching")
            index[variant] = label
    return index


def ontology_from_annotations(rows):
    """Derive {Code: [Sub-code, ...]} from gold records (used when no ontology is given)."""
    ontology = {}
    for row in rows:
        for ann in row["annotations"]:
            subs = ontology.setdefault(ann["Code"], [])
            if ann["Sub-code"] not in subs:
                subs.append(ann["Sub-code"])
    return ontology


class LabelInventory:
    """Code and Sub-code inventories with normalization and optional fuzzy mapping.

    Sub-code names may repeat under several Codes; the Sub-code metric scores
    Sub-code names, and the joint pair metric scores hierarchical attachment.
    """

    def __init__(self, ontology, fuzzy=True):
        if not isinstance(ontology, dict) or not ontology:
            raise ValueError("ontology must map Code strings to lists of Sub-code strings")
        self.ontology = {code: list(subs) for code, subs in ontology.items()}
        self.codes = list(self.ontology)
        self.subcodes = list(dict.fromkeys(s for subs in self.ontology.values() for s in subs))
        self.fuzzy = bool(fuzzy)
        self._codes = _index(self.codes, "Code")
        self._subcodes = _index(self.subcodes, "Sub-code")
        self._valid_pairs = {(c, s) for c, subs in self.ontology.items() for s in subs}

    def _map(self, raw, index, cutoff):
        key = canonical_label(raw)
        if not key:
            return None
        for variant in (key, key.replace("/", "")):
            if variant in index:
                return index[variant]
        if not self.fuzzy:
            return None
        if _HAS_RAPIDFUZZ:
            match = process.extractOne(key, list(index), scorer=fuzz.QRatio, score_cutoff=cutoff)
            return index[match[0]] if match else None
        best = difflib.get_close_matches(key, list(index), n=1, cutoff=cutoff / 100)
        return index[best[0]] if best else None

    def code(self, raw):
        return self._map(raw, self._codes, CODE_FUZZY_CUTOFF)

    def subcode(self, raw):
        return self._map(raw, self._subcodes, SUBCODE_FUZZY_CUTOFF)

    def valid_pair(self, code, subcode):
        return (code, subcode) in self._valid_pairs


# ---------------------------------------------------------------------------
# Span matching
# ---------------------------------------------------------------------------
def _tokens(text):
    return set(text.lower().split())


def span_match(predicted, gold, threshold=0.6):
    p, g = _tokens(predicted), _tokens(gold)
    if not p or not g:  # an empty span supports nothing
        return False
    return p <= g or g <= p or len(p & g) / len(p | g) >= threshold


def _prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def score(pairs, inventory, span_threshold=0.6, unknown_labels="ignore"):
    """Score (gold_annotations, predicted_output) pairs.

    gold_annotations: list of {Code, Sub-code, Span} with labels in the inventory.
    predicted_output: raw generated text, a parsed list, or a {"results": [...]} dict.
    unknown_labels: "ignore" drops predicted labels that match no inventory label
      (the convention used for the manuscript); "count" scores each distinct
      unmatched predicted label as a false positive.
    """
    if unknown_labels not in UNKNOWN_POLICIES:
        raise ValueError(f"unknown_labels must be one of {UNKNOWN_POLICIES}")
    if not 0 < span_threshold <= 1:
        raise ValueError("span threshold must be in (0, 1]")
    counts = {key: [0, 0, 0] for key in ("code", "subcode", "code_subcode", "span")}
    unknown = {"code": 0, "subcode": 0}
    invalid_pairs = 0
    for gold, predicted in pairs:
        predicted = parse_output(predicted)
        g_code, g_sub, g_pair = set(), set(), set()
        for ann in gold:
            code, sub = inventory.code(ann["Code"]), inventory.subcode(ann["Sub-code"])
            if code is None or sub is None:
                raise ValueError(f"gold label outside the scoring inventory: {ann['Code']!r}/{ann['Sub-code']!r}")
            g_code.add(code); g_sub.add(sub); g_pair.add((code, sub))
        p_code, p_sub, p_pair = set(), set(), set()
        for ann in predicted:
            raw_code, raw_sub = ann.get("Code"), ann.get("Sub-code")
            code, sub = inventory.code(raw_code), inventory.subcode(raw_sub)
            if code is None:
                unknown["code"] += 1
            if sub is None:
                unknown["subcode"] += 1
            count = unknown_labels == "count"
            if code is not None:
                p_code.add(code)
            elif count:
                p_code.add(("<unknown>", canonical_label(raw_code)))
            if sub is not None:
                p_sub.add(sub)
            elif count:
                p_sub.add(("<unknown>", canonical_label(raw_sub)))
            if code is not None and sub is not None:
                p_pair.add((code, sub))
                invalid_pairs += not inventory.valid_pair(code, sub)
            elif count:
                p_pair.add(("<unknown>", canonical_label(raw_code), canonical_label(raw_sub)))
        for key, g, p in (("code", g_code, p_code), ("subcode", g_sub, p_sub), ("code_subcode", g_pair, p_pair)):
            for j, value in enumerate((len(g & p), len(p - g), len(g - p))):
                counts[key][j] += value
        gold_spans = [a["Span"] for a in gold if isinstance(a.get("Span"), str)]
        pred_spans = [a.get("Span") for a in predicted if isinstance(a.get("Span"), str)]
        used, matched = set(), 0
        for span in pred_spans:
            for j, target in enumerate(gold_spans):
                if j not in used and span_match(span, target, span_threshold):
                    used.add(j)
                    matched += 1
                    break
        for j, value in enumerate((matched, len(pred_spans) - matched, len(gold_spans) - matched)):
            counts["span"][j] += value
    metrics = {key: _prf(*value) for key, value in counts.items()}
    return metrics, {"unknown_predicted_code_labels": unknown["code"],
                     "unknown_predicted_subcode_labels": unknown["subcode"],
                     "predicted_pairs_outside_hierarchy": invalid_pairs}

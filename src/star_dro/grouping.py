"""Group construction independent of loss attribution (main §4.4; supplement S4–S5)."""
from collections import Counter
from dataclasses import dataclass
import json

GROUPINGS = ("code", "subcode", "code_subcode", "num_annotations", "code_subcode_num_annotations")
ALIASES = {"num_span": "num_annotations", "code_subcode_num_span": "code_subcode_num_annotations"}


def canonical_groupby(value):
    value = ALIASES.get(value, value)
    if value not in GROUPINGS:
        raise ValueError(f"groupby must be one of {', '.join(GROUPINGS)}")
    return value


def annotation_keys(annotations, groupby="code"):
    """One key per annotation; count means annotation objects, not unique spans.

    JSON tuples prevent collisions when a label contains a delimiter. Sub-code
    grouping deliberately merges identical child names across different parents.
    Joint grouping preserves parent identity. No Cartesian products are formed.
    """
    groupby = canonical_groupby(groupby)
    n = len(annotations)
    if groupby == "num_annotations":
        return [f"NA_{n}"] * n
    if groupby == "code":
        return [a["Code"] for a in annotations]
    if groupby == "subcode":
        return [a["Sub-code"] for a in annotations]
    return [json.dumps([a["Code"], a["Sub-code"]] + ([n] if groupby.endswith("num_annotations") else []),
                       ensure_ascii=False, separators=(",", ":")) for a in annotations]


@dataclass(frozen=True)
class GroupInventory:
    """Fit on training only and reuse unchanged for validation and resumption."""
    groupby: str
    names: tuple
    rare_keys: tuple = ()
    min_count: int = 1

    @classmethod
    def fit(cls, records, groupby="code", min_count=1):
        groupby = canonical_groupby(groupby)
        if isinstance(min_count, bool) or not isinstance(min_count, int) or min_count < 1:
            raise ValueError("grouping.min_count must be a positive integer")
        # Count sample memberships, as in the reference notebook. Within-record
        # repeats contribute once to the threshold, then separately to ann CE.
        counts = Counter()
        for row in records:
            keys = annotation_keys(row["annotations"], groupby)
            if not keys and groupby == "num_annotations":
                keys = ["NA_0"]
            counts.update(set(keys))
        rare = tuple(sorted(k for k, n in counts.items() if n < min_count))
        names = set(counts) - set(rare)
        if rare:
            if "__OTHER__" in counts:
                raise ValueError("__OTHER__ is reserved when rare groups are pooled")
            names.add("__OTHER__")
        if not names:
            raise ValueError("training has no groups under the chosen grouping")
        return cls(groupby, tuple(sorted(names)), rare, min_count)

    def encode(self, annotations):
        lookup = {name: i for i, name in enumerate(self.names)}
        keys = annotation_keys(annotations, self.groupby)

        def index(key):
            if key in self.rare_keys:
                key = "__OTHER__"
            if key not in lookup:
                # Pool unseen groups only if training explicitly created a pool.
                if self.rare_keys:
                    key = "__OTHER__"
                else:
                    raise ValueError(f"group absent from the training inventory: {key}; "
                                     "review grouping/splits or fit an explicit rare-group pool")
            return lookup[key]

        ids = [index(key) for key in keys]
        members = sorted(set(ids))
        if not keys and self.groupby == "num_annotations":
            members = [index("NA_0")]
        return ids, members

    def to_dict(self):
        return {"groupby": self.groupby, "names": list(self.names),
                "rare_keys": list(self.rare_keys), "min_count": self.min_count}

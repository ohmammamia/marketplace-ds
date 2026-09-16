"""
Reusable data-validation layer (RAP-style). Every pipeline runs its inputs
through `validate()` before any modelling. Failures are explicit objects,
never silent coercions; the caller decides whether a failure is blocking.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import pandas as pd


@dataclass
class Check:
    name: str
    passed: bool
    severity: str          # "error" | "warning"
    detail: str = ""


@dataclass
class ValidationReport:
    dataset: str
    checks: list[Check] = field(default_factory=list)

    def add(self, name, passed, severity="error", detail=""):
        self.checks.append(Check(name, bool(passed), severity, detail))

    @property
    def errors(self):
        return [c for c in self.checks if not c.passed and c.severity == "error"]

    @property
    def warnings(self):
        return [c for c in self.checks if not c.passed and c.severity == "warning"]

    def raise_if_failed(self):
        if self.errors:
            msg = "\n".join(f"  - {c.name}: {c.detail}" for c in self.errors)
            raise ValueError(f"Validation failed for '{self.dataset}':\n{msg}")

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([c.__dict__ for c in self.checks])


@dataclass
class Schema:
    required: dict[str, str]                       # column -> pandas dtype kind ('i','f','O','M','b')
    key: str | None = None
    ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    categories: dict[str, set] = field(default_factory=dict)
    max_missing: dict[str, float] = field(default_factory=dict)  # column -> allowed fraction


KIND = {"i": "integer", "f": "float", "O": "object", "M": "datetime", "b": "bool"}


def validate(df: pd.DataFrame, schema: Schema, name: str,
             references: dict[str, pd.Series] | None = None) -> ValidationReport:
    rep = ValidationReport(name)
    # 1. schema / types
    for col, kind in schema.required.items():
        if col not in df.columns:
            rep.add(f"column_present:{col}", False, detail="missing column")
            continue
        actual = df[col].dtype.kind
        ok = actual == kind or (kind == "f" and actual == "i") or (kind == "i" and actual == "f" and df[col].dropna().mod(1).eq(0).all())
        rep.add(f"dtype:{col}", ok, "warning", detail=f"expected {KIND[kind]}, got {df[col].dtype}")
    # 2. key uniqueness
    if schema.key and schema.key in df:
        n_dup = int(df[schema.key].duplicated().sum())
        rep.add("unique_key", n_dup == 0, detail=f"{n_dup} duplicated {schema.key}")
    # 3. missingness
    for col, allowed in schema.max_missing.items():
        if col in df:
            frac = float(df[col].isna().mean())
            rep.add(f"missing:{col}", frac <= allowed, "warning", detail=f"{frac:.1%} missing (allowed {allowed:.0%})")
    # 4. ranges
    for col, (lo, hi) in schema.ranges.items():
        if col in df:
            s = df[col].dropna()
            bad = int(((s < lo) | (s > hi)).sum())
            rep.add(f"range:{col}", bad == 0, detail=f"{bad} values outside [{lo}, {hi}]")
    # 5. categories
    for col, allowed in schema.categories.items():
        if col in df:
            bad = set(df[col].dropna().unique()) - allowed
            rep.add(f"categories:{col}", not bad, "warning", detail=f"unexpected: {sorted(map(str, bad))[:5]}")
    # 6. referential integrity
    for col, ref in (references or {}).items():
        if col in df:
            orphans = int((~df[col].isin(set(ref))).sum())
            rep.add(f"referential:{col}", orphans == 0, detail=f"{orphans} orphan rows")
    return rep

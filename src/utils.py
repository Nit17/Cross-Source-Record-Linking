from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Literal

import pandas as pd
from pydantic import BaseModel, Field, ValidationError

# --- Cleaning helpers ---

def normalize_id(text: object) -> str:
    if text is None:
        return ""
    s = str(text)
    return re.sub(r"\D", "", s)


def clean_email(email: object) -> str:
    if email is None:
        return ""
    return str(email).strip().replace(" ", "").lower()


def extract_domain(email: object) -> str:
    e = clean_email(email)
    if "@" in e:
        return e.split("@", 1)[1]
    return ""


def to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


# --- Pydantic Configs ---

class MappingA(BaseModel):
    invoice_id: str
    customer_email: str
    invoice_date: str
    total_amount: str
    po_number: Optional[str] = None
    tax_amount: Optional[str] = None
    currency: Optional[str] = None


class MappingB(BaseModel):
    ref_code: str
    email: str
    doc_date: str
    grand_total: str
    purchase_order: Optional[str] = None
    tax_amount: Optional[str] = None
    currency: Optional[str] = None


class RulesConfig(BaseModel):
    amount_tolerance: float = Field(0.001, ge=0)
    date_tolerance: int = Field(2, ge=0)
    name_similarity: float = Field(0.85, ge=0, le=1)
    domain_similarity: float = Field(0.9, ge=0, le=1)
    domain_similarity_min: Optional[float] = Field(default=None, ge=0, le=1)
    # weights for scoring (will be normalized)
    weight_amount: float = Field(0.6, ge=0, le=1)
    weight_date: float = Field(0.3, ge=0, le=1)
    weight_name: float = Field(0.1, ge=0, le=1)
    weight_domain: float = Field(0.0, ge=0, le=1)

    def to_weights(self):
        from .rules import ScoreWeights
        total = max(1e-9, self.weight_amount + self.weight_date + self.weight_name + self.weight_domain)
        return ScoreWeights(
            amount=self.weight_amount / total,
            date=self.weight_date / total,
            name=self.weight_name / total,
            domain=self.weight_domain / total,
        )


class AppPreset(BaseModel):
    mapping_a: MappingA
    mapping_b: MappingB
    rules: RulesConfig
    rule_order: List[str] = Field(default_factory=lambda: [
        "exact_id",
        "canonical_id",
        "composite",
        "fuzzy",
    ])
    patterns: Optional[List[Dict]] = None  # persisted raw pattern dicts


# --- Schema autodetect ---

CANDIDATES_A = {
    "invoice_id": ["invoice_id", "invoice", "inv", "inv_id", "id"],
    "customer_email": ["customer_email", "email", "customeremail", "client_email"],
    "invoice_date": ["invoice_date", "date", "doc_date", "invoice_dt"],
    "total_amount": ["total_amount", "amount", "grand_total", "total", "amt"],
    "po_number": ["po_number", "po", "purchase_order", "po_no"],
    "tax_amount": ["tax_amount", "tax", "vat", "gst", "sales_tax", "taxamt", "tax_amt"],
    "currency": ["currency", "curr", "ccy", "iso_currency", "currency_code", "currencycode"],
}

CANDIDATES_B = {
    "ref_code": ["ref_code", "reference", "ref", "invoice_id", "id"],
    "email": ["email", "customer_email", "client_email"],
    "doc_date": ["doc_date", "date", "invoice_date"],
    "grand_total": ["grand_total", "amount", "total", "amt"],
    "purchase_order": ["purchase_order", "po", "po_number", "po_no"],
    "tax_amount": ["tax_amount", "tax", "vat", "gst", "sales_tax", "taxamt", "tax_amt"],
    "currency": ["currency", "curr", "ccy", "iso_currency", "currency_code", "currencycode"],
}


def _best_match(columns: List[str], candidates: List[str]) -> Optional[str]:
    cols = [c.lower() for c in columns]
    for cand in candidates:
        if cand in cols:
            # exact first
            return columns[cols.index(cand)]
    # fallback: simple contains
    for cand in candidates:
        for i, c in enumerate(cols):
            if cand in c:
                return columns[i]
    return None


def autodetect_a(df: pd.DataFrame) -> Optional[MappingA]:
    cols = list(df.columns)
    mapping: Dict[str, Optional[str]] = {}
    for key, cand in CANDIDATES_A.items():
        mapping[key] = _best_match(cols, cand)
    if None in (mapping["invoice_id"], mapping["customer_email"], mapping["invoice_date"], mapping["total_amount"]):
        return None
    return MappingA(**{k: v for k, v in mapping.items() if v is not None})


def autodetect_b(df: pd.DataFrame) -> Optional[MappingB]:
    cols = list(df.columns)
    mapping: Dict[str, Optional[str]] = {}
    for key, cand in CANDIDATES_B.items():
        mapping[key] = _best_match(cols, cand)
    if None in (mapping["ref_code"], mapping["email"], mapping["doc_date"], mapping["grand_total"]):
        return None
    return MappingB(**{k: v for k, v in mapping.items() if v is not None})


# --- Simple diffs ---

def diff_fields(a: str, b: str) -> List[Tuple[int, int]]:
    """Return simplistic difference spans between two strings for UI highlight.
    This can be replaced with a proper diff algorithm later.
    """
    spans: List[Tuple[int, int]] = []
    if a == b:
        return spans
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    j = 0
    while j < n and a[len(a)-1-j] == b[len(b)-1-j]:
        j += 1
    if i < len(a) - j:
        spans.append((i, len(a) - j))
    return spans


# --- Pattern adoption ---

class Pattern(BaseModel):
    kind: Literal["strip_prefix", "strip_suffix", "regex_replace", "extract_digits", "extract_token"]
    value: Optional[str] = None         # for prefix/suffix or regex pattern
    replacement: Optional[str] = None   # for regex replacement
    sep: Optional[str] = None           # for extract_token
    index: Optional[int] = None         # token index
    apply_to: Literal["A", "B", "both"] = "B"
    enabled: bool = True


def apply_patterns(text: object, patterns: List[Pattern]) -> str:
    s = "" if text is None else str(text)
    for p in patterns or []:
        if hasattr(p, 'enabled') and not p.enabled:
            continue
        if p.apply_to not in ("both",):
            # Application side will be enforced by caller choosing patterns list for that side
            pass
        if p.kind == "strip_prefix" and p.value and s.startswith(p.value):
            s = s[len(p.value):]
        elif p.kind == "strip_suffix" and p.value and s.endswith(p.value):
            s = s[: -len(p.value)]
        elif p.kind == "regex_replace" and p.value is not None:
            try:
                s = re.sub(p.value, p.replacement or "", s)
            except re.error:
                continue
        elif p.kind == "extract_digits":
            s = re.sub(r"\D", "", s)
        elif p.kind == "extract_token" and p.sep is not None and p.index is not None:
            parts = s.split(p.sep)
            if 0 <= p.index < len(parts):
                s = parts[p.index]
    return s


def suggest_pattern(a_id: object, b_ref: object) -> Optional[Pattern]:
    """Heuristic: suggest a simple transformation to turn B ref into A id."""
    a_str = "" if a_id is None else str(a_id)
    b_str = "" if b_ref is None else str(b_ref)
    if not a_str or not b_str:
        return None
    # exact digits match -> extract_digits
    if re.sub(r"\D", "", b_str) == re.sub(r"\D", "", a_str) and re.sub(r"\D", "", b_str) != b_str:
        return Pattern(kind="extract_digits", apply_to="B")
    # suffix removal
    if b_str.endswith(a_str):
        prefix = b_str[:-len(a_str)]
        if prefix:
            return Pattern(kind="strip_prefix", value=prefix, apply_to="B")
    # prefix removal
    if b_str.startswith(a_str):
        suffix = b_str[len(a_str):]
        if suffix:
            return Pattern(kind="strip_suffix", value=suffix, apply_to="B")
    # substring token by delimiter
    for sep in ["-", "_", "/", " "]:
        parts = b_str.split(sep)
        for idx, tok in enumerate(parts):
            if tok == a_str:
                return Pattern(kind="extract_token", sep=sep, index=idx, apply_to="B")
    # regex fallback: capture digits
    return Pattern(kind="regex_replace", value=r".*?(\d+).*", replacement=r"\1", apply_to="B")

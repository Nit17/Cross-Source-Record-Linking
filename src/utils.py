from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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


class MappingB(BaseModel):
    ref_code: str
    email: str
    doc_date: str
    grand_total: str
    purchase_order: Optional[str] = None


class RulesConfig(BaseModel):
    amount_tolerance: float = Field(0.001, ge=0)
    date_tolerance: int = Field(2, ge=0)
    name_similarity: float = Field(0.85, ge=0, le=1)
    domain_similarity: float = Field(0.9, ge=0, le=1)


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


# --- Schema autodetect ---

CANDIDATES_A = {
    "invoice_id": ["invoice_id", "invoice", "inv", "inv_id", "id"],
    "customer_email": ["customer_email", "email", "customeremail", "client_email"],
    "invoice_date": ["invoice_date", "date", "doc_date", "invoice_dt"],
    "total_amount": ["total_amount", "amount", "grand_total", "total", "amt"],
    "po_number": ["po_number", "po", "purchase_order", "po_no"],
}

CANDIDATES_B = {
    "ref_code": ["ref_code", "reference", "ref", "invoice_id", "id"],
    "email": ["email", "customer_email", "client_email"],
    "doc_date": ["doc_date", "date", "invoice_date"],
    "grand_total": ["grand_total", "amount", "total", "amt"],
    "purchase_order": ["purchase_order", "po", "po_number", "po_no"],
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

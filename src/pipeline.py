from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import pandas as pd

from .utils import MappingA, MappingB, RulesConfig, normalize_id, clean_email, to_datetime, to_numeric
from .rules import ScoreWeights, name_similarity


@dataclass
class MatchResult:
    a_row: dict
    b_row: dict
    tier: str
    rationale: str
    score: float | None = None


ProgressFn = Callable[[float, str], None]


def run_pipeline(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    map_a: MappingA,
    map_b: MappingB,
    rules: RulesConfig,
    progress: ProgressFn | None = None,
    sample_n: int | None = None,
) -> Tuple[List[MatchResult], List[MatchResult], pd.DataFrame, pd.DataFrame]:
    """Run tiered matching with optional progress callback and sampling."""
    if sample_n:
        df_a = df_a.head(sample_n)
        df_b = df_b.head(sample_n)

    def step(p: float, msg: str):
        if progress:
            progress(p, msg)

    step(0.05, "Preprocessing")
    a = df_a.copy().reset_index().rename(columns={"index": "_a_idx"})
    b = df_b.copy().reset_index().rename(columns={"index": "_b_idx"})

    a["canonical_id"] = a[map_a.invoice_id].astype(str).apply(normalize_id)
    b["canonical_id"] = b[map_b.ref_code].astype(str).apply(normalize_id)
    a["clean_email"] = a[map_a.customer_email].astype(str).apply(clean_email)
    b["clean_email"] = b[map_b.email].astype(str).apply(clean_email)
    a["date"] = to_datetime(a[map_a.invoice_date])
    b["date"] = to_datetime(b[map_b.doc_date])
    a["amount"] = to_numeric(a[map_a.total_amount])
    b["amount"] = to_numeric(b[map_b.grand_total])

    matched: List[MatchResult] = []
    suspects: List[MatchResult] = []
    used_a: set[int] = set()
    used_b: set[int] = set()

    # Tier 1: exact id
    step(0.2, "Tier 1: exact ids")
    exact = a.merge(b, left_on=map_a.invoice_id, right_on=map_b.ref_code, how="inner", suffixes=("_a", "_b"))
    if not exact.empty:
        for _, r in exact.drop_duplicates(subset=["_a_idx"]).iterrows():
            if r["_a_idx"] in used_a or r["_b_idx"] in used_b:
                continue
            matched.append(MatchResult(r.to_dict(), r.to_dict(), "exact_id", "Exact match on primary identifier.", 1.0))
            used_a.add(r["_a_idx"]) 
            used_b.add(r["_b_idx"]) 

    # Tier 2: canonical id
    step(0.4, "Tier 2: canonical ids")
    rem_a = a[~a["_a_idx"].isin(used_a)]
    rem_b = b[~b["_b_idx"].isin(used_b)]
    canon = rem_a.merge(rem_b, on="canonical_id", how="inner", suffixes=("_a", "_b"))
    if not canon.empty:
        for _, r in canon.drop_duplicates(subset=["_a_idx"]).iterrows():
            if r["_a_idx"] in used_a or r["_b_idx"] in used_b:
                continue
            matched.append(MatchResult(r.to_dict(), r.to_dict(), "canonical_id", "Match on normalized numeric identifier.", 0.99))
            used_a.add(r["_a_idx"]) 
            used_b.add(r["_b_idx"]) 

    # Tier 3: composite with tie-breaker by score
    step(0.65, "Tier 3: composite scoring")
    rem_a = a[~a["_a_idx"].isin(used_a)]
    rem_b = b[~b["_b_idx"].isin(used_b)]
    if not rem_a.empty and not rem_b.empty:
        cand = rem_a.merge(rem_b, on="clean_email", how="inner", suffixes=("_a", "_b"))
        if not cand.empty:
            cand["amount_diff_pct"] = (cand["amount_a"] - cand["amount_b"]).abs() / cand["amount_a"].replace(0, pd.NA)
            cand["date_diff_days"] = (cand["date_a"] - cand["date_b"]).abs().dt.days

            elig = cand[(cand["amount_diff_pct"] <= rules.amount_tolerance) & (cand["date_diff_days"] <= rules.date_tolerance)]
            if not elig.empty:
                # simple greedy by score
                weights = ScoreWeights()
                elig["score"] = (1 - elig["amount_diff_pct"]) * weights.amount + (1 - (elig["date_diff_days"] / 30.0).clip(0, 1)) * weights.date
                elig = elig.sort_values("score", ascending=False)
                seen_a: set[int] = set()
                seen_b: set[int] = set()
                for _, r in elig.iterrows():
                    if r["_a_idx"] in used_a or r["_b_idx"] in used_b:
                        continue
                    if r["_a_idx"] in seen_a or r["_b_idx"] in seen_b:
                        suspects.append(MatchResult(rem_a.loc[rem_a["_a_idx"] == r["_a_idx"]].iloc[0].to_dict(), rem_b.loc[rem_b["_b_idx"] == r["_b_idx"]].iloc[0].to_dict(), "composite", "Competing candidate (tie)", None))
                        continue
                    ra = rem_a.loc[rem_a["_a_idx"] == r["_a_idx"]].iloc[0]
                    rb = rem_b.loc[rem_b["_b_idx"] == r["_b_idx"]].iloc[0]
                    matched.append(MatchResult(ra.to_dict(), rb.to_dict(), "composite", f"Email match; date ≤ {rules.date_tolerance} days and amount ≤ {rules.amount_tolerance*100:.2f}%.", float(r["score"])) )
                    used_a.add(int(r["_a_idx"]))
                    used_b.add(int(r["_b_idx"]))
                    seen_a.add(int(r["_a_idx"]))
                    seen_b.add(int(r["_b_idx"]))

            # near-misses
            near = cand[((cand["amount_diff_pct"] <= rules.amount_tolerance) & (cand["date_diff_days"] > rules.date_tolerance)) | ((cand["amount_diff_pct"] > rules.amount_tolerance) & (cand["date_diff_days"] <= rules.date_tolerance))]
            for _, r in near.iterrows():
                ra = rem_a.loc[rem_a["_a_idx"] == r["_a_idx"]].iloc[0]
                rb = rem_b.loc[rem_b["_b_idx"] == r["_b_idx"]].iloc[0]
                suspects.append(MatchResult(ra.to_dict(), rb.to_dict(), "composite", "Email matches; one of date/amount outside tolerance.", None))

    # Tier 4: fuzzy name/domain backup
    step(0.85, "Tier 4: fuzzy (name/domain)")
    rem_a = a[~a["_a_idx"].isin(used_a)]
    rem_b = b[~b["_b_idx"].isin(used_b)]
    # Optional: look for likely name columns
    name_cols_a = [c for c in a.columns if "name" in c.lower() or "customer" in c.lower()]
    name_cols_b = [c for c in b.columns if "name" in c.lower() or "client" in c.lower()]
    a_name_col = name_cols_a[0] if name_cols_a else None
    b_name_col = name_cols_b[0] if name_cols_b else None

    if a_name_col and b_name_col and not rem_a.empty and not rem_b.empty:
        cand = rem_a.assign(_name_a=rem_a[a_name_col]).merge(
            rem_b.assign(_name_b=rem_b[b_name_col]), how="cross"
        )
        if not cand.empty:
            cand["name_sim"] = cand.apply(lambda r: name_similarity(r["_name_a"], r["_name_b"]), axis=1)
            cand = cand[cand["name_sim"] >= rules.name_similarity]
            cand = cand.sort_values("name_sim", ascending=False)
            seen_a: set[int] = set()
            seen_b: set[int] = set()
            for _, r in cand.iterrows():
                if r["_a_idx"] in used_a or r["_b_idx"] in used_b:
                    continue
                if r["_a_idx"] in seen_a or r["_b_idx"] in seen_b:
                    suspects.append(MatchResult(rem_a.loc[rem_a["_a_idx"] == r["_a_idx"]].iloc[0].to_dict(), rem_b.loc[rem_b["_b_idx"] == r["_b_idx"]].iloc[0].to_dict(), "fuzzy", "Competing candidate (tie)", float(r["name_sim"])) )
                    continue
                ra = rem_a.loc[rem_a["_a_idx"] == r["_a_idx"]].iloc[0]
                rb = rem_b.loc[rem_b["_b_idx"] == r["_b_idx"]].iloc[0]
                matched.append(MatchResult(ra.to_dict(), rb.to_dict(), "fuzzy", "Fuzzy name/domain similarity.", float(r["name_sim"])) )
                used_a.add(int(r["_a_idx"]))
                used_b.add(int(r["_b_idx"]))
                seen_a.add(int(r["_a_idx"]))
                seen_b.add(int(r["_b_idx"]))

    step(1.0, "Done")
    return matched, suspects, a[~a["_a_idx"].isin(used_a)], b[~b["_b_idx"].isin(used_b)]

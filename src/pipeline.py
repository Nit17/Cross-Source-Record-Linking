from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Sequence

import pandas as pd

from .utils import MappingA, MappingB, RulesConfig, normalize_id, clean_email, to_datetime, to_numeric, Pattern, apply_patterns
from .rules import ScoreWeights, name_similarity, email_domain_similarity, composite_score


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
    rule_order: Sequence[str] | None = None,
    patterns_a: List[Pattern] | None = None,
    patterns_b: List[Pattern] | None = None,
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

    # Apply adopted patterns to id/ref columns if present
    a_id_series = a[map_a.invoice_id].astype(str)
    b_ref_series = b[map_b.ref_code].astype(str)
    if patterns_a:
        a_id_series = a_id_series.apply(lambda x: apply_patterns(x, patterns_a))
    if patterns_b:
        b_ref_series = b_ref_series.apply(lambda x: apply_patterns(x, patterns_b))
    # store transformed ids for use in exact tier
    a["_id_transformed"] = a_id_series
    b["_ref_transformed"] = b_ref_series

    a["canonical_id"] = a_id_series.apply(normalize_id)
    b["canonical_id"] = b_ref_series.apply(normalize_id)
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

    # Define tiers and dynamic order
    tiers: Dict[str, Callable[[], None]] = {}

    def tier_exact():
        step(0.2, "Tier: exact ids")
        exact = a.merge(b, left_on="_id_transformed", right_on="_ref_transformed", how="inner", suffixes=("_a", "_b"))
        if not exact.empty:
            for _, r in exact.drop_duplicates(subset=["_a_idx"]).iterrows():
                if r["_a_idx"] in used_a or r["_b_idx"] in used_b:
                    continue
                matched.append(MatchResult(r.to_dict(), r.to_dict(), "exact_id", "Exact match on primary identifier.", 1.0))
                used_a.add(r["_a_idx"]) 
                used_b.add(r["_b_idx"]) 

    tiers["exact_id"] = tier_exact

    def tier_canonical():
        step(0.4, "Tier: canonical ids")
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

    tiers["canonical_id"] = tier_canonical

    def tier_composite():
        step(0.65, "Tier: composite scoring")
        rem_a = a[~a["_a_idx"].isin(used_a)]
        rem_b = b[~b["_b_idx"].isin(used_b)]
        if rem_a.empty or rem_b.empty:
            return
        cand = rem_a.merge(rem_b, on="clean_email", how="inner", suffixes=("_a", "_b"))
        if cand.empty:
            return
        cand["amount_diff_pct"] = (cand["amount_a"] - cand["amount_b"]).abs() / cand["amount_a"].replace(0, pd.NA)
        cand["date_diff_days"] = (cand["date_a"] - cand["date_b"]).abs().dt.days
        # emails are equal in this join, so domain similarity is perfect
        cand["domain_sim"] = 1.0
        cand["name_sim"] = 0.0  # placeholder if name columns are unknown here

        elig = cand[(cand["amount_diff_pct"] <= rules.amount_tolerance) & (cand["date_diff_days"] <= rules.date_tolerance)]
        if not elig.empty:
            weights = rules.to_weights() if hasattr(rules, 'to_weights') else ScoreWeights()
            elig["score"] = elig.apply(lambda r: composite_score(float(r["amount_diff_pct"] or 0), float(r["date_diff_days"] or 0), float(r["name_sim"] or 0), float(r["domain_sim"] or 0), weights), axis=1)
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

        near = cand[((cand["amount_diff_pct"] <= rules.amount_tolerance) & (cand["date_diff_days"] > rules.date_tolerance)) | ((cand["amount_diff_pct"] > rules.amount_tolerance) & (cand["date_diff_days"] <= rules.date_tolerance))]
        for _, r in near.iterrows():
            ra = rem_a.loc[rem_a["_a_idx"] == r["_a_idx"]].iloc[0]
            rb = rem_b.loc[rem_b["_b_idx"] == r["_b_idx"]].iloc[0]
            suspects.append(MatchResult(ra.to_dict(), rb.to_dict(), "composite", "Email matches; one of date/amount outside tolerance.", None))

    tiers["composite"] = tier_composite

    def tier_fuzzy():
        step(0.85, "Tier: fuzzy (name/domain)")
        rem_a = a[~a["_a_idx"].isin(used_a)]
        rem_b = b[~b["_b_idx"].isin(used_b)]
        name_cols_a = [c for c in a.columns if "name" in c.lower() or "customer" in c.lower()]
        name_cols_b = [c for c in b.columns if "name" in c.lower() or "client" in c.lower()]
        a_name_col = name_cols_a[0] if name_cols_a else None
        b_name_col = name_cols_b[0] if name_cols_b else None
        if not (a_name_col and b_name_col and not rem_a.empty and not rem_b.empty):
            return
        cand = rem_a.assign(_name_a=rem_a[a_name_col]).merge(
            rem_b.assign(_name_b=rem_b[b_name_col]), how="cross"
        )
        if cand.empty:
            return
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

    tiers["fuzzy"] = tier_fuzzy

    # Execute tiers according to order
    order: Sequence[str] = rule_order or ["exact_id", "canonical_id", "composite", "fuzzy"]
    for key in order:
        if key in tiers:
            tiers[key]()
    step(1.0, "Done")
    return matched, suspects, a[~a["_a_idx"].isin(used_a)], b[~b["_b_idx"].isin(used_b)]

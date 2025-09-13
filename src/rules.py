from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd
from rapidfuzz import fuzz

from .utils import extract_domain


@dataclass
class ScoreWeights:
    amount: float = 0.6
    date: float = 0.3
    name: float = 0.08
    domain: float = 0.02


def email_domain_similarity(a_email: str, b_email: str) -> float:
    ad = extract_domain(a_email)
    bd = extract_domain(b_email)
    if not ad or not bd:
        return 0.0
    return fuzz.partial_ratio(ad, bd) / 100.0


def name_similarity(a_name: str, b_name: str) -> float:
    if not a_name or not b_name:
        return 0.0
    return fuzz.token_set_ratio(str(a_name), str(b_name)) / 100.0


def composite_score(amount_diff_pct: float, date_diff_days: float, name_sim: float, domain_sim: float, weights: ScoreWeights) -> float:
    return (
        (1 - min(1.0, amount_diff_pct)) * weights.amount
        + (1 - min(1.0, date_diff_days / 30.0)) * weights.date
    + name_sim * weights.name
    + domain_sim * weights.domain
    )

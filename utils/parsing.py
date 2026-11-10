import re
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd


def parse_score(value: Any) -> tuple[float | None, bool, str | None]:
    """Parse a UPDRS score cell into (score, dual_attempt_flag, raw_value)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None, False, None

    if isinstance(value, datetime):
        return None, False, value.isoformat()

    if isinstance(value, (int, np.integer)):
        score = float(value)
        if 0 <= score <= 4:
            return score, False, str(int(score))
        return None, False, str(value)

    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return None, False, None
        if 0 <= value <= 4 and float(value).is_integer():
            return float(value), False, str(int(value))
        return None, False, str(value)

    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None, False, None

    # Dual-attempt patterns: "1/2(done twice)", "3/2 (done twice)", "3(1st), 2(2nd)"
    dual_attempt = bool(
        re.search(r"done twice|1st|2nd|/", text, flags=re.IGNORECASE)
        or "," in text
    )

    numbers = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", text)]
    if not numbers:
        return None, dual_attempt, text

    valid = [n for n in numbers if 0 <= n <= 4]
    if not valid:
        return None, dual_attempt, text

    return float(np.mean(valid)), dual_attempt, text


# Backward-compatible import path; see utils.id_mapping for implementation.
from utils.id_mapping import build_id_mapping  # noqa: E402,F401

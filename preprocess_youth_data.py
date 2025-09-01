"""
Preprocessing utility for the 20th Korea Youth Risk Behavior Survey (2024).

This module exposes a single function ``preprocess_youth_data`` which takes a
pandas ``DataFrame`` corresponding to the 2024 Youth Risk Behavior Survey raw
data and returns a cleaned and numerically encoded ``DataFrame`` ready for
modelling.  The transformation pipeline encapsulates several domain‐specific
rules:

1. **Handling skip patterns** – survey logic sometimes directs respondents past
   follow‑up items, leaving structural missing values in the raw data.  All
   missing values are replaced with ``-1`` to distinguish "not asked" from
   other valid responses.
2. **Multi‑select questions** – questions allowing "select all that apply"
   responses are typically stored as a set of binary columns sharing the same
   prefix and differing only by a numeric suffix.  Any non‑zero value in these
   columns indicates the option was selected; unselected options should be
   coded as ``0``.  To enforce this binary semantics the preprocessor
   coerces any value greater than ``0`` to ``1``.
3. **Continuous measurements** – when both height (``HT``) and weight
   (``WT``) variables are present the body mass index (BMI) is computed as
   ``weight_kg / (height_m ** 2)``.  The original weight column is removed
   afterwards, but the height column is retained.
4. **Categorical encoding** – many survey responses are stored as small
   integers representing ordered or unordered categories (e.g., 1–5 Likert
   scales).  With the exception of truly binary indicators (0/1), these
   categorical variables are one‑hot encoded.  Column names are preserved
   during encoding by appending the category value to the original name
   (e.g., ``M_SAD`` → ``M_SAD_1``, ``M_SAD_2``).  Numeric variables with
   greater cardinality (more than 10 unique values) are treated as
   continuous and left unchanged.

The resulting ``DataFrame`` contains only numeric columns suitable for
traditional machine learning algorithms as well as the bespoke interpretable
ANN described in the accompanying notebooks.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

def _identify_multi_select_columns(columns: Iterable[str]) -> Dict[str, List[str]]:
    """Group columns by common base name if they follow a ``prefix_suffix`` pattern.

    The Youth Risk Behavior Survey encodes "select all that apply" items as a
    series of columns whose names share a prefix and end with an underscore and
    an integer (e.g., ``PA_1``, ``PA_2``, ``PA_3``).  This helper uses a
    regular expression to collect such columns into a dictionary keyed by the
    prefix.  Only prefixes with more than one associated column are returned.

    Parameters
    ----------
    columns : Iterable[str]
        Column names from the raw DataFrame.

    Returns
    -------
    Dict[str, List[str]]
        A mapping from prefix to a list of full column names.  Singletons are
        ignored since they do not represent multi‑select groups.
    """
    pattern = re.compile(r"^(?P<prefix>.+?)_(?P<suffix>\d+)$")
    groups: Dict[str, List[str]] = defaultdict(list)
    for col in columns:
        m = pattern.match(col)
        if m:
            prefix = m.group("prefix")
            groups[prefix].append(col)
    # Discard prefixes that only occur once (not multi‑select)
    return {p: sorted(cols) for p, cols in groups.items() if len(cols) > 1}


def _determine_column_types(
    df: pd.DataFrame,
    multi_select_cols: Iterable[str],
    *,
    nominal_cols: Iterable[str] | None = None,
) -> Tuple[List[str], List[str]]:
    """Heuristically separate columns into categorical and continuous.

    A numeric column is considered continuous if it contains a large number
    of distinct values (more than 10) or if its data type is float.  Binary
    indicators (0/1 or -1/0/1) are treated as continuous and left unchanged.
    All other numeric columns with a small number of unique values (e.g., 1–5
    Likert scales) are considered categorical and slated for one‑hot encoding.
    Object or string typed columns are always considered categorical.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame after multi‑select processing and missing value handling.
    multi_select_cols : Iterable[str]
        A flat iterable of all columns participating in multi‑select groups.

    Returns
    -------
    Tuple[List[str], List[str]]
        Two lists: the first containing names of categorical columns, and the
        second containing names of continuous columns.
    """
    cat_cols: List[str] = []
    num_cols: List[str] = []
    multi_set = set(multi_select_cols)
    nominal_set = set(col.upper() for col in nominal_cols) if nominal_cols else set()
    for col in df.columns:
        # skip multi‑select columns: already binary encoded
        if col in multi_set:
            num_cols.append(col)
            continue
        # Explicitly treat certain variables as continuous regardless of
        # cardinality.  Height (HT) and derived BMI are measurements rather
        # than categorical codes.  Additional variables can be added here
        # in future (e.g., AGE) if required.
        if col.upper() in {"HT", "BMI"}:
            num_cols.append(col)
            continue
        # Force columns explicitly listed by the user as nominal into the
        # categorical set.  ``nominal_cols`` is case‑insensitive.
        if col.upper() in nominal_set:
            cat_cols.append(col)
            continue
        series = df[col]
        # treat object/string columns as categorical
        if series.dtype == object:
            cat_cols.append(col)
            continue
        # numeric types
        # drop missing placeholder -1 when counting unique values
        unique_vals = series[series != -1].dropna().unique()
        if len(unique_vals) == 0:
            # degenerate constant column
            num_cols.append(col)
            continue
        # if all unique values are within {0,1}
        if set(unique_vals).issubset({0, 1}):
            num_cols.append(col)
            continue
        # Determine if unique values are all integers (e.g., Likert scales may
        # be stored as floats after missing value imputation).  ``astype(int)``
        # will fail for true non‑integer floats (e.g., 1.5), so we use
        # ``np.isclose`` for robustness.
        all_integers = True
        for val in unique_vals:
            if not np.isclose(val, int(val)):
                all_integers = False
                break
        # Treat as continuous if it has many unique values or contains non‑integer
        # values (e.g., continuous measurements) even if dtype is float.  This
        # preserves variables like BMI, hours slept, etc.  Small integer
        # scales (e.g., 1–5) are treated as categorical.
        if len(unique_vals) > 10 or not all_integers:
            num_cols.append(col)
        else:
            cat_cols.append(col)
    return cat_cols, num_cols


def preprocess_youth_data(df: pd.DataFrame, nominal_cols: Iterable[str] | None = None) -> pd.DataFrame:
    """Preprocess raw Youth Risk Behavior Survey data for modelling.

    This function implements the transformations described in the module
    docstring.  It does not mutate its input; instead a transformed copy is
    returned.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame containing the 2024 survey responses.  Columns should
        correspond to variable names described in the survey metadata.
    nominal_cols : Iterable[str] or None, optional
        An optional collection of column names that should always be treated
        as nominal (non‑ordered) categorical variables and therefore one‑hot
        encoded.  This allows the caller to keep ordinal or Likert scale
        variables in their numeric form while encoding truly nominal items
        (e.g., 거주형태, 가족동거 여부) even if they are encoded as small
        integers in the raw data.  Matching is case‑insensitive.

    Returns
    -------
    pd.DataFrame
        A numerically encoded DataFrame with BMI computed, multi‑select
        questions binarised, missing values filled with ``-1`` and
        categorical variables one‑hot encoded.
    """
    if df is None:
        raise ValueError("Input DataFrame cannot be None")
    # Work on a copy to avoid side‑effects
    data = df.copy()

    # ------------------------------------------------------------------
    # 1. Compute BMI and drop weight column if both height and weight exist
    # Height is recorded in centimetres (HT), weight in kilograms (WT)
    height_col_candidates = [c for c in data.columns if c.upper() == "HT"]
    weight_col_candidates = [c for c in data.columns if c.upper() == "WT"]
    if height_col_candidates and weight_col_candidates:
        height_col = height_col_candidates[0]
        weight_col = weight_col_candidates[0]
        # Convert to numeric and handle non‑numeric gracefully
        height_cm = pd.to_numeric(data[height_col], errors="coerce")
        weight_kg = pd.to_numeric(data[weight_col], errors="coerce")
        # Compute BMI (weight_kg / (height_m ** 2))
        # Avoid division by zero or NaNs; result will be NaN which will be filled later
        bmi = weight_kg / (height_cm / 100.0) ** 2
        data["BMI"] = bmi
        # Drop weight column but keep height
        data = data.drop(columns=[weight_col])

    # ------------------------------------------------------------------
    # 2. Replace structural missing values created by skip patterns with -1
    # We'll defer filling until after multi‑select processing so that NaNs in
    # multi‑select columns can still be coerced to 0 properly

    # ------------------------------------------------------------------
    # 3. Identify multi‑select question columns and binarise them
    multi_groups = _identify_multi_select_columns(data.columns)
    multi_cols = [col for cols in multi_groups.values() for col in cols]
    for base, cols in multi_groups.items():
        # In a multi‑select group a respondent may skip the entire question due
        # to earlier branching logic.  If all options in the group are
        # missing for a row we preserve NaN so it can later be filled with
        # ``-1`` to encode "not asked".  Otherwise missing entries are
        # interpreted as unselected and replaced with 0.  Any positive value
        # indicates selection and is collapsed to 1.
        group_df = data[cols]
        # mask identifying rows where every option is missing
        all_missing = group_df.isna().all(axis=1)
        # For rows with at least one non‑missing value, fill NaN with 0 and
        # coerce values to numeric
        # Using apply(pd.to_numeric) rowwise would be expensive; instead
        # vectorise operations per column
        for col in cols:
            # Work on a copy of the column
            col_series = pd.to_numeric(data[col], errors="coerce")
            # For rows where all options are missing, preserve NaN
            # For other rows, replace NaN with 0
            col_series = col_series.where(all_missing, col_series.fillna(0))
            # Collapse to binary: positive values → 1, zero or negative → 0
            data[col] = col_series.apply(lambda x: 1 if (x is not np.nan and x > 0) else (np.nan if np.isnan(x) else 0))

    # ------------------------------------------------------------------
    # 4. Fill remaining NaNs with -1 to encode skipped items
    data = data.fillna(-1)

    # ------------------------------------------------------------------
    # 5. Determine which columns to one‑hot encode and which to keep numeric
    cat_cols, num_cols = _determine_column_types(data, multi_cols, nominal_cols=nominal_cols)

    # ------------------------------------------------------------------
    # 6. One‑hot encode categorical columns; leave numeric columns untouched
    if cat_cols:
        # pandas.get_dummies drops the original categorical columns and
        # generates new ones named ``<col>_<value>``.  We keep dummy_na=False
        # because missing categories have already been encoded as -1; the -1
        # category will be treated like any other integer and produce its own
        # dummy column if present in the data.
        data = pd.get_dummies(
            data,
            columns=cat_cols,
            prefix=cat_cols,
            prefix_sep="_",
            dummy_na=False,
        )
    # Ensure column order is deterministic: continuous columns first, then
    # categorical dummies sorted alphabetically
    numeric_cols_present = [c for c in num_cols if c in data.columns]
    dummy_cols = [c for c in data.columns if c not in numeric_cols_present]
    data = data[numeric_cols_present + dummy_cols]

    # Cast all columns to float for modelling convenience
    # (binary dummies remain 0/1 but cast to float implicitly)
    data = data.astype(float)

    return data
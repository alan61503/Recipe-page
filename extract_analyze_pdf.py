from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
from pypdf import PdfReader
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


PDF_PATH = Path(__file__).with_name("round 1.pdf")
OUTPUT_DIR = Path(__file__).with_name("analysis_output")
OUTPUT_DIR.mkdir(exist_ok=True)
COLUMN_NAMES_PATH = Path(__file__).with_name("column_names.csv")


def extract_rows_from_pdf(pdf_path: Path) -> List[List[str]]:
    reader = PdfReader(str(pdf_path))
    rows: List[List[str]] = []
    numeric_line = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

    for page in reader.pages:
        text = page.extract_text() or ""
        for line in text.splitlines():
            tokens = numeric_line.findall(line)
            if len(tokens) >= 3:
                rows.append(tokens)
    return rows


def normalize_row_lengths(rows: List[List[str]]) -> List[List[str]]:
    if not rows:
        return rows
    lengths = [len(r) for r in rows]
    mode_len = pd.Series(lengths).mode().iloc[0]
    return [r for r in rows if len(r) == mode_len]


def load_column_names(col_count: int) -> List[str]:
    if COLUMN_NAMES_PATH.exists():
        names = (
            pd.read_csv(COLUMN_NAMES_PATH, header=None)
            .iloc[:, 0]
            .astype(str)
            .tolist()
        )
        if len(names) >= col_count:
            return names[:col_count]
    return [f"col_{i+1}" for i in range(col_count)]


def to_dataframe(rows: List[List[str]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    col_count = max(len(r) for r in rows)
    columns = load_column_names(col_count)
    padded_rows = [r + [None] * (col_count - len(r)) for r in rows]
    df = pd.DataFrame(padded_rows, columns=columns)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(axis=1, how="all")
    df = df.dropna(axis=0, how="any")
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    return df


def select_numeric_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def create_composite_indices(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    if len(numeric_cols) < 3:
        return df

    variances = df[numeric_cols].var().sort_values(ascending=False)
    top_cols = variances.index[: max(3, min(5, len(numeric_cols)))].tolist()

    scaler = StandardScaler()
    z = scaler.fit_transform(df[top_cols])
    z_df = pd.DataFrame(z, columns=[f"z_{c}" for c in top_cols])

    df = df.copy()
    df["composite_z_index"] = z_df.mean(axis=1)

    weights = variances.loc[top_cols] / variances.loc[top_cols].sum()
    df["composite_weighted_index"] = (df[top_cols] * weights.values).sum(axis=1)

    return df


def identify_high_value_segments(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    if score_col not in df.columns:
        return df
    q90 = df[score_col].quantile(0.90)
    q95 = df[score_col].quantile(0.95)
    df = df.copy()
    df["segment"] = np.where(
        df[score_col] >= q95,
        "top_5%",
        np.where(df[score_col] >= q90, "top_10%", "baseline"),
    )
    return df


def find_monetizable_groups(df: pd.DataFrame, numeric_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if len(numeric_cols) < 3:
        return df, pd.DataFrame()

    X_raw = df[numeric_cols].astype(float)
    X_raw = np.sign(X_raw) * np.log1p(np.abs(X_raw))
    finite_mask = np.isfinite(X_raw).all(axis=1)
    df = df.loc[finite_mask].copy()
    if df.empty:
        return df, pd.DataFrame()

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw.loc[finite_mask])
    if not np.isfinite(X).all():
        keep = np.isfinite(X).all(axis=1)
        X = X[keep]
        df = df.iloc[keep].copy()
        if df.empty:
            return df, pd.DataFrame()

    k = 4 if len(df) >= 50 else 3
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)

    df = df.copy()
    df["cluster"] = clusters

    cluster_summary = (
        df.groupby("cluster")[numeric_cols]
        .agg(["mean", "median", "count"])
        .sort_values((numeric_cols[0], "count"), ascending=False)
    )
    return df, cluster_summary


def build_insights(df: pd.DataFrame, score_col: str) -> List[str]:
    insights: List[str] = []

    if df.empty:
        return ["No usable numeric rows were extracted from the PDF."]

    insights.append(f"Rows analyzed: {len(df)}")

    if score_col in df.columns:
        insights.append(
            f"High-value threshold (90th percentile of {score_col}): {df[score_col].quantile(0.90):.3f}"
        )
        insights.append(
            f"Top 5% cutoff ({score_col}): {df[score_col].quantile(0.95):.3f}"
        )

        segment_counts = df["segment"].value_counts(dropna=False).to_dict()
        insights.append(f"Segment distribution: {segment_counts}")

        top5 = df[df["segment"] == "top_5%"].head(5)
        if not top5.empty:
            insights.append("Top 5 rows in top_5% segment (first 5):")
            insights.extend(top5.to_string(index=False).splitlines())

    if "cluster" in df.columns:
        cluster_sizes = df["cluster"].value_counts().to_dict()
        insights.append(f"Cluster sizes (potential monetizable groups): {cluster_sizes}")

        high_clusters = (
            df.groupby("cluster")[score_col]
            .mean()
            .sort_values(ascending=False)
            .head(2)
            .index.tolist()
            if score_col in df.columns
            else []
        )
        if high_clusters:
            insights.append(f"High-value clusters by {score_col}: {high_clusters}")

    return insights


def main() -> None:
    rows = extract_rows_from_pdf(PDF_PATH)
    raw_df = to_dataframe(rows)
    raw_df.to_csv(OUTPUT_DIR / "raw_extracted.csv", index=False)
    pd.DataFrame({"column_name": raw_df.columns}).to_csv(
        OUTPUT_DIR / "column_names_used.csv", index=False
    )

    rows = normalize_row_lengths(rows)
    df = to_dataframe(rows)
    df = clean_dataset(df)

    numeric_cols = select_numeric_columns(df)
    df = create_composite_indices(df, numeric_cols)

    score_col = "composite_z_index" if "composite_z_index" in df.columns else (numeric_cols[-1] if numeric_cols else "")
    if score_col:
        df = identify_high_value_segments(df, score_col)

    df, cluster_summary = find_monetizable_groups(df, numeric_cols)

    df.to_csv(OUTPUT_DIR / "cleaned_dataset.csv", index=False)
    if not cluster_summary.empty:
        cluster_summary.to_csv(OUTPUT_DIR / "cluster_summary.csv")

    insights = build_insights(df, score_col)
    (OUTPUT_DIR / "insights.txt").write_text("\n".join(insights), encoding="utf-8")

    print("\n".join(insights))
    print(f"\nSaved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

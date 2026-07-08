import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


DATA_PATH = "/Users/sarayu/Downloads/loan_prediction.csv"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "univariant_output")
STATS_CSV = os.path.join(OUTPUT_DIR, "univariate_feature_summary.csv")


def ensure_output_dir() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)


def plot_numeric_feature(df: pd.DataFrame, col: str) -> dict:
    s = df[col]
    s_nonnull = s.dropna()

    missing_count = int(s.isna().sum())
    if s_nonnull.empty:
        return {
            "feature": col,
            "type": "numeric",
            "n_non_null": 0,
            "missing_count": missing_count,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "25%": np.nan,
            "median": np.nan,
            "75%": np.nan,
            "max": np.nan,
        }

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(s_nonnull, kde=True, ax=ax, color="#4C72B0")
    ax.set_title(f"Univariate Distribution (Numeric): {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Count")
    fig.tight_layout()

    safe_col = col.replace("/", "_")
    fig_path = os.path.join(OUTPUT_DIR, f"numeric_{safe_col}.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    desc = s_nonnull.describe(percentiles=[0.25, 0.5, 0.75])
    return {
        "feature": col,
        "type": "numeric",
        "n_non_null": int(s_nonnull.shape[0]),
        "missing_count": missing_count,
        "mean": float(desc.get("mean", np.nan)),
        "std": float(desc.get("std", np.nan)),
        "min": float(desc.get("min", np.nan)),
        "25%": float(desc.get("25%", np.nan)),
        "median": float(desc.get("50%", np.nan)),
        "75%": float(desc.get("75%", np.nan)),
        "max": float(desc.get("max", np.nan)),
    }


def plot_categorical_feature(df: pd.DataFrame, col: str, max_categories: int = 25) -> dict:
    s = df[col]
    s_nonnull = s.dropna()

    missing_count = int(s.isna().sum())
    if s_nonnull.empty:
        return {
            "feature": col,
            "type": "categorical",
            "n_non_null": 0,
            "missing_count": missing_count,
            "n_unique_non_null": 0,
            "top_category": np.nan,
            "top_category_count": np.nan,
        }

    counts = s_nonnull.value_counts(dropna=True)

    if len(counts) > max_categories:
        counts_plot = counts.iloc[:max_categories].copy()
        other_count = int(counts.iloc[max_categories:].sum())
        counts_plot = pd.concat([counts_plot, pd.Series({"Other": other_count})])
    else:
        counts_plot = counts

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=counts_plot.index.astype(str), y=counts_plot.values, ax=ax, color="#55A868")
    ax.set_title(f"Univariate Distribution (Categorical): {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()

    safe_col = col.replace("/", "_")
    fig_path = os.path.join(OUTPUT_DIR, f"categorical_{safe_col}.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    top_cat = counts.index[0]
    top_count = int(counts.iloc[0])

    return {
        "feature": col,
        "type": "categorical",
        "n_non_null": int(s_nonnull.shape[0]),
        "missing_count": missing_count,
        "n_unique_non_null": int(s_nonnull.nunique()),
        "top_category": top_cat,
        "top_category_count": top_count,
    }


def run_univariate_analysis() -> None:
    ensure_output_dir()

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}. Update DATA_PATH in this script to your CSV path."
        )

    df = pd.read_csv(DATA_PATH)

    # Drop completely empty columns
    df = df.dropna(axis=1, how="all").copy()

    results = []
    for col in df.columns:
        if df[col].isna().all():
            continue
        if is_numeric(df[col]):
            results.append(plot_numeric_feature(df, col))
        else:
            results.append(plot_categorical_feature(df, col))

    pd.DataFrame(results).to_csv(STATS_CSV, index=False)

    print("Univariate analysis complete.")
    print(f"Plots saved to: {OUTPUT_DIR}")
    print(f"Summary saved to: {STATS_CSV}")


if __name__ == "__main__":
    sns.set_theme(style="whitegrid")
    run_univariate_analysis()


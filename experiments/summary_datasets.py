# summary_datasets.py

from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path("/NOBACKUP/aumona/projects/RF-GAP-Python/data")

# Assumes label is in the first column, as in your dataprep pipeline
LABEL_COL_IDX = 0


def resolve_dataset_groups(data_dir: Path):
    """
    Resolve datasets by base name.

    Rules:
    - if <base>_train.parquet and <base>_test.parquet exist, use them as a predefined split
    - otherwise, if <base>.parquet exists, use it as a single dataset
    """
    groups = {}

    for path in sorted(data_dir.glob("*.parquet")):
        stem = path.stem

        if stem.endswith("_train"):
            base = stem[:-6]
            groups.setdefault(base, {"train": None, "test": None, "single": None})
            groups[base]["train"] = path
        elif stem.endswith("_test"):
            base = stem[:-5]
            groups.setdefault(base, {"train": None, "test": None, "single": None})
            groups[base]["test"] = path
        else:
            base = stem
            groups.setdefault(base, {"train": None, "test": None, "single": None})
            groups[base]["single"] = path

    # if a predefined split exists, ignore any single-file version
    for base, g in groups.items():
        if g["train"] is not None and g["test"] is not None:
            g["single"] = None

    return groups


def dataset_summary(base_name: str, group: dict):
    """
    Return one summary row with:
    - data name
    - number of samples
    - number of test samples (only for predefined split)
    - number of features
    - number of classes
    """
    if group["train"] is not None and group["test"] is not None:
        df_train = pd.read_parquet(group["train"])
        df_test = pd.read_parquet(group["test"])

        n_train = len(df_train)
        n_test = len(df_test)
        n_samples = n_train + n_test

        label_col = df_train.columns[LABEL_COL_IDX]
        n_features = df_train.shape[1] - 1

        # count classes over the union of train and test labels
        y_all = pd.concat([df_train.iloc[:, LABEL_COL_IDX], df_test.iloc[:, LABEL_COL_IDX]], ignore_index=True)
        n_classes = y_all.nunique(dropna=True)

        return {
            "dataset": base_name,
            "n_samples": n_samples,
            "n_test_samples": n_test,
            "n_features": n_features,
            "n_classes": n_classes,
            "predefined_split": True,
        }

    if group["single"] is not None:
        df = pd.read_parquet(group["single"])

        n_samples = len(df)
        n_test = np.nan
        n_features = df.shape[1] - 1
        n_classes = df.iloc[:, LABEL_COL_IDX].nunique(dropna=True)

        return {
            "dataset": base_name,
            "n_samples": n_samples,
            "n_test_samples": n_test,
            "n_features": n_features,
            "n_classes": n_classes,
            "predefined_split": False,
        }

    return None


def format_int_or_dash(x):
    if pd.isna(x):
        return "--"
    return f"{int(x)}"


def make_latex_table(df_summary: pd.DataFrame) -> str:
    df_disp = df_summary.copy()

    df_disp["n_samples"] = df_disp["n_samples"].map(format_int_or_dash)
    df_disp["n_test_samples"] = df_disp["n_test_samples"].map(format_int_or_dash)
    df_disp["n_features"] = df_disp["n_features"].map(format_int_or_dash)
    df_disp["n_classes"] = df_disp["n_classes"].map(format_int_or_dash)

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\begin{tabular}{lrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Dataset & Samples & Test samples & Features & Classes \\")
    lines.append(r"\midrule")

    for _, row in df_disp.iterrows():
        lines.append(
            f"{row['dataset']} & {row['n_samples']} & {row['n_test_samples']} & "
            f"{row['n_features']} & {row['n_classes']} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Summary of datasets used in the experiments. "
                 r'The "Test samples" column is only applicable to datasets with a predefined train/test split.}')
    lines.append(r"\label{tab:dataset_summary}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def main():
    groups = resolve_dataset_groups(DATA_DIR)

    rows = []
    for base_name, group in sorted(groups.items()):
        row = dataset_summary(base_name, group)
        if row is not None:
            rows.append(row)

    df_summary = pd.DataFrame(rows)
    df_summary = df_summary.sort_values("dataset").reset_index(drop=True)

    print("Dataset summary:\n")
    print(df_summary[["dataset", "n_samples", "n_test_samples", "n_features", "n_classes"]])

    latex_table = make_latex_table(df_summary)

    out_path = DATA_DIR.parent / "experiments" / "dataset_summary_table.tex"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(latex_table, encoding="utf-8")

    print("\nLaTeX table:\n")
    print(latex_table)
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
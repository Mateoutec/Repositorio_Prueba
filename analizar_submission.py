import argparse
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {"Id", "Target"}


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo: {path}")

    df = pd.read_csv(path)
    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Faltan columnas requeridas: {missing}")

    return df


def analyze_submission(submission_path: Path, train_path: Path | None = None) -> None:
    submission = load_csv(submission_path)

    print(f"Archivo analizado: {submission_path}")
    print(f"Filas: {len(submission):,}")
    print(f"Columnas: {list(submission.columns)}")
    print()

    print("Valores nulos por columna:")
    print(submission.isna().sum())
    print()

    duplicate_ids = submission["Id"].duplicated().sum()
    print(f"Ids duplicados: {duplicate_ids:,}")
    print(f"Targets unicos: {submission['Target'].nunique():,}")
    print()

    print("Distribucion de Target:")
    target_counts = (
        submission["Target"]
        .value_counts(dropna=False)
        .sort_index()
        .rename_axis("Target")
        .reset_index(name="Count")
    )
    target_counts["Percent"] = target_counts["Count"] / len(submission) * 100
    print(target_counts.to_string(index=False, formatters={"Percent": "{:.2f}%".format}))
    print()

    if train_path:
        train = load_csv(train_path)
        train_targets = set(train["Target"].dropna().unique())
        submission_targets = set(submission["Target"].dropna().unique())

        missing_from_submission = sorted(train_targets - submission_targets)
        extra_in_submission = sorted(submission_targets - train_targets)

        print(f"Comparacion contra: {train_path}")
        print(f"Clases en train: {len(train_targets):,}")
        print(f"Clases en submission: {len(submission_targets):,}")
        print(f"Clases de train ausentes en submission: {missing_from_submission}")
        print(f"Clases extra en submission: {extra_in_submission}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analiza un archivo submission.csv usando pandas."
    )
    parser.add_argument(
        "--submission",
        default="submission.csv",
        type=Path,
        help="Ruta del CSV de submission. Por defecto: submission.csv",
    )
    parser.add_argument(
        "--train",
        default=None,
        type=Path,
        help="Ruta opcional de train.csv para comparar clases.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    analyze_submission(args.submission, args.train)

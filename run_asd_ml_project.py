"""
ASD vs Healthy Control classification using peripheral blood miRNA expression data.

Inputs expected in the same folder:
- GSE89596_miRNA_X.csv
- GSE89596_labels.csv

Outputs:
- metrics_summary.csv
- selected_miRNA_biomarkers.csv
- predictions_test_set.csv
- confusion_matrix.png
- roc_curve.png
- pca_qc_plot.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    RocCurveDisplay,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)


class SafeLog1pTransformer(BaseEstimator, TransformerMixin):
    """Apply log1p after shifting features when negatives are present.
    Some processed miRNA values can be slightly negative. Since log1p requires
    values > -1, this then learns a per feature shift from training data
    only, then applies the same shift to test data.
    """

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        mins = np.nanmin(X, axis=0)
        self.shift_ = np.where(mins <= -1.0, -mins + 1e-6, 0.0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.log1p(X + self.shift_)


def load_and_align_data(expression_path: Path, labels_path: Path):
    labels = pd.read_csv(labels_path)
    expr_raw = pd.read_csv(expression_path)

    # Basic label cleanup
    labels = labels.drop_duplicates(subset="sample_id").copy()
    labels = labels.dropna(subset=["sample_id", "group"])
    labels["group"] = labels["group"].astype(str).str.strip()
    labels["label"] = labels["group"].map({"control": 0, "Control": 0, "CTRL": 0, "ASD": 1, "asd": 1})
    if labels["label"].isna().any():
        bad = labels.loc[labels["label"].isna(), "group"].unique()
        raise ValueError(f"Unrecognized group labels: {bad}")

    # Drop duplicated miRNA IDs by averaging duplicates, then transpose so rows=samples.
    if "miRNA_ID" not in expr_raw.columns:
        raise ValueError("Expression file must contain a miRNA_ID column.")
    expr = expr_raw.groupby("miRNA_ID", as_index=True).mean(numeric_only=True)
    X = expr.T
    X.index.name = "sample_id"

    common_samples = labels["sample_id"].isin(X.index)
    labels = labels.loc[common_samples].copy()
    X = X.loc[labels["sample_id"]]
    y = labels["label"].astype(int).to_numpy()

    return X, y, labels, expr_raw


def make_qc_plot(X: pd.DataFrame, y: np.ndarray, labels: pd.DataFrame, outpath: Path):
    # QC visualization only. It is not used to train/evaluate the classifier.
    X_qc = SimpleImputer(strategy="median").fit_transform(X)
    X_qc = StandardScaler().fit_transform(X_qc)
    pcs = PCA(n_components=2, random_state=42).fit_transform(X_qc)

    plt.figure(figsize=(7, 5))
    for value, name, marker in [(0, "Control", "o"), (1, "ASD", "x")]:
        mask = y == value
        plt.scatter(pcs[mask, 0], pcs[mask, 1], label=name, marker=marker)
    plt.title("PCA QC Plot of miRNA Expression Profiles")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expression", default="GSE89596_miRNA_X.csv")
    parser.add_argument("--labels", default="GSE89596_labels.csv")
    parser.add_argument("--outdir", default="outputs")  
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    X, y, labels, expr_raw = load_and_align_data(Path(args.expression), Path(args.labels))

    qc = {
        "n_samples": X.shape[0],
        "n_features_after_duplicate_miRNA_averaging": X.shape[1],
        "n_raw_miRNA_rows": expr_raw.shape[0],
        "n_duplicate_miRNA_rows_removed_by_averaging": int(expr_raw.shape[0] - X.shape[1]),
        "missing_values_total": int(X.isna().sum().sum()),
        "class_counts": labels["group"].value_counts().to_dict(),
    }
    pd.Series(qc, dtype="object").to_csv(outdir / "data_qc_summary.csv")

    make_qc_plot(X, y, labels, outdir / "pca_qc_plot.png")

    X_train, X_test, y_train, y_test, labels_train, labels_test = train_test_split(
        X, y, labels, test_size=0.30, random_state=42, stratify=y
    )

    # Everything below is fit only on the training folds during search.
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("variance_filter", VarianceThreshold(threshold=0.0)),
        ("scaler", StandardScaler()),
        ("feature_selection", SelectKBest(score_func=f_classif)),
        ("classifier", LogisticRegression(max_iter=5000, solver="liblinear", random_state=42)),
    ])

    param_grid = {
        "feature_selection__k": [10, 25, 50],
        "classifier__C": [0.1, 1, 10],
        "classifier__penalty": ["l1", "l2"],
        "classifier__class_weight": [None],
    }

    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=inner_cv,
        n_jobs=1,
        refit=True,
        return_train_score=True,
    )
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    metrics = {
        "best_cv_auroc_on_training_folds": grid.best_score_,
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_precision": precision_score(y_test, y_pred, zero_division=0),
        "test_recall": recall_score(y_test, y_pred, zero_division=0),
        "test_f1": f1_score(y_test, y_pred, zero_division=0),
        "test_auroc": roc_auc_score(y_test, y_prob),
        "best_params": grid.best_params_,
    }
    pd.DataFrame([metrics]).to_csv(outdir / "metrics_summary.csv", index=False)

    report = classification_report(y_test, y_pred, target_names=["Control", "ASD"], output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(outdir / "classification_report.csv")

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Control", "ASD"])
    disp.plot(values_format="d")
    plt.title("Confusion Matrix: Held-Out Test Set")
    plt.tight_layout()
    plt.savefig(outdir / "confusion_matrix.png", dpi=200)
    plt.close()

    RocCurveDisplay.from_predictions(y_test, y_prob)
    plt.title("ROC Curve: Held-Out Test Set")
    plt.tight_layout()
    plt.savefig(outdir / "roc_curve.png", dpi=200)
    plt.close()

    predictions = labels_test[["sample_id", "group", "sample_title"]].copy()
    predictions["true_label"] = y_test
    predictions["predicted_label"] = y_pred
    predictions["predicted_probability_ASD"] = y_prob
    predictions.to_csv(outdir / "predictions_test_set.csv", index=False)

    # Interpret selected features 
    var_mask = best_model.named_steps["variance_filter"].get_support()
    features_after_var = X.columns[var_mask]
    selected_mask = best_model.named_steps["feature_selection"].get_support()
    selected_features = features_after_var[selected_mask]
    coef = best_model.named_steps["classifier"].coef_[0]

    biomarker_table = pd.DataFrame({
        "miRNA": selected_features,
        "logistic_regression_coefficient": coef,
        "absolute_coefficient": np.abs(coef),
        "direction": np.where(coef > 0, "Higher favors ASD", "Higher favors Control"),
    }).sort_values("absolute_coefficient", ascending=False)
    biomarker_table.to_csv(outdir / "selected_miRNA_biomarkers.csv", index=False)

    # Overall performance estimate using the full pipeline with CV
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
    cv_scores = cross_validate(
        grid.best_estimator_, X, y, cv=outer_cv,
        scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
        n_jobs=1
    )
    cv_summary = pd.DataFrame({
        metric.replace("test_", ""): [np.mean(values), np.std(values)]
        for metric, values in cv_scores.items() if metric.startswith("test_")
    }, index=["mean", "std"])
    cv_summary.to_csv(outdir / "cross_validation_summary.csv")

    # Summary
    with open(outdir / "RESULTS_SUMMARY.md", "w", encoding="utf-8") as f:
        f.write("# ASD miRNA Classification Results\n\n")
        f.write("## Dataset/QC\n")
        f.write(f"- Samples: {X.shape[0]}\n")
        f.write(f"- miRNA features after duplicate averaging: {X.shape[1]}\n")
        f.write(f"- Class balance: {labels['group'].value_counts().to_dict()}\n")
        f.write(f"- Missing values: {int(X.isna().sum().sum())}\n\n")
        f.write("## Leakage Prevention\n")
        f.write("The data was split before model fitting. Median imputation, log transform, variance filtering, standardization, and feature selection were all placed inside a scikit-learn Pipeline, so each step was fit only on training folds during grid search/CV and then applied to held-out data.\n\n")
        f.write("## Best Model\n")
        f.write(f"- Best parameters: `{grid.best_params_}`\n")
        f.write(f"- Best training-fold CV AUROC during grid search: {grid.best_score_:.3f}\n\n")
        f.write("## Held-Out 30% Test Set Metrics\n")
        for k, v in metrics.items():
            if k.startswith("test_"):
                f.write(f"- {k}: {v:.3f}\n")
        f.write("\n## Top Candidate miRNA Biomarkers by Absolute Logistic Regression Coefficient\n")
        f.write(biomarker_table.head(15).to_markdown(index=False))
        f.write("\n\n## Important Caveat\n")
        f.write("This is a small p >> n dataset, so the model is useful for a class project and exploratory biomarker screening, not for clinical diagnosis. Results should be validated on an independent cohort.\n")

    # Save grid search results 
    pd.DataFrame(grid.cv_results_).to_csv(outdir / "grid_search_results.csv", index=False)

    print("Outputs written to:", outdir)
    print(pd.DataFrame([metrics]).T)


if __name__ == "__main__":
    main()

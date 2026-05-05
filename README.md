# ASD miRNA Classification Project

This project classifies Autism Spectrum Disorder (ASD) vs. healthy control using peripheral blood miRNA expression data.

## Files
- `run_asd_ml_project.py`: complete executable ML pipeline.
- `RESULTS_SUMMARY.md`: project-ready written results summary.
- `metrics_summary.csv`: held-out test metrics and best hyperparameters.
- `classification_report.csv`: precision/recall/F1 by class.
- `selected_miRNA_biomarkers.csv`: selected candidate miRNAs ranked by logistic regression coefficient magnitude.
- `predictions_test_set.csv`: test-set predictions and ASD probabilities.
- `pca_qc_plot.png`: PCA QC visualization.
- `confusion_matrix.png`: held-out test confusion matrix.
- `roc_curve.png`: held-out test ROC curve.
- `cross_validation_summary.csv`: 5-fold CV summary of the final pipeline.
- `grid_search_results.csv`: full GridSearchCV table.

## How to run
```bash
pip install -r requirements.txt
python run_asd_ml_project.py --expression GSE89596_miRNA_X.csv --labels GSE89596_labels.csv --outdir results
```

## Main leakage-control design
The script splits the dataset first. Imputation, variance filtering, scaling, feature selection, and logistic regression are inside one scikit-learn `Pipeline`, so these operations are learned only from the training data during cross-validation and then applied to held-out data.

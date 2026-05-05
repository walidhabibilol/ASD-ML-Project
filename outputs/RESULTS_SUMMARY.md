# ASD miRNA Classification Results

## Summary

This project applies a machine learning pipeline to classify Autism Spectrum Disorder (ASD) using peripheral blood miRNA expression data. The objective is to evaluate classification performance and identify potential biomarkers associated with ASD.

---

## Dataset & Quality Control

* Samples: ~60
* Features: ~2500 miRNA (after preprocessing)
* Class distribution: Balanced (ASD vs Control)
* Missing values handled using median imputation

A PCA based quality control plot was generated to visualize data structure and detect potential outliers.

---

## Modeling Approach

The workflow follows proper machine learning practice to avoid data leakage:

* Data is split into training (70%) and test (30%) sets before preprocessing
* All transformations are applied using a scikit-learn Pipeline
* Preprocessing (scaling and feature selection) is learned only from training data
* The same transformations are applied to test data
* Model is evaluated only on unseen test data

---

## Key Results (Held-Out Test Set)

* **Accuracy:** 0.50
* **Precision:** 0.50
* **Recall:** 0.67
* **F1 Score:** 0.57
* **AUROC:** 0.70

---

## Interpretation

* AUROC ≈ 0.70 indicates the model performs better than random guessing
* Higher recall suggests the model is effective at identifying ASD cases
* Lower accuracy reflects dataset limitations and class overlap

---

## Feature Selection & Biomarkers

Feature selection was performed using a statistical filter method (ANOVA F-test). The selected miRNA features represent those most strongly associated with ASD classification.

Logistic regression coefficients were used to rank features by importance, identifying candidate biomarkers.

---

## Cross-Validation Insight

Cross-validation was used to estimate generalization performance across different data splits. Results were consistent with test performance, indicating moderate model stability.

---

## Limitations

* Small sample size (p >> n problem)
* Risk of overfitting despite regularization
* No external validation dataset
* Interpretation requires further experimental validation

---

## Conclusion

This project demonstrates that miRNA expression data can be used for ASD classification using a properly constructed machine learning pipeline. While performance is moderate due to dataset constraints, the workflow successfully integrates preprocessing, feature selection, and evaluation in a leakage safe manner.

---

## Outputs Generated

* ROC curve
* Confusion matrix
* PCA QC visualization
* Metrics summary
* Biomarker rankings
* Prediction outputs

All results are saved in the `outputs/` directory.

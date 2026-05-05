# ASD miRNA Classification Project

## Overview

This project develops a machine learning pipeline to classify Autism Spectrum Disorder (ASD) versus healthy controls using peripheral blood miRNA expression data. The goal is to evaluate predictive performance and identify potential biomarker candidates associated with ASD.

---

## Dataset

* ~60 samples (ASD vs Control)
* ~2500 miRNA features
* High-dimensional dataset (p >> n problem)

---

## Methodology

The workflow strictly follows proper machine learning practice to avoid data leakage:

1. **Train-Test Split (70/30)**

   * Data is split before any preprocessing.

2. **Preprocessing (Training Data Only)**

   * Standardization 
   * Feature selection using ANOVA F test

3. **Model**

   * Logistic Regression classifier

4. **Pipeline**

   * All preprocessing and modeling steps are combined using a scikit-learn Pipeline to ensure transformations are learned only from training data and applied consistently to test data.

5. **Evaluation**

   * Performed on held out test set only
   * Metrics: Accuracy, Precision, Recall, F1-score, AUROC

---

## Results

* **Accuracy:** 0.50
* **Precision:** 0.50
* **Recall:** 0.67
* **F1 Score:** 0.57
* **AUROC:** 0.70

The model demonstrates moderate predictive performance and performs better than random guessing, with relatively strong recall in detecting ASD cases.

---

## Key Concepts Demonstrated

* Classification
* Train/test split and proper evaluation
* Feature selection 
* Logistic regression modeling
* Data leakage prevention using pipelines

---

## How to Run

### 1. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 2. Run the project

```bash
python run_asd_ml_project.py
```

---

## Outputs

Generated in the `outputs/` folder:

* `roc_curve.png` – ROC curve visualization
* `confusion_matrix.png` – confusion matrix
* `pca_qc_plot.png` – data quality visualization
* `metrics_summary.csv` – performance metrics
* `selected_miRNA_biomarkers.csv` – important features
* `predictions_test_set.csv` – model predictions
* `RESULTS_SUMMARY.md` – detailed results report

---

## Limitations

* Small dataset (~60 samples)
* High dimensionality relative to sample size
* Potential overfitting despite feature selection
* Results are exploratory and not clinically validated

---

## Conclusion

This project demonstrates a complete machine learning workflow for biological data classification. While performance is limited by dataset size, the pipeline correctly implements core ML principles and provides a foundation for future biomarker discovery.

---

## Author

Walid Habibi, Sonal Chakraborty, Rhea Rajmanna

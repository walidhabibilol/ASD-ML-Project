# ASD miRNA Classification Results

## Dataset/QC
- Samples: 60
- miRNA features after duplicate averaging: 2549
- Class balance: {'ASD': 30, 'Control': 30}
- Missing values: 0

## Leakage Prevention
The data was split before model fitting. Median imputation, log transform, variance filtering, standardization, and feature selection were all placed inside a scikit-learn Pipeline, so each step was fit only on training folds during grid search/CV and then applied to held-out data.

## Best Model
- Best parameters: `{'classifier__C': 0.1, 'classifier__class_weight': None, 'classifier__penalty': 'l2', 'feature_selection__k': 10}`
- Best training-fold CV AUROC during grid search: 0.562

## Held-Out 30% Test Set Metrics
- test_accuracy: 0.500
- test_precision: 0.500
- test_recall: 0.667
- test_f1: 0.571
- test_auroc: 0.704

## Top Candidate miRNA Biomarkers by Absolute Logistic Regression Coefficient
| miRNA            |   logistic_regression_coefficient |   absolute_coefficient | direction             |
|:-----------------|----------------------------------:|-----------------------:|:----------------------|
| hsa-miR-5189-5p  |                        -0.188731  |              0.188731  | Higher favors Control |
| hsa-miR-1288-3p  |                        -0.113137  |              0.113137  | Higher favors Control |
| hsa-miR-6780a-5p |                        -0.100757  |              0.100757  | Higher favors Control |
| hsa-miR-5581-5p  |                        -0.0940733 |              0.0940733 | Higher favors Control |
| hsa-miR-4716-3p  |                        -0.0912956 |              0.0912956 | Higher favors Control |
| hsa-miR-1305     |                        -0.0892914 |              0.0892914 | Higher favors Control |
| hsa-miR-3156-5p  |                        -0.0799512 |              0.0799512 | Higher favors Control |
| hsa-miR-6512-5p  |                        -0.0758208 |              0.0758208 | Higher favors Control |
| hsa-miR-3125     |                        -0.0584306 |              0.0584306 | Higher favors Control |
| hsa-miR-6734-5p  |                        -0.0550957 |              0.0550957 | Higher favors Control |

## Important Caveat
This is a small p >> n dataset, so the model is useful for a class project and exploratory biomarker screening, not for clinical diagnosis. Results should be validated on an independent cohort.

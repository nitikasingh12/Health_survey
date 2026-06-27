
Age Group Classification — Nutrition Health Survey
Problem Statement
Predict whether a respondent is an Adult (0) or Senior (1) based on health indicators from the NHANES dataset.
Dataset
FileRowsDescriptionTrain_dataset.csv2016Training data with targetTest_dataset.csv312Test data without targetSample_submission.csv312Submission format
Approach
1. EDA

Class imbalance: 1638 Adults vs 314 Seniors (5:1 ratio)
Seniors show higher glucose, insulin resistance, and diabetes rate
Missing values present across all features

2. Feature Engineering (7 → 26 features)

BMI flags (WHO thresholds) — obese, overweight, underweight
Glucose flags (ADA thresholds) — prediabetes, diabetes, OGTT
HOMA-IR proxy for insulin resistance
Composite metabolic risk score (0–5)
Interaction features — BMI×Glucose, Insulin×Risk

3. Preprocessing

Median imputation for missing values
SMOTE to balance classes (1638 vs 1638)

4. Models

XGBoost
Gradient Boosting
Random Forest
LightGBM
Stacking Ensemble (LGBM + XGB + GB + RF → Logistic Regression meta model)

5. Threshold Tuning

Tuned decision threshold on validation set optimizing F1-Macro
Default 0.5 threshold caused almost no Senior predictions

Results
MetricScoreCV Accuracy~87%CV F1-Macro~87%Public Score40
How to Run
bashpip install xgboost lightgbm scikit-learn imbalanced-learn pandas numpy
python best_submission.py
Or open directly in Google Colab:

🔗 https://colab.research.google.com/drive/1aYpMM-kpVBUfrbFRQCwdnYlCKd2GaNvn?usp=sharing
Requirements
pandas
numpy
scikit-learn
xgboost
lightgbm
imbalanced-learn
matplotlib
seaborn
File Structure
├── Train_dataset.csv
├── Test_dataset.csv
├── Sample_submission.csv
├── best_submission.py
├── submission.csv
└── README.md

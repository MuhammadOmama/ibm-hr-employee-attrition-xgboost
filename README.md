# IBM HR Analytics: Corporate Employee Attrition Prediction

An industry-standard, end-to-end data science pipeline built to predict employee attrition using advanced gradient boosting (**XGBoost**) and game-theoretic model interpretability (**SHAP**). 

This repository demonstrates how to handle heavily imbalanced corporate datasets, design robust cross-validation workflows, and translate complex machine learning decisions into actionable human resources strategies.

## 📊 Project Milestones & Performance
* **Baseline Rescue:** Upgraded a standard unweighted Random Forest baseline (Minority F1-Score: `0.2296`) to a robust weighted **XGBoost Classifier**, increasing the minority F1-score to **`0.5447`**.
* **Generalization Power:** Achieved a highly stable **ROC-AUC Score of `0.7566`** on a completely hidden final test set, proving the pipeline is highly generalizable and free from data leakage.
* **Proactive Catch Rate:** Attained a **Recall of `49%`** on the attrition class, enabling HR to proactively identify and engage nearly half of all future flight-risk employees before they resign.

## 🛠️ Data Pipeline Architecture
1. **Defensive Preprocessing:** Handled potential missing values securely using **Grouped Medians** (calculated by `JobRole` and `JobLevel` to protect continuous feature distributions from executive salary outliers) and **Modes**.
2. **Feature Engineering:** Log-transformed heavily right-skewed variables (`MonthlyIncome`, `TotalWorkingYears`) and applied standard scaling.
3. **Encoding Pipeline:** Evaluated categorical dimensions; applied manual **Ordinal Mapping** to sequential columns (`BusinessTravel`) and **One-Hot Encoding** to nominal features (`Department`, `JobRole`) while evading the dummy variable trap.
4. **Validation Strategy:** Set aside a 20% validation split and leveraged a strict **Stratified 5-Fold Cross-Validation** loop during model optimization to ensure consistent class distributions across data splits.

## 🔍 Key Business Insights (via SHAP)
Rather than delivering a "black-box" model, this project implements the **SHAP** library to uncover the exact directional drivers behind employee turnover:
* **OverTime:** The absolute #1 driver of attrition. High overtime correlates heavily with extreme attrition risk (burnout indicator).
* **Compensation:** Low monthly income and minimal stock option levels serve as massive push factors forcing early-career staff to look outside the company.
* **Commute Strain:** Employees with longer distances from home demonstrate an increased propensity to quit over time.
  <img width="779" height="540" alt="download" src="https://github.com/user-attachments/assets/1a71a260-19b2-46d0-a1c1-b17399a4f419" />


## 🚀 Technologies Used
* Python 3.12
* Pandas & NumPy (Data Wrangling)
* Scikit-Learn (Preprocessing & Validation)
* XGBoost (Advanced Gradient Boosting)
* SHAP (Model Interpretability & Explanation)
* Matplotlib & Seaborn (Data Visualization)

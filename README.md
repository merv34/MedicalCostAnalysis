# MedicalCostAnalysis
My final project for Aygaz Machine Learning Bootcamp
Dataset:  https://www.kaggle.com/datasets/mirichoi0218/insurance

# 🏥 Medical Insurance Cost Prediction Analysis

This project employs machine learning regression techniques to predict individual medical insurance costs based on demographic and lifestyle indicators. By analyzing a dataset of 1,338 beneficiaries, the project identifies key cost drivers and builds a predictive model with high accuracy.

## 📌 Project Overview

* **Dataset:** 1,338 records with 7 features (Age, Sex, BMI, Children, Smoker, Region, Charges).
* **Objective:** To predict the continuous variable `charges` using regression algorithms.
* **Final Model:** Optimized Random Forest Regressor.
* **Performance:** $R^2$ Score of **0.914**.

## 📊 Exploratory Data Analysis (EDA) & Insights

Before modeling, a comprehensive analysis was conducted to understand the data distribution and correlations:

* **Smoker vs. Non-Smoker:** Smoking status was identified as the strongest determinant of higher insurance charges.
* **BMI Analysis:** BMI follows a near-normal distribution. However, outliers were detected at the upper end of the scale.
* **Regional Trends:** The Southeast region was found to have the highest number of smokers and children compared to other regions.
* **Correlation:** A strong positive correlation exists between smoking habits and medical costs, while Age and BMI show moderate correlations.

## ⚙️ Data Preprocessing Pipeline

To ensure model stability and performance, the following preprocessing steps were applied:

1.  **Outlier Handling:** Outliers in the `bmi` feature were detected and removed using the **Z-Score method** (threshold > 3), ensuring the model wasn't skewed by extreme values.
2.  **Encoding:** Categorical variables (`sex`, `smoker`, `region`) were converted into numerical format using **One-Hot Encoding**.
3.  **Feature Scaling:** The dataset was normalized using **MinMaxScaler** to bring all features to a similar scale, optimizing the performance of distance-based algorithms.

## 🤖 Model Selection & Optimization

Three different regression models were evaluated using **5-Fold Cross-Validation** to test for generalization:

| Model | Mean RMSE Score |
|-------|-----------------|
| **Random Forest Regressor** | **~4,852** |
| Linear Regression | ~6,068 |
| Decision Tree Regressor | ~6,589 |

### Hyperparameter Tuning
The Random Forest model outperformed the others and was selected for further optimization. **GridSearchCV** was used to find the best hyperparameters:
* `n_estimators`: 300
* `min_samples_leaf`: 4
* `min_samples_split`: 10

**Result:** The optimization process further reduced the RMSE to **~4,580**.

## 🏆 Final Results

The optimized Random Forest model achieved outstanding performance on the test set:

* **R² Score:** `0.914` (The model explains **91.4%** of the variance in medical costs).
* **Mean Absolute Error (MAE):** `1,906.24`
* **Mean Squared Error (MSE):** `12,492,748`

## 🛠️ Technologies Used

* **Python** (Data Science ecosystem)
* **Pandas & NumPy** (Data Manipulation)
* **Seaborn & Matplotlib** (Data Visualization)
* **Scikit-Learn** (Machine Learning, Preprocessing, GridSearch)
* **SciPy** (Statistical Analysis)

---
*This project demonstrates a complete end-to-end data science workflow, from raw data cleaning and outlier detection to model deployment and evaluation.*

# 📉 Customer Churn Prediction – Machine Learning Competition Project


![Python](https://img.shields.io/badge/Python-3.8-blue?logo=python)  
![Machine Learning](https://img.shields.io/badge/ML-Classification-success?logo=scikit-learn)  
![Competition](https://img.shields.io/badge/Kaggle-Churn_Prediction-orange?logo=kaggle)  
![Model Performance](https://img.shields.io/badge/F1_Score-0.6258-purple?logo=data)  
![Status](https://img.shields.io/badge/Project-Completed-brightgreen)


This repository presents an end-to-end machine learning pipeline developed for predicting **customer churn** in a financial institution. The task was to identify whether a customer would exit (close their account) or stay, based on demographic, geographic, and account-related attributes.  

This project strengthened my skills in **imbalanced classification**, handling missing data, encoding strategies, and experimenting with multiple ML models, including advanced boosting ensembles.

---

## 📊 Problem Statement

The objective was to build a classification model that predicts the likelihood of customer churn.  

- **Target Variable**:  
  - `exit_status = 1 → Customer exited (churned)`  
  - `exit_status = 0 → Customer stayed`  

The dataset presented challenges such as **missing values, duplicates, outliers, and class imbalance (~20% churners vs 80% stayers).**

---

## 🧠 Key Contributions

- Cleaned and preprocessed raw training and testing datasets  
- Handled missing values using **grouped median imputation** (with country, age bins, and product count) + **missingness indicators**  
- Detected and removed duplicates & treated outliers  
- Explored EDA and visualized key relationships:
  - Higher churn in customers aged 41–60  
  - Germany had the highest churn rate  
  - Customers with 2 products had lowest churn; >2 products showed very high churn  
- Addressed **class imbalance** using stratified splits and appropriate evaluation metrics  
- Encoded categorical variables and scaled numerical features  
- Built, trained, and compared multiple classifiers  
- Hyperparameter-tuned Gradient Boosting, XGBoost, and HistGradientBoosting  
- Explored **ensemble techniques**: Voting Classifier, Stacking, and Blending  
- Achieved a final **F1 score of 0.6258**

---

## 🚀 Models Implemented

| Model                         | F1 Score (Validation) |
|-------------------------------|------------------------|
| Logistic Regression           | 0.4927                 |
| K Nearest Neighbors           | 0.5768                 |
| Decision Tree                 | 0.5082                 |
| Random Forest (Tuned)         | 0.6124                 |
| AdaBoost                      | 0.5952                 |
| Gradient Boosting (Tuned)     | 0.6124                 |
| XGBoost (Tuned)               | 0.6104                 |
| HistGradient Boosting (Tuned) | 0.6138                 |
| LightGBM                      | 0.6125                 |
| **Final Ensemble (Stack/Blend)** | **0.6149**             |

> The final solution combined multiple boosting models using **blending**, achieving the best F1 score of **0.6258** on the final Kaggle submission.

---

## 📈 Evaluation Metrics

- **F1 Score (primary)** – due to class imbalance  
- Precision, Recall, Accuracy (for additional context)  
- Confusion Matrix for churn vs non-churn predictions  

---

## 🛠️ Technology Stack

- Python 3  
- Pandas, NumPy  
- scikit-learn  
- XGBoost, LightGBM, HistGradientBoosting, AdaBoost  
- Matplotlib & Seaborn for visualization  

---

## 💡 Learnings and Takeaways

- Grouped imputation + missingness indicators helped preserve predictive power  
- **Class imbalance handling** was critical — accuracy alone was misleading  
- Ensemble methods (stacking/blending) significantly improved results over single models  
- Visual EDA revealed non-intuitive churn drivers (e.g., high churn among 3+ product customers)  

---

## 📌 Final Reflections

This project was an exciting dive into **imbalanced classification problems**. It pushed me to test multiple ensemble approaches and hyperparameter tuning strategies, ultimately achieving a strong F1 score. The learnings here directly build on my previous competition and give me confidence for future challenges.  

---

🔗 You can check out the Kaggle Notebook version here:  
👉 [Customer Churn Prediction Notebook](https://www.kaggle.com/code/drishya23f3001900/iitm-ka2-23f3001900)

🧠 *If you’re tackling class imbalance or customer churn problems, this project demonstrates practical strategies from data cleaning to advanced ensembling.*

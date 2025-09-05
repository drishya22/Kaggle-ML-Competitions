# ✈️ Flight Price Prediction – Machine Learning Competition Project

![Python](https://img.shields.io/badge/Python-3.8-blue?logo=python)
![Machine Learning](https://img.shields.io/badge/ML-Regression-success?logo=scikit-learn)
![Competition](https://img.shields.io/badge/Kaggle-First_Competition-orange?logo=kaggle)
![Model Performance](https://img.shields.io/badge/R²_Score-0.973-purple?logo=data)
![Status](https://img.shields.io/badge/Project-Completed-brightgreen)

This repository presents an end-to-end machine learning pipeline developed for a flight fare prediction challenge. As my first experience participating in a real-world ML competition, this project provided valuable insights into handling complex datasets, feature engineering, model building, and performance evaluation.

## 📊 Problem Statement

The objective was to construct a regression model capable of accurately estimating **flight ticket prices** using structured data containing both categorical and numerical variables. The dataset included challenges such as missing values, inconsistent formats, and outliers—making it a robust problem for developing practical ML skills.

## 🧠 Key Contributions

- Cleaned and preprocessed raw competition data
- Imputed missing values using suitable strategies
- Encoded categorical features via label encoding and one-hot encoding
- Scaled numerical features using `StandardScaler`
- Built and tested multiple regression models, including advanced ensemble methods
- Tuned hyperparameters for selected models to improve accuracy
- Achieved a final **R² score of 0.973** on the test dataset

## 🚀 Models Implemented

| Model               | R² Score |
|--------------------|----------|
| Random Forest       | 0.971    |
| XGBoost             | 0.970    |
| CatBoost            | 0.968    |
| Extra Trees         | 0.966    |
| LightGBM            | 0.966    |
| Ridge Regression    | 0.912    |
| Linear Regression   | 0.912    |

> The final solution employed an **ensemble of Random Forest and XGBoost**, combining strengths from both tree-based approaches to optimize prediction accuracy.

## 📈 Evaluation Metrics

- Coefficient of Determination (**R² Score**)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)

## 🛠️ Technology Stack

- Python 3
- Pandas, NumPy
- scikit-learn
- XGBoost, CatBoost, LightGBM
- Matplotlib and Seaborn for data visualization

## 💡 Learnings and Takeaways

- Navigating messy, real-world datasets requires careful attention to detail
- Feature selection and model tuning are critical for achieving performance gains
- Ensemble learning techniques can provide robust improvements over baseline models
- Small optimizations can compound meaningfully in predictive modeling

## 📌 Final Reflections

Participating in this competition was an enriching experience that helped solidify my foundational machine learning knowledge. I’m proud of the final results and excited to continue learning, exploring, and contributing to more applied ML challenges in the future.

---
You can check out my Kaggle Notebook:
🔗 [Kaggle Notebook](https://www.kaggle.com/code/drishya23f3001900/iitm-ka1-23f3001900)


🧠 *If you're exploring regression challenges or new to ML competitions, this project may offer helpful perspectives on workflow and strategy.*

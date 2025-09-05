# 🍔 Eatery Review Rating – Machine Learning Competition Project

![Python](https://img.shields.io/badge/Python-3.8-blue?logo=python)
![Machine Learning](https://img.shields.io/badge/ML-Classification-success?logo=scikit-learn)
![Competition](https://img.shields.io/badge/Kaggle-Competition-orange?logo=kaggle)
![Model Performance](https://img.shields.io/badge/Accuracy-0.6952-purple?logo=data)
![Status](https://img.shields.io/badge/Project-Completed-brightgreen)

This repository contains a machine learning pipeline developed for a Kaggle competition where the task was to **predict customer review ratings for eateries** from text reviews.  

The project stands out because instead of traditional text vectorization (TF-IDF, Bag-of-Words), it leverages **Sentence Transformers** to generate semantic embeddings, making it a strong step into modern NLP approaches.

---

## 📊 Problem Statement

The goal was to build a **classification model** that predicts the star rating (1–5) a customer would assign to an eatery based on their written review.  
The dataset required handling of **imbalanced classes**, **long unstructured text**, and effective **feature extraction** for classification.

---

## 🧠 Key Contributions

- Cleaned and preprocessed the raw review dataset (handled missing values, duplicates, outliers)  
- Conducted extensive **EDA**:  
  - Distribution of ratings  
  - Review length vs rating trends  
  - Frequent words and patterns in reviews  
  - Word cloud visualization of positive vs negative sentiment  
- **Converted textual reviews into numerical embeddings** using:  
  - **Sentence Transformer (`all-MiniLM-L6-v2`)** → generated **384-dimensional semantic embeddings** for each review  
  - **PCA (n=100)** → reduced dimensionality for efficiency while retaining variance  
- Trained multiple classification models on embeddings  
- Tuned hyperparameters and evaluated on validation/test sets  
- Achieved a final **Accuracy of 0.6952** on the test set  

---

## 🚀 Models Implemented

| Model                         | Accuracy |
|-------------------------------|----------|
| Logistic Regression           | 0.6477   |
| SGD Classifier                | 0.6432   |
| Random Forest (Untuned)       | 0.6943   |
| Random Forest (Tuned)         | 0.6947   |
| ExtraTrees                    | 0.6926   |
| Gradient Boosting             | ~0.67    |
| XGBoost (Untuned)             | ~0.68    |
| XGBoost (Tuned)               | 0.7096   |
| HistGradient Boosting         | 0.6667   |
| HistGradient Boosting (Tuned) | 0.6979   |
| LightGBM                      | 0.6843   |
| **Final Weighted Ensemble**   | **0.7109** |


> The final solution used a **weighted ensemble** of XGBoost (60%), HistGradientBoosting (30%), and Random Forest (10%), achieving the best **validation score of 0.7109**.  
> The final **submission accuracy on Kaggle was 0.6952**.


---

## 📈 Evaluation Metrics

- **Accuracy** (primary competition metric)  
- Additional checks with F1, Precision, and Recall for imbalance insights  

---

## 🛠️ Technology Stack

- Python 3  
- Pandas, NumPy  
- scikit-learn  
- XGBoost, GradientBoosting, HistGradientBoosting  
- Matplotlib & Seaborn for EDA  
- **NLP**: Sentence Transformers (`all-MiniLM-L6-v2`), PCA for dimensionality reduction  

---

## 💡 Learnings and Takeaways

- Using **semantic embeddings** from transformers captures context better than word-frequency methods like TF-IDF  
- Dimensionality reduction (PCA) is essential for balancing performance and efficiency with embeddings  
- Boosting-based classifiers (especially **HistGradientBoosting**) adapt well to high-dimensional embedding features  
- Accuracy can be pushed further with deeper transformer models (e.g., DistilBERT, BERT) and advanced techniques  

---

## 📌 Final Reflections

This competition was a major step into **modern NLP for text classification**.  
Instead of relying on simple vectorizers, this project explored **sentence-level embeddings**, leading to a competitive **0.6952 accuracy**.  

I’m proud of implementing transformer-based embeddings in a Kaggle competition setting and excited to continue exploring cutting-edge NLP approaches.

---

You can check out the Kaggle Notebook:  
🔗 [Kaggle Notebook](https://www.kaggle.com/code/drishya23f3001900/23f3001900-ka3)

🧠 *If you’re looking to apply transformer embeddings in ML competitions, this project is a practical example of combining modern NLP with traditional ML classifiers.*

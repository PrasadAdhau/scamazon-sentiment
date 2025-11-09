<!-- ============================================================ -->
<!--            SCAMAZON: AMAZON REVIEW SENTIMENT ANALYZER         -->
<!-- ============================================================ -->

# 💬 Scam-azon: Amazon Review Sentiment Analyzer  

![Python](https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white)
![BERT](https://img.shields.io/badge/-BERT-FF6F00?logo=pytorch&logoColor=white)
![DistilBERT](https://img.shields.io/badge/-DistilBERT-FFA500?logo=transformers&logoColor=white)
![NLP](https://img.shields.io/badge/-NLP-4B8BBE)
![Tkinter](https://img.shields.io/badge/-Tkinter-008080)
![GCP](https://img.shields.io/badge/-GCP-4285F4?logo=google-cloud&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

---

Analyze sentiment in Amazon reviews for **Books** and **Electronics** using **fine-tuned transformer models (BERT & DistilBERT)** and an intuitive **Tkinter GUI**. This project demonstrates the use of **interpretable AI** for real-world sentiment analysis.

---

## 📘 Overview

**Scam-azon** automates review sentiment classification, helping identify patterns and emotional tones across large datasets.  
It compares multiple models, visualizes probabilities, and interprets aspect-level insights (e.g., *battery*, *story*, *durability*).

---

## 🧩 Key Features

✅ Dual-domain Support – Books & Electronics  
✅ Model Comparison – Fine-tuned BERT, DistilBERT, Logistic Regression  
✅ Aspect-Based Sentiment – Detects product-specific opinions  
✅ Tkinter GUI – Clean, responsive desktop interface  
✅ Visualization – Displays probability bars for model confidence  

---

## 🧱 Project Structure

```bash
scamazon-sentiment/
│
├── data/
│ ├── amazon_reviews_us_Books_v1_02.tsv
│ └── amazon_reviews_us_Electronics_v1_00.tsv
│
├── notebooks/
│ └── train_final.ipynb
│
├── src/
│ └── demo.py
│
├── requirements.txt
├── .gitignore
└── README.md
```
---

## 📊 Model Training (train_final.ipynb)

This notebook covers:
- Data preprocessing & cleaning
- Label encoding (Positive, Neutral, Negative)
- Fine-tuning BERT & DistilBERT
- Evaluation metrics (Accuracy, F1-score, Confusion Matrix)
- Model saving & loading for the GUI

---

## 🧮 Results Summary

- Model	Dataset	Accuracy	Key Notes
- DistilBERT	Books	0.96	Fast & lightweight
- BERT	Electronics	0.97	Most accurate overall
- Logistic Regression	Books	0.89	Strong baseline
- VADER	Both	0.83	Quick rule-based method

---

## 🧰 Tech Stack

Category	Tools
- Languages	Python, SQL
- ML/NLP	BERT, DistilBERT, VADER, Scikit-learn, NLTK
- Data Processing	Pandas, NumPy
- Visualization	Matplotlib, Seaborn
- Interface	Tkinter, ttkthemes
- Cloud	Google Cloud Platform (GCP)
- Version Control	Git, GitHub

---

## 🧩 Future Improvements

- Add SHAP or LIME for advanced explainability
- Deploy GUI as a web app (Streamlit/Flask)
- Expand datasets (Movies, Clothing, etc.)
- Automate model retraining for continuous improvement


---

### 📊 Sentiment Prediction Example

| Input Review | Predicted Sentiment | Model | Confidence |
|---------------|--------------------|--------|-------------|
| “The story was gripping and beautifully written.” | Positive 😊 | DistilBERT | 0.97 |
| “Battery life is poor and doesn’t last long.” | Negative 😞 | BERT | 0.94 |
| “Average quality, nothing special.” | Neutral 😐 | Logistic Regression | 0.78 |

---


### 🧠 Insights

- **Books:** Emotional tone dominates — most 5★ reviews discuss *story* and *characters*.  
- **Electronics:** Sentiment trends depend heavily on *battery life* and *durability*.  
- **Explainability:** Tokens like *amazing*, *poor*, *disappointed* heavily influence model confidence.

---


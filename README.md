# SBA Loan Default Risk Prediction (ML Pipeline)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](https://jupyter.org/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Sklearn-lightgrey?logo=scikit-learn)](https://scikit-learn.org/)

A complete end-to-end machine learning pipeline for predicting **SBA loan default risk** using real-world Small Business Administration (SBA) data. This repository includes two phases designed to demonstrate structured data processing, model building, evaluation, and business interpretation.

---

## 📁 Project Structure

```
SBA Loan Default Risk Prediction/
│
├── Phase 1/
│   ├── notebook/
│   │   ├── Phase_1_Model Training.ipynb
│   │   └── Phase_1_Scoring Function.ipynb
│   └── artifacts/
│       ├── LogisticRegressionModel.pkl
│       ├── one_hot_encoder.pkl
│       ├── woe_encoder.pkl
│       └── Shiva Kumar Reddy_Koppula_sxk230064_Project_1_Scoring Function.py
│
├── Phase 2/
│   ├── Notebook/
│   │   ├── Phase_2_Final Model Training.ipynb
│   │   └── Phase_2_Scoring Function.ipynb
│   └── artifacts/
│       ├── artifacts_dict_file.pkl
│       └── best_model/
```

---

## Objective

Build machine learning models that predict the likelihood of SBA loan defaults, and evaluate their performance to extract actionable insights for financial decision-making.

---

## Methodology

Both Phase 1 and Phase 2 follow a structured machine learning workflow:

1. Data Loading & Cleaning
2. Exploratory Data Analysis (EDA)
3. Feature Engineering (including WOE, OHE)
4. Model Training (Logistic Regression, Random Forest)
5. Evaluation (AUC, Accuracy, Confusion Matrix)
6. Scoring function for holdout data
7. Model Export & Deployment Readiness

---

## Feature Engineering Techniques

- Weight-of-Evidence (WOE) encoding
- One-hot encoding (OHE)
- Binning and transformation
- Null handling and outlier treatment

---

## Business Context

Loan defaults can impact small businesses and financial institutions significantly. Predicting them early enables risk mitigation and better lending decisions. This project aims to develop predictive models using machine learning to flag high-risk SBA loan applications.

---

## Notebooks

- 📘 Phase 1:
  - [`Phase_1_Model Training.ipynb`](./Phase%201/notebook/Phase_1_Model%20Training.ipynb)
  - [`Phase_1_Scoring Function.ipynb`](./Phase%201/notebook/Phase_1_Scoring%20Function.ipynb)

- 📘 Phase 2:
  - [`Phase_2_Final Model Training.ipynb`](./Phase%202/Notebook/Phase_2_Final%20Model%20Training.ipynb)
  - [`Phase_2_Scoring Function.ipynb`](./Phase%202/Notebook/Phase_2_Scoring%20Function.ipynb)

---

## Highlights

| Metric              | Phase 1       | Phase 2       |
|---------------------|----------------|----------------|
| Accuracy            | 84%            | 85%            |
| AUC Score           | 0.78           | 0.81           |
| Best Model          | Logistic Regression | Random Forest |
| Imbalance Handling  | No             | Yes            |

---

## Tech Stack

- Python 3.8+
- Jupyter Notebook
- pandas, NumPy, scikit-learn
- Matplotlib, Seaborn
- WOE & OHE encoding
- Pickle-based model saving

---

## 📤 How to Run

1. Clone this repository:
```bash
git clone https://github.com/shivakrdy/SBA-Loan-Default-Risk-Prediction-ML-Pipeline.git
cd SBA-Loan-Default-Risk-Prediction-ML-Pipeline
```

2. Open Jupyter Notebook and run either phase:
```bash
jupyter notebook
```

---

## 👨‍💻 Author

**Shiva Kumar Reddy Koppula**  
Graduate Student – Business Analytics  
University of Texas at Dallas  
GitHub: [@shivakrdy](https://github.com/shivakrdy)

---

## Future Improvements

- SHAP/LIME for model interpretability
- Streamlit-based dashboard for business users
- Full CI/CD pipeline for automated model updates

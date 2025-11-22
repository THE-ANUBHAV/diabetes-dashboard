# 🩺 Diabetes Prediction Dashboard

An interactive **Streamlit Dashboard** for training, comparing, and visualizing multiple machine learning models (RandomForest, XGBoost, LightGBM, SVM, Logistic Regression) for **Diabetes Prediction**.

---

## 🔗 Live App  
👉 **[Click here to open the Diabetes Dashboard](https://diabetes-dashboard-ml.streamlit.app/)**

---

## 🚀 Features

- **Notebook-Mode**  
  High-accuracy mode including *glyhb* features.

- **Safe-Mode**  
  Leakage-free evaluation for real-world reliability.

- **Model Comparison Visuals**  
  - ROC Curves  
  - Confusion Matrices  
  - Radar Charts  

- **Explainability (XAI)**  
  SHAP visualizations for supported models.

- **Interactive Prediction Form**  
  Predict diabetes probability based on user input.

---

## 🔧 How to Run Locally

```bash
pip install -r requirements.txt
streamlit run streamlit_diabetes_dashboard.py

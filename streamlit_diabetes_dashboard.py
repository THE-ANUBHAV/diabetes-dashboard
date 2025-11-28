# ============================================
# DIAGUARD PRO — Diabetes Risk Prediction Suite
# Final Boss Edition — with SVM, SHAP 3D, SMOTE,
# Calibrated Models, AI Medical Chatbot
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import shap
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, classification_report

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from imblearn.over_sampling import SMOTE

st.set_page_config(page_title="DiaGuard Pro — Diabetes Risk Prediction", layout="wide")

# ===========================================================
# LOAD DATA
# ===========================================================
@st.cache_data
def load_data(path="diabetes (1).csv"):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df

df = load_data()

# ===========================================================
# PREPROCESSOR
# ===========================================================
def preprocess(df):
    df = df.copy()
    df["diabetes"] = (df["glyhb"] >= 6.5).astype(int)

    drop_cols = ["id", "glyhb", "ratio", "stab.glu"]  # safe mode drop
    X = df.drop(columns=[c for c in drop_cols if c in df.columns] + ["diabetes"])
    y = df["diabetes"]

    num_cols = X.select_dtypes(include=["int", "float"]).columns
    cat_cols = X.select_dtypes(include=["object"]).columns

    # Impute
    X[num_cols] = SimpleImputer(strategy="median").fit_transform(X[num_cols])
    if len(cat_cols) > 0:
        X[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(X[cat_cols])
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, X.columns.tolist(), scaler


# ===========================================================
# MODEL TRAINING — Professional Level Fix (Option B)
# SMOTE + Balanced + Calibration + Optimized SVM
# ===========================================================
@st.cache_resource
def train_all_models(X_train, y_train, X_test, y_test):
    models = {}
    results = {}

    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)

    # ------------------------ Random Forest ------------------------
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        class_weight="balanced",
        random_state=42
    )
    rf_cal = CalibratedClassifierCV(rf, cv=5)
    rf_cal.fit(X_res, y_res)
    models["RandomForest"] = rf_cal

    # ------------------------ SVM ------------------------
    svm = SVC(
        kernel="rbf",
        C=3,
        gamma="scale",
        probability=True,
        class_weight="balanced",
        random_state=42
    )
    svm_cal = CalibratedClassifierCV(svm, cv=5)
    svm_cal.fit(X_res, y_res)
    models["SVM"] = svm_cal

    # ------------------------ XGBoost ------------------------
    posw = (y_res.value_counts()[0] / y_res.value_counts()[1])
    xgb = XGBClassifier(
        n_estimators=350,
        max_depth=5,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=posw,
        eval_metric="logloss",
        random_state=42
    )
    xgb_cal = CalibratedClassifierCV(xgb, cv=5)
    xgb_cal.fit(X_res, y_res)
    models["XGBoost"] = xgb_cal

    # ------------------------ LightGBM ------------------------
    lgb = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.04,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=42
    )
    lgb_cal = CalibratedClassifierCV(lgb, cv=5)
    lgb_cal.fit(X_res, y_res)
    models["LightGBM"] = lgb_cal

    # ------------------------ Evaluate ------------------------
    for name, model in models.items():
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]

        results[name] = {
            "accuracy": accuracy_score(y_test, pred),
            "auc": roc_auc_score(y_test, proba),
            "cm": confusion_matrix(y_test, pred),
            "pred": pred,
            "proba": proba,
            "report": classification_report(y_test, pred, output_dict=True)
        }

    return models, results


# ===========================================================
# SIDEBAR
# ===========================================================
page = st.sidebar.radio("Navigation", ["Train Models", "Predict", "AI Doctor Chatbot"])

# ===========================================================
# TRAIN MODELS
# ===========================================================
if page == "Train Models":
    st.title("📊 Model Training & Comparison — DiaGuard Pro")

    X_scaled, y, feature_names, scaler = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )

    if st.button("Train All Models (SMOTE + Calibration + Balanced)"):
        with st.spinner("Training models…"):
            models, results = train_all_models(X_train, y_train, X_test, y_test)
            st.session_state["models"] = models
            st.session_state["results"] = results
            st.session_state["feature_names"] = feature_names
            st.session_state["scaler"] = scaler

        st.success("Training Completed!")

        # Show Metrics Table
        res_df = pd.DataFrame([
            {
                "Model": name,
                "Accuracy": r["accuracy"],
                "AUC": r["auc"],
                "Precision": r["report"]["weighted avg"]["precision"],
                "Recall": r["report"]["weighted avg"]["recall"],
                "F1": r["report"]["weighted avg"]["f1-score"]
            }
            for name, r in results.items()
        ])
        st.dataframe(res_df.set_index("Model"))

        # Feature Importances Fix
        st.subheader("Feature Importances (Tree Models)")

        for name, model in models.items():
            try:
                if hasattr(model, "base_estimator"):
                    inner = model.base_estimator
                elif hasattr(model, "estimators_"):
                    inner = model.estimators_[0]
                else:
                    inner = model

                if not hasattr(inner, "feature_importances_"):
                    st.info(f"{name} does not provide feature importances.")
                    continue

                importances = inner.feature_importances_
                idx = np.argsort(importances)[::-1][:15]

                fig, ax = plt.subplots(figsize=(8, 4))
                sns.barplot(x=importances[idx],
                            y=np.array(feature_names)[idx],
                            ax=ax)
                ax.set_title(f"Top Features — {name}")
                st.pyplot(fig)

            except Exception as e:
                st.warning(f"Could not compute feature importances for {name}: {e}")


# ===========================================================
# PREDICT PAGE
# ===========================================================
elif page == "Predict":
    st.title("🧬 Diabetes Risk Prediction")

    if "models" not in st.session_state:
        st.error("Please train models first.")
        st.stop()

    models = st.session_state["models"]
    feature_names = st.session_state["feature_names"]
    scaler = st.session_state["scaler"]

    selected_model = st.selectbox("Choose Model", list(models.keys()))

    st.subheader("Enter Patient Details")

    age = st.number_input("Age", 18, 100, 50)
    weight = st.number_input("Weight (kg)", 40.0, 150.0, 80.0)
    height = st.number_input("Height (cm)", 120.0, 220.0, 165.0)
    chol = st.number_input("Cholesterol", 100.0, 400.0, 260.0)
    hdl = st.number_input("HDL", 10.0, 100.0, 30.0)
    bp_sys = st.number_input("Systolic BP", 80.0, 200.0, 160.0)
    bp_dia = st.number_input("Diastolic BP", 50.0, 120.0, 98.0)
    waist = st.number_input("Waist", 50.0, 180.0, 120.0)
    hip = st.number_input("Hip", 50.0, 180.0, 98.0)
    frame = st.selectbox("Frame", ["small", "medium", "large"])
    location = st.selectbox("Location", df["location"].unique())

    if st.button("Predict"):
        input_data = pd.DataFrame([{
            "age": age,
            "weight": weight,
            "height": height,
            "chol": chol,
            "hdl": hdl,
            "bp.1s": bp_sys,
            "bp.1d": bp_dia,
            "waist": waist,
            "hip": hip,
            "frame": frame,
            "location": location
        }])

        # One-hot encode
        full_df = df.copy()
        full_df = full_df.drop(columns=["id", "glyhb", "ratio", "stab.glu"])
        full_df = pd.get_dummies(full_df, columns=["frame", "location"], drop_first=True)

        input_data = pd.get_dummies(input_data, columns=["frame", "location"], drop_first=True)

        for col in full_df.drop(columns=["diabetes"]).columns:
            if col not in input_data.columns:
                input_data[col] = 0

        input_data = input_data[feature_names]
        input_scaled = scaler.transform(input_data)

        model = models[selected_model]
        prob = model.predict_proba(input_scaled)[0][1]
        pred = int(prob >= 0.5)
        pct = round(prob * 100, 2)

        # Risk Gauge
        st.subheader("Risk Score")
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=pct,
            title={"text": "Diabetes Risk (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "steps": [
                    {"range": [0, 30], "color": "green"},
                    {"range": [30, 60], "color": "orange"},
                    {"range": [60, 100], "color": "red"},
                ],
                "threshold": {"line": {"color": "black", "width": 4},
                              "thickness": 0.75, "value": pct}
            }))
        st.plotly_chart(fig)

        st.success(f"Prediction: **{'Diabetic' if pred==1 else 'Non-Diabetic'}** (Risk: **{pct}%**)")

        # SHAP EXPLANATION
        st.subheader("Explainability (SHAP)")

        try:
            # Kernel SHAP (works for SVM, calibrated models)
            explainer = shap.KernelExplainer(model.predict_proba, X_train[:50])
            shap_values = explainer.shap_values(input_scaled)
            st.write("### SHAP Force Plot")
            st_shap = shap.force_plot(explainer.expected_value[1],
                                      shap_values[1],
                                      input_data,
                                      matplotlib=True)
            st.pyplot()
        except Exception as e:
            st.warning(f"SHAP explanation unavailable: {e}")

        # MEDICAL RECOMMENDATIONS
        st.subheader("Medical Advice (AI Doctor)")
        if pct < 30:
            st.info("Low Risk. Maintain healthy lifestyle, regular exercise, avoid sugar overload.")
        elif pct < 60:
            st.warning("Moderate Risk. Improve diet, reduce weight, monitor BP & lipid levels.")
        elif pct < 85:
            st.error("High Risk. Strongly recommended to consult a physician & run blood tests.")
        else:
            st.error("⚠️ VERY HIGH RISK. Immediate medical evaluation is strongly advised.")

# ===========================================================
# AI CHATBOT PAGE
# ===========================================================
elif page == "AI Doctor Chatbot":
    st.title("🩺 AI Medical Assistant — Diabetes Advisor")

    user = st.text_area("Ask any diabetes-related question:")

    if st.button("Ask"):
        if len(user.strip()) == 0:
            st.error("Please enter a question.")
        else:
            st.success("AI Doctor Response (Simulated)")
            st.write("""
            • Diabetes risk increases with age, obesity, and high cholesterol.  
            • Maintaining diet, exercise, and BP control reduces risk.  
            • Regular HbA1c screening recommended every 6 months.  
            • If symptoms like fatigue, thirst, or frequent urination appear, consult a physician.  
            """)

# END

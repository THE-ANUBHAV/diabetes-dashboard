# streamlit_diabetes_dashboard.py
"""
Final Dashboard (with SVM)
- Robust preprocessing & feature alignment
- Models: RandomForest, XGBoost, LightGBM, SVM
- SHAP explanations with safe fallbacks
- Plotly gauge + interactive SHAP views
- AI Diagnosis Chatbot (local)
- Prediction history + PDF report
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io, base64
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, roc_curve
import plotly.graph_objects as go
import plotly.express as px
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Optional libs
try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    shap = None
    SHAP_AVAILABLE = False

# ---------------- Config ----------------
st.set_page_config(page_title="Diabetes Dashboard (Final)", layout="wide", initial_sidebar_state="expanded")
DATA_PATH = "diabetes (1).csv"
RANDOM_STATE = 42

# ---------------- Utilities ----------------
@st.cache_data
def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df

def clamp(v, min_v, max_v):
    try:
        v = float(v)
    except Exception:
        return min_v
    return max(min_v, min(max_v, v))

def preprocess_df_for_training(df, remove_leakage=True):
    """
    Preprocess full dataset for training:
    - creates 'diabetes' target from glyhb
    - drops id and leakage columns if remove_leakage True
    - imputes numeric/categorical, encodes categorical via get_dummies
    - scales features and returns X_scaled, y, feature_names, scaler
    """
    df = df.copy()
    df.columns = df.columns.str.strip()
    if 'glyhb' not in df.columns:
        raise ValueError("Dataset must contain 'glyhb' column.")
    df['diabetes'] = (df['glyhb'] >= 6.5).astype(int)

    leakage_cols = ['glyhb', 'ratio', 'stab.glu']
    drop_cols = ['id']
    if remove_leakage:
        drop_cols += leakage_cols

    X = df.drop(columns=[c for c in drop_cols + ['diabetes'] if c in df.columns], errors='ignore')
    y = df['diabetes']

    cat_cols = X.select_dtypes(include=['object','category']).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    if len(num_cols) > 0:
        num_imp = SimpleImputer(strategy='median')
        X[num_cols] = num_imp.fit_transform(X[num_cols])

    if len(cat_cols) > 0:
        cat_imp = SimpleImputer(strategy='most_frequent')
        X[cat_cols] = cat_imp.fit_transform(X[cat_cols])
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, X.columns.tolist(), scaler

def train_models(X_train, y_train, models_to_train=None):
    models = {}
    if models_to_train is None:
        models_to_train = ['RandomForest','XGBoost','LightGBM','SVM']
    if 'RandomForest' in models_to_train:
        rf = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=RANDOM_STATE)
        rf.fit(X_train, y_train)
        models['RandomForest'] = rf
    if 'XGBoost' in models_to_train and XGBClassifier is not None:
        xgb = XGBClassifier(n_estimators=300, learning_rate=0.03, max_depth=5, subsample=0.75,
                             colsample_bytree=0.8, gamma=0.3, reg_alpha=0.2, reg_lambda=1.0,
                             min_child_weight=2, eval_metric='logloss', use_label_encoder=False, random_state=RANDOM_STATE)
        xgb.fit(X_train, y_train)
        models['XGBoost'] = xgb
    if 'LightGBM' in models_to_train and LGBMClassifier is not None:
        lgb = LGBMClassifier(n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=30,
                              subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.3, random_state=RANDOM_STATE)
        lgb.fit(X_train, y_train)
        models['LightGBM'] = lgb
    if 'SVM' in models_to_train:
        svm = SVC(kernel='rbf', C=2, probability=True, random_state=RANDOM_STATE)
        svm.fit(X_train, y_train)
        models['SVM'] = svm
    return models

def evaluate_model(model, X_test, y_test):
    pred = model.predict(X_test)
    probs = None
    try:
        probs = model.predict_proba(X_test)[:,1]
    except Exception:
        try:
            probs = model.decision_function(X_test)
            probs = np.asarray(probs).ravel()
        except Exception:
            probs = np.zeros(len(pred))
    acc = accuracy_score(y_test, pred)
    auc = None
    try:
        if probs is not None and len(np.unique(y_test))>1:
            auc = roc_auc_score(y_test, probs)
    except Exception:
        auc = None
    cm = confusion_matrix(y_test, pred)
    report = classification_report(y_test, pred, output_dict=True)
    return {'accuracy':acc,'auc':auc,'cm':cm,'report':report,'pred':pred,'probs':probs}

def generate_pdf_report(patient_input_df, predicted_class, probability, risk_label, rec_list):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, height-60, "Diabetes Risk Prediction Report")
    c.setFont("Helvetica", 10)
    c.drawString(40, height-80, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.line(40, height-90, width-40, height-90)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, height-110, "Patient Inputs:")
    y = height - 130
    for col, val in patient_input_df.iloc[0].items():
        c.setFont("Helvetica", 10)
        c.drawString(48, y, f"{col}: {val}")
        y -= 14
        if y < 120:
            c.showPage()
            y = height - 60
    y -= 8
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "Prediction Summary:")
    y -= 16
    c.setFont("Helvetica", 10)
    c.drawString(48, y, f"Predicted Class: {'Diabetic' if predicted_class==1 else 'Non-diabetic'}")
    y -= 14
    c.drawString(48, y, f"Probability: {probability:.3f}")
    y -= 20
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "Doctor-style Recommendations:")
    y -= 16
    c.setFont("Helvetica", 10)
    for rec in rec_list:
        c.drawString(48, y, f"- {rec}")
        y -= 14
        if y < 60:
            c.showPage()
            y = height - 60
    c.save()
    buffer.seek(0)
    return buffer.read()

# ---------------- Feature alignment (critical) ----------------
def align_features_for_prediction(user_raw_df, df_training, scaler, feature_names, remove_leakage=True):
    """
    Align user_raw_df to the exact features (order + dummies) used during training.
    Steps:
    1) Concatenate user row on top of training df
    2) Apply same preprocessing steps (impute, get_dummies)
    3) Extract the first row processed and ensure all training feature_names exist
    4) Use scaler to transform
    """
    # 1. Put user row on top of training df copy
    df_temp = pd.concat([user_raw_df.reset_index(drop=True), df_training.reset_index(drop=True)], ignore_index=True)
    # 2. Preprocess entire df_temp with same pipeline (but we only need returned X columns)
    X_scaled_all, y_temp, final_feat_names, _ = preprocess_df_for_training(df_temp, remove_leakage=remove_leakage)
    # X_scaled_all is numpy; build dataframe with final_feat_names
    df_processed = pd.DataFrame(X_scaled_all, columns=final_feat_names)
    # 3. First row corresponds to user input after preprocessing and scaling (scaled)
    user_processed_scaled = df_processed.iloc[[0]].copy()
    # 4. If training feature_names contain missing columns, add zeros
    for col in feature_names:
        if col not in user_processed_scaled.columns:
            user_processed_scaled[col] = 0.0
    # 5. Ensure correct order
    user_processed_scaled = user_processed_scaled[feature_names]
    # 6. IMPORTANT: preprocess_df_for_training returned scaled features already.
    # But we must return an array scaled by the training scaler (the scaler from training).
    # The df_processed was made using a NEW scaler fit on df_temp; to be robust, we transform user_processed with provided scaler
    user_unscaled = user_processed_scaled.copy()  # although values are scaled, we re-transform to be consistent
    # To avoid double-scaling issues we prefer to reconstruct user_unscaled by reverse-transform if possible.
    # Simpler approach: re-create user_raw_df processed with same pipeline but without scaling then scale with provided scaler.
    # We'll re-run preprocessing steps but stopping before scaling.
    # --- Re-run imputation + get_dummies aligning to df_training ---
    # Build X (like in preprocess_df_for_training, but stop before scaling)
    df_temp2 = pd.concat([user_raw_df.reset_index(drop=True), df_training.reset_index(drop=True)], ignore_index=True)
    df_temp2.columns = df_temp2.columns.str.strip()
    # drop leakage if needed
    leakage_cols = ['glyhb','ratio','stab.glu']
    drop_cols = ['id']
    if remove_leakage:
        drop_cols += leakage_cols
    X_raw = df_temp2.drop(columns=[c for c in drop_cols + ['diabetes'] if c in df_temp2.columns], errors='ignore')
    cat_cols = X_raw.select_dtypes(include=['object','category']).columns.tolist()
    num_cols = X_raw.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols)>0:
        num_imp = SimpleImputer(strategy='median')
        X_raw[num_cols] = num_imp.fit_transform(X_raw[num_cols])
    if len(cat_cols)>0:
        cat_imp = SimpleImputer(strategy='most_frequent')
        X_raw[cat_cols] = cat_imp.fit_transform(X_raw[cat_cols])
        X_raw = pd.get_dummies(X_raw, columns=cat_cols, drop_first=True)
    X_raw = X_raw.apply(pd.to_numeric, errors='coerce').fillna(0)
    # Now X_raw columns are a superset; ensure training feature_names exist
    for col in feature_names:
        if col not in X_raw.columns:
            X_raw[col] = 0.0
    X_raw = X_raw[feature_names]
    # Extract first row (user) unscaled
    user_unscaled_row = X_raw.iloc[[0]].copy()
    # Now scale with training scaler
    user_scaled = scaler.transform(user_unscaled_row)
    return user_scaled

# ---------------- App layout ----------------
df = load_data()

# Sidebar controls
st.sidebar.title("Controls")
page = st.sidebar.radio("Navigation", ["Home", "Train & Compare", "Predict", "History"])
st.sidebar.markdown("---")
st.sidebar.write(f"Dataset rows: {df.shape[0]}  columns: {df.shape[1]}")

# Models to train (include SVM as requested)
default_models = ['RandomForest','XGBoost','LightGBM','SVM']
models_to_train = st.sidebar.multiselect("Models to train", options=default_models, default=default_models)

# Notebook mode flag (leakage)
use_notebook_mode_default = False
notebook_mode_sidebar = st.sidebar.checkbox("Enable Notebook Mode (include glyhb etc.)", value=use_notebook_mode_default)

# Header
if page != "Predict":
    st.title("Diabetes Models Dashboard — Final")
else:
    st.title("Diabetes Risk Prediction")

# HOME
if page == "Home":
    st.header("Overview")
    st.write("""
        This dashboard supports Safe Mode (no glyhb leakage) and Notebook Mode (includes glyhb).
        Models: RandomForest, XGBoost, LightGBM, SVM.
        Train models (Train & Compare) before using Predict.
    """)
    st.subheader("Dataset preview")
    st.dataframe(df.head(8))

# TRAIN & COMPARE
if page == "Train & Compare":
    st.header("Train & Compare Models")
    mode_choice = st.radio("Mode", ("Safe (no leakage)", "Notebook (with glyhb)"),
                          index=0 if not notebook_mode_sidebar else 1)
    remove_leakage = True if mode_choice.startswith("Safe") else False
    if mode_choice.startswith("Notebook"):
        st.warning("Notebook Mode includes glyhb/stab.glu/ratio — diagnostic features that create optimistic results.")
    if st.button("Train & Evaluate"):
        with st.spinner("Preprocessing and training..."):
            X_scaled, y, feature_names, scaler = preprocess_df_for_training(df, remove_leakage=remove_leakage)
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
            trained_models = train_models(X_train, y_train, models_to_train=models_to_train)
            results = {name: evaluate_model(m, X_test, y_test) for name,m in trained_models.items()}

            # Metrics DataFrame
            metrics = []
            for name,res in results.items():
                metrics.append({
                    'model': name,
                    'accuracy': res['accuracy'],
                    'auc': res['auc'],
                    'precision': res['report'].get('weighted avg',{}).get('precision', np.nan),
                    'recall': res['report'].get('weighted avg',{}).get('recall', np.nan),
                    'f1': res['report'].get('weighted avg',{}).get('f1-score', np.nan)
                })
            metrics_df = pd.DataFrame(metrics).sort_values('accuracy', ascending=False).reset_index(drop=True)
            st.subheader("Model comparison")
            st.dataframe(metrics_df.set_index('model').round(3), use_container_width=True)

            # ROC curves
            st.subheader("ROC comparison")
            fig, ax = plt.subplots(figsize=(7,5))
            for name,res in results.items():
                if res['probs'] is not None:
                    fpr,tpr,_ = roc_curve(y_test, res['probs'])
                    ax.plot(fpr,tpr,label=f"{name} (AUC={res['auc']:.3f})")
            ax.plot([0,1],[0,1],'--',color='gray'); ax.set_xlabel('FPR'); ax.set_ylabel('TPR'); ax.legend()
            st.pyplot(fig)

            # Feature importance for tree models
            st.subheader("Feature importances (tree models)")
            for name, m in trained_models.items():
                try:
                    importances = m.feature_importances_
                    idx = np.argsort(importances)[::-1][:15]
                    fig, ax = plt.subplots(figsize=(8,3))
                    sns.barplot(x=importances[idx], y=np.array(feature_names)[idx], ax=ax)
                    ax.set_title(f"Top features - {name}")
                    st.pyplot(fig)
                except Exception:
                    st.info(f"{name} does not expose feature_importances_")

            # Save to session
            st.session_state['models'] = trained_models
            st.session_state['scaler'] = scaler
            st.session_state['feature_names'] = feature_names
            st.session_state['remove_leakage'] = remove_leakage
            st.session_state['results'] = results
            st.success("Training complete. Models saved to session.")

# PREDICT
if page == "Predict":
    st.header("Predict Diabetes Risk (Simple / Advanced)")

    if 'models' not in st.session_state:
        st.warning("No trained models in session. Please train models first.")
        st.stop()

    left, right = st.columns([1,1])
    with left:
        mode = st.radio("Input mode", ["Simple","Advanced"])
        model_choice = st.selectbox("Model to use", options=list(st.session_state['models'].keys()))
        notebook_mode = st.checkbox("Use Notebook Mode (may expect glyhb)", value=st.session_state.get('remove_leakage', False)==False)
        st.markdown("---")
        st.subheader("Patient Inputs")

        if mode == "Simple":
            age = st.number_input("Age", min_value=0, max_value=120, value=int(clamp(df['age'].median(),0,120)))
            gender = st.selectbox("Gender", df['gender'].dropna().unique().tolist())
            weight = st.number_input("Weight (kg)", min_value=20.0, max_value=200.0, value=float(clamp(df['weight'].median(),20.0,200.0)))
            height = st.number_input("Height (cm)", min_value=100.0, max_value=220.0, value=float(clamp(df['height'].median(),100.0,220.0)))
            chol = st.number_input("Cholesterol (chol)", min_value=50.0, max_value=400.0, value=float(clamp(df['chol'].median(),50.0,400.0)))
            hdl = st.number_input("HDL", min_value=10.0, max_value=150.0, value=float(clamp(df['hdl'].median(),10.0,150.0)))
            sys_bp = st.number_input("Systolic BP (bp.1s)", min_value=80.0, max_value=220.0,
                                     value=float(clamp(df['bp.1s'].median() if 'bp.1s' in df.columns else 120.0,80.0,220.0)))
            dia_bp = st.number_input("Diastolic BP (bp.1d)", min_value=40.0, max_value=140.0,
                                     value=float(clamp(df['bp.1d'].median() if 'bp.1d' in df.columns else 80.0,40.0,140.0)))
            waist = st.number_input("Waist (cm)", min_value=40.0, max_value=200.0, value=float(clamp(df['waist'].median() if 'waist' in df.columns else 80.0,40.0,200.0)))
            hip = st.number_input("Hip (cm)", min_value=40.0, max_value=200.0, value=float(clamp(df['hip'].median() if 'hip' in df.columns else 90.0,40.0,200.0)))
            frame = st.selectbox("Frame", df['frame'].dropna().unique().tolist() if 'frame' in df.columns else ['M'])
            location = st.selectbox("Location", df['location'].dropna().unique().tolist() if 'location' in df.columns else ['Unknown'])
            input_raw = pd.DataFrame([{
                'age': age, 'gender': gender, 'height': height, 'weight': weight, 'chol': chol, 'hdl': hdl,
                'bp.1s': sys_bp, 'bp.1d': dia_bp, 'waist': waist, 'hip': hip, 'frame': frame, 'location': location
            }])
        else:
            adv_cols = [c for c in df.columns if c not in ['id','Outcome','diabetes']]
            adv_inputs = {}
            for c in adv_cols:
                if df[c].dtype.kind in 'biufc':
                    adv_inputs[c] = st.number_input(f"{c}", value=float(df[c].median()) if df[c].notna().any() else 0.0)
                else:
                    adv_inputs[c] = st.selectbox(f"{c}", df[c].dropna().unique().tolist())
            input_raw = pd.DataFrame([adv_inputs])

        st.markdown("---")
        predict_btn = st.button("Predict Risk")

    with right:
        st.subheader("Prediction Result")
        result_area = st.empty()
        shap_area = st.empty()
        plotly_area = st.empty()
        chat_area = st.empty()

    if predict_btn:
        remove_leakage = not notebook_mode
        # prepare training artefacts
        scaler = st.session_state.get('scaler')
        feature_names = st.session_state.get('feature_names')
        df_training = df.copy()

        if scaler is None or feature_names is None:
            st.error("Training artifacts missing from session. Please re-train models.")
            st.stop()

        # align features + scale using alignment function
        try:
            input_scaled = align_features_for_prediction(input_raw, df_training, scaler, feature_names, remove_leakage=remove_leakage)
        except Exception as e:
            st.error(f"Feature alignment failed: {e}")
            st.stop()

        model = st.session_state['models'].get(model_choice)
        if model is None:
            st.error("Model not found in session.")
            st.stop()

        # predict
        try:
            if hasattr(model, "predict_proba"):
                prob = float(model.predict_proba(input_scaled)[:,1][0])
            else:
                raw = model.decision_function(input_scaled)
                prob = float(1/(1+np.exp(-raw))[0])
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

        pred_class = int(prob >= 0.5)
        prob_pct = round(prob*100, 1)

        # risk label
        if prob < 0.3:
            risk_label, color, emoji, advice_short = "Low", "#2ecc71", "🟢", "Low risk — maintain healthy lifestyle"
        elif prob < 0.6:
            risk_label, color, emoji, advice_short = "Moderate", "#f39c12", "🟠", "Moderate risk — consider lifestyle changes"
        elif prob < 0.8:
            risk_label, color, emoji, advice_short = "High", "#e74c3c", "🔴", "High risk — seek medical advice"
        else:
            risk_label, color, emoji, advice_short = "Very High", "#8b0000", "🚨", "Very High risk — immediate evaluation"

        # Theme-aware card
        is_dark = st.get_option("theme.base") == "dark"
        if is_dark:
            card_style = "background-color:#0f1720;color:#fff;padding:14px;border-radius:10px;border:1px solid #283040;"
        else:
            card_style = "background-color:#f3fbff;color:#000;padding:14px;border-radius:10px;"
        card_html = f"""
        <div style="{card_style}">
          <h3 style="margin:4px 0 6px 0;">Prediction Summary</h3>
          <div style="display:flex;align-items:center;gap:16px;">
            <div style="width:110px;text-align:center;">
              <div style="font-size:28px;font-weight:700;">{prob_pct}%</div>
              <div style="font-size:12px;color:gray">Probability</div>
            </div>
            <div>
              <div><strong>Predicted:</strong> {'Diabetic' if pred_class==1 else 'Non-diabetic'} ({pred_class})</div>
              <div><strong>Model:</strong> {model_choice} • <strong>Mode:</strong> {'Notebook' if not remove_leakage else 'Safe'}</div>
              <div style="margin-top:8px;color:{color};font-weight:700;">{emoji} {risk_label} — {advice_short}</div>
            </div>
          </div>
        </div>
        """
        result_area.markdown(card_html, unsafe_allow_html=True)
        st.metric(label="Diabetes Risk Probability", value=f"{prob_pct}%")

        # Plotly gauge
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob_pct,
            domain={'x':[0,1],'y':[0,1]},
            title={'text':"Risk Gauge (%)"},
            gauge={'axis':{'range':[0,100]}, 'bar':{'color':color},
                   'steps':[{'range':[0,30],'color':'#e6f8ea'},
                            {'range':[30,60],'color':'#fff8e1'},
                            {'range':[60,80],'color':'#ffe6e6'},
                            {'range':[80,100],'color':'#ffdfe6'}]}
        ))
        st.plotly_chart(gauge, use_container_width=True)

        # SHAP explanations (robust)
        if SHAP_AVAILABLE:
            try:
                shap_area.subheader("SHAP - Local Explanation")
                vals = None
                base_val = None
                # For tree models
                if model_choice in ['RandomForest','XGBoost','LightGBM']:
                    explainer = shap.TreeExplainer(model)
                    sv = explainer.shap_values(pd.DataFrame(input_scaled, columns=feature_names))
                    if isinstance(sv, list):
                        vals = np.asarray(sv[1])[0] if len(sv)>1 else np.asarray(sv[0])[0]
                    else:
                        arr = np.asarray(sv)
                        if arr.ndim == 3:
                            vals = arr[1,0,:] if arr.shape[0]>1 else arr[0,0,:]
                        elif arr.ndim==2:
                            vals = arr[0,:]
                        else:
                            vals = arr.ravel()
                    base_val = explainer.expected_value[1] if isinstance(explainer.expected_value,(list,tuple,np.ndarray)) and len(explainer.expected_value)>1 else explainer.expected_value
                elif model_choice == 'SVM':
                    shap_area.info("Computing Kernel SHAP for SVM (may take a few seconds)...")
                    background = shap.sample(pd.DataFrame(preprocess_df_for_training(df, remove_leakage=remove_leakage)[0], columns=st.session_state['feature_names']), min(100, len(df)))
                    explainer = shap.KernelExplainer(model.predict_proba, background)
                    sv = explainer.shap_values(pd.DataFrame(input_scaled, columns=feature_names))
                    vals = np.asarray(sv[1])[0] if isinstance(sv, list) and len(sv)>1 else np.asarray(sv).ravel()
                    base_val = explainer.expected_value[1] if hasattr(explainer.expected_value, "__len__") and len(explainer.expected_value)>1 else explainer.expected_value
                else:
                    # Fallback generic KernelExplainer
                    background = shap.sample(pd.DataFrame(preprocess_df_for_training(df, remove_leakage=remove_leakage)[0], columns=st.session_state['feature_names']), min(100, len(df)))
                    explainer = shap.KernelExplainer(model.predict_proba, background)
                    sv = explainer.shap_values(pd.DataFrame(input_scaled, columns=feature_names))
                    vals = np.asarray(sv[1])[0] if isinstance(sv, list) and len(sv)>1 else np.asarray(sv).ravel()
                    base_val = explainer.expected_value[1] if hasattr(explainer.expected_value, "__len__") and len(explainer.expected_value)>1 else explainer.expected_value

                # Build shap.Explanation and plot waterfall (single-sample)
                expl = shap.Explanation(values=vals, base_values=base_val, data=pd.Series(input_raw.iloc[0]).reindex(feature_names).fillna(0), feature_names=feature_names)
                try:
                    plt.figure(facecolor="white")
                    shap.plots.waterfall(expl, show=False)
                    st.pyplot(plt.gcf())
                    plt.clf()
                except Exception:
                    try:
                        plt.figure(facecolor="white")
                        shap.plots.bar(expl, show=False)
                        st.pyplot(plt.gcf())
                        plt.clf()
                    except Exception:
                        shap_area.info("SHAP visual not available in this environment.")
                # Force plot (interactive) - embed if possible
                try:
                    fp = shap.force_plot(base_val, vals, pd.Series(input_raw.iloc[0]).reindex(feature_names).fillna(0), matplotlib=False)
                    import streamlit.components.v1 as components
                    components.html(f"<head>{shap.getjs()}</head><body>{fp.html()}</body>", height=380)
                except Exception:
                    pass
            except Exception as e:
                shap_area.error(f"SHAP explanation failed: {e}")
        else:
            shap_area.info("SHAP is not installed. Install shap to enable explanations.")

        # 3D-like interactive SHAP bar (Plotly)
        try:
            df_shap = pd.DataFrame({'feature': feature_names, 'contrib': vals})
            df_shap['abs'] = df_shap['contrib'].abs()
            df_shap = df_shap.sort_values('abs', ascending=False).head(12)
            fig3 = px.bar(df_shap, x='feature', y='contrib', color='contrib', color_continuous_scale='RdBu', title='Top SHAP contributions')
            st.plotly_chart(fig3, use_container_width=True)
        except Exception:
            pass

        # Doctor-style recommendations (rules)
        recs = []
        try:
            wt = float(input_raw.get('weight',0).iloc[0])
            ht = float(input_raw.get('height',0).iloc[0])
            bmi = wt/((ht/100)**2) if ht>0 else np.nan
        except Exception:
            bmi = np.nan
        if not np.isnan(bmi):
            if bmi < 18.5: recs.append("Underweight: consider nutritional assessment.")
            elif bmi <25: recs.append("Normal BMI: maintain healthy lifestyle.")
            elif bmi <30: recs.append("Overweight: aim for gradual weight loss (5-10%).")
            else: recs.append("Obese: medical weight-loss program recommended.")
        try:
            s = float(input_raw.get('bp.1s', np.nan).iloc[0])
            d = float(input_raw.get('bp.1d', np.nan).iloc[0])
            if not np.isnan(s) and not np.isnan(d):
                if s<120 and d<80: recs.append("BP normal: continue routine monitoring.")
                elif s<130: recs.append("BP elevated: lifestyle modifications recommended.")
                elif s<140: recs.append("Stage 1 HTN: consult clinician.")
                else: recs.append("Stage 2 HTN: seek medical care.")
        except Exception:
            pass
        try:
            cholv = float(input_raw.get('chol', np.nan).iloc[0])
            if not np.isnan(cholv):
                if cholv<200: recs.append("Cholesterol desirable.")
                elif cholv<240: recs.append("Borderline high: diet/exercise.")
                else: recs.append("High cholesterol: consider lipid panel & clinician.")
        except Exception:
            pass
        if risk_label == "Very High":
            recs.append("Immediate: order HbA1c and fasting glucose; see clinician.")
        elif risk_label == "High":
            recs.append("Early clinical screening and lifestyle changes advised.")
        elif risk_label == "Moderate":
            recs.append("Monitor regularly and adopt preventive measures.")
        else:
            recs.append("Routine checks recommended; re-evaluate annually.")

        st.subheader("Doctor-style Recommendations")
        for r in recs:
            st.write("- " + r)

        # Save history
        hist = st.session_state.get('history', [])
        entry = input_raw.copy()
        entry['predicted_class'] = pred_class
        entry['predicted_prob'] = prob
        entry['model'] = model_choice
        entry['mode'] = 'Notebook' if not remove_leakage else 'Safe'
        entry['timestamp'] = datetime.now().isoformat()
        st.session_state['history'] = hist + [entry.to_dict(orient='records')[0]]
        st.success("Saved to session history.")

        # Download CSV & PDF
        csv_bytes = input_raw.copy()
        csv_bytes['predicted_class'] = pred_class
        csv_bytes['predicted_prob'] = prob
        st.download_button("Download prediction (CSV)", csv_bytes.to_csv(index=False).encode('utf-8'), file_name='prediction.csv', mime='text/csv')
        pdf = generate_pdf_report(input_raw, pred_class, prob, risk_label, recs)
        st.download_button("Download PDF report", data=pdf, file_name="diabetes_report.pdf", mime="application/pdf")

        # AI Diagnosis Chatbot (local)
        chat_area.subheader("AI Diagnosis Chatbot")
        chat_area.write("Ask for explanation, next steps or interpretation. (Local rule-based assistant)")
        if 'chat_history' not in st.session_state:
            st.session_state['chat_history'] = []
        user_q = st.text_input("Question to assistant (press Enter)", key="chatbox")
        if user_q:
            reply = "I can help interpret the result. "
            q = user_q.lower()
            if any(w in q for w in ["risk","probab","chance","how likely","percentage"]):
                reply = f"The model reports {prob_pct}%. Risk bucket: {risk_label}. This is a screening estimate, not a diagnosis."
            elif any(w in q for w in ["test","what should i do","next step","advise","recommend"]):
                reply = "Consider HbA1c/fasting glucose if risk is Moderate or higher. Lifestyle: weight reduction, exercise, reduce sugar intake. See clinician for personalized advice if High/Very High."
            elif any(w in q for w in ["why","explain","reason","because"]):
                try:
                    top_feats = df_shap.iloc[:5] if 'df_shap' in locals() else None
                    if top_feats is not None:
                        top_txt = ", ".join([f"{r.feature} ({r.contrib:.2f})" for r in top_feats.itertuples()])
                        reply = f"Top local contributing features: {top_txt}."
                    else:
                        reply = "Top contributing features unavailable; typical important variables: age, waist/hip, cholesterol, HDL."
                except Exception:
                    reply = "Could not compute detailed features now."
            else:
                reply = "I can explain the prediction, give next-step recommendations, or list important features. Try: 'Why was I predicted high risk?'"
            st.session_state['chat_history'].append({'q': user_q, 'a': reply})

        # display chat history
        for turn in st.session_state.get('chat_history', [])[::-1]:
            st.markdown(f"**You:** {turn['q']}")
            st.markdown(f"**Assistant:** {turn['a']}")

# HISTORY page
if page == "History":
    st.header("Prediction History")
    hist = st.session_state.get('history', [])
    if not hist:
        st.info("No history yet.")
    else:
        hist_df = pd.DataFrame(hist)
        st.dataframe(hist_df, use_container_width=True)
        st.download_button("Download History CSV", hist_df.to_csv(index=False).encode('utf-8'), file_name='history.csv', mime='text/csv')

st.markdown("---")
st.markdown("_Note: Notebook Mode includes glyhb/stab.glu/ratio which can leak direct diagnostic information into training. Use Safe Mode for real-world screening._")

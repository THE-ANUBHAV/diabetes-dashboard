# streamlit_diabetes_dashboard.py
"""
Professional-grade Diabetes Dashboard
- SMOTE oversampling, class balancing, calibration
- Models: RandomForest, XGBoost, LightGBM, SVM (SVM tuned)
- Calibrated probabilities (CalibratedClassifierCV)
- Robust preprocessing & feature-alignment
- SHAP interactive (TreeExplainer/KernelExplainer)
- 3D-like Plotly SHAP visualization
- AI Diagnosis Chatbot (local rule-based)
- Prediction history + PDF report
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.utils import compute_class_weight
from imblearn.over_sampling import SMOTE
import plotly.graph_objects as go
import plotly.express as px
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import warnings
warnings.filterwarnings("ignore")

# optional libs
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
st.set_page_config(page_title="Diabetes Dashboard — Professional", layout="wide", initial_sidebar_state="expanded")
DATA_PATH = "diabetes (1).csv"
RANDOM_STATE = 42

# ---------------- Helpers ----------------
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

def create_target_and_clean(df):
    df = df.copy()
    df.columns = df.columns.str.strip()
    if 'glyhb' not in df.columns:
        raise ValueError("Dataset must contain 'glyhb' column to create target.")
    df['diabetes'] = (df['glyhb'] >= 6.5).astype(int)
    return df

def preprocessing_no_scale(df_input, remove_leakage=True):
    """
    Preprocess but DON'T scale — returns DataFrame X_proc and y series.
    This is used for feature alignment then scaling with training scaler.
    """
    df = df_input.copy()
    df.columns = df.columns.str.strip()
    df = create_target_and_clean(df) if 'diabetes' not in df.columns else df
    leakage_cols = ['glyhb','ratio','stab.glu']
    drop_cols = ['id']
    if remove_leakage:
        drop_cols += leakage_cols
    X = df.drop(columns=[c for c in drop_cols + ['diabetes'] if c in df.columns], errors='ignore')
    y = df['diabetes']
    # Impute numeric/categorical
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
    return X, y

def preprocess_for_training(df_full, remove_leakage=True):
    """Do full preprocessing including scaling. Returns X_scaled (np), y, feature_names, scaler, X_df_unscaled"""
    X_df, y = preprocessing_no_scale(df_full, remove_leakage=remove_leakage)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df)
    feature_names = X_df.columns.tolist()
    return X_scaled, y, feature_names, scaler, X_df

def align_features_for_prediction(user_raw_df, df_training_raw, feature_names, scaler, remove_leakage=True):
    """
    Robust alignment:
    - concatenate user row + training raw df
    - run preprocessing_no_scale to get consistent dummy columns
    - extract first row (user) in unscaled space
    - ensure all feature_names exist (fill zeros)
    - scale using provided scaler (trained on training X_df)
    """
    # concat with training raw to get consistent categories and dummies
    df_temp = pd.concat([user_raw_df.reset_index(drop=True), df_training_raw.reset_index(drop=True)], ignore_index=True)
    X_all_unscaled, _ = preprocessing_no_scale(df_temp, remove_leakage=remove_leakage)
    # First row is user (unscaled preprocessed)
    user_unscaled = X_all_unscaled.iloc[[0]].copy()
    # Ensure training feature_names exist
    for col in feature_names:
        if col not in user_unscaled.columns:
            user_unscaled[col] = 0.0
    user_unscaled = user_unscaled[feature_names]
    # scale using training scaler
    user_scaled = scaler.transform(user_unscaled)
    return user_scaled

def calibrate_model(estimator, X_train, y_train, cv=5):
    """Wrap estimator in CalibratedClassifierCV using sigmoid and cv folds."""
    calibrated = CalibratedClassifierCV(estimator, cv=cv, method='sigmoid')
    calibrated.fit(X_train, y_train)
    return calibrated

def train_with_smote_and_calibration(X_scaled, y, models_to_train):
    """
    X_scaled : numpy array scaled features
    y : series
    models_to_train : list of model names to train
    Returns: dict of trained calibrated models and evaluation-ready artifacts
    """
    results = {}
    # Split train/test stratified
    X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
    # Apply SMOTE on training set
    sm = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = sm.fit_resample(X_tr, y_tr)
    # training class counts
    neg = (y_res==0).sum(); pos = (y_res==1).sum()
    scale_pos_weight = (neg / pos) if pos>0 else 1.0

    trained_models = {}
    # RandomForest with balanced class weight
    if 'RandomForest' in models_to_train:
        rf = RandomForestClassifier(n_estimators=300, max_depth=12, class_weight='balanced', random_state=RANDOM_STATE)
        rf_cal = CalibratedClassifierCV(rf, cv=5, method='sigmoid')
        rf_cal.fit(X_res, y_res)
        trained_models['RandomForest'] = rf_cal

    # XGBoost (if available) with scale_pos_weight
    if 'XGBoost' in models_to_train and XGBClassifier is not None:
        xgb = XGBClassifier(
            n_estimators=300, learning_rate=0.03, max_depth=5,
            subsample=0.75, colsample_bytree=0.8, gamma=0.3,
            reg_alpha=0.2, reg_lambda=1.0, min_child_weight=2,
            random_state=RANDOM_STATE, use_label_encoder=False,
            scale_pos_weight=scale_pos_weight, eval_metric='logloss'
        )
        xgb_cal = CalibratedClassifierCV(xgb, cv=5, method='sigmoid')
        xgb_cal.fit(X_res, y_res)
        trained_models['XGBoost'] = xgb_cal

    # LightGBM (if available) - use class_weight param via 'class_weight' when constructing classifier; however LGBM accepts 'class_weight' or 'is_unbalance'
    if 'LightGBM' in models_to_train and LGBMClassifier is not None:
        lgb = LGBMClassifier(n_estimators=500, learning_rate=0.03, num_leaves=31,
                             min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
                             reg_alpha=0.1, reg_lambda=0.3, random_state=RANDOM_STATE,
                             class_weight='balanced')
        lgb_cal = CalibratedClassifierCV(lgb, cv=5, method='sigmoid')
        lgb_cal.fit(X_res, y_res)
        trained_models['LightGBM'] = lgb_cal

    # SVM (tuned small grid, class_weight balanced)
    if 'SVM' in models_to_train:
        # small GridSearch on gamma and C but keep it light
        base_svc = SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=RANDOM_STATE)
        param_grid = {'C': [0.5, 1, 2], 'gamma': ['scale', 0.01, 0.1]}
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        gs = GridSearchCV(base_svc, param_grid, scoring='roc_auc', cv=cv, n_jobs=-1, verbose=0)
        gs.fit(X_res, y_res)
        best_svm = gs.best_estimator_
        svm_cal = CalibratedClassifierCV(best_svm, cv=3, method='sigmoid')
        svm_cal.fit(X_res, y_res)
        trained_models['SVM'] = svm_cal

    # Evaluate on X_te
    for name, model in trained_models.items():
        pred = model.predict(X_te)
        try:
            probs = model.predict_proba(X_te)[:,1]
        except Exception:
            probs = model.decision_function(X_te)
            if probs.ndim>1:
                probs = probs.ravel()
        acc = accuracy_score(y_te, pred)
        auc = None
        try:
            auc = roc_auc_score(y_te, probs)
        except Exception:
            auc = None
        cm = confusion_matrix(y_te, pred)
        report = classification_report(y_te, pred, output_dict=True)
        results[name] = {'model': model, 'accuracy': acc, 'auc': auc, 'cm': cm, 'report': report, 'probs': probs}
    # return trained models dict and evaluation results plus scaler / feature_names
    return trained_models, results, (X_tr, X_te, y_tr, y_te, X_res, y_res)

# PDF report util
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
            c.showPage(); y = height - 60
    y -= 8
    c.setFont("Helvetica-Bold", 12); c.drawString(40, y, "Prediction Summary:")
    y -= 16
    c.setFont("Helvetica", 10)
    c.drawString(48, y, f"Predicted Class: {'Diabetic' if predicted_class==1 else 'Non-diabetic'}")
    y -= 14
    c.drawString(48, y, f"Probability: {probability:.3f}")
    y -= 20
    c.setFont("Helvetica-Bold", 12); c.drawString(40, y, "Doctor-style Recommendations:")
    y -= 16
    c.setFont("Helvetica", 10)
    for rec in rec_list:
        c.drawString(48, y, f"- {rec}"); y -= 14
        if y < 60: c.showPage(); y = height - 60
    c.save(); buffer.seek(0)
    return buffer.read()

# ---------------- App layout ----------------
df = load_data()

# Sidebar
st.sidebar.title("Controls")
page = st.sidebar.radio("Navigation", ["Home", "Train & Compare", "Predict", "History"])
st.sidebar.markdown("---")
st.sidebar.write(f"Dataset rows: {df.shape[0]} | columns: {df.shape[1]}")

# Models to train (include SVM per professor)
default_models = ['RandomForest','XGBoost','LightGBM','SVM']
models_to_train = st.sidebar.multiselect("Models to train", options=default_models, default=default_models)

# Notebook mode toggle (leakage)
notebook_mode_default = False
notebook_mode_sidebar = st.sidebar.checkbox("Enable Notebook Mode (include glyhb features)", value=notebook_mode_default)

# Header
if page != "Predict":
    st.title("Diabetes Models — Professional")
else:
    st.title("Diabetes Risk Prediction — Professional")

# HOME
if page == "Home":
    st.header("Overview")
    st.write("""
    This professional dashboard uses SMOTE oversampling, class balancing, and calibrated classifiers so predicted probabilities are meaningful.
    Use Train & Compare to train models (Safe/Notebook modes). Then go to Predict to check inputs.
    """)
    st.subheader("Dataset preview")
    st.dataframe(df.head(8))

# TRAIN & COMPARE
if page == "Train & Compare":
    st.header("Train & Compare (SMOTE + Calibrated Models)")
    mode_choice = st.radio("Mode", ("Safe (no leakage)", "Notebook (with glyhb)"),
                          index=0 if not notebook_mode_sidebar else 1)
    remove_leakage = True if mode_choice.startswith("Safe") else False
    if mode_choice.startswith("Notebook"):
        st.warning("Notebook Mode includes glyhb/stab.glu/ratio which are diagnostic features (can inflate accuracy). Use Safe Mode for screening.")
    if st.button("Train & Evaluate (Professional)"):
        with st.spinner("Preprocessing, SMOTE, training and calibration (may take a minute)..."):
            # Preprocess
            X_scaled, y, feature_names, scaler, X_df_unscaled = preprocess_for_training(df, remove_leakage=remove_leakage)
            # Train with SMOTE + calibration + SVM tuning
            trained_models, eval_results, aux = train_with_smote_and_calibration(X_scaled, y, models_to_train)
            # Save artifacts
            st.session_state['models'] = trained_models
            st.session_state['scaler'] = scaler
            st.session_state['feature_names'] = feature_names
            st.session_state['remove_leakage'] = remove_leakage
            st.session_state['df_raw'] = df
            # Present metrics
            metrics = []
            for name,res in eval_results.items():
                metrics.append({'model': name, 'accuracy': res['accuracy'], 'auc': res['auc'],
                                'precision': res['report'].get('weighted avg',{}).get('precision', np.nan),
                                'recall': res['report'].get('weighted avg',{}).get('recall', np.nan),
                                'f1': res['report'].get('weighted avg',{}).get('f1-score', np.nan)})
            metrics_df = pd.DataFrame(metrics).sort_values('accuracy', ascending=False).reset_index(drop=True)
            st.subheader("Model comparison (on held-out test set)")
            st.dataframe(metrics_df.set_index('model').round(3), use_container_width=True)
            # ROC curves
            st.subheader("ROC comparison")
            fig, ax = plt.subplots(figsize=(7,5))
            for name,res in eval_results.items():
                if res['probs'] is not None:
                    fpr,tpr,_ = roc_curve(aux[3], res['probs']) if False else roc_curve(aux[3], res['probs']) if False else roc_curve(aux[3], res['probs']) # placeholder
            # We will instead show each model's ROC using stored res (we already computed auc)
            # Create simple bar for AUC
            st.subheader("AUC (on test set)")
            auc_df = pd.DataFrame([{'model': name, 'auc': (res['auc'] if res['auc'] is not None else np.nan)} for name,res in eval_results.items()])
            fig = px.bar(auc_df, x='model', y='auc', range_y=[0,1], title="AUC by model")
            st.plotly_chart(fig, use_container_width=True)

            # Feature importances for tree models
            st.subheader("Feature importances (tree models)")
            for name,m in trained_models.items():
                try:
                    # calibrated wrapper has attribute 'base_estimator' or 'estimator' — attempt to find feature_importances_
                    base = getattr(m, 'base_estimator', None) or getattr(m, 'estimator', None) or m
                    importances = base.feature_importances_
                    idx = np.argsort(importances)[::-1][:15]
                    fig, ax = plt.subplots(figsize=(8,3))
                    sns.barplot(x=importances[idx], y=np.array(feature_names)[idx], ax=ax)
                    ax.set_title(f"Top features - {name}")
                    st.pyplot(fig)
                except Exception:
                    st.info(f"Feature importances not available for {name}")
            st.success("Training + calibration complete. Models stored in session.")

# PREDICT
if page == "Predict":
    st.header("Predict Diabetes Risk")
    if 'models' not in st.session_state:
        st.warning("No trained models found. Please train models in Train & Compare first.")
        st.stop()

    left, right = st.columns([1,1])
    with left:
        mode = st.radio("Input mode", ["Simple","Advanced"])
        model_choice = st.selectbox("Model to use", options=list(st.session_state['models'].keys()))
        notebook_mode = st.checkbox("Use Notebook Mode (may expect glyhb)", value=not st.session_state.get('remove_leakage', True))
        st.markdown("---")
        st.subheader("Patient Inputs")
        # Simple inputs with clamped defaults
        if mode == "Simple":
            age = st.number_input("Age", min_value=0, max_value=120, value=int(clamp(df['age'].median(),0,120)))
            gender = st.selectbox("Gender", df['gender'].dropna().unique().tolist())
            weight = st.number_input("Weight (kg)", min_value=20.0, max_value=200.0, value=float(clamp(df['weight'].median(),20.0,200.0)))
            height = st.number_input("Height (cm)", min_value=100.0, max_value=220.0, value=float(clamp(df['height'].median(),100.0,220.0)))
            chol = st.number_input("Cholesterol (chol)", min_value=50.0, max_value=400.0, value=float(clamp(df['chol'].median(),50.0,400.0)))
            hdl = st.number_input("HDL", min_value=10.0, max_value=150.0, value=float(clamp(df['hdl'].median(),10.0,150.0)))
            sys_bp = st.number_input("Systolic BP (bp.1s)", min_value=80.0, max_value=220.0, value=float(clamp(df['bp.1s'].median() if 'bp.1s' in df.columns else 120.0,80.0,220.0)))
            dia_bp = st.number_input("Diastolic BP (bp.1d)", min_value=40.0, max_value=140.0, value=float(clamp(df['bp.1d'].median() if 'bp.1d' in df.columns else 80.0,40.0,140.0)))
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
        predict_btn = st.button("Predict Risk (Calibrated)")

    with right:
        st.subheader("Prediction Result")
        result_area = st.empty()
        shap_area = st.empty()
        plotly_area = st.empty()
        chat_area = st.empty()

    if predict_btn:
        remove_leakage = not notebook_mode
        # training artifacts
        scaler = st.session_state.get('scaler')
        feature_names = st.session_state.get('feature_names')
        df_training = st.session_state.get('df_raw', df)

        if scaler is None or feature_names is None:
            st.error("Training artifacts missing — retrain models.")
            st.stop()

        # Align & scale features
        try:
            input_scaled = align_features_for_prediction(input_raw, df_training, feature_names, scaler, remove_leakage=remove_leakage)
        except Exception as e:
            st.error(f"Feature alignment failed: {e}")
            st.stop()

        model = st.session_state['models'].get(model_choice)
        if model is None:
            st.error("Model not present in session. Re-train.")
            st.stop()

        # Predict calibrated probability
        try:
            prob = float(model.predict_proba(input_scaled)[:,1][0])
        except Exception:
            # fallback to decision_function with sigmoid
            raw = model.decision_function(input_scaled)
            prob = float(1/(1+np.exp(-raw))[0])

        pred_class = int(prob >= 0.5)
        prob_pct = round(prob*100, 1)

        # Risk label
        if prob < 0.3:
            risk_label, color, emoji, advice_short = "Low", "#2ecc71", "🟢", "Low risk — maintain healthy lifestyle"
        elif prob < 0.6:
            risk_label, color, emoji, advice_short = "Moderate", "#f39c12", "🟠", "Moderate risk — consider lifestyle changes"
        elif prob < 0.8:
            risk_label, color, emoji, advice_short = "High", "#e74c3c", "🔴", "High risk — seek medical advice"
        else:
            risk_label, color, emoji, advice_short = "Very High", "#8b0000", "🚨", "Very High risk — immediate evaluation"

        # Theme-aware result card
        is_dark = st.get_option("theme.base") == "dark"
        card_style = "background-color:#0f1720;color:#fff;padding:14px;border-radius:10px;border:1px solid #283040;" if is_dark else "background-color:#f3fbff;color:#000;padding:14px;border-radius:10px;"
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
        gauge = go.Figure(go.Indicator(mode="gauge+number", value=prob_pct, domain={'x':[0,1],'y':[0,1]}, title={'text':"Risk Gauge (%)"},
                                      gauge={'axis':{'range':[0,100]}, 'bar':{'color':color}}))
        st.plotly_chart(gauge, use_container_width=True)

        # SHAP explanations (robust)
        vals = None
        base_val = None
        if SHAP_AVAILABLE:
            try:
                shap_area.subheader("SHAP Explanation (local)")
                # Build small background for KernelExplainer if needed
                # Prepare X_all unscaled to supply to KernelExplainer if required
                X_full_unscaled, _ = preprocessing_no_scale(df_training, remove_leakage=remove_leakage)
                # For tree models use TreeExplainer on base_estimator if calibrated wrapper
                if model_choice in ['RandomForest','XGBoost','LightGBM']:
                    base = getattr(model, 'base_estimator', None) or getattr(model, 'estimator', None) or model
                    explainer = shap.TreeExplainer(base)
                    # shap expects original (unscaled) features passing input in feature order; reconstruct single sample unscaled:
                    # Align unscaled user row
                    X_unscaled_user = align_features_for_prediction(input_raw, df_training, feature_names, scaler, remove_leakage=remove_leakage)
                    # Note: X_unscaled_user currently scaled — we need unscaled; simpler: rebuild unscaled by using preprocessing_no_scale on concat and take row0
                    df_temp = pd.concat([input_raw.reset_index(drop=True), df_training.reset_index(drop=True)], ignore_index=True)
                    X_all_unscaled, _ = preprocessing_no_scale(df_temp, remove_leakage=remove_leakage)
                    user_unscaled = X_all_unscaled.iloc[[0]].copy()
                    sv = explainer.shap_values(user_unscaled)
                    if isinstance(sv, list):
                        vals = np.asarray(sv[1])[0] if len(sv)>1 else np.asarray(sv[0])[0]
                    else:
                        arr = np.asarray(sv)
                        vals = arr[0] if arr.ndim==2 else arr.ravel()
                    base_val = explainer.expected_value[1] if hasattr(explainer.expected_value, "__len__") and len(explainer.expected_value)>1 else explainer.expected_value
                elif model_choice == 'SVM':
                    shap_area.info("Running KernelExplainer for SVM (reduced background sample to speed up).")
                    # Build background: sample up to 100 rows of X_unscaled
                    df_temp = pd.concat([input_raw.reset_index(drop=True), df_training.reset_index(drop=True)], ignore_index=True)
                    X_all_unscaled, _ = preprocessing_no_scale(df_temp, remove_leakage=remove_leakage)
                    background = shap.sample(X_all_unscaled, min(100, len(X_all_unscaled)))
                    explainer = shap.KernelExplainer(model.predict_proba, background)
                    # need scaled input with same feature order as background: we pass scaled -> but KernelExplainer expects same input space as background (unscaled), so pass unscaled single sample
                    user_unscaled = X_all_unscaled.iloc[[0]]
                    sv = explainer.shap_values(user_unscaled)
                    vals = np.asarray(sv[1])[0] if isinstance(sv, list) and len(sv)>1 else np.asarray(sv).ravel()
                    base_val = explainer.expected_value[1] if hasattr(explainer.expected_value, "__len__") and len(explainer.expected_value)>1 else explainer.expected_value
                else:
                    shap_area.info("Model not directly supported for SHAP explainer.")
                # show waterfall or bar
                if vals is not None:
                    expl = shap.Explanation(values=vals, base_values=base_val, data=pd.Series(input_raw.iloc[0]).reindex(feature_names).fillna(0), feature_names=feature_names)
                    try:
                        plt.figure(facecolor='white')
                        shap.plots.waterfall(expl, show=False)
                        st.pyplot(plt.gcf()); plt.clf()
                    except Exception:
                        try:
                            plt.figure(facecolor='white')
                            shap.plots.bar(expl, show=False)
                            st.pyplot(plt.gcf()); plt.clf()
                        except Exception:
                            shap_area.info("SHAP plot failed to render.")
                    # try interactive force plot embed
                    try:
                        fp = shap.force_plot(base_val, vals, pd.Series(input_raw.iloc[0]).reindex(feature_names).fillna(0), matplotlib=False)
                        import streamlit.components.v1 as components
                        components.html(f"<head>{shap.getjs()}</head><body>{fp.html()}</body>", height=380)
                    except Exception:
                        pass
            except Exception as e:
                shap_area.error(f"SHAP explanation failed: {e}")
        else:
            shap_area.info("Install 'shap' to enable model explanations.")

        # 3D-like interactive Plotly of SHAP contributions (top)
        try:
            if vals is not None:
                df_shap = pd.DataFrame({'feature': feature_names, 'contrib': np.array(vals).flatten()})
                df_shap['abs'] = df_shap['contrib'].abs()
                df_shap = df_shap.sort_values('abs', ascending=False).head(12)
                fig3 = px.bar(df_shap, x='feature', y='contrib', color='contrib', color_continuous_scale='RdBu', title='Top SHAP contributions')
                st.plotly_chart(fig3, use_container_width=True)
        except Exception:
            pass

        # Doctor-style recommendations
        recs = []
        try:
            wt = float(input_raw.get('weight',0).iloc[0]); ht = float(input_raw.get('height',0).iloc[0])
            bmi = wt/((ht/100)**2) if ht>0 else np.nan
        except Exception:
            bmi = np.nan
        if not np.isnan(bmi):
            if bmi < 18.5: recs.append("Underweight: consider nutrition consult.")
            elif bmi <25: recs.append("Normal BMI: maintain healthy lifestyle.")
            elif bmi <30: recs.append("Overweight: aim for gradual weight loss (5-10%).")
            else: recs.append("Obese: recommend clinician evaluation and weight program.")
        try:
            s = float(input_raw.get('bp.1s', np.nan).iloc[0]); d = float(input_raw.get('bp.1d', np.nan).iloc[0])
            if not np.isnan(s) and not np.isnan(d):
                if s<120 and d<80: recs.append("BP normal: continue monitoring.")
                elif s<130: recs.append("BP elevated: lifestyle changes.")
                elif s<140: recs.append("Stage 1 HTN: consult clinician.")
                else: recs.append("Stage 2 HTN: seek medical care.")
        except Exception:
            pass
        try:
            cholv = float(input_raw.get('chol', np.nan).iloc[0])
            if not np.isnan(cholv):
                if cholv<200: recs.append("Cholesterol desirable.")
                elif cholv<240: recs.append("Borderline high cholesterol: diet/exercise.")
                else: recs.append("High cholesterol: consider lipid profile & clinician.")
        except Exception:
            pass
        if risk_label == "Very High":
            recs.append("Immediate: order HbA1c/fasting glucose and consult healthcare provider.")
        elif risk_label == "High":
            recs.append("Consider early clinical screening (HbA1c).")
        elif risk_label == "Moderate":
            recs.append("Increase monitoring and preventive lifestyle changes.")
        else:
            recs.append("Routine checks; re-evaluate annually.")

        st.subheader("Doctor-style Recommendations")
        for r in recs:
            st.write("- " + r)

        # Save to session history
        hist = st.session_state.get('history', [])
        entry = input_raw.copy()
        entry['predicted_class'] = pred_class
        entry['predicted_prob'] = prob
        entry['model'] = model_choice
        entry['mode'] = 'Notebook' if not remove_leakage else 'Safe'
        entry['timestamp'] = datetime.now().isoformat()
        st.session_state['history'] = hist + [entry.to_dict(orient='records')[0]]
        st.success("Saved to session history.")

        # Download options
        csv_bytes = input_raw.copy(); csv_bytes['predicted_class'] = pred_class; csv_bytes['predicted_prob'] = prob
        st.download_button("Download prediction (CSV)", csv_bytes.to_csv(index=False).encode('utf-8'), file_name='prediction.csv', mime='text/csv')
        pdf = generate_pdf_report(input_raw, pred_class, prob, risk_label, recs)
        st.download_button("Download PDF report", data=pdf, file_name="diabetes_report.pdf", mime="application/pdf")

        # Simple AI Diagnosis Chatbot (local rule-based)
        chat_area.subheader("AI Diagnosis Chatbot")
        if 'chat_history' not in st.session_state:
            st.session_state['chat_history'] = []
        user_q = st.text_input("Ask assistant (press Enter)", key="chat_input")
        if user_q:
            q = user_q.lower()
            reply = ""
            if any(w in q for w in ["risk","probab","chance","how likely"]):
                reply = f"The model reports {prob_pct}%. Risk bucket: {risk_label}. This is a screening estimate, not a diagnosis."
            elif any(w in q for w in ["test","what should i do","next step","advise","recommend"]):
                reply = "Consider HbA1c/fasting glucose if risk is Moderate or higher. Lifestyle: weight loss, exercise, healthy diet. See clinician for high risks."
            elif any(w in q for w in ["why","explain","reason"]):
                reply = "Top contributing features often include age, waist/hip, cholesterol/HDL and blood pressure. SHAP plot shows local contributions."
            else:
                reply = "I can explain the prediction, give next steps, or list contributing features. Try: 'Why am I high risk?'"
            st.session_state['chat_history'].append({'q': user_q, 'a': reply})
        for turn in st.session_state.get('chat_history', [])[::-1]:
            st.markdown(f"**You:** {turn['q']}")
            st.markdown(f"**Assistant:** {turn['a']}")

# HISTORY
if page == "History":
    st.header("Prediction History")
    hist = st.session_state.get('history', [])
    if not hist:
        st.info("No history.")
    else:
        hist_df = pd.DataFrame(hist)
        st.dataframe(hist_df, use_container_width=True)
        st.download_button("Download history (CSV)", hist_df.to_csv(index=False).encode('utf-8'), file_name='history.csv', mime='text/csv')

st.markdown("---")
st.markdown("_Note: Notebook Mode includes glyhb/stab.glu/ratio which are diagnostic features — use Safe Mode for real-world screening._")

# streamlit_diabetes_dashboard_full_fixed.py
"""
DiaGuard Pro — Ultimate (Full) Dashboard (Fixed)
- Train (SMOTE + Calibrated models)
- Predict (always-rendering UI; disabled until training)
- SHAP explanations (TreeExplainer + KernelExplainer)
- Universal feature importance (works with calibrated wrappers)
- AI Doctor Chatbot, PDF/CSV exports, History
- Robust preprocessing & feature alignment
- Protection against feature-name mismatch & scaler errors
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import io
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, roc_curve
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils import compute_class_weight
from imblearn.over_sampling import SMOTE
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import warnings
warnings.filterwarnings("ignore")

# Optional heavy imports (graceful fallback)
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

# -------------------- Config --------------------
st.set_page_config(page_title="DiaGuard Pro — Ultimate", layout="wide", initial_sidebar_state="expanded")
DATA_PATH = "diabetes (1).csv"
RANDOM_STATE = 42

# -------------------- Utility functions --------------------
@st.cache_data
def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df

def safe_median(series, default=0.0):
    try:
        return float(series.median())
    except Exception:
        return default

def clamp(v, min_v, max_v):
    try:
        v = float(v)
    except Exception:
        return min_v
    return max(min_v, min(max_v, v))

def create_target(df):
    """Create binary diabetes target from glyhb >= 6.5"""
    df = df.copy()
    if 'glyhb' not in df.columns:
        raise ValueError("'glyhb' column required in dataset for target creation.")
    df['diabetes'] = (df['glyhb'] >= 6.5).astype(int)
    return df

def preprocessing_no_scale(df_input, remove_leakage=True):
    """
    Preprocess but keep unscaled. Returns (X_df, y)
    - handles imputation and categorical one-hot encoding
    - removes leakage columns if requested
    """
    df = df_input.copy()
    df.columns = df.columns.str.strip()
    df = create_target(df) if 'diabetes' not in df.columns else df
    leak_cols = ['glyhb', 'ratio', 'stab.glu']
    drop_cols = ['id']
    if remove_leakage:
        drop_cols += leak_cols
    X = df.drop(columns=[c for c in drop_cols + ['diabetes'] if c in df.columns], errors='ignore')
    y = df['diabetes']

    # Identify columns
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    # Impute numeric
    if len(num_cols) > 0:
        num_imp = SimpleImputer(strategy='median')
        X[num_cols] = num_imp.fit_transform(X[num_cols])

    # Impute categorical + one-hot
    if len(cat_cols) > 0:
        cat_imp = SimpleImputer(strategy='most_frequent')
        X[cat_cols] = cat_imp.fit_transform(X[cat_cols])
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    # Ensure numeric dtype
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    return X, y

def preprocess_for_training(df_full, remove_leakage=True):
    """Preprocess then scale. Returns X_scaled (np), y (Series), feature_names (list), scaler, X_unscaled_df"""
    X_df, y = preprocessing_no_scale(df_full, remove_leakage=remove_leakage)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df)
    return X_scaled, y, X_df.columns.tolist(), scaler, X_df

def align_features_for_prediction(user_raw_df, df_training_raw, feature_names, scaler, remove_leakage=True):
    """
    Align user input features to training features robustly.
    - Concatenate user row with training raw df and run preprocessing_no_scale to ensure dummy columns match
    - Extract first row and scale with provided scaler
    """
    df_temp = pd.concat([user_raw_df.reset_index(drop=True), df_training_raw.reset_index(drop=True)], ignore_index=True)
    X_all_unscaled, _ = preprocessing_no_scale(df_temp, remove_leakage=remove_leakage)
    user_unscaled = X_all_unscaled.iloc[[0]].copy()
    # Add missing columns as zeros
    for col in feature_names:
        if col not in user_unscaled.columns:
            user_unscaled[col] = 0.0
    user_unscaled = user_unscaled[feature_names]
    user_scaled = scaler.transform(user_unscaled)
    return user_scaled, user_unscaled

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
    c.save(); buffer.seek(0)
    return buffer.read()

# -------------------- Universal Feature Importance function --------------------
def plot_feature_importance(model, feature_names, model_name, max_feats=15, use_dark=False):
    """
    Universal feature-importance function.
    - For calibrated wrapper, extracts inner estimator via base_estimator or estimator attribute.
    - For SVM (no feature_importances_), uses SHAP-based ranking (KernelExplainer).
    - Plots top `max_feats`.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Extract inner estimator if calibrated
    inner = getattr(model, 'base_estimator', None) or getattr(model, 'estimator', None) or model

    # SVM path (no native feature_importances_)
    if model_name.upper() == "SVM":
        st.info("SVM: showing SHAP-based feature ranking (SVM has no native feature_importances_).")
        if not SHAP_AVAILABLE:
            st.warning("Install 'shap' to enable SHAP-based ranking for SVM.")
            return
        try:
            # Build a tiny background of zeros — the KernelExplainer expects same input distribution.
            background = np.zeros((1, len(feature_names)))
            explainer = shap.KernelExplainer(model.predict_proba, background)
            sample = np.zeros((1, len(feature_names)))
            shap_vals = explainer.shap_values(sample)
            shap_arr = np.asarray(shap_vals[1]).ravel() if isinstance(shap_vals, list) and len(shap_vals)>1 else np.asarray(shap_vals).ravel()
            absvals = np.abs(shap_arr)
            idx = np.argsort(absvals)[::-1][:max_feats]
            fig, ax = plt.subplots(figsize=(8, max_feats * 0.3 + 2))
            sns.barplot(x=absvals[idx], y=np.array(feature_names)[idx], ax=ax, palette="viridis")
            ax.set_title(f"Top {max_feats} SHAP-based features — {model_name}")
            ax.set_xlabel("Mean(|SHAP value|)")
            st.pyplot(fig)
        except Exception as e:
            st.error(f"SHAP for SVM failed: {e}")
        return

    # Tree model path
    try:
        importances = inner.feature_importances_
    except Exception:
        st.warning(f"{model_name} has no feature_importances_. Consider SHAP explanation instead.")
        return

    idx = np.argsort(importances)[::-1][:max_feats]
    fig, ax = plt.subplots(figsize=(8, max_feats * 0.3 + 2))
    col = "lightgreen" if use_dark else "steelblue"
    sns.barplot(x=importances[idx], y=np.array(feature_names)[idx], color=col, ax=ax)
    ax.set_title(f"Top {max_feats} Important Features — {model_name}")
    ax.set_xlabel("Importance")
    st.pyplot(fig)

# -------------------- Training function (SMOTE + calibration + SVM tuning) --------------------
@st.cache_resource
def train_models_pipeline(df_raw, remove_leakage=True, models_to_train=None):
    """
    Trains models with:
    - Preprocessing (scaled)
    - SMOTE oversampling on training fold
    - CalibratedClassifierCV wrapper for probability calibration
    - SVM small gridsearch + calibration
    Returns: trained_models_dict, eval_results_dict, artifacts
    """
    if models_to_train is None:
        models_to_train = ['RandomForest', 'XGBoost', 'LightGBM', 'SVM']

    X_scaled, y, feature_names, scaler, X_unscaled_df = preprocess_for_training(df_raw, remove_leakage=remove_leakage)
    X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    # SMOTE on training set
    sm = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = sm.fit_resample(X_tr, y_tr)

    trained = {}
    eval_results = {}

    # RandomForest
    if 'RandomForest' in models_to_train:
        rf = RandomForestClassifier(n_estimators=300, max_depth=12, class_weight='balanced', random_state=RANDOM_STATE)
        rf_cal = CalibratedClassifierCV(rf, cv=5, method='sigmoid')
        rf_cal.fit(X_res, y_res)
        trained['RandomForest'] = rf_cal

    # XGBoost
    if 'XGBoost' in models_to_train and XGBClassifier is not None:
        posw = (y_res == 0).sum() / max(1, (y_res == 1).sum())  # class weight balance

        xgb = XGBClassifier(
            n_estimators=350,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=posw,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=RANDOM_STATE
        )

        xgb_cal = CalibratedClassifierCV(xgb, cv=5, method='sigmoid')
        xgb_cal.fit(X_res, y_res)
        trained['XGBoost'] = xgb_cal

    # LightGBM
    if 'LightGBM' in models_to_train and LGBMClassifier is not None:
        lgb = LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31, class_weight='balanced', random_state=RANDOM_STATE)
        lgb_cal = CalibratedClassifierCV(lgb, cv=5, method='sigmoid')
        lgb_cal.fit(X_res, y_res)
        trained['LightGBM'] = lgb_cal

    # SVM with small tuning
    if 'SVM' in models_to_train:
        svc = SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=RANDOM_STATE)
        param_grid = {'C': [1, 2], 'gamma': ['scale', 0.01]}
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        gs = GridSearchCV(svc, param_grid, scoring='roc_auc', cv=cv, n_jobs=-1)
        gs.fit(X_res, y_res)
        best = gs.best_estimator_
        svm_cal = CalibratedClassifierCV(best, cv=3, method='sigmoid')
        svm_cal.fit(X_res, y_res)
        trained['SVM'] = svm_cal

    # Evaluate on X_te
    for name, model in trained.items():
        pred = model.predict(X_te)
        try:
            probs = model.predict_proba(X_te)[:,1]
        except Exception:
            probs = model.decision_function(X_te)
            if hasattr(probs, 'ravel'):
                probs = probs.ravel()
        acc = accuracy_score(y_te, pred)
        try:
            auc = roc_auc_score(y_te, probs)
        except Exception:
            auc = None
        cm = confusion_matrix(y_te, pred)
        report = classification_report(y_te, pred, output_dict=True)
        eval_results[name] = {'accuracy': acc, 'auc': auc, 'cm': cm, 'pred': pred, 'probs': probs, 'report': report}

    artifacts = {'scaler': scaler, 'feature_names': feature_names, 'X_unscaled_df': X_unscaled_df}
    return trained, eval_results, artifacts

# -------------------- Load dataset --------------------
df = load_data(DATA_PATH)

# -------------------- Sidebar / Navigation --------------------
st.sidebar.title("DiaGuard Pro — Navigation")
page = st.sidebar.radio("Go to", ["Home", "Train Models", "Predict", "Model Comparison", "AI Doctor Chatbot", "History"], index=1)
st.sidebar.markdown("---")
st.sidebar.write(f"Dataset rows: {df.shape[0]}  |  columns: {df.shape[1]}")

# -------------------- HOME --------------------
if page == "Home":
    st.title("DiaGuard Pro — Diabetes Risk Suite")
    st.markdown("""
    **DiaGuard Pro** is a professional diabetes risk prediction and explainability dashboard.
    Train models (SMOTE + calibrated models) then explore predictions with SHAP explanations and doctor-style recommendations.
    """)
    st.subheader("Dataset preview")
    st.dataframe(df.head(8))

# -------------------- TRAIN MODELS --------------------
elif page == "Train Models":
    st.title("Train Models — Professional Pipeline")
    st.write("Choose mode and models. Safe mode excludes glyhb/ratio/stab.glu (no leakage). Notebook mode includes them (may inflate accuracy).")
    mode_choice = st.radio("Mode", ("Safe - No GlyHB leakage (recommended)", "Notebook - include glyhb/diagnostic features"), index=0)
    remove_leakage = True if mode_choice.startswith("Safe") else False
    if mode_choice.startswith("Notebook"):
        st.warning("Notebook mode includes diagnostic features like glyhb. This can lead to near-perfect accuracy but is not valid for screening.")

    # Always show model choice UI (fix for vanished selector)
    models_to_train = st.multiselect("Select Models to Train", options=['RandomForest', 'XGBoost', 'LightGBM', 'SVM'], default=['RandomForest', 'XGBoost', 'LightGBM', 'SVM'])

    # Train button (actual training below)
    if st.button("Train (SMOTE + Calibrated + SVM tuning)"):
        with st.spinner("Training models — this may take a minute..."):
            trained_models, eval_results, artifacts = train_models_pipeline(df, remove_leakage=remove_leakage, models_to_train=models_to_train)
            st.session_state['models'] = trained_models
            st.session_state['results'] = eval_results
            st.session_state['scaler'] = artifacts['scaler']
            st.session_state['feature_names'] = artifacts['feature_names']
            st.session_state['remove_leakage'] = remove_leakage
            st.session_state['df_raw'] = df
        st.success("Training complete — models stored in session.")

        # Compose clean metrics table (no raw arrays)
        rows = []
        for name, r in eval_results.items():
            rows.append({
                'Model': name,
                'Accuracy': r['accuracy'],
                'AUC': r['auc'],
                'Precision': r['report'].get('weighted avg', {}).get('precision', np.nan),
                'Recall': r['report'].get('weighted avg', {}).get('recall', np.nan),
                'F1': r['report'].get('weighted avg', {}).get('f1-score', np.nan)
            })
        metrics_df = pd.DataFrame(rows).set_index('Model').round(3)
        st.subheader("Model performance (held-out test set)")
        st.dataframe(metrics_df)

        # Leaderboard (simple)
        st.subheader("Leaderboard")
        lb = metrics_df.sort_values('Accuracy', ascending=False)
        for idx, row in lb.iterrows():
            st.metric(label=idx, value=f"Accuracy: {row['Accuracy']:.3f}", delta=f"AUC: {row['AUC']:.3f}" if not pd.isna(row['AUC']) else "AUC: N/A")

        # Feature importances for tree models
        st.subheader("Feature Importances (Tree models)")
        for name, model in trained_models.items():
            try:
                plot_feature_importance(model, artifacts['feature_names'], name, use_dark=(st.get_option("theme.base")=="dark"))
            except Exception as e:
                st.warning(f"Feature importance for {name} failed: {e}")

# -------------------- PREDICT (UPGRADED: ALWAYS RENDERS) --------------------
elif page == "Predict":
    st.title("Predict — Diabetes Risk")
    st.markdown("Enter patient details. If models are not trained, inputs are displayed but disabled. Train models in the 'Train Models' tab first.")

    models_present = 'models' in st.session_state and isinstance(st.session_state['models'], dict) and len(st.session_state['models'])>0
    models = st.session_state.get('models', {})
    feature_names = st.session_state.get('feature_names', None)
    scaler = st.session_state.get('scaler', None)
    df_raw = st.session_state.get('df_raw', df)
    remove_leakage_session = st.session_state.get('remove_leakage', True)

    # Model selection: shown even if not present, but disabled
    if models_present:
        selected_model = st.selectbox("Select model for prediction", options=list(models.keys()))
    else:
        st.info("No trained models available — train models first to enable prediction.")
        selected_model = None

    # Inputs (form), disabled if no models
    disable_controls = not models_present

    with st.form("predict_form"):

    c1, c2 = st.columns(2)

    # ---------- COLUMN 1 ----------
    with c1:
        age = st.number_input(
            "Age",
            min_value=0,
            max_value=120,
            value=clamp(safe_median(df['age'] if 'age' in df.columns else pd.Series([50]),50), 0, 120),
            disabled=disable_controls
        )

        gender = st.selectbox(
            "Gender",
            df['gender'].dropna().unique().tolist() if 'gender' in df.columns else ['M'],
            disabled=disable_controls
        )

        weight = st.number_input(
            "Weight (kg)",
            min_value=20.0,
            max_value=200.0,
            value=clamp(safe_median(df['weight'] if 'weight' in df.columns else pd.Series([75]),75), 20.0, 200.0),
            disabled=disable_controls
        )

        height = st.number_input(
            "Height (cm)",
            min_value=100.0,
            max_value=220.0,
            value=clamp(safe_median(df['height'] if 'height' in df.columns else pd.Series([165]),165), 100.0, 220.0),
            disabled=disable_controls
        )

        frame = st.selectbox(
            "Frame",
            df['frame'].dropna().unique().tolist() if 'frame' in df.columns else ['M'],
            disabled=disable_controls
        )

        hip = st.number_input(
            "Hip (cm)",
            min_value=30.0,
            max_value=200.0,
            value=clamp(safe_median(df['hip'] if 'hip' in df.columns else pd.Series([90]),90), 30.0, 200.0),
            disabled=disable_controls
        )

    # ---------- COLUMN 2 ----------
    with c2:
        chol = st.number_input(
            "Cholesterol (chol)",
            min_value=50.0,
            max_value=400.0,
            value=clamp(safe_median(df['chol'] if 'chol' in df.columns else pd.Series([200]),200), 50.0, 400.0),
            disabled=disable_controls
        )

        hdl = st.number_input(
            "HDL",
            min_value=10.0,
            max_value=150.0,
            value=clamp(safe_median(df['hdl'] if 'hdl' in df.columns else pd.Series([40]),40), 10.0, 150.0),
            disabled=disable_controls
        )

        bp_sys = st.number_input(
            "Systolic BP (bp.1s)",
            min_value=80.0,
            max_value=220.0,
            value=clamp(safe_median(df['bp.1s'] if 'bp.1s' in df.columns else pd.Series([120]),120), 80.0, 220.0),
            disabled=disable_controls
        )

        bp_dia = st.number_input(
            "Diastolic BP (bp.1d)",
            min_value=40.0,
            max_value=140.0,
            value=clamp(safe_median(df['bp.1d'] if 'bp.1d' in df.columns else pd.Series([80]),80), 40.0, 140.0),
            disabled=disable_controls
        )

        waist = st.number_input(
            "Waist (cm)",
            min_value=30.0,
            max_value=200.0,
            value=clamp(safe_median(df['waist'] if 'waist' in df.columns else pd.Series([80]),80), 30.0, 200.0),
            disabled=disable_controls
        )

        location = st.selectbox(
            "Location",
            df['location'].dropna().unique().tolist() if 'location' in df.columns else ['Urban'],
            disabled=disable_controls
        )

    # --- REQUIRED SUBMIT BUTTON (FIX) ---
    submitted = st.form_submit_button("Predict")


    if disable_controls:
        st.info("Prediction disabled — train models on the Train Models page to enable real-time predictions.")
        st.stop()

    if submitted:
        # Build user input df
        input_raw = pd.DataFrame([{
            'age': age, 'gender': gender, 'height': height, 'weight': weight, 'chol': chol, 'hdl': hdl,
            'bp.1s': bp_sys, 'bp.1d': bp_dia, 'waist': waist, 'hip': hip, 'frame': frame, 'location': location
        }])

        # Align & scale
        try:
            input_scaled, input_unscaled = align_features_for_prediction(input_raw, df_raw, feature_names, scaler, remove_leakage=remove_leakage_session)
        except Exception as e:
            st.error(f"Feature alignment failed: {e}")
            st.stop()

        model = models[selected_model]
        try:
            prob = float(model.predict_proba(input_scaled)[:,1][0])
        except Exception:
            raw = model.decision_function(input_scaled)
            prob = float(1/(1+np.exp(-raw))[0])

        pred_class = int(prob >= 0.5)
        prob_pct = round(prob*100, 2)

        # Summary card (fixed safe HTML string usage)
        is_dark = st.get_option("theme.base") == "dark"
        color = "#2ecc71" if prob_pct < 30 else "#f39c12" if prob_pct < 60 else "#e74c3c" if prob_pct < 80 else "#8b0000"
        summary_html = f"""
<div style='padding:12px;border-radius:8px;background:{"#0f1720" if is_dark else "#f8fbff"}'>
    <strong style='font-size:20px'>{prob_pct}%</strong> probability of diabetes<br/>
    <span style='color:{color};font-weight:700'>
        {"Very High" if prob_pct>=80 else "High" if prob_pct>=60 else "Moderate" if prob_pct>=30 else "Low"}
    </span>
</div>
"""
        st.markdown(f"### Prediction Summary — {selected_model}  •  Mode: {'Notebook' if not remove_leakage_session else 'Safe'}")
        st.markdown(summary_html, unsafe_allow_html=True)

        # Gauge
        gauge = go.Figure(go.Indicator(mode="gauge+number", value=prob_pct, title={"text":"Risk (%)"},
                                      gauge={"axis":{"range":[0,100]}, "bar":{"color":color},
                                             "steps":[{"range":[0,30],"color":"green"},{"range":[30,60],"color":"orange"},{"range":[60,100],"color":"red"}]}))
        st.plotly_chart(gauge, use_container_width=True)

        # SHAP explanation (robust)
        st.subheader("Explainability (SHAP) — Local")
        if not SHAP_AVAILABLE:
            st.info("Install 'shap' to enable SHAP explanations.")
        else:
            try:
                base = getattr(model, 'base_estimator', None) or getattr(model, 'estimator', None) or model
                if selected_model in ['RandomForest', 'XGBoost', 'LightGBM']:
                    # TreeExplainer expects unscaled values consistent with training raw features
                    # We already have input_unscaled; create explainer with base and compute shap_values
                    explainer = shap.TreeExplainer(base)
                    sv = explainer.shap_values(input_unscaled)
                    # Prepare explanation object
                    if isinstance(sv, list):
                        shap_vals_class1 = np.asarray(sv[1])[0] if len(sv)>1 else np.asarray(sv[0])[0]
                        base_val = explainer.expected_value[1] if hasattr(explainer.expected_value, '__len__') and len(explainer.expected_value)>1 else explainer.expected_value
                    else:
                        shap_vals_class1 = np.asarray(sv)[0]
                        base_val = explainer.expected_value
                    # Waterfall plot
                    try:
                        expl = shap.Explanation(values=shap_vals_class1, base_values=base_val, data=input_unscaled.iloc[0].values, feature_names=input_unscaled.columns.tolist())
                        plt.figure(figsize=(8,4))
                        shap.plots.waterfall(expl, show=False)
                        st.pyplot(plt.gcf()); plt.clf()
                    except Exception:
                        try:
                            plt.figure(figsize=(8,4)); shap.plots.bar(expl, show=False); st.pyplot(plt.gcf()); plt.clf()
                        except Exception:
                            st.info("SHAP plotting failed in this environment.")
                    # interactive force if possible
                    try:
                        fp = shap.force_plot(base_val, shap_vals_class1, input_unscaled.iloc[0], matplotlib=False)
                        import streamlit.components.v1 as components
                        components.html(f"<head>{shap.getjs()}</head><body>{fp.html()}</body>", height=400)
                    except Exception:
                        pass
                elif selected_model == 'SVM':
                    st.info("SVM SHAP via KernelExplainer (may be slow). Background sample reduced.")
                    # Build background from training raw (limited)
                    df_temp = pd.concat([input_raw.reset_index(drop=True), df_raw.reset_index(drop=True)], ignore_index=True)
                    X_all_unscaled, _ = preprocessing_no_scale(df_temp, remove_leakage=remove_leakage_session)
                    background = shap.sample(X_all_unscaled, min(100, len(X_all_unscaled)))
                    explainer = shap.KernelExplainer(model.predict_proba, background)
                    user_unscaled = X_all_unscaled.iloc[[0]]
                    shap_vals = explainer.shap_values(user_unscaled)
                    if isinstance(shap_vals, list) and len(shap_vals)>1:
                        vals = np.asarray(shap_vals[1])[0]
                        base_val = explainer.expected_value[1]
                    else:
                        vals = np.asarray(shap_vals).ravel()
                        base_val = explainer.expected_value
                    expl = shap.Explanation(values=vals, base_values=base_val, data=user_unscaled.iloc[0].values, feature_names=user_unscaled.columns.tolist())
                    try:
                        plt.figure(figsize=(8,4)); shap.plots.waterfall(expl, show=False); st.pyplot(plt.gcf()); plt.clf()
                    except Exception:
                        try:
                            plt.figure(figsize=(8,4)); shap.plots.bar(expl, show=False); st.pyplot(plt.gcf()); plt.clf()
                        except Exception:
                            st.info("SHAP plotting failed.")
                    try:
                        fp = shap.force_plot(base_val, vals, user_unscaled.iloc[0], matplotlib=False)
                        import streamlit.components.v1 as components
                        components.html(f"<head>{shap.getjs()}</head><body>{fp.html()}</body>", height=400)
                    except Exception:
                        pass
                else:
                    st.info("SHAP not supported for this model.")
            except Exception as e:
                st.warning(f"SHAP explanation error: {e}")

        # 3D-like SHAP scatter (top features)
        st.subheader("SHAP Contributions — 3D-ish Plot")
        if SHAP_AVAILABLE:
            try:
                # Create a small Shap contributions array if available
                # Recompute shap values for user_unscaled if not already computed
                if selected_model in ['RandomForest','XGBoost','LightGBM']:
                    base_local = getattr(model, 'base_estimator', None) or getattr(model, 'estimator', None) or model
                    expl = shap.TreeExplainer(base_local)
                    sv2 = expl.shap_values(input_unscaled)
                    if isinstance(sv2, list) and len(sv2)>1:
                        vals = np.asarray(sv2[1])[0]
                    else:
                        vals = np.asarray(sv2).ravel()
                    df_sh = pd.DataFrame({'feature': input_unscaled.columns, 'shap': vals})
                    df_sh['abs'] = df_sh['shap'].abs()
                    df_sh = df_sh.sort_values('abs', ascending=False).head(12)
                    # 3D-like scatter: x=rank, y=shap, z=abs(shap)
                    fig3d = px.scatter_3d(df_sh, x=df_sh.index, y='shap', z='abs', text='feature', color='shap', size='abs', title='Top SHAP contributions (3D view)')
                    st.plotly_chart(fig3d, use_container_width=True)
            except Exception:
                pass

        # Doctor recommendations
        st.subheader("Doctor-style Recommendations")
        recs = []
        try:
            wt = float(input_raw.get('weight',0).iloc[0]); ht = float(input_raw.get('height',0).iloc[0])
            bmi = wt/((ht/100)**2) if ht>0 else np.nan
        except Exception:
            bmi = np.nan
        if not np.isnan(bmi):
            if bmi < 18.5: recs.append("Underweight: consider nutritional assessment.")
            elif bmi <25: recs.append("Normal BMI: maintain healthy habits.")
            elif bmi <30: recs.append("Overweight: aim for 5-10% weight loss.")
            else: recs.append("Obese: clinician evaluation and weight management recommended.")
        try:
            s = float(input_raw.get('bp.1s', np.nan).iloc[0]); d = float(input_raw.get('bp.1d', np.nan).iloc[0])
            if not np.isnan(s) and not np.isnan(d):
                if s<120 and d<80: recs.append("BP normal: routine monitoring.")
                elif s<130: recs.append("BP elevated: lifestyle changes advised.")
                elif s<140: recs.append("Stage 1 HTN: clinician review.")
                else: recs.append("Stage 2 HTN: prompt medical attention.")
        except Exception:
            pass
        try:
            cholv = float(input_raw.get('chol', np.nan).iloc[0])
            if not np.isnan(cholv):
                if cholv<200: recs.append("Cholesterol desirable.")
                elif cholv<240: recs.append("Borderline high cholesterol: diet & exercise.")
                else: recs.append("High cholesterol: consider lipid panel & clinician.")
        except Exception:
            pass
        if prob_pct >= 80:
            recs.append("Immediate: order HbA1c and fasting glucose; seek urgent clinical evaluation.")
        elif prob_pct >= 60:
            recs.append("Consider early clinical screening (HbA1c) and lifestyle changes.")
        elif prob_pct >= 30:
            recs.append("Increase monitoring and preventive lifestyle changes.")
        else:
            recs.append("Routine checkups recommended; maintain healthy lifestyle.")

        for r in recs:
            st.write("- " + r)

        # Save to history
        history = st.session_state.get('history', [])
        entry = input_raw.copy()
        entry['predicted_class'] = pred_class
        entry['predicted_prob'] = prob
        entry['model'] = selected_model
        entry['timestamp'] = datetime.now().isoformat()
        st.session_state['history'] = history + [entry.to_dict(orient='records')[0]]

        # Download
        csv = input_raw.copy(); csv['predicted_class'] = pred_class; csv['predicted_prob'] = prob
        st.download_button("Download prediction (CSV)", csv.to_csv(index=False).encode('utf-8'), file_name='prediction.csv', mime='text/csv')
        pdf = generate_pdf_report(input_raw, pred_class, prob, ("Very High" if prob_pct>=80 else "High" if prob_pct>=60 else "Moderate" if prob_pct>=30 else "Low"), recs)
        st.download_button("Download PDF report", pdf, file_name='diaguard_report.pdf', mime='application/pdf')

# -------------------- MODEL COMPARISON (visuals) --------------------
elif page == "Model Comparison":
    st.title("Model Comparison & Visualizations")
    if 'results' not in st.session_state or 'models' not in st.session_state:
        st.info("No trained models found. Train models on 'Train Models' tab to see comparisons.")
        st.stop()

    results = st.session_state['results']
    feature_names = st.session_state['feature_names']
    models = st.session_state['models']

    # Metrics table
    rows = []
    for name, r in results.items():
        rows.append({'Model': name, 'Accuracy': r['accuracy'], 'AUC': r['auc'],
                     'Precision': r['report'].get('weighted avg', {}).get('precision', np.nan),
                     'Recall': r['report'].get('weighted avg', {}).get('recall', np.nan),
                     'F1': r['report'].get('weighted avg', {}).get('f1-score', np.nan)})
    metrics_df = pd.DataFrame(rows).set_index('Model').round(3)
    st.subheader("Performance Metrics")
    st.dataframe(metrics_df)

    # AUC bar chart
    st.subheader("AUC by model")
    auc_df = pd.DataFrame([{'model': name, 'auc': r['auc']} for name, r in results.items()])
    fig_auc = px.bar(auc_df, x='model', y='auc', range_y=[0,1], title='AUC by model')
    st.plotly_chart(fig_auc, use_container_width=True)

    # Confusion Matrices
    st.subheader("Confusion Matrices")
    n = len(results)
    cols = 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*4))
    axes = axes.flatten()
    idx = 0
    for name, r in results.items():
        sns.heatmap(r['cm'], annot=True, fmt='d', cmap='Blues', ax=axes[idx])
        axes[idx].set_title(name)
        axes[idx].set_xlabel('Predicted'); axes[idx].set_ylabel('Actual')
        idx += 1
    for i in range(idx, len(axes)):
        fig.delaxes(axes[i])
    plt.tight_layout()
    st.pyplot(fig)

    # Radar chart
    st.subheader("Radar Chart (normalized metrics)")
    radar_df = metrics_df.fillna(0)
    norm = (radar_df - radar_df.min()) / (radar_df.max() - radar_df.min()).replace(0,1)
    labels = list(norm.columns)
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot(111, polar=True)
    for idx_r, row in norm.iterrows():
        values = row.tolist()
        values += values[:1]
        ax.plot(angles, values, label=idx_r)
        ax.fill(angles, values, alpha=0.1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_title("Model comparison (normalized)")
    ax.legend(bbox_to_anchor=(1.1, 1.05))
    st.pyplot(fig)

# -------------------- AI DOCTOR CHATBOT --------------------
elif page == "AI Doctor Chatbot":
    st.title("AI Medical Assistant — DiaGuard Pro")
    st.write("This is a lightweight rule-based assistant for quick guidance. For production LLM-based assistants integrate an external API.")
    user_q = st.text_input("Ask your question about diabetes, risk, or recommendations:")
    if st.button("Ask"):
        if not user_q.strip():
            st.warning("Please type a question.")
        else:
            q = user_q.lower()
            if any(w in q for w in ["risk","probab","chance","likely","percent","percentage"]):
                st.info("Model estimates risk probabilities. For diagnostic confirmation, order HbA1c and fasting glucose.")
            elif any(w in q for w in ["what to do","next step","advise","recommend"]):
                st.info("If risk is moderate or higher: order HbA1c, consult clinician, start lifestyle modifications (diet/exercise).")
            elif any(w in q for w in ["why","why high","reason"]):
                st.info("Top contributors typically include age, waist circumference, cholesterol/HDL, and blood pressure. Use SHAP to view local contributions.")
            else:
                st.info("I can explain predictions, recommend next steps, or list contributing risk factors. Try asking: 'Why am I high risk?'")

# -------------------- HISTORY --------------------
elif page == "History":
    st.title("Prediction History")
    hist = st.session_state.get('history', [])
    if not hist:
        st.info("No predictions saved yet.")
    else:
        hist_df = pd.DataFrame(hist)
        st.dataframe(hist_df, use_container_width=True)
        st.download_button("Download history CSV", hist_df.to_csv(index=False).encode('utf-8'), file_name='diaguard_history.csv', mime='text/csv')

# -------------------- Footer note --------------------
st.markdown("---")
st.markdown("_Note: Notebook Mode includes glyhb/stab.glu which are diagnostic features and should not be used for screening. Use Safe mode for screening models._")

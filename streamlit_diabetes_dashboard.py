# streamlit_diabetes_dashboard.py
"""
DiaGuard Pro — Upgraded Predict Page
Professional-grade Diabetes Dashboard (SMOTE + Calibration + SVM + SHAP)
Predict page will render even if models aren't trained: controls are disabled until training is done.
Other features preserved: training, SMOTE, calibration, SHAP, AI Assistant, PDF export, history.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
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
from imblearn.over_sampling import SMOTE
import plotly.graph_objects as go
import plotly.express as px
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import warnings
warnings.filterwarnings("ignore")

# Optional packages (skip gracefully if missing)
try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None
try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None

# ---------------- App config ----------------
st.set_page_config(page_title="DiaGuard Pro — Diabetes Risk Prediction", layout="wide", initial_sidebar_state="expanded")
RANDOM_STATE = 42
DATA_PATH = "diabetes (1).csv"

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

def create_target_and_clean(df):
    df = df.copy()
    df.columns = df.columns.str.strip()
    if 'glyhb' not in df.columns:
        raise ValueError("Dataset must contain 'glyhb' column to create target.")
    df['diabetes'] = (df['glyhb'] >= 6.5).astype(int)
    return df

def preprocessing_no_scale(df_input, remove_leakage=True):
    df = df_input.copy()
    df.columns = df.columns.str.strip()
    df = create_target_and_clean(df) if 'diabetes' not in df.columns else df
    leakage_cols = ['glyhb','ratio','stab.glu']
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
    return X, y

def preprocess_for_training(df_full, remove_leakage=True):
    X_df, y = preprocessing_no_scale(df_full, remove_leakage=remove_leakage)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df)
    feature_names = X_df.columns.tolist()
    return X_scaled, y, feature_names, scaler, X_df

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

def align_features_for_prediction(user_raw_df, df_training_raw, feature_names, scaler, remove_leakage=True):
    # Concatenate and run preprocessing_no_scale on concat to ensure dummies match
    df_temp = pd.concat([user_raw_df.reset_index(drop=True), df_training_raw.reset_index(drop=True)], ignore_index=True)
    X_all_unscaled, _ = preprocessing_no_scale(df_temp, remove_leakage=remove_leakage)
    user_unscaled = X_all_unscaled.iloc[[0]].copy()
    for col in feature_names:
        if col not in user_unscaled.columns:
            user_unscaled[col] = 0.0
    user_unscaled = user_unscaled[feature_names]
    user_scaled = scaler.transform(user_unscaled)
    return user_scaled

# ----------------- Training routine (SMOTE + Calibrated) -----------------
@st.cache_resource
def train_with_smote_and_calibration(df_raw, remove_leakage=True, models_to_train=None):
    if models_to_train is None:
        models_to_train = ['RandomForest','XGBoost','LightGBM','SVM']
    X_scaled, y, feature_names, scaler, X_unscaled_df = preprocess_for_training(df_raw, remove_leakage=remove_leakage)
    X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    sm = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = sm.fit_resample(X_tr, y_tr)
    models = {}
    results = {}

    # RandomForest
    if 'RandomForest' in models_to_train:
        rf = RandomForestClassifier(n_estimators=300, max_depth=12, class_weight='balanced', random_state=RANDOM_STATE)
        rf_cal = CalibratedClassifierCV(rf, cv=5, method='sigmoid')
        rf_cal.fit(X_res, y_res)
        models['RandomForest'] = rf_cal

    # XGBoost
    if 'XGBoost' in models_to_train and XGBClassifier is not None:
        posw = (y_res==0).sum() / max(1, (y_res==1).sum())
        xgb = XGBClassifier(n_estimators=300, learning_rate=0.03, max_depth=5, subsample=0.75, colsample_bytree=0.8, scale_pos_weight=posw, use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_STATE)
        xgb_cal = CalibratedClassifierCV(xgb, cv=5, method='sigmoid')
        xgb_cal.fit(X_res, y_res)
        models['XGBoost'] = xgb_cal

    # LightGBM
    if 'LightGBM' in models_to_train and LGBMClassifier is not None:
        lgb = LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31, class_weight='balanced', random_state=RANDOM_STATE)
        lgb_cal = CalibratedClassifierCV(lgb, cv=5, method='sigmoid')
        lgb_cal.fit(X_res, y_res)
        models['LightGBM'] = lgb_cal

    # SVM (tuned small grid)
    if 'SVM' in models_to_train:
        base_svc = SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=RANDOM_STATE)
        # small grid to keep runtime reasonable
        param_grid = {'C':[1,2], 'gamma':['scale', 0.01]}
        gs = GridSearchCV(base_svc, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
        gs.fit(X_res, y_res)
        best = gs.best_estimator_
        svm_cal = CalibratedClassifierCV(best, cv=3, method='sigmoid')
        svm_cal.fit(X_res, y_res)
        models['SVM'] = svm_cal

    # Evaluate
    for name, model in models.items():
        pred = model.predict(X_te)
        try:
            probs = model.predict_proba(X_te)[:,1]
        except Exception:
            probs = model.decision_function(X_te)
            if probs.ndim>1:
                probs = probs.ravel()
        results[name] = {
            'accuracy': accuracy_score(y_te, pred),
            'auc': roc_auc_score(y_te, probs) if len(np.unique(y_te))>1 else None,
            'cm': confusion_matrix(y_te, pred),
            'pred': pred,
            'probs': probs,
            'report': classification_report(y_te, pred, output_dict=True)
        }
    artifacts = {
        'scaler': scaler,
        'feature_names': feature_names,
        'X_unscaled_df': X_unscaled_df,
        'X_train_unscaled_shape': X_tr.shape
    }
    return models, results, artifacts

# ---------------- load dataset ----------------
df = load_data(DATA_PATH)

# ---------------- Sidebar & navigation ----------------
st.sidebar.header("Navigation")
page = st.sidebar.radio('', ['Train Models', 'Predict', 'AI Doctor Chatbot', 'History'])

st.sidebar.markdown("---")
st.sidebar.write(f"Dataset: {df.shape[0]} rows | {df.shape[1]} cols")
st.sidebar.markdown("Note: Use 'Train Models' to (re)train models. Predict page will be enabled after training.")

# ---------------- TRAIN MODELS page ----------------
if page == 'Train Models':
    st.title("Train & Compare Models — DiaGuard Pro (Professional)")
    mode_choice = st.radio("Mode", ("Safe - no glyhb leakage", "Notebook - include glyhb (diagnostic)"), index=0)
    remove_leakage = True if mode_choice.startswith("Safe") else False
    if mode_choice.startswith("Notebook"):
        st.warning("Notebook Mode includes glyhb/stab.glu/ratio which are diagnostic features and can inflate accuracy.")
    models_to_train = st.multiselect("Models to train", options=['RandomForest','XGBoost','LightGBM','SVM'], default=['RandomForest','XGBoost','LightGBM','SVM'])

    if st.button("Train (SMOTE + Calibrated Models)"):
        with st.spinner("Training models with SMOTE, calibration and small SVM tuning (this may take a minute)..."):
            trained_models, eval_results, artifacts = train_with_smote_and_calibration(df, remove_leakage=remove_leakage, models_to_train=models_to_train)
            # Save to session
            st.session_state['models'] = trained_models
            st.session_state['results'] = eval_results
            st.session_state['scaler'] = artifacts['scaler']
            st.session_state['feature_names'] = artifacts['feature_names']
            st.session_state['remove_leakage'] = remove_leakage
            st.session_state['df_raw'] = df
        st.success("Training complete and models saved in session.")

        # Display results summary
        rows = []
        for name,res in eval_results.items():
            rows.append({
                'model': name,
                'accuracy': res['accuracy'],
                'auc': res['auc'],
                'precision': res['report'].get('weighted avg', {}).get('precision', np.nan),
                'recall': res['report'].get('weighted avg', {}).get('recall', np.nan),
                'f1': res['report'].get('weighted avg', {}).get('f1-score', np.nan)
            })
        metrics_df = pd.DataFrame(rows).set_index('model').round(3)
        st.subheader("Model Performance (on held-out test set)")
        st.dataframe(metrics_df)

        # Feature importances (use base_estimator)
        st.subheader("Feature Importances (Tree Models)")
        for name, model in trained_models.items():
            try:
                # unwrap calibrated wrapper
                inner = getattr(model, 'base_estimator', None) or getattr(model, 'estimator', None) or model
                if not hasattr(inner, 'feature_importances_'):
                    st.info(f"{name} does not provide feature_importances_. Use SHAP for explanations.")
                    continue
                importances = inner.feature_importances_
                idx = np.argsort(importances)[::-1][:15]
                fig, ax = plt.subplots(figsize=(8,3))
                sns.barplot(x=importances[idx], y=np.array(artifacts['feature_names'])[idx], ax=ax)
                ax.set_title(f"Top Features — {name}")
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"Could not show feature importances for {name}: {e}")

# ---------------- PREDICT page (UPGRADED to always render UI) ----------------
elif page == 'Predict':
    st.title("Diabetes Risk Prediction — DiaGuard Pro")
    st.markdown("Enter patient details and compute risk. If models are not trained yet, controls will be disabled. Train models on the 'Train Models' page.")

    models_present = 'models' in st.session_state and isinstance(st.session_state['models'], dict) and len(st.session_state['models'])>0
    models = st.session_state['models'] if models_present else {}
    feature_names = st.session_state.get('feature_names', None)
    scaler = st.session_state.get('scaler', None)
    df_raw = st.session_state.get('df_raw', df)

    # Model selector (disabled until models available)
    if models_present:
        selected_model = st.selectbox("Choose Model", options=list(models.keys()))
    else:
        st.info("No trained models found. Train models on the 'Train Models' page to enable prediction.")
        # display a disabled selectbox lookalike
        selected_model = None

    # Predict input controls - disabled if no models
    disable_controls = not models_present

    st.subheader("Patient Inputs")
    with st.form(key="predict_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=0, max_value=120, value=int(clamp(df['age'].median(),0,120)), disabled=disable_controls)
            gender = st.selectbox("Gender", df['gender'].dropna().unique().tolist(), disabled=disable_controls)
            weight = st.number_input("Weight (kg)", min_value=20.0, max_value=200.0, value=float(clamp(df['weight'].median(),20.0,200.0)), disabled=disable_controls)
            height = st.number_input("Height (cm)", min_value=100.0, max_value=220.0, value=float(clamp(df['height'].median(),100.0,220.0)), disabled=disable_controls)
            frame = st.selectbox("Frame", df['frame'].dropna().unique().tolist() if 'frame' in df.columns else ['M'], disabled=disable_controls)
        with col2:
            chol = st.number_input("Cholesterol (chol)", min_value=50.0, max_value=400.0, value=float(clamp(df['chol'].median(),50.0,400.0)), disabled=disable_controls)
            hdl = st.number_input("HDL", min_value=10.0, max_value=150.0, value=float(clamp(df['hdl'].median(),10.0,150.0)), disabled=disable_controls)
            sys_bp = st.number_input("Systolic BP (bp.1s)", min_value=80.0, max_value=220.0, value=float(clamp(df['bp.1s'].median() if 'bp.1s' in df.columns else 120.0,80.0,220.0)), disabled=disable_controls)
            dia_bp = st.number_input("Diastolic BP (bp.1d)", min_value=40.0, max_value=140.0, value=float(clamp(df['bp.1d'].median() if 'bp.1d' in df.columns else 80.0,40.0,140.0)), disabled=disable_controls)
        waist = st.number_input("Waist (cm)", min_value=30.0, max_value=200.0, value=float(clamp(df['waist'].median() if 'waist' in df.columns else 80.0,30.0,200.0)), disabled=disable_controls)
        hip = st.number_input("Hip (cm)", min_value=30.0, max_value=200.0, value=float(clamp(df['hip'].median() if 'hip' in df.columns else 90.0,30.0,200.0)), disabled=disable_controls)
        location_options = df['location'].dropna().unique().tolist() if 'location' in df.columns else ['Unknown']
        location = st.selectbox("Location", location_options, disabled=disable_controls)
        submitted = st.form_submit_button("Predict Risk", disabled=disable_controls)

    # If controls disabled, show helpful CTA
    if disable_controls:
        st.info("Prediction controls are disabled because no trained models were found. Click 'Train' on the Train Models page, then return here.")
        # show last training metrics if available
        if 'results' in st.session_state:
            st.write("Last training metrics are available in session.")
        st.stop()  # important: stop further execution to avoid using missing artifacts

    # If we reach here, controls are enabled and model(s) present
    # Build input dataframe
    input_raw = pd.DataFrame([{
        'age': age, 'gender': gender, 'height': height, 'weight': weight, 'chol': chol, 'hdl': hdl,
        'bp.1s': sys_bp, 'bp.1d': dia_bp, 'waist': waist, 'hip': hip, 'frame': frame, 'location': location
    }])

    # Align features + scale
    try:
        input_scaled = align_features_for_prediction(input_raw, df_raw, st.session_state['feature_names'], st.session_state['scaler'], remove_leakage=st.session_state.get('remove_leakage', True))
    except Exception as e:
        st.error(f"Failed to align features for prediction: {e}")
        st.stop()

    # Predict
    model = st.session_state['models'][selected_model]
    try:
        prob = float(model.predict_proba(input_scaled)[:,1][0])
    except Exception:
        raw = model.decision_function(input_scaled)
        prob = float(1/(1+np.exp(-raw))[0])
    pred_class = int(prob >= 0.5)
    prob_pct = round(prob*100, 2)

    # Display result
    st.markdown("---")
    st.subheader("Prediction Summary")
    st.markdown(f"**Model:** {selected_model} | **Mode:** {'Notebook' if not st.session_state.get('remove_leakage', True) else 'Safe'}")
    st.metric(label="Diabetes risk probability", value=f"{prob_pct}%")
    # gauge
    color = "green" if prob_pct < 30 else "orange" if prob_pct < 60 else "red"
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob_pct,
        domain={'x':[0,1],'y':[0,1]},
        title={'text':"Risk Gauge (%)"},
        gauge={'axis':{'range':[0,100]}, 'bar':{'color':color}}
    ))
    st.plotly_chart(gauge, use_container_width=True)

    # SHAP explanation
    st.subheader("Explainability (SHAP)")
    if shap is None:
        st.info("shap not installed. Install shap for explanations.")
    else:
        try:
            # For tree models use TreeExplainer on base_estimator if calibrated wrapper
            base_model = getattr(model, 'base_estimator', None) or getattr(model, 'estimator', None) or model
            if selected_model in ['RandomForest','XGBoost','LightGBM']:
                # build unscaled user row for SHAP tree explainer
                df_temp = pd.concat([input_raw.reset_index(drop=True), df_raw.reset_index(drop=True)], ignore_index=True)
                X_unscaled_all, _ = preprocessing_no_scale(df_temp, remove_leakage=st.session_state.get('remove_leakage', True))
                user_unscaled = X_unscaled_all.iloc[[0]]
                explainer = shap.TreeExplainer(base_model)
                shap_values = explainer.shap_values(user_unscaled)
                # single sample waterfall
                if isinstance(shap_values, list):
                    sv = shap_values[1] if len(shap_values)>1 else shap_values[0]
                    base_val = explainer.expected_value[1] if hasattr(explainer.expected_value, '__len__') and len(explainer.expected_value)>1 else explainer.expected_value
                    expl = shap.Explanation(values=np.asarray(sv)[0], base_values=base_val, data=user_unscaled.iloc[0].values, feature_names=user_unscaled.columns.tolist())
                else:
                    arr = np.asarray(shap_values)
                    expl = shap.Explanation(values=arr[0], base_values=explainer.expected_value, data=user_unscaled.iloc[0].values, feature_names=user_unscaled.columns.tolist())
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
                        st.info("SHAP plot not available in this environment.")
                # interactive force (html)
                try:
                    fp = shap.force_plot(base_val, np.asarray(sv)[0], user_unscaled.iloc[0], matplotlib=False)
                    import streamlit.components.v1 as components
                    components.html(f"<head>{shap.getjs()}</head><body>{fp.html()}</body>", height=380)
                except Exception:
                    pass
            elif selected_model == 'SVM':
                st.info("Kernel SHAP for SVM (background sample reduced for speed). This may be slow.")
                df_temp = pd.concat([input_raw.reset_index(drop=True), df_raw.reset_index(drop=True)], ignore_index=True)
                X_unscaled_all, _ = preprocessing_no_scale(df_temp, remove_leakage=st.session_state.get('remove_leakage', True))
                background = shap.sample(X_unscaled_all, min(100, len(X_unscaled_all)))
                explainer = shap.KernelExplainer(model.predict_proba, background)
                user_unscaled = X_unscaled_all.iloc[[0]]
                shap_values = explainer.shap_values(user_unscaled)
                vals = np.asarray(shap_values[1])[0] if isinstance(shap_values, list) and len(shap_values)>1 else np.asarray(shap_values).ravel()
                base_val = explainer.expected_value[1] if hasattr(explainer.expected_value, '__len__') and len(explainer.expected_value)>1 else explainer.expected_value
                expl = shap.Explanation(values=vals, base_values=base_val, data=user_unscaled.iloc[0].values, feature_names=user_unscaled.columns.tolist())
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
                        st.info("SHAP plot not available for SVM in this environment.")
                try:
                    fp = shap.force_plot(base_val, vals, user_unscaled.iloc[0], matplotlib=False)
                    import streamlit.components.v1 as components
                    components.html(f"<head>{shap.getjs()}</head><body>{fp.html()}</body>", height=380)
                except Exception:
                    pass
            else:
                st.info("Model not supported for SHAP in this environment.")
        except Exception as e:
            st.warning(f"SHAP failed: {e}")

    # medical recommendations
    st.subheader("Doctor-style Recommendations")
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
        s = float(input_raw.get('bp.1s', np.nan).iloc[0]); d = float(input_raw.get('bp.1d', np.nan).iloc[0])
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
    if prob_pct < 30:
        recs.append("Routine checks recommended; re-evaluate annually.")
    elif prob_pct < 60:
        recs.append("Consider lifestyle modifications and monitoring.")
    elif prob_pct < 80:
        recs.append("Early clinical screening and further blood tests advised.")
    else:
        recs.append("Immediate medical evaluation recommended.")

    for r in recs:
        st.write("- " + r)

    # save history
    hist = st.session_state.get('history', [])
    entry = input_raw.copy()
    entry['predicted_class'] = pred_class
    entry['predicted_prob'] = prob
    entry['model'] = selected_model
    entry['timestamp'] = datetime.now().isoformat()
    st.session_state['history'] = hist + [entry.to_dict(orient='records')[0]]

    # downloads
    csv_bytes = input_raw.copy(); csv_bytes['predicted_class'] = pred_class; csv_bytes['predicted_prob'] = prob
    st.download_button("Download prediction (CSV)", csv_bytes.to_csv(index=False).encode('utf-8'), file_name='prediction.csv', mime='text/csv')
    pdf = generate_pdf_report(input_raw, pred_class, prob, ("Very High" if prob_pct>=80 else "High" if prob_pct>=60 else "Moderate" if prob_pct>=30 else "Low"), recs)
    st.download_button("Download PDF report", data=pdf, file_name="diabetes_report.pdf", mime="application/pdf")

# ---------------- AI Doctor Chatbot ----------------
elif page == 'AI Doctor Chatbot':
    st.title("AI Medical Assistant — DiaGuard Pro")
    st.write("A simple local rule-based assistant. For more advanced chatbots integrate a hosted LLM or API.")
    user_q = st.text_input("Ask a question about diabetes risk or results:")
    if st.button("Ask"):
        if not user_q.strip():
            st.warning("Please enter a question.")
        else:
            q = user_q.lower()
            if any(w in q for w in ["risk","probab","chance","percentage","likely"]):
                st.info("Risk is estimated by the trained model. For a full diagnosis, perform clinical tests (HbA1c, fasting glucose).")
            elif any(w in q for w in ["what to do","next step","recommend","advise"]):
                st.info("If risk is moderate or higher: order HbA1c, consult clinician for personalized plan, adopt lifestyle changes.")
            else:
                st.info("I can explain model results, suggest next steps, and list features that typically contribute to high risk (age, BMI, waist, cholesterol).")

# ---------------- History ----------------
elif page == 'History':
    st.title("Prediction History")
    hist = st.session_state.get('history', [])
    if not hist:
        st.info("No predictions yet.")
    else:
        hist_df = pd.DataFrame(hist)
        st.dataframe(hist_df, use_container_width=True)
        st.download_button("Download history CSV", hist_df.to_csv(index=False).encode('utf-8'), file_name='history.csv', mime='text/csv')

st.markdown("---")
st.markdown("_Note: Notebook Mode includes glyhb/stab.glu/ratio which may cause optimistic accuracy. Use Safe Mode for real-world screening._")

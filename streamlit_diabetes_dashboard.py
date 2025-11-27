# streamlit_diabetes_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import time
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, roc_curve
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import plotly.graph_objects as go

# Optional packages
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

# ---- Config ----
st.set_page_config(page_title="Diabetes Models Dashboard", layout="wide", initial_sidebar_state="expanded")
DATA_PATH = "diabetes (1).csv"  # adjust if needed
RANDOM_STATE = 42

# ---- Utility / helper functions ----
@st.cache_data
def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df

def preprocess_df(df, remove_leakage=True):
    """Return X_scaled, y, feature_names, scaler.
       remove_leakage=True -> drop glyhb, ratio, stab.glu (safe mode)
       remove_leakage=False -> notebook mode (keep glyhb etc)"""
    df = df.copy()
    df.columns = df.columns.str.strip()
    if 'glyhb' not in df.columns:
        raise ValueError("Dataset must contain 'glyhb' column for target creation.")
    df['diabetes'] = (df['glyhb'] >= 6.5).astype(int)

    leakage_cols = ['glyhb', 'ratio', 'stab.glu']
    drop_cols = ['id']
    if remove_leakage:
        drop_cols += leakage_cols

    X = df.drop(columns=[c for c in drop_cols + ['diabetes'] if c in df.columns], errors='ignore')
    y = df['diabetes']

    # Impute numeric and categorical
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
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

@st.cache_resource
def train_models_cached(X_train, y_train, models_to_train=None):
    return train_models(X_train, y_train, models_to_train=models_to_train)

def train_models(X_train, y_train, models_to_train=None):
    models = {}
    if models_to_train is None:
        models_to_train = ['RandomForest', 'LogisticRegression', 'SVM', 'XGBoost', 'LightGBM']

    if 'RandomForest' in models_to_train:
        rf = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=RANDOM_STATE)
        rf.fit(X_train, y_train)
        models['RandomForest'] = rf

    if 'LogisticRegression' in models_to_train:
        lr = LogisticRegression(class_weight='balanced', solver='liblinear', C=0.3, max_iter=500, random_state=RANDOM_STATE)
        lr.fit(X_train, y_train)
        models['LogisticRegression'] = lr

    if 'SVM' in models_to_train:
        svm = SVC(kernel='rbf', C=2, probability=True, random_state=RANDOM_STATE)
        svm.fit(X_train, y_train)
        models['SVM'] = svm

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
        if probs is not None and len(np.unique(y_test)) > 1:
            auc = roc_auc_score(y_test, probs)
    except Exception:
        auc = None

    cm = confusion_matrix(y_test, pred)
    report = classification_report(y_test, pred, output_dict=True)
    return {'accuracy': acc, 'auc': auc, 'cm': cm, 'report': report, 'pred': pred, 'probs': probs}

def safe_shap_explanation(model, X_background, inp_df, model_name):
    """Return a shap.Explanation object (single-sample) or raise informative exception."""
    if not SHAP_AVAILABLE:
        raise RuntimeError("shap is not installed")

    # Determine explainer & get shap_values
    try:
        if model_name in ['RandomForest','LightGBM','XGBoost']:
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(inp_df)  # might be list/array
            base = explainer.expected_value
        elif model_name == 'LogisticRegression':
            # LinearExplainer expects training data (background)
            try:
                explainer = shap.LinearExplainer(model, X_background, feature_dependence="independent")
                sv = explainer.shap_values(inp_df)
                base = explainer.expected_value
            except Exception:
                explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X_background, min(100, len(X_background))))
                sv = explainer.shap_values(inp_df)
                base = explainer.expected_value
        else:
            # fallback to KernelExplainer for SVM or others
            explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X_background, min(100, len(X_background))))
            sv = explainer.shap_values(inp_df)
            base = explainer.expected_value

        # Normalize to single-sample positive-class SHAP vector
        # sv could be: list (n_classes) each (n_samples,n_features) or array
        if isinstance(sv, list):
            # pick class 1 if exists else class 0
            idx = 1 if len(sv) > 1 else 0
            arr = np.asarray(sv[idx])
            shap_vec = arr[0]
            base_val = base[idx] if isinstance(base, (list, tuple, np.ndarray)) and len(base) > idx else base
        else:
            arr = np.asarray(sv)
            if arr.ndim == 3:
                # (n_classes, n_samples, n_features)
                shap_vec = arr[1,0,:] if arr.shape[0] > 1 else arr[0,0,:]
                base_val = base[1] if hasattr(base, "__len__") and len(base) > 1 else base
            elif arr.ndim == 2:
                # (n_samples, n_features)
                shap_vec = arr[0,:]
                base_val = base
            else:
                shap_vec = arr.ravel()
                base_val = base

        expl = shap.Explanation(values=shap_vec, base_values=base_val, data=inp_df.iloc[0])
        return expl
    except Exception as e:
        raise RuntimeError(f"SHAP explanation failed: {e}")

def generate_pdf_report(patient_input_df, predicted_class, probability, risk_label, rec_list):
    """Create a simple PDF report in memory and return bytes."""
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, height-60, "Diabetes Risk Prediction Report")
    c.setFont("Helvetica", 10)
    c.drawString(40, height-80, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.line(40, height-90, width-40, height-90)

    # Patient inputs
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, height-110, "Patient Inputs:")
    c.setFont("Helvetica", 10)
    y = height-130
    for col, val in patient_input_df.iloc[0].items():
        c.drawString(48, y, f"{col}: {val}")
        y -= 14
        if y < 120:
            c.showPage()
            y = height - 60

    # Prediction summary
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

# ---- App Layout & Logic ----
df = load_data()

# Sidebar - Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Train & Compare", "Predict", "History"])

# Sidebar - global controls
st.sidebar.markdown("### Global Options")
use_leakage_default = False
use_leakage_sidebar = st.sidebar.checkbox("Enable Notebook Mode (include glyhb-derived features)", value=use_leakage_default)
models_to_train = st.sidebar.multiselect("Models to train", options=['RandomForest','LogisticRegression','SVM','XGBoost','LightGBM'],
                                         default=['RandomForest','XGBoost','LightGBM'])
st.sidebar.markdown("---")
st.sidebar.write(f"Dataset rows: {df.shape[0]}  columns: {df.shape[1]}")

# Top header
if page != 'Predict':
    st.title("Diabetes Models — Comparison & Prediction")
    st.markdown("Interactive dashboard: Train multiple models, compare metrics, and predict individual risk.")
else:
    st.title("Diabetes Risk Prediction")
    st.markdown("Use the left panel to enter patient data (Simple or Advanced). Tip: Train models first under Train & Compare.")

# HOME page content
if page == "Home":
    col1, col2 = st.columns([2,1])
    with col1:
        st.header("Project Overview")
        st.write("""
        This dashboard trains and compares multiple classifiers for diabetes risk prediction.
        Two modes are provided:
        - **Notebook Mode** (includes glyhb/stab.glu/ratio) — reproduces notebook experiments (may include leakage).
        - **Safe Mode** (drops glyhb/stab.glu/ratio) — realistic non-invasive screening.
        """)
        st.markdown("**Quick start:** Train models in Train & Compare → Inspect metrics → Use Predict to try inputs.")
    with col2:
        st.header("Dataset Snapshot")
        st.dataframe(df.head(10), use_container_width=True)

# TRAIN & COMPARE
if page == "Train & Compare":
    st.header("Train & Evaluate Models")
    st.write("Choose mode and press **Train & Evaluate**. Training is cached to speed up repeat runs.")

    mode_toggle = st.radio("Mode", ("Safe (no leakage)", "Notebook (with glyhb)"),
                          index=0 if not use_leakage_sidebar else 1)
    remove_leakage = True if mode_toggle.startswith("Safe") else False

    if mode_toggle.startswith("Notebook"):
        st.warning("Notebook Mode includes glyhb-derived features which directly influence the target — results will be optimistic.")

    if st.button("Train & Evaluate Models"):
        with st.spinner("Preprocessing and training..."):
            X_scaled, y, feature_names, scaler = preprocess_df(df, remove_leakage=remove_leakage)
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

            trained_models = train_models(X_train, y_train, models_to_train=models_to_train)
            results = {}
            for name, m in trained_models.items():
                results[name] = evaluate_model(m, X_test, y_test)

            # metrics
            metrics_df = pd.DataFrame([{
                'model': name,
                'accuracy': res['accuracy'],
                'auc': res['auc'],
                'precision': res['report'].get('weighted avg', {}).get('precision', np.nan),
                'recall': res['report'].get('weighted avg', {}).get('recall', np.nan),
                'f1': res['report'].get('weighted avg', {}).get('f1-score', np.nan)
            } for name,res in results.items()]).sort_values('accuracy', ascending=False).reset_index(drop=True)

            st.subheader("Model Comparison Table")
            st.dataframe(metrics_df.set_index('model').round(3), use_container_width=True)

            # Advanced comparison visuals
            st.markdown("---")
            st.subheader("Advanced Comparison Graphs")
            tabs = st.tabs(["Metrics", "Confusion Matrices", "ROC Comparison", "Radar", "Leaderboard"])
            with tabs[0]:
                fig, ax = plt.subplots(1,2,figsize=(14,4))
                sns.barplot(x='model', y='accuracy', data=metrics_df, ax=ax[0])
                ax[0].set_ylim(0,1)
                ax[0].set_title('Accuracy')
                sns.barplot(x='model', y='auc', data=metrics_df, ax=ax[1])
                ax[1].set_ylim(0,1)
                ax[1].set_title('AUC')
                st.pyplot(fig)

            with tabs[1]:
                n = len(results)
                cols = 2
                rows = (n + cols - 1)//cols
                fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*4))
                axes = axes.flatten()
                for i, (name, res) in enumerate(results.items()):
                    sns.heatmap(res['cm'], annot=True, fmt='d', cmap='Blues', ax=axes[i])
                    axes[i].set_title(name)
                    axes[i].set_xlabel('Predicted')
                    axes[i].set_ylabel('Actual')
                for j in range(i+1, len(axes)):
                    fig.delaxes(axes[j])
                plt.tight_layout()
                st.pyplot(fig)

            with tabs[2]:
                fig, ax = plt.subplots(figsize=(8,6))
                for name, res in results.items():
                    if res['probs'] is not None:
                        fpr, tpr, _ = roc_curve(y_test, res['probs'])
                        ax.plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.3f})")
                ax.plot([0,1],[0,1],'--', color='gray')
                ax.set_xlabel('FPR'); ax.set_ylabel('TPR'); ax.set_title('ROC Comparison'); ax.legend()
                st.pyplot(fig)

            with tabs[3]:
                radar_df = metrics_df.set_index('model')[['accuracy','precision','recall','f1','auc']].fillna(0)
                norm = (radar_df - radar_df.min())/(radar_df.max()-radar_df.min()).replace(0,1)
                labels = list(norm.columns)
                angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
                angles += angles[:1]
                fig = plt.figure(figsize=(6,6))
                ax = fig.add_subplot(111, polar=True)
                for idx, row in norm.iterrows():
                    values = row.tolist(); values += values[:1]
                    ax.plot(angles, values, label=idx); ax.fill(angles, values, alpha=0.1)
                ax.set_thetagrids(np.degrees(angles[:-1]), labels)
                ax.set_title('Metrics Radar')
                ax.legend(bbox_to_anchor=(1.1,1.05))
                st.pyplot(fig)

            with tabs[4]:
                st.subheader("Leaderboard")
                for i,row in metrics_df.iterrows():
                    st.metric(label=row['model'], value=f"Acc {row['accuracy']:.3f}", delta=f"AUC {row['auc']:.3f}" if not pd.isna(row['auc']) else "AUC N/A")

            # Feature importances for tree models
            st.subheader("Feature Importances (top 15)")
            for name,m in trained_models.items():
                try:
                    importances = m.feature_importances_
                    idx = np.argsort(importances)[::-1][:15]
                    fig, ax = plt.subplots(figsize=(8,3))
                    sns.barplot(x=importances[idx], y=np.array(feature_names)[idx], ax=ax)
                    ax.set_title(f"Top features - {name}")
                    st.pyplot(fig)
                except Exception:
                    st.info(f"{name} does not expose feature_importances_")

            # Save to session state
            st.session_state['models'] = trained_models
            st.session_state['scaler'] = scaler
            st.session_state['feature_names'] = feature_names
            st.session_state['use_leakage'] = not remove_leakage  # True if notebook mode
            st.success("Training complete. Models are available in session.")

# PREDICT page
if page == "Predict":
    st.header("Predict Diabetes Risk (Simple / Advanced)")

    if 'models' not in st.session_state:
        st.warning("No trained models found. Please train models in Train & Compare first.")
        st.stop()

    # Input layout
    left, right = st.columns([1,1])
    with left:
        mode = st.radio("Input Mode", ["Simple", "Advanced"], index=0)
        model_choice = st.selectbox("Choose model", options=list(st.session_state['models'].keys()))
        notebook_mode = st.checkbox("Use Notebook Mode (may expect glyhb features)", value=st.session_state.get('use_leakage', False))
        st.markdown("---")
        st.subheader("Patient Information (Inputs)")

        if mode == "Simple":
            # pick sensible defaults from df
            age = st.number_input("Age", 0, 120, int(df['age'].median()))
            gender = st.selectbox("Gender", df['gender'].dropna().unique().tolist())
            weight = st.number_input("Weight (kg)", 20.0, 200.0, float(df['weight'].median()))
            height = st.number_input("Height (cm)", 100.0, 220.0, float(df['height'].median()))
            chol = st.number_input("Cholesterol (chol)", 50.0, 400.0, float(df['chol'].median()))
            hdl = st.number_input("HDL", 10.0, 150.0, float(df['hdl'].median()))
            sys_bp = st.number_input("Systolic BP (bp.1s)", 80.0, 220.0, float(df['bp.1s'].median()) if 'bp.1s' in df.columns else 120.0)
            dia_bp = st.number_input("Diastolic BP (bp.1d)", 40.0, 140.0, float(df['bp.1d'].median()) if 'bp.1d' in df.columns else 80.0)
            waist = st.number_input("Waist (cm)", 40.0, 200.0, float(df['waist'].median()) if 'waist' in df.columns else 80.0)
            hip = st.number_input("Hip (cm)", 40.0, 200.0, float(df['hip'].median()) if 'hip' in df.columns else 90.0)
            frame = st.selectbox("Frame", df['frame'].dropna().unique().tolist() if 'frame' in df.columns else ['M'])
            location = st.selectbox("Location", df['location'].dropna().unique().tolist())
            input_df = pd.DataFrame([{
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
            input_df = pd.DataFrame([adv_inputs])

        st.markdown("---")
        predict_btn = st.button("Predict Risk")

    with right:
        st.subheader("Prediction Result")
        result_slot = st.empty()
        shap_slot = st.empty()

    # Prediction logic
    if predict_btn:
        # Preprocess: use the selected mode's preprocessing (safe vs notebook)
        remove_leakage = not notebook_mode  # notebook_mode True => remove_leakage False
        try:
            X_all, y_all, feat_names, scaler = preprocess_df(df, remove_leakage=remove_leakage)
        except Exception as e:
            st.error(f"Preprocessing failed: {e}")
            st.stop()

        # Align input to feat_names
        for col in feat_names:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[feat_names]
        input_df = input_df.apply(pd.to_numeric, errors='coerce').fillna(0)

        # Scale using session scaler if exists (from training), else the one from preprocess
        scaler_used = st.session_state.get('scaler', scaler)
        input_scaled = scaler_used.transform(input_df)

        # get model
        model = st.session_state['models'].get(model_choice)
        if model is None:
            st.error("Selected model not available in session. Re-train models first.")
            st.stop()

        # Predict probability
        try:
            if hasattr(model, "predict_proba"):
                prob = float(model.predict_proba(input_scaled)[:,1][0])
            else:
                raw = model.decision_function(input_scaled)
                prob = float(1 / (1 + np.exp(-raw))[0])
        except Exception as e:
            st.error(f"Model prediction failed: {e}")
            st.stop()

        pred_class = int(prob >= 0.5)
        prob_pct = round(prob*100, 1)

        # Risk label
        if prob < 0.3:
            risk_label, color = "Low", "#2ecc71"
            short_msg = "Low risk — maintain healthy lifestyle"
        elif prob < 0.6:
            risk_label, color = "Moderate", "#f39c12"
            short_msg = "Moderate risk — consider lifestyle changes"
        elif prob < 0.8:
            risk_label, color = "High", "#e74c3c"
            short_msg = "High risk — seek medical advice"
        else:
            risk_label, color = "Very High", "#8b0000"
            short_msg = "Very high risk — immediate clinical evaluation recommended"

        # Display polished card
        card = f"""
        <div style="background: linear-gradient(135deg,#f3fbff 0%, #e6f2ff 100%);
                    padding:16px;border-radius:10px;box-shadow:0 6px 20px rgba(0,0,0,0.08);">
          <h3 style="margin:4px 0 8px 0;">Prediction Summary</h3>
          <div style="display:flex;gap:16px;align-items:center">
            <div style="width:120px;text-align:center">
              <div style="font-size:28px;font-weight:700">{prob_pct}%</div>
              <div style="font-size:12px;color:gray">Probability</div>
            </div>
            <div style="flex:1">
              <div><strong>Predicted:</strong> {'Diabetic' if pred_class==1 else 'Non-diabetic'} ({pred_class})</div>
              <div><strong>Model:</strong> {model_choice} • <strong>Mode:</strong> {'Notebook' if not remove_leakage else 'Safe'}</div>
              <div style="margin-top:8px;"><span style="color:{color};font-weight:700">{risk_label}</span> — {short_msg}</div>
            </div>
          </div>
        </div>
        """
        result_slot.markdown(card, unsafe_allow_html=True)
        st.metric(label="Diabetes Risk Probability", value=f"{prob_pct}%")

        # Plotly circular gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob_pct,
            domain={'x':[0,1], 'y':[0,1]},
            title={'text':"Risk Gauge (%)"},
            gauge={'axis': {'range':[0,100]},
                   'bar': {'color': color},
                   'steps': [
                       {'range':[0,30], 'color': "#e6f8ea"},
                       {'range':[30,60], 'color': "#fff5e6"},
                       {'range':[60,80], 'color': "#ffe6e6"},
                       {'range':[80,100], 'color': "#ffe6e6"}]
                  }
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)

        # SHAP explanation (robust)
        if SHAP_AVAILABLE and model_choice in ['RandomForest','LightGBM','XGBoost','LogisticRegression','SVM']:
            try:
                expl = safe_shap_explanation(model, X_all, input_df, model_choice)
                shap_slot.subheader("Local Explanation (SHAP)")
                # Try waterfall; if fails, fallback to bar
                try:
                    plt.clf()
                    shap.plots.waterfall(expl, show=False)
                    st.pyplot(plt.gcf())
                except Exception:
                    plt.clf()
                    shap.plots.bar(expl, show=False)
                    st.pyplot(plt.gcf())
            except Exception as e:
                shap_slot.info(f"SHAP explanation unavailable: {e}")
        else:
            shap_slot.info("SHAP not available for this model or shap not installed.")

        # Doctor-style recommendations rules
        recs = []
        # BMI
        try:
            wt = float(input_df.get('weight', 0).iloc[0])
            ht = float(input_df.get('height', 0).iloc[0])
            bmi = wt / ((ht/100)**2) if ht > 0 else np.nan
        except Exception:
            bmi = np.nan
        if not np.isnan(bmi):
            if bmi < 18.5:
                recs.append("Underweight: consider nutritional assessment.")
            elif bmi < 25:
                recs.append("Normal BMI: maintain balanced diet and activity.")
            elif bmi < 30:
                recs.append("Overweight: aim for gradual weight reduction (5-10%).")
            else:
                recs.append("Obese: consider medical weight-loss program and clinician consult.")

        # BP
        try:
            s = float(input_df.get('bp.1s', np.nan).iloc[0])
            d = float(input_df.get('bp.1d', np.nan).iloc[0])
            if not np.isnan(s) and not np.isnan(d):
                if s < 120 and d < 80:
                    recs.append("BP: normal; continue regular monitoring.")
                elif s < 130:
                    recs.append("BP: elevated; lifestyle changes advised.")
                elif s < 140:
                    recs.append("BP: stage 1; discuss with clinician.")
                else:
                    recs.append("BP: stage 2; immediate clinical evaluation advised.")
        except Exception:
            pass

        # Cholesterol
        try:
            cholv = float(input_df.get('chol', np.nan).iloc[0])
            if not np.isnan(cholv):
                if cholv < 200:
                    recs.append("Cholesterol: desirable. Maintain healthy diet.")
                elif cholv < 240:
                    recs.append("Cholesterol: borderline high; consider diet/exercise.")
                else:
                    recs.append("Cholesterol: high; clinical lipid evaluation recommended.")
        except Exception:
            pass

        # Risk-specific
        if risk_label == "Very High":
            recs.append("Immediate action: request HbA1c/fasting glucose tests and consult healthcare provider.")
        elif risk_label == "High":
            recs.append("Consider early clinical screening (HbA1c) and lifestyle interventions.")
        elif risk_label == "Moderate":
            recs.append("Increase monitoring frequency and adopt preventive measures.")
        else:
            recs.append("Continue routine checks; re-evaluate annually or earlier if symptoms appear.")

        st.subheader("Doctor-style Recommendations")
        for r in recs:
            st.write("- " + r)

        # Save prediction to history in session_state
        hist = st.session_state.get('history', [])
        rec_text = " | ".join(recs[:3])  # short version
        hist_entry = input_df.copy()
        hist_entry['predicted_class'] = pred_class
        hist_entry['predicted_prob'] = prob
        hist_entry['model'] = model_choice
        hist_entry['mode'] = 'Notebook' if not remove_leakage else 'Safe'
        hist_entry['timestamp'] = datetime.now().isoformat()
        st.session_state['history'] = hist + [hist_entry.to_dict(orient='records')[0]]
        st.success("Prediction saved to history.")

        # Download CSV of this prediction
        csv_bytes = input_df.copy()
        csv_bytes['predicted_class'] = pred_class
        csv_bytes['predicted_prob'] = prob
        csv = csv_bytes.to_csv(index=False).encode('utf-8')
        st.download_button("Download prediction (CSV)", csv, file_name="prediction.csv", mime="text/csv")

        # Generate PDF
        pdf_bytes = generate_pdf_report(input_df, pred_class, prob, risk_label, recs)
        st.download_button("Download PDF Report", data=pdf_bytes, file_name="diabetes_report.pdf", mime="application/pdf")

# HISTORY page
if page == "History":
    st.header("Prediction History")
    hist = st.session_state.get('history', [])
    if not hist:
        st.info("No predictions saved yet.")
    else:
        hist_df = pd.DataFrame(hist)
        st.dataframe(hist_df, use_container_width=True)
        st.download_button("Download History (CSV)", hist_df.to_csv(index=False).encode('utf-8'), "history.csv", "text/csv")

# Footer
st.markdown("---")
st.markdown("**Note:** Notebook Mode includes glyhb-derived features which can leak diagnostic information into training and cause optimistic results. Use Safe Mode for real-world screening.")

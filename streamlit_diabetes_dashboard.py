# streamlit_diabetes_dashboard_full.py
"""
Complete upgraded Streamlit dashboard:
- Train & Compare (Safe and Notebook modes)
- Predict page with SHAP (force + 3D-like Plotly view)
- Doctor-style recommendations
- AI Diagnosis Chatbot (local, rule-based + uses model)
- Prediction history + PDF report
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io, base64, time
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
st.set_page_config(page_title="Diabetes Models Dashboard (Full)", layout="wide", initial_sidebar_state="expanded")
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

def preprocess_df(df, remove_leakage=True):
    """Return X_scaled, y, feature_names, scaler"""
    df = df.copy()
    df.columns = df.columns.str.strip()
    if 'glyhb' not in df.columns:
        raise ValueError("Dataset must contain 'glyhb' column.")
    df['diabetes'] = (df['glyhb'] >= 6.5).astype(int)

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
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, X.columns.tolist(), scaler

def train_models(X_train, y_train, models_to_train=None):
    if models_to_train is None:
        models_to_train = ['RandomForest','LogisticRegression','SVM','XGBoost','LightGBM']
    models = {}
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

# ---------------- App ----------------
df = load_data()

# Sidebar navigation & global
st.sidebar.title("Controls")
page = st.sidebar.radio("Navigation", ["Home","Train & Compare","Predict","History"])
st.sidebar.markdown("---")
use_leakage_default = False
enable_notebook_mode = st.sidebar.checkbox("Enable Notebook Mode (include glyhb features)", value=use_leakage_default)
models_to_train = st.sidebar.multiselect("Models to train", options=['RandomForest','LogisticRegression','SVM','XGBoost','LightGBM'],
                                         default=['RandomForest','XGBoost','LightGBM'])
st.sidebar.write(f"Dataset rows: {df.shape[0]}  columns: {df.shape[1]}")

# header
if page != "Predict":
    st.title("Diabetes Models — Comparison & Prediction")
else:
    st.title("Diabetes Risk Prediction")

# Home
if page == "Home":
    st.header("Project overview")
    st.write("This dashboard trains models and provides an interactive prediction interface.")
    st.subheader("Dataset preview")
    st.dataframe(df.head(8))

# Train & Compare
if page == "Train & Compare":
    st.header("Train & Evaluate Models")
    mode_choice = st.radio("Mode", ("Safe (no leakage)", "Notebook (with glyhb)"),
                          index=0 if not enable_notebook_mode else 1)
    remove_leakage = True if mode_choice.startswith("Safe") else False
    if mode_choice.startswith("Notebook"):
        st.warning("Notebook mode includes glyhb/stab.glu/ratio — results will be optimistic (diagnostic).")
    if st.button("Train & Evaluate"):
        with st.spinner("Preprocessing and training..."):
            X_scaled, y, feature_names, scaler = preprocess_df(df, remove_leakage=remove_leakage)
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
            models = train_models(X_train, y_train, models_to_train=models_to_train)
            results = {name:evaluate_model(m, X_test, y_test) for name,m in models.items()}

            # metrics df
            metrics_df = pd.DataFrame([{
                'model': name,
                'accuracy': res['accuracy'],
                'auc': res['auc'],
                'precision': res['report'].get('weighted avg',{}).get('precision', np.nan),
                'recall': res['report'].get('weighted avg',{}).get('recall', np.nan),
                'f1': res['report'].get('weighted avg',{}).get('f1-score', np.nan)
            } for name,res in results.items()]).sort_values('accuracy', ascending=False).reset_index(drop=True)
            st.subheader("Model Comparison")
            st.dataframe(metrics_df.set_index('model').round(3))

            # ROC comparison
            st.subheader("ROC Comparison")
            fig, ax = plt.subplots(figsize=(7,5))
            for name,res in results.items():
                if res['probs'] is not None:
                    fpr,tpr,_ = roc_curve(y_test, res['probs'])
                    ax.plot(fpr,tpr,label=f"{name} (AUC={res['auc']:.3f})")
            ax.plot([0,1],[0,1],'--',color='gray'); ax.set_xlabel('FPR'); ax.set_ylabel('TPR'); ax.legend()
            st.pyplot(fig)

            # feature importance
            st.subheader("Feature Importances (tree models)")
            for name,m in models.items():
                try:
                    importances = m.feature_importances_
                    idx = np.argsort(importances)[::-1][:15]
                    fig, ax = plt.subplots(figsize=(8,3))
                    sns.barplot(x=importances[idx], y=np.array(feature_names)[idx], ax=ax)
                    ax.set_title(f"Top features - {name}")
                    st.pyplot(fig)
                except Exception:
                    st.info(f"{name} does not expose feature_importances_")

            # save session
            st.session_state['models'] = models
            st.session_state['scaler'] = scaler
            st.session_state['feature_names'] = feature_names
            st.session_state['remove_leakage'] = remove_leakage
            st.session_state['results'] = results
            st.success("Training complete and models saved in session.")

# Predict
if page == "Predict":
    st.header("Predict Diabetes Risk (Simple / Advanced)")

    if 'models' not in st.session_state:
        st.warning("No trained models found in session. Please train models first.")
        st.stop()

    left, right = st.columns([1,1])
    with left:
        mode = st.radio("Input mode", ["Simple","Advanced"])
        model_choice = st.selectbox("Model to use", list(st.session_state['models'].keys()))
        notebook_mode = st.checkbox("Use Notebook Mode (may expect glyhb)", value=not enable_notebook_mode)
        st.markdown("---")
        st.subheader("Patient Inputs")

        # SIMPLE MODE inputs (safe clamped defaults)
        if mode == "Simple":
            age = st.number_input("Age", min_value=0, max_value=120, value=int(clamp(df['age'].median(), 0, 120)))
            gender = st.selectbox("Gender", df['gender'].dropna().unique().tolist())
            weight = st.number_input("Weight (kg)", min_value=20.0, max_value=200.0, value=float(clamp(df['weight'].median(), 20.0, 200.0)))
            height = st.number_input("Height (cm)", min_value=100.0, max_value=220.0, value=float(clamp(df['height'].median(), 100.0, 220.0)))
            chol = st.number_input("Cholesterol (chol)", min_value=50.0, max_value=400.0, value=float(clamp(df['chol'].median(), 50.0, 400.0)))
            hdl = st.number_input("HDL", min_value=10.0, max_value=150.0, value=float(clamp(df['hdl'].median(), 10.0, 150.0)))
            sys_bp = st.number_input("Systolic BP (bp.1s)", min_value=80.0, max_value=220.0, value=float(clamp(df['bp.1s'].median() if 'bp.1s' in df.columns else 120.0, 80.0, 220.0)))
            dia_bp = st.number_input("Diastolic BP (bp.1d)", min_value=40.0, max_value=140.0, value=float(clamp(df['bp.1d'].median() if 'bp.1d' in df.columns else 80.0, 40.0, 140.0)))
            waist = st.number_input("Waist (cm)", min_value=40.0, max_value=200.0, value=float(clamp(df['waist'].median() if 'waist' in df.columns else 80.0,40.0,200.0)))
            hip = st.number_input("Hip (cm)", min_value=40.0, max_value=200.0, value=float(clamp(df['hip'].median() if 'hip' in df.columns else 90.0,40.0,200.0)))
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
        result_area = st.empty()
        shap_area = st.empty()
        force_area = st.empty()
        chatbot_area = st.empty()

    # prediction handling
    if predict_btn:
        remove_leakage = not notebook_mode
        try:
            X_all, y_all, feat_names, scaler = preprocess_df(df, remove_leakage=remove_leakage)
        except Exception as e:
            st.error(f"Preprocessing failed: {e}")
            st.stop()

        # align features
        for c in feat_names:
            if c not in input_df.columns:
                input_df[c] = 0
        input_df = input_df[feat_names]
        input_df = input_df.apply(pd.to_numeric, errors='coerce').fillna(0)

        scaler_used = st.session_state.get('scaler', scaler)
        input_scaled = scaler_used.transform(input_df)

        model = st.session_state['models'].get(model_choice)
        if model is None:
            st.error("Model not found in session. Retrain models.")
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

        pred_class = int(prob>=0.5)
        prob_pct = round(prob*100,1)

        # risk label
        if prob < 0.3:
            risk_label, color, emoji, advice_short = "Low", "#2ecc71", "🟢", "Low risk — maintain healthy lifestyle"
        elif prob < 0.6:
            risk_label, color, emoji, advice_short = "Moderate", "#f39c12", "🟠", "Moderate risk — consider lifestyle changes"
        elif prob < 0.8:
            risk_label, color, emoji, advice_short = "High", "#e74c3c", "🔴", "High risk — seek medical advice"
        else:
            risk_label, color, emoji, advice_short = "Very High", "#8b0000", "🚨", "Very High risk — immediate evaluation"

        # theme aware card (works in dark mode)
        is_dark = st.get_option("theme.base") == "dark"
        if is_dark:
            card_style = f"background-color:#0f1720;color:#fff;padding:14px;border-radius:10px;border:1px solid #283040;"
        else:
            card_style = f"background-color:#f3fbff;color:#000;padding:14px;border-radius:10px;"

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

        # plotly gauge
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

        # SHAP explanations
        if SHAP_AVAILABLE:
            try:
                shap_area.subheader("SHAP - Local Explanation (Waterfall & Force)")
                # build explainer according to model type
                if model_choice in ['RandomForest','XGBoost','LightGBM']:
                    explainer = shap.TreeExplainer(model)
                    sv = explainer.shap_values(input_df)
                    # pick class 1 if list; else handle arr
                    if isinstance(sv, list):
                        vals = sv[1][0] if len(sv)>1 else sv[0][0]
                    else:
                        if np.asarray(sv).ndim == 3:
                            vals = sv[1,0,:]
                        else:
                            vals = sv[0] if np.asarray(sv).shape[0]>1 else sv[0]
                    base_val = explainer.expected_value[1] if isinstance(explainer.expected_value,(list,tuple,np.ndarray)) and len(explainer.expected_value)>1 else explainer.expected_value
                    expl_single = shap.Explanation(values=vals, base_values=base_val, data=input_df.iloc[0], feature_names=input_df.columns.tolist())
                    # waterfall
                    plt.figure(figsize=(7,4), facecolor='white')
                    try:
                        shap.plots.waterfall(expl_single, show=False)
                        st.pyplot(plt.gcf())
                        plt.clf()
                    except Exception:
                        shap.plots.bar(expl_single, show=False)
                        st.pyplot(plt.gcf())
                        plt.clf()
                    # force plot (interactive)
                    try:
                        # shap.force_plot returns HTML/js -- embed using components
                        force_html = shap.plots.force(expl_single, matplotlib=False, show=False)
                        # shap.plots.force with JS returns a Javascript object if shap JS available; fallback to shap.force_plot
                        # Use shap.force_plot (older API) to create html
                        fp = shap.force_plot(base_val, vals, input_df.iloc[0], matplotlib=False)
                        shap_html = f"<head>{shap.getjs()}</head><body>{fp.html()}</body>"
                        import streamlit.components.v1 as components
                        components.html(shap_html, height=350)
                    except Exception:
                        shap_area.info("Interactive SHAP force plot not available in this environment.")
                elif model_choice == 'LogisticRegression':
                    explainer = shap.LinearExplainer(model, X_all, feature_dependence="independent")
                    sv = explainer.shap_values(input_df)
                    vals = sv[0]
                    base_val = explainer.expected_value
                    expl_single = shap.Explanation(values=vals, base_values=base_val, data=input_df.iloc[0], feature_names=input_df.columns.tolist())
                    try:
                        shap.plots.waterfall(expl_single, show=False)
                        st.pyplot(plt.gcf()); plt.clf()
                    except Exception:
                        shap.plots.bar(expl_single, show=False); st.pyplot(plt.gcf()); plt.clf()
                    # force plot (try)
                    try:
                        fp = shap.force_plot(base_val, vals, input_df.iloc[0], matplotlib=False)
                        import streamlit.components.v1 as components
                        components.html(f"<head>{shap.getjs()}</head><body>{fp.html()}</body>", height=350)
                    except Exception:
                        pass
                else:
                    # SVM or others: use KernelExplainer fallback
                    shap_area.info("Using KernelExplainer for model explanation (may take a bit).")
                    background = shap.sample(X_all, min(100, len(X_all)))
                    explainer = shap.KernelExplainer(model.predict_proba, background)
                    sv = explainer.shap_values(input_df)
                    # pick class 1
                    vals = sv[1][0] if isinstance(sv, list) and len(sv)>1 else np.asarray(sv).ravel()
                    base_val = explainer.expected_value[1] if isinstance(explainer.expected_value, (list,tuple,np.ndarray)) and len(explainer.expected_value)>1 else explainer.expected_value
                    expl_single = shap.Explanation(values=vals, base_values=base_val, data=input_df.iloc[0], feature_names=input_df.columns.tolist())
                    try:
                        shap.plots.waterfall(expl_single, show=False); st.pyplot(plt.gcf()); plt.clf()
                    except Exception:
                        shap.plots.bar(expl_single, show=False); st.pyplot(plt.gcf()); plt.clf()
                    # try force plot
                    try:
                        fp = shap.force_plot(base_val, vals, input_df.iloc[0], matplotlib=False)
                        import streamlit.components.v1 as components
                        components.html(f"<head>{shap.getjs()}</head><body>{fp.html()}</body>", height=350)
                    except Exception:
                        pass
            except Exception as e:
                shap_area.error(f"SHAP explanation failed: {e}")
        else:
            shap_area.info("SHAP not installed. Install 'shap' to enable interactive explanations.")

        # 3D-like SHAP view (Plotly) - contributions per feature
        try:
            contribs = np.array(vals).flatten()
            feat_names_plot = input_df.columns.tolist()
            df_shap = pd.DataFrame({'feature': feat_names_plot, 'contrib': contribs})
            df_shap['abs'] = df_shap['contrib'].abs()
            df_shap = df_shap.sort_values('abs', ascending=False).head(12)
            fig3d = px.bar(df_shap, x='feature', y='contrib', color='contrib', color_continuous_scale='RdBu', title='Top SHAP contributions (interactive)')
            st.plotly_chart(fig3d, use_container_width=True)
        except Exception:
            pass

        # doctor-style recommendations (rules)
        recs = []
        try:
            wt = float(input_df.get('weight', 0).iloc[0])
            ht = float(input_df.get('height', 0).iloc[0])
            bmi = wt/((ht/100)**2) if ht>0 else np.nan
        except Exception:
            bmi = np.nan
        if not np.isnan(bmi):
            if bmi < 18.5: recs.append("Underweight: consider nutrition consult.")
            elif bmi <25: recs.append("Normal BMI: maintain diet and exercise.")
            elif bmi <30: recs.append("Overweight: target gradual weight reduction (5-10%).")
            else: recs.append("Obese: structured weight-loss program and clinician consult advised.")
        try:
            s = float(input_df.get('bp.1s', np.nan).iloc[0])
            d = float(input_df.get('bp.1d', np.nan).iloc[0])
            if not np.isnan(s) and not np.isnan(d):
                if s<120 and d<80: recs.append("BP normal: continue monitoring.")
                elif s<130: recs.append("Elevated BP: lifestyle changes recommended.")
                elif s<140: recs.append("Stage 1 HTN: discuss with clinician.")
                else: recs.append("Stage 2 HTN: seek medical care.")
        except Exception:
            pass
        try:
            cholv = float(input_df.get('chol', np.nan).iloc[0])
            if not np.isnan(cholv):
                if cholv <200: recs.append("Cholesterol desirable.")
                elif cholv <240: recs.append("Borderline high cholesterol: diet/lifestyle.")
                else: recs.append("High cholesterol: clinical lipid evaluation recommended.")
        except Exception:
            pass
        if risk_label == "Very High":
            recs.append("Immediate: order HbA1c/fasting glucose and consult clinician.")
        elif risk_label == "High":
            recs.append("Early screening advised (HbA1c).")
        elif risk_label == "Moderate":
            recs.append("Increase monitoring and adopt preventive lifestyle changes.")
        else:
            recs.append("Routine checks; re-evaluate annually.")

        st.subheader("Doctor-style Recommendations")
        for r in recs:
            st.write("- " + r)

        # Save history
        hist = st.session_state.get('history', [])
        entry = input_df.copy()
        entry['predicted_class'] = pred_class
        entry['predicted_prob'] = prob
        entry['model'] = model_choice
        entry['mode'] = 'Notebook' if not remove_leakage else 'Safe'
        entry['timestamp'] = datetime.now().isoformat()
        st.session_state['history'] = hist + [entry.to_dict(orient='records')[0]]
        st.success("Saved to session history.")

        # Download CSV and PDF
        csv_bytes = input_df.copy()
        csv_bytes['predicted_class'] = pred_class
        csv_bytes['predicted_prob'] = prob
        st.download_button("Download prediction (CSV)", csv_bytes.to_csv(index=False).encode('utf-8'), file_name='prediction.csv', mime='text/csv')

        pdf = generate_pdf_report(input_df, pred_class, prob, risk_label, recs)
        st.download_button("Download PDF report", data=pdf, file_name="diabetes_report.pdf", mime="application/pdf")

        # AI Diagnosis Chatbot (local rule-based assistant)
        chatbot_area.subheader("AI Diagnosis Chatbot")
        chatbot_area.write("Ask the assistant for simple explanations, next steps, or interpretation of the prediction.")
        # We'll implement a small local chat interface
        if 'chat_history' not in st.session_state:
            st.session_state['chat_history'] = []
        user_msg = st.text_input("Type your question to the assistant and press Enter:", key="chat_input")
        if user_msg:
            # simple rule-based replies using prediction and features
            reply = "I'm an assistant — here's what I can say:\n"
            # if asking about risk
            low_q = any(w in user_msg.lower() for w in ["risk","probab","how likely","chance"])
            action_q = any(w in user_msg.lower() for w in ["what should i do","advise","recommend","next step","test"])
            explain_q = any(w in user_msg.lower() for w in ["why","explain","because","reason","important","feature"])
            if low_q:
                reply = f"The model reports a probability of {prob_pct}%. Risk bucket: {risk_label}. "
                reply += "This is a screening estimate and not a diagnosis."
            elif action_q:
                reply = "Based on the risk and your inputs: "
                reply += "1) Consider ordering HbA1c or fasting glucose if risk is Moderate or higher.\n"
                reply += "2) Lifestyle: weight management, reduce sugar intake, increase activity.\n"
                reply += "3) See a clinician for personalized treatment if risk is High/Very High."
            elif explain_q:
                # use top shap features for explanation if available
                try:
                    top_feats = df_shap = None
                    try:
                        df_shap = pd.DataFrame({'feature': input_df.columns.tolist(), 'contrib': list(vals)})
                        df_shap['abs'] = df_shap['contrib'].abs()
                        df_shap = df_shap.sort_values('abs', ascending=False).head(5)
                        top_feats = ", ".join([f"{row.feature} ({row.contrib:.2f})" for row in df_shap.itertuples()])
                        reply = f"The top contributing features (local) are: {top_feats}. Positive contrib increases risk; negative decreases risk."
                    except Exception:
                        reply = "I couldn't compute detailed feature contributions, but typical important factors include age, waist/hip, cholesterol and HDL."
                except Exception:
                    reply = "No explanation available."
            else:
                reply = "I can explain the prediction, give advice, and suggest next steps. Try asking 'Why was I predicted high risk?' or 'What should I do next?'"

            st.session_state['chat_history'].append({"user":user_msg,"bot":reply})
        # display chat history
        for turn in st.session_state.get('chat_history', [])[::-1]:
            st.markdown(f"**You:** {turn['user']}")
            st.markdown(f"**Assistant:** {turn['bot']}")

# History page
if page == "History":
    st.header("Prediction History")
    hist = st.session_state.get('history', [])
    if not hist:
        st.info("No history.")
    else:
        hist_df = pd.DataFrame(hist)
        st.dataframe(hist_df, use_container_width=True)
        st.download_button("Download History CSV", hist_df.to_csv(index=False).encode('utf-8'), "history.csv", "text/csv")

st.markdown("---")
st.markdown("_Note: Notebook Mode includes glyhb/stab.glu/ratio which can leak diagnostic info into predictions — use Safe mode for screening._")

# streamlit_diabetes_fixed.py
"""
DiaGuard Pro — Ultimate (Full) Dashboard (Fixed & Enhanced)
- Fixes Feature Importance (unwraps CalibratedClassifierCV)
- Fixes SHAP integration
- Adds File Uploader
- Uses Plotly for Feature Graphs
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

# Sklearn imports
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

import warnings
warnings.filterwarnings("ignore")

# Optional imports
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    shap = None
    SHAP_AVAILABLE = False

# -------------------- Config --------------------
st.set_page_config(page_title="DiaGuard Pro — Fixed", layout="wide", initial_sidebar_state="expanded")
DEFAULT_PATH = "diabetes (1).csv"
RANDOM_STATE = 42

# -------------------- Utility Functions --------------------

def get_base_model(model):
    """
    Unwraps a CalibratedClassifierCV to get the actual fitted estimator
    for Feature Importance and SHAP.
    """
    # Check if it is a CalibratedClassifierCV
    if hasattr(model, 'calibrated_classifiers_') and model.calibrated_classifiers_:
        # Return the estimator from the first split
        return model.calibrated_classifiers_[0].estimator
    # Check if it is a GridSearchCV or RandomizedSearchCV
    elif hasattr(model, 'best_estimator_'):
        return model.best_estimator_
    # Return model as is
    return model

@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            df.columns = df.columns.str.strip()
            return df
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return None
    
    # Fallback to local path if no upload
    try:
        df = pd.read_csv(DEFAULT_PATH)
        df.columns = df.columns.str.strip()
        return df
    except FileNotFoundError:
        return None

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
    df = df.copy()
    if 'glyhb' in df.columns:
        df['diabetes'] = (df['glyhb'] >= 6.5).astype(int)
    return df

def preprocessing_no_scale(df_input, remove_leakage=True):
    df = df_input.copy()
    df.columns = df.columns.str.strip()
    df = create_target(df) if 'glyhb' in df.columns else df
    
    leak_cols = ['glyhb', 'ratio', 'stab.glu']
    drop_cols = ['id']
    if remove_leakage:
        drop_cols += leak_cols
    
    # Drop columns that exist in df
    cols_to_drop = [c for c in drop_cols + ['diabetes'] if c in df.columns]
    X = df.drop(columns=cols_to_drop, errors='ignore')
    
    y = df['diabetes'] if 'diabetes' in df.columns else None

    # Identification
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    # Impute
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
    if y is None:
        raise ValueError("Target column could not be created. Ensure dataset has 'glyhb'.")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df)
    return X_scaled, y, X_df.columns.tolist(), scaler, X_df

def align_features_for_prediction(user_raw_df, df_training_raw, feature_names, scaler, remove_leakage=True):
    # Concat to ensure one-hot encoding columns match
    df_temp = pd.concat([user_raw_df.reset_index(drop=True), df_training_raw.reset_index(drop=True)], ignore_index=True)
    X_all_unscaled, _ = preprocessing_no_scale(df_temp, remove_leakage=remove_leakage)
    
    # Extract user row (first row)
    user_unscaled = X_all_unscaled.iloc[[0]].copy()
    
    # Ensure all columns from training exist, fill missing with 0
    for col in feature_names:
        if col not in user_unscaled.columns:
            user_unscaled[col] = 0.0
            
    # Reorder columns to match training exactly
    user_unscaled = user_unscaled[feature_names]
    
    # Scale
    user_scaled = scaler.transform(user_unscaled)
    return user_scaled, user_unscaled

def plot_feature_importance_plotly(model, feature_names, model_name, max_feats=15):
    """
    Robust Feature Importance using Plotly.
    Unwraps CalibratedClassifierCV to find the actual tree model.
    """
    base_model = get_base_model(model)

    importances = None
    
    # 1. Try native feature_importances_ (Trees)
    if hasattr(base_model, 'feature_importances_'):
        importances = base_model.feature_importances_
        
    # 2. Try coef_ (Linear models, though SVM RBF doesn't have it)
    elif hasattr(base_model, 'coef_'):
        importances = np.abs(base_model.coef_[0])
        
    if importances is not None:
        # Create DataFrame
        fi_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=True).tail(max_feats)
        
        fig = px.bar(fi_df, x='Importance', y='Feature', orientation='h', 
                     title=f"Top {max_feats} Features ({model_name})",
                     color='Importance', color_continuous_scale='Viridis')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"Model '{model_name}' (type: {type(base_model).__name__}) does not provide native feature importances. Use SHAP below for insights.")

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
        if y < 100:
            c.showPage(); y = height - 60
            
    y -= 10
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "Prediction Summary:")
    y -= 16
    c.setFont("Helvetica", 10)
    c.drawString(48, y, f"Prediction: {'Diabetic Risk' if predicted_class==1 else 'Low Risk'}")
    y -= 14
    c.drawString(48, y, f"Probability: {probability:.1%}")
    y -= 14
    c.drawString(48, y, f"Risk Level: {risk_label}")
    
    y -= 25
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "Recommendations:")
    y -= 16
    c.setFont("Helvetica", 10)
    for rec in rec_list:
        c.drawString(48, y, f"- {rec}")
        y -= 14
        if y < 50:
            c.showPage(); y = height - 60
            
    c.save(); buffer.seek(0)
    return buffer.read()

# -------------------- Training Pipeline --------------------
@st.cache_resource
def train_models_pipeline(df_raw, remove_leakage=True, models_to_train=None):
    if models_to_train is None:
        models_to_train = ['RandomForest', 'XGBoost', 'LightGBM', 'SVM']

    X_scaled, y, feature_names, scaler, X_unscaled_df = preprocess_for_training(df_raw, remove_leakage=remove_leakage)
    X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    sm = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = sm.fit_resample(X_tr, y_tr)

    trained = {}
    eval_results = {}

    # 1. Random Forest
    if 'RandomForest' in models_to_train:
        rf = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced', random_state=RANDOM_STATE)
        rf_cal = CalibratedClassifierCV(rf, cv=3, method='sigmoid')
        rf_cal.fit(X_res, y_res)
        trained['RandomForest'] = rf_cal

    # 2. XGBoost
    if 'XGBoost' in models_to_train and XGBClassifier is not None:
        xgb = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, eval_metric='logloss', random_state=RANDOM_STATE)
        xgb_cal = CalibratedClassifierCV(xgb, cv=3, method='sigmoid')
        xgb_cal.fit(X_res, y_res)
        trained['XGBoost'] = xgb_cal

    # 3. LightGBM
    if 'LightGBM' in models_to_train and LGBMClassifier is not None:
        lgb = LGBMClassifier(n_estimators=200, learning_rate=0.05, random_state=RANDOM_STATE, verbose=-1)
        lgb_cal = CalibratedClassifierCV(lgb, cv=3, method='sigmoid')
        lgb_cal.fit(X_res, y_res)
        trained['LightGBM'] = lgb_cal

    # 4. SVM
    if 'SVM' in models_to_train:
        svc = SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=RANDOM_STATE)
        # Simple fit then calibrate
        svm_cal = CalibratedClassifierCV(svc, cv=3, method='sigmoid')
        svm_cal.fit(X_res, y_res)
        trained['SVM'] = svm_cal

    # Evaluate
    for name, model in trained.items():
        pred = model.predict(X_te)
        probs = model.predict_proba(X_te)[:,1]
        
        acc = accuracy_score(y_te, pred)
        auc = roc_auc_score(y_te, probs)
        cm = confusion_matrix(y_te, pred)
        report = classification_report(y_te, pred, output_dict=True)
        
        eval_results[name] = {'accuracy': acc, 'auc': auc, 'cm': cm, 'report': report}

    artifacts = {'scaler': scaler, 'feature_names': feature_names, 'X_unscaled_df': X_unscaled_df}
    return trained, eval_results, artifacts

# -------------------- Main App Logic --------------------

st.sidebar.title("DiaGuard Pro")
uploaded_file = st.sidebar.file_uploader("Upload CSV Data (Optional)", type=["csv"])
df = load_data(uploaded_file)

if df is None:
    st.error("No dataset found. Please upload a CSV file or ensure 'diabetes (1).csv' is in the folder.")
    st.stop()

page = st.sidebar.radio("Navigation", ["Home", "Train Models", "Predict", "Compare Models", "Chatbot"], index=1)
st.sidebar.info(f"Data Loaded: {df.shape[0]} rows, {df.shape[1]} cols")

# --- HOME ---
if page == "Home":
    st.title("DiaGuard Pro — Diabetes AI Suite")
    st.markdown("""
    **Professional Diabetes Risk Prediction Dashboard**
    
    1. **Train**: Build calibrated models with SMOTE balancing.
    2. **Predict**: Real-time risk scoring with SHAP explanations.
    3. **Compare**: View ROC-AUC, Confusion Matrices, and Feature Importance.
    """)
    st.dataframe(df.head())

# --- TRAIN MODELS ---
elif page == "Train Models":
    st.title("Model Training Studio")
    
    mode = st.radio("Training Mode", ["Safe (No Leakage)", "Notebook (Includes glyhb)"], index=0)
    remove_leakage = (mode == "Safe (No Leakage)")
    
    if mode.startswith("Notebook"):
        st.warning("Warning: 'Notebook' mode uses diagnostic variables (glyhb) which are usually unknown before diagnosis. Use 'Safe' for screening.")

    available_models = ['RandomForest', 'SVM']
    if XGBClassifier: available_models.append('XGBoost')
    if LGBMClassifier: available_models.append('LightGBM')
    
    selected_models = st.multiselect("Choose Algorithms", available_models, default=available_models)
    
    if st.button("Start Training Pipeline"):
        with st.spinner("Training & Calibrating Models..."):
            models, results, artifacts = train_models_pipeline(df, remove_leakage, selected_models)
            
            st.session_state['models'] = models
            st.session_state['results'] = results
            st.session_state['artifacts'] = artifacts
            st.session_state['remove_leakage'] = remove_leakage
            st.session_state['df_raw'] = df
            
            st.success("Training Complete!")
            
            # Leaderboard
            res_list = []
            for k, v in results.items():
                res_list.append({'Model': k, 'Accuracy': v['accuracy'], 'AUC': v['auc']})
            res_df = pd.DataFrame(res_list).sort_values(by='AUC', ascending=False)
            st.table(res_df)

            # Feature Importance Graphs (Now Working!)
            st.subheader("Global Feature Importance")
            cols = st.columns(2)
            for i, (name, model) in enumerate(models.items()):
                with cols[i % 2]:
                    plot_feature_importance_plotly(model, artifacts['feature_names'], name)

# --- PREDICT ---
elif page == "Predict":
    st.title("Clinical Prediction Interface")
    
    if 'models' not in st.session_state:
        st.warning("Please train models first in the 'Train Models' tab.")
        st.stop()
        
    models = st.session_state['models']
    scaler = st.session_state['artifacts']['scaler']
    feature_names = st.session_state['artifacts']['feature_names']
    remove_leakage = st.session_state['remove_leakage']
    df_raw = st.session_state['df_raw']
    
    sel_model_name = st.selectbox("Select Model", list(models.keys()))
    curr_model = models[sel_model_name]

    # Input Form
    with st.form("input_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age", 10, 100, 50)
            gender = st.selectbox("Gender", df['gender'].unique() if 'gender' in df else ['male', 'female'])
            weight = st.number_input("Weight (kg)", 30.0, 200.0, 75.0)
            height = st.number_input("Height (cm)", 100.0, 220.0, 170.0)
        with c2:
            chol = st.number_input("Cholesterol", 50.0, 500.0, 200.0)
            hdl = st.number_input("HDL", 10.0, 150.0, 50.0)
            bp_sys = st.number_input("Systolic BP", 80, 220, 120)
            bp_dia = st.number_input("Diastolic BP", 40, 140, 80)
        with c3:
            waist = st.number_input("Waist (cm)", 50.0, 150.0, 90.0)
            hip = st.number_input("Hip (cm)", 50.0, 150.0, 100.0)
            frame = st.selectbox("Frame", df['frame'].unique() if 'frame' in df else ['medium'])
            location = st.selectbox("Location", df['location'].unique() if 'location' in df else ['Louisa'])
            
        submit = st.form_submit_button("Analyze Risk")
        
    if submit:
        # Create Input DF
        user_input = pd.DataFrame([{
            'age': age, 'gender': gender, 'weight': weight, 'height': height,
            'chol': chol, 'hdl': hdl, 'bp.1s': bp_sys, 'bp.1d': bp_dia,
            'waist': waist, 'hip': hip, 'frame': frame, 'location': location
        }])
        
        # Align Features
        X_user_scaled, X_user_unscaled = align_features_for_prediction(user_input, df_raw, feature_names, scaler, remove_leakage)
        
        # Predict
        prob = curr_model.predict_proba(X_user_scaled)[0][1]
        pred_cls = 1 if prob >= 0.5 else 0
        risk_pct = prob * 100
        
        # Display Result
        st.divider()
        col_res, col_gauge = st.columns([1, 1])
        
        with col_res:
            color = "green" if risk_pct < 40 else "orange" if risk_pct < 70 else "red"
            st.markdown(f"<h2 style='color:{color}'>Probability: {risk_pct:.2f}%</h2>", unsafe_allow_html=True)
            st.write(f"Class: **{'Diabetic' if pred_cls else 'Non-Diabetic'}**")
            
            # Recommendations
            recs = []
            if prob > 0.6: recs.append("Consult a doctor immediately.")
            if chol > 240: recs.append("High cholesterol detected.")
            if bp_sys > 140: recs.append("Hypertension detected.")
            if waist > 100: recs.append("Central obesity detected.")
            
            if recs:
                st.info("Recommendations:\n" + "\n".join([f"- {r}" for r in recs]))
            else:
                st.success("Vitals look generally healthy.")

        with col_gauge:
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = risk_pct,
                title = {'text': "Diabetes Risk"},
                gauge = {'axis': {'range': [0, 100]},
                         'bar': {'color': color},
                         'steps': [
                             {'range': [0, 40], 'color': "lightgreen"},
                             {'range': [40, 70], 'color': "khaki"},
                             {'range': [70, 100], 'color': "salmon"}]}))
            st.plotly_chart(fig, use_container_width=True)

        # SHAP EXPLANATION (FIXED)
        if SHAP_AVAILABLE:
            st.subheader("Why this prediction?")
            try:
                base_est = get_base_model(curr_model)
                
                # SHAP works best with Tree models
                if sel_model_name in ['RandomForest', 'XGBoost', 'LightGBM']:
                    explainer = shap.TreeExplainer(base_est)
                    shap_values = explainer.shap_values(X_user_unscaled)
                    
                    # Handle different SHAP return types (list for classification vs array)
                    if isinstance(shap_values, list):
                        vals = shap_values[1][0]
                        expected_val = explainer.expected_value[1]
                    else:
                        vals = shap_values[0]
                        expected_val = explainer.expected_value

                    # Visuals
                    st_shap_col1, st_shap_col2 = st.columns(2)
                    with st_shap_col1:
                        st.write("**Waterfall Plot (Local Interpretation)**")
                        fig_water, ax = plt.subplots()
                        shap.plots.waterfall(shap.Explanation(
                            values=vals, 
                            base_values=expected_val, 
                            data=X_user_unscaled.iloc[0], 
                            feature_names=X_user_unscaled.columns
                        ), show=False)
                        st.pyplot(fig_water)
                        
                    with st_shap_col2:
                         # Simple Bar summary of SHAP
                        st.write("**Impact of Features**")
                        shap_df = pd.DataFrame({
                            'Feature': X_user_unscaled.columns,
                            'Impact': vals
                        }).sort_values(by='Impact', key=abs, ascending=True).tail(10)
                        
                        fig_bar = px.bar(shap_df, x='Impact', y='Feature', orientation='h')
                        st.plotly_chart(fig_bar, use_container_width=True)
                        
                else:
                    st.info("SHAP visualization is optimized for Tree-based models (RF, XGB, LGBM).")
            except Exception as e:
                st.error(f"Could not generate SHAP plot: {e}")

        # PDF Download
        pdf_data = generate_pdf_report(user_input, pred_cls, prob, "High" if prob>0.7 else "Low", recs)
        st.download_button("Download Report (PDF)", pdf_data, file_name="report.pdf", mime="application/pdf")

# --- COMPARE ---
elif page == "Compare Models":
    st.title("Model Comparison")
    if 'results' not in st.session_state:
        st.warning("Train models first.")
        st.stop()
        
    results = st.session_state['results']
    
    # ROC AUC Comparison
    st.subheader("AUC Performance")
    auc_data = {name: res['auc'] for name, res in results.items()}
    fig_auc = px.bar(x=list(auc_data.keys()), y=list(auc_data.values()), 
                     labels={'x':'Model', 'y':'ROC AUC Score'}, title="Model Accuracy Comparison")
    st.plotly_chart(fig_auc, use_container_width=True)
    
    # Confusion Matrix Gallery
    st.subheader("Confusion Matrices")
    cols = st.columns(2)
    for i, (name, res) in enumerate(results.items()):
        with cols[i % 2]:
            st.write(f"**{name}**")
            fig_cm = px.imshow(res['cm'], text_auto=True, color_continuous_scale='Blues',
                               labels=dict(x="Predicted", y="Actual"))
            st.plotly_chart(fig_cm, use_container_width=True, key=f"cm_{name}")

# --- CHATBOT ---
elif page == "Chatbot":
    st.title("AI Doctor Assistant")
    st.markdown("Ask questions about diabetes risk factors or the model's logic.")
    
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "Hello! I can explain diabetes risk factors or interpret your results. What would you like to know?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        # Simple Rule-Based Logic (Placeholder for LLM)
        response = "I'm a rule-based assistant. "
        p = prompt.lower()
        if "risk" in p:
            response += "Risk is calculated based on Glucose (glyhb), Age, Cholesterol, and BMI."
        elif "reduce" in p or "lower" in p:
            response += "To lower risk: maintain healthy weight, exercise 150min/week, and reduce sugar intake."
        elif "symptoms" in p:
            response += "Common symptoms: increased thirst, frequent urination, hunger, fatigue, and blurred vision."
        else:
            response += "Please consult a medical professional for specific advice."
            
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)

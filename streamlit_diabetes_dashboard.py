import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, roc_curve

# Optional imports
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
    SHAP_AVAILABLE = False

# ---- Configuration ----
DATA_PATH = "diabetes (1).csv"  # local dataset path (use this as file URL)
RANDOM_STATE = 42

st.set_page_config(page_title="Diabetes Models Dashboard", layout="wide")

# ---- Utility functions ----
@st.cache_data
def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df

@st.cache_data
def get_unique_values(df, col):
    return df[col].dropna().unique().tolist()

@st.cache_resource
def train_models(X_train, y_train, X_val=None, y_val=None, models_to_train=None):
    """Train models using notebook's exact hyperparameters. Returns trained models dict."""
    models = {}
    if models_to_train is None:
        models_to_train = ['RandomForest', 'LogisticRegression', 'SVM', 'XGBoost', 'LightGBM']

    if 'RandomForest' in models_to_train:
        rf = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=RANDOM_STATE)
        rf.fit(X_train, y_train)
        models['RandomForest'] = rf

    if 'LogisticRegression' in models_to_train:
        lr = LogisticRegression(class_weight='balanced', solver='liblinear', C=0.3, random_state=RANDOM_STATE, max_iter=500)
        lr.fit(X_train, y_train)
        models['LogisticRegression'] = lr

    if 'SVM' in models_to_train:
        svm = SVC(kernel='rbf', C=2, probability=True, random_state=RANDOM_STATE)
        svm.fit(X_train, y_train)
        models['SVM'] = svm

    if XGBClassifier is not None and 'XGBoost' in models_to_train:
        xgb = XGBClassifier(n_estimators=300, learning_rate=0.03, max_depth=5, subsample=0.75,
                             colsample_bytree=0.8, gamma=0.3, reg_alpha=0.2, reg_lambda=1.0,
                             min_child_weight=2, eval_metric='logloss', random_state=RANDOM_STATE)
        xgb.fit(X_train, y_train)
        models['XGBoost'] = xgb

    if LGBMClassifier is not None and 'LightGBM' in models_to_train:
        lgb = LGBMClassifier(n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=30,
                              subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.3,
                              random_state=RANDOM_STATE)
        lgb.fit(X_train, y_train)
        models['LightGBM'] = lgb

    return models


def preprocess_df(df, remove_leakage=False):
    """Preprocessing matched to notebook pipeline.
    If remove_leakage==False, includes glyhb-related features (notebook mode).
    If remove_leakage==True, drops glyhb, ratio, stab.glu (safe mode).
    """
    df = df.copy()
    df.columns = df.columns.str.strip()
    if 'glyhb' not in df.columns:
        raise ValueError('glyhb column not found in dataframe')
    df['diabetes'] = (df['glyhb'] >= 6.5).astype(int)

    leakage_cols = ['glyhb', 'ratio', 'stab.glu']
    drop_cols = ['id']
    # remove_leakage True -> drop leakage cols (safe mode)
    if remove_leakage:
        drop_cols += leakage_cols

    X = df.drop(columns=[col for col in drop_cols + ['diabetes'] if col in df.columns])
    y = df['diabetes']

    # Imputation
    cat_cols = X.select_dtypes(include=['object','category']).columns.tolist()
    num_cols = X.select_dtypes(include=['number']).columns.tolist()

    if len(num_cols) > 0:
        num_imp = SimpleImputer(strategy='median')
        X[num_cols] = num_imp.fit_transform(X[num_cols])

    if len(cat_cols) > 0:
        cat_imp = SimpleImputer(strategy='most_frequent')
        X[cat_cols] = cat_imp.fit_transform(X[cat_cols])
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    X = X.apply(pd.to_numeric, errors='coerce')
    X.fillna(0, inplace=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, X.columns.tolist(), scaler


def evaluate_model(model, X_test, y_test):
    """Evaluate a trained model and return common metrics and predictions."""
    pred = model.predict(X_test)
    probs = None
    try:
        probs = model.predict_proba(X_test)[:, 1]
    except Exception:
        # some models may not support predict_proba; try decision_function, else zeros
        try:
            probs = model.decision_function(X_test)
            # decision_function may return shape (n_samples,) or (n_samples,); ensure 1d
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

# Note: the training function is already defined above (cached variant). Remove the duplicate/garbled train_models below to avoid redefinition.
# ---- App Layout ----
# Top-level navigation (separate pages)
page = st.sidebar.radio('Navigation', ['Home', 'Train & Compare', 'Predict'], index=1)

# Show title only on Home/Train pages; Predict page will have its own header
if page != 'Predict':
    st.title("Diabetes Models — Comparison & Prediction)
    st.markdown("A Streamlit dashboard to train, compare models and predict diabetes using your dataset.")
else:
    st.title("Diabetes Risk Prediction")
    st.markdown("Enter patient details on the left (Simple or Advanced mode) and choose a model to get a risk score.")

# Added enhanced model comparison visualizations

st.title("Diabetes Models — Comparison & Prediction)
st.markdown("A Streamlit dashboard to train, compare models and predict diabetes using your dataset.")

df = load_data()

# Sidebar controls
st.sidebar.header("Configuration")
use_leakage = st.sidebar.checkbox("Include glyhb-related features (may cause perfect accuracy)", value=False)
models_to_train = st.sidebar.multiselect("Models to train/compare", options=['RandomForest','LogisticRegression','SVM','XGBoost','LightGBM'], default=['RandomForest','LightGBM','XGBoost'])

if 'XGBoost' in models_to_train and XGBClassifier is None:
    st.sidebar.warning("XGBoost not installed in environment. It will be skipped.")
if 'LightGBM' in models_to_train and LGBMClassifier is None:
    st.sidebar.warning("LightGBM not installed in environment. It will be skipped.")

st.sidebar.write("Dataset path:")
st.sidebar.code(DATA_PATH)

train_button = st.sidebar.button("Train & Evaluate Models")

# Show dataset overview
with st.expander("Dataset Preview & Info", expanded=False):
    st.write(df.head())
    st.write(df.describe(include='all'))
    st.markdown(f"**Total rows:** {df.shape[0]}  |  **Columns:** {df.shape[1]}")

# Train models on button press
if train_button:
    with st.spinner("Preprocessing and training models — this may take a moment..."):
        X_scaled, y, feature_names, scaler = preprocess_df(df, remove_leakage=not use_leakage)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

        trained_models = train_models(X_train, y_train, X_val=X_test, y_val=y_test, models_to_train=models_to_train)

        # Evaluate models
        results = {}
        for name, m in trained_models.items():
            results[name] = evaluate_model(m, X_test, y_test)

        # Metrics table
        metrics_df = pd.DataFrame([{
            'model': name,
            'accuracy': res['accuracy'],
            'auc': res['auc'],
            'precision': res['report'].get('weighted avg', {}).get('precision', None),
            'recall': res['report'].get('weighted avg', {}).get('recall', None),
            'f1': res['report'].get('weighted avg', {}).get('f1-score', None)
        } for name, res in results.items()])
        metrics_df = metrics_df.sort_values('accuracy', ascending=False)

        st.subheader("Model Comparison")
        st.dataframe(metrics_df.set_index('model'))

        # Enhanced comparison visualizations: create separate pages/tabs
        st.markdown("---")
        st.subheader("Advanced Comparison Graphs")
        tabs = st.tabs(["Metrics Comparison","Confusion Matrices","ROC Comparison","Radar Chart","Leaderboard"])

        # --- Metrics Comparison Tab ---
        with tabs[0]:
            fig, ax = plt.subplots(1,2,figsize=(14,5))
            sns.barplot(x='model', y='accuracy', data=metrics_df, ax=ax[0])
            ax[0].set_ylim(0,1)
            ax[0].set_title('Test Accuracy by Model')

            sns.barplot(x='model', y='auc', data=metrics_df, ax=ax[1])
            ax[1].set_ylim(0,1)
            ax[1].set_title('Test AUC by Model')
            st.pyplot(fig)

            # Precision/Recall/F1 grouped bar
            prf = metrics_df.melt(id_vars=['model'], value_vars=['precision','recall','f1'], var_name='metric', value_name='value')
            fig, ax = plt.subplots(figsize=(10,5))
            sns.barplot(x='model', y='value', hue='metric', data=prf, ax=ax)
            ax.set_ylim(0,1)
            ax.set_title('Precision / Recall / F1 (weighted avg)')
            st.pyplot(fig)

        # --- Confusion Matrices Tab ---
        with tabs[1]:
            n = len(results)
            cols = 2
            rows = (n + 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*4))
            axes = axes.flatten()
            for ax_idx, (name, res) in enumerate(results.items()):
                sns.heatmap(res['cm'], annot=True, fmt='d', cmap='Blues', ax=axes[ax_idx])
                axes[ax_idx].set_title(name)
                axes[ax_idx].set_xlabel('Predicted')
                axes[ax_idx].set_ylabel('Actual')
            for i in range(len(results), len(axes)):
                fig.delaxes(axes[i])
            plt.tight_layout()
            st.pyplot(fig)

        # --- ROC Comparison Tab ---
        with tabs[2]:
            fig, ax = plt.subplots(figsize=(8,6))
            for name, res in results.items():
                if res['probs'] is not None:
                    fpr, tpr, _ = roc_curve(y_test, res['probs'])
                    ax.plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.3f})")
            ax.plot([0,1],[0,1],'--', color='gray')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curves Comparison')
            ax.legend()
            st.pyplot(fig)

        # --- Radar Chart Tab ---
        with tabs[3]:
            # prepare radar chart data
            radar_df = metrics_df.set_index('model')[['accuracy','precision','recall','f1','auc']].fillna(0)
            # normalize to 0-1
            norm = (radar_df - radar_df.min()) / (radar_df.max() - radar_df.min()).replace(0,1)
            labels = list(norm.columns)
            angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
            angles += angles[:1]

            fig = plt.figure(figsize=(6,6))
            ax = fig.add_subplot(111, polar=True)
            for idx, row in norm.iterrows():
                values = row.tolist()
                values += values[:1]
                ax.plot(angles, values, label=idx)
                ax.fill(angles, values, alpha=0.1)
            ax.set_thetagrids(np.degrees(angles[:-1]), labels)
            ax.set_title('Model Metrics Radar Chart (normalized)')
            ax.legend(bbox_to_anchor=(1.1, 1.05))
            st.pyplot(fig)

        # --- Leaderboard Tab ---
        with tabs[4]:
            st.markdown("## Leaderboard")
            for i, row in metrics_df.reset_index().iterrows():
                st.metric(label=row['model'], value=f"Accuracy: {row['accuracy']:.3f}", delta=f"AUC: {row['auc']:.3f}" if pd.notnull(row['auc']) else "AUC: N/A")

        # Feature importance (if available)
        st.subheader("Feature Importances")
        for name, m in trained_models.items():
            try:
                importances = m.feature_importances_
                idx = np.argsort(importances)[::-1][:15]
                fig, ax = plt.subplots(figsize=(8,4))
                sns.barplot(x=importances[idx], y=np.array(feature_names)[idx], ax=ax)
                ax.set_title(f'Top Features - {name}')
                st.pyplot(fig)
            except Exception:
                st.info(f"{name} does not expose feature_importances_.")

        # Save models into session_state
        st.session_state['models'] = trained_models
        st.session_state['scaler'] = scaler
        st.session_state['feature_names'] = feature_names
        st.session_state['use_leakage'] = use_leakage

# Prediction panel
st.sidebar.header("Predict Diabetes")
if 'models' in st.session_state:
    selected_model_name = st.sidebar.selectbox('Choose model for prediction', options=list(st.session_state['models'].keys()))
else:
    selected_model_name = None

st.sidebar.markdown("Enter patient features to get a diabetes prediction")
# Build input form dynamically from original df columns (excluding target and removed cols)
input_cols = [c for c in df.columns if c not in ['id','glyhb','diabetes']]
# For simplicity, we'll ask a subset of inputs commonly used
if page == 'Predict':
    # -------------------------
    # Predict Page UI (Medical + Futuristic mix)
    # -------------------------
    st.markdown("### Diabetes Risk Prediction")
    st.write("Use **Simple** mode for quick entry or **Advanced** to provide all available features. Choose the trained model and Notebook/Safe mode used during training.")

    # Layout: left sidebar for inputs, right column for results
    left, right = st.columns([1, 1])

    with left:
        st.subheader("Input Mode & Model")
        mode = st.radio("Select mode:", ("Simple", "Advanced"), index=0)
        model_choice = st.selectbox("Model to use for prediction:", options=list(st.session_state.get('models', {}).keys()) if 'models' in st.session_state else ['RandomForest'])
        notebook_mode = st.checkbox("Use Notebook mode (may include glyhb features, higher accuracy but possible leakage)", value=st.session_state.get('use_leakage', False))

        st.markdown("---")
        st.subheader("Patient Details")
        # Simple mode inputs
        if mode == 'Simple':
            age = st.number_input('Age', min_value=0, max_value=120, value=50)
            gender = st.selectbox('Gender', options=df['gender'].dropna().unique().tolist())
            weight = st.number_input('Weight (kg)', value=float(df['weight'].median()))
            height = st.number_input('Height (cm)', value=float(df['height'].median()))
            chol = st.number_input('Cholesterol', value=float(df['chol'].median()))
            hdl = st.number_input('HDL', value=float(df['hdl'].median()))
            sys_bp = st.number_input('Systolic BP (bp.1s)', value=float(df['bp.1s'].median()) if 'bp.1s' in df.columns else 120.0)
            dia_bp = st.number_input('Diastolic BP (bp.1d)', value=float(df['bp.1d'].median()) if 'bp.1d' in df.columns else 80.0)
            waist = st.number_input('Waist (cm)', value=float(df['waist'].median()))
            hip = st.number_input('Hip (cm)', value=float(df['hip'].median()))
            frame = st.selectbox('Frame', options=df['frame'].dropna().unique().tolist() if 'frame' in df.columns else ['M'])
            location = st.selectbox('Location', options=df['location'].dropna().unique().tolist())
        else:
            # Advanced: build inputs dynamically from dataframe columns (excluding id and target)
            adv_cols = [c for c in df.columns if c not in ['id','glyhb','Outcome','diabetes']]
            adv_vals = {}
            for c in adv_cols:
                if df[c].dtype.kind in 'biufc':
                    adv_vals[c] = st.number_input(f"{c}", value=float(df[c].median()) if not df[c].isna().all() else 0.0)
                else:
                    vals = df[c].dropna().unique().tolist()
                    adv_vals[c] = st.selectbox(f"{c}", options=vals)

        st.markdown("---")
        predict_button = st.button('Predict Risk', type='primary')

    with right:
        st.subheader("Prediction Result")
        result_area = st.empty()
        explanation_area = st.empty()

    # Prediction handling
    if predict_button:
        # Build input row
        if mode == 'Simple':
            inp = pd.DataFrame([{
                'age': age,
                'gender': gender,
                'height': height,
                'weight': weight,
                'chol': chol,
                'hdl': hdl,
                'bp.1s': sys_bp if 'bp.1s' in df.columns else np.nan,
                'bp.1d': dia_bp if 'bp.1d' in df.columns else np.nan,
                'waist': waist if 'waist' in df.columns else np.nan,
                'hip': hip if 'hip' in df.columns else np.nan,
                'frame': frame,
                'location': location
            }])
        else:
            inp = pd.DataFrame([adv_vals])

        # Ensure columns exist and align with training features
        # Preprocess like training pipeline
        use_leak = notebook_mode
        # Prepare a copy of df to use preprocess_df for consistent encoding
        try:
            X_all, y_all, feat_names, scaler = preprocess_df(df, remove_leakage=not use_leak)
        except Exception as e:
            st.error(f"Preprocessing failed: {e}")
            raise

        # If notebook_mode (use_leak True) we included glyhb-derived features in training; but user inputs likely lack glyhb, so warn
        if use_leak:
            st.warning('Notebook mode may expect glyhb-derived features which are not available from manual input — predictions may be less reliable.')

        # Align input columns to feat_names
        # Impute missing columns in inp
        for col in feat_names:
            if col not in inp.columns:
                inp[col] = 0
        inp = inp[feat_names]

        # Coerce numeric
        inp = inp.apply(pd.to_numeric, errors='coerce').fillna(0)

        # Scale using the scaler from training (we stored scaler in session_state during training)
        if 'scaler' in st.session_state:
            scaler_used = st.session_state['scaler']
        else:
            scaler_used = scaler
        inp_scaled = scaler_used.transform(inp)

        # Get model
        if 'models' not in st.session_state or model_choice not in st.session_state['models']:
            st.error('Model not trained yet. Please train models first on Train & Compare page.')
        else:
            model = st.session_state['models'][model_choice]
            # Predict
            try:
                prob = None
                if hasattr(model, 'predict_proba'):
                    prob = model.predict_proba(inp_scaled)[:,1][0]
                else:
                    # fallback to decision_function
                    prob = model.decision_function(inp_scaled)[0]
                    # map decision to 0-1 via sigmoid
                    prob = 1/(1+np.exp(-prob))
            except Exception as e:
                st.error(f"Model prediction failed: {e}")
                raise

            cls = int(prob >= 0.5)
            # Risk category
            if prob < 0.3:
                cat = ('Low', 'green', '😀', 'Keep healthy — low risk')
            elif prob < 0.6:
                cat = ('Moderate', 'orange', '🙂', 'Consider lifestyle changes')
            elif prob < 0.8:
                cat = ('High', 'red', '⚠️', 'Seek medical advice')
            else:
                cat = ('Very High', 'darkred', '🚨', 'Immediate medical evaluation recommended')

            # Animated progress meter
            p = int(round(prob*100))
            prog = st.progress(0)
            for i in range(p+1):
                prog.progress(i)
            result_area.markdown(f"### Predicted class: **{cls}**  ")
            result_area.markdown(f"### Probability: **{prob:.3f}** ({p}%)")
            # colored badge
            result_area.markdown(f"<h3 style='color:{cat[1]};'> {cat[2]} {cat[0]} - {cat[3]}</h3>", unsafe_allow_html=True)

            # Show simple feature contribution (for tree models use SHAP if available)
            if SHAP_AVAILABLE and model_choice in ['RandomForest','XGBoost','LightGBM','LogisticRegression']:
                try:
                    explanation_area.subheader('Local explanation (SHAP)')
                    if model_choice in ['RandomForest','LightGBM','XGBoost']:
                        explainer = shap.TreeExplainer(model)
                        shap_vals = explainer.shap_values(inp)
                        # summary plot
                        fig_shap = shap.plots.waterfall(shap.Explanation(values=shap_vals[0], base_values=explainer.expected_value, data=inp), show=False)
                        explanation_area.pyplot(plt.gcf())
                    else:
                        explainer = shap.LinearExplainer(model, inp)
                        shap_vals = explainer.shap_values(inp)
                        shap.summary_plot(shap_vals, inp, show=False)
                        explanation_area.pyplot(plt.gcf())
                except Exception as e:
                    explanation_area.info('SHAP explanation failed: ' + str(e))
            else:
                explanation_area.info('SHAP explanation not available for this model or shap not installed.')

            # Download result
            out = inp.copy()
            out['predicted_class'] = cls
            out['predicted_prob'] = prob
            csv = out.to_csv(index=False)
            st.download_button('Download prediction (CSV)', csv, file_name='prediction.csv', mime='text/csv')

    st.markdown("---")



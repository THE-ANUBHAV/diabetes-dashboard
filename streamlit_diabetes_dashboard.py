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
    pred = model.predict(X_test)
    probs = None
    try:
        probs = model.predict_proba(X_test)[:, 1]
    except Exception:
        # some models may not support predict_proba
        try:
            probs = model.decision_function(X_test)
        except Exception:
            probs = np.zeros(len(pred))

    acc = accuracy_score(y_test, pred)
    auc = None
    try:
        auc = roc_auc_score(y_test, probs) if (probs is not None and len(np.unique(y_test)) > 1) else None
    except Exception:
        auc = None
    cm = confusion_matrix(y_test, pred)
    report = classification_report(y_test, pred, output_dict=True)
    return {'accuracy': acc, 'auc': auc, 'cm': cm, 'report': report, 'pred': pred, 'probs': probs}

# ---- App Layout ----
# Added enhanced model comparison visualizations

st.title("Diabetes Models — Comparison & Prediction (Beautiful UI)")
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
with st.sidebar.form('predict_form'):
    age = st.number_input('Age', min_value=0, max_value=120, value=50)
    chol = st.number_input('Cholesterol', value=float(df['chol'].median()))
    stab_glu = st.number_input('Fasting Glucose (stab.glu)', value=float(df['stab.glu'].median()))
    hdl = st.number_input('HDL', value=float(df['hdl'].median()))
    ratio = st.number_input('Ratio', value=float(df['ratio'].median()))
    weight = st.number_input('Weight', value=float(df['weight'].median()))
    height = st.number_input('Height', value=float(df['height'].median()))
    waist = st.number_input('Waist', value=float(df['waist'].median()))
    hip = st.number_input('Hip', value=float(df['hip'].median()))
    frame = st.selectbox('Frame', options=df['frame'].dropna().unique().tolist())
    location = st.selectbox('Location', options=df['location'].dropna().unique().tolist())
    gender = st.selectbox('Gender', options=df['gender'].dropna().unique().tolist())

    submit_pred = st.form_submit_button('Predict')

if submit_pred:
    if 'models' not in st.session_state:
        st.error('Train models first using the sidebar "Train & Evaluate Models" button')
    else:
        # Create single-row DataFrame
        inp = pd.DataFrame([{ 'chol': chol, 'stab.glu': stab_glu, 'hdl': hdl, 'ratio': ratio,
                              'weight': weight, 'height': height, 'waist': waist, 'hip': hip,
                              'frame': frame, 'location': location, 'gender': gender, 'age': age }])

        # Preprocess consistent with training
        use_leakage_now = st.session_state.get('use_leakage', False)
        # If leakage removed during training, drop those fields
        if not use_leakage_now:
            for c in ['glyhb','ratio','stab.glu']:
                if c in inp.columns:
                    inp = inp.drop(columns=[c])

        # Impute categorical
        cat_cols = inp.select_dtypes(include=['object']).columns.tolist()
        num_cols = inp.select_dtypes(include=['number']).columns.tolist()

        if len(num_cols) > 0:
            num_imp = SimpleImputer(strategy='median')
            inp[num_cols] = num_imp.fit_transform(inp[num_cols])

        if len(cat_cols) > 0:
            cat_imp = SimpleImputer(strategy='most_frequent')
            inp[cat_cols] = cat_imp.fit_transform(inp[cat_cols])
            inp = pd.get_dummies(inp, columns=cat_cols, drop_first=True)

        # Align features to training feature names
        feat_names = st.session_state['feature_names']
        for fn in feat_names:
            if fn not in inp.columns:
                inp[fn] = 0
        inp = inp[feat_names]

        # Scale
        scaler = st.session_state['scaler']
        inp_scaled = scaler.transform(inp)

        model = st.session_state['models'][selected_model_name]
        pred = model.predict(inp_scaled)[0]
        prob = None
        try:
            prob = model.predict_proba(inp_scaled)[0][1]
        except Exception:
            try:
                prob = model.decision_function(inp_scaled)[0]
            except Exception:
                prob = None

        st.success(f"Predicted class: {pred} \nProbability (if available): {prob}")

        # SHAP explanation (if available and supported)
        if SHAP_AVAILABLE:
            try:
                # Only run SHAP for supported model types (tree-based or linear)
                supported_for_shap = ['RandomForest', 'XGBoost', 'LightGBM', 'LogisticRegression']
                if selected_model_name in supported_for_shap:
                    st.subheader('Local SHAP Explanation')
                    # pick appropriate explainer
                    if selected_model_name in ['RandomForest','LightGBM','XGBoost']:
                        explainer = shap.TreeExplainer(model)
                    elif selected_model_name == 'LogisticRegression':
                        explainer = shap.LinearExplainer(model, model._validate_data(inp_scaled, reset=False))
                    else:
                        explainer = None

                    if explainer is not None:
                        shap_vals = explainer.shap_values(inp)
                        try:
                            # summary / waterfall
                            st.pyplot(shap.plots.waterfall(shap.Explanation(values=shap_vals[0], base_values=explainer.expected_value, data=inp), show=False))
                        except Exception:
                            # fallback: summary plot
                            shap.summary_plot(shap_vals, inp, show=False)
                            st.pyplot(plt.gcf())
                    else:
                        st.info('SHAP explainer not available for this model type.')
                else:
                    st.info(f'SHAP explanation not supported for {selected_model_name}.')
            except Exception as e:
                st.info('SHAP explanation failed: ' + str(e))
        else:
            st.info('SHAP not installed. Install shap to get feature explanations.')

st.markdown("---")

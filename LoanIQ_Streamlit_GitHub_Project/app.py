from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

warnings.filterwarnings("ignore")
st.set_page_config(page_title="LoanIQ | Loan Approval Analytics", page_icon="🏦", layout="wide")

st.markdown("""
<style>
    .stApp {background: #0b1020; color: #edf2ff;}
    [data-testid="stSidebar"] {background: #11182b;}
    .hero {padding: 1.3rem 1.5rem; border-radius: 18px; background: linear-gradient(120deg,#172554,#312e81,#164e63); margin-bottom: 1rem;}
    .hero h1 {margin: 0; color: white; font-size: 2.1rem;}
    .hero p {margin: .45rem 0 0; color: #dbeafe;}
    div[data-testid="stMetric"] {background: #151e33; border: 1px solid #293653; padding: 15px; border-radius: 14px;}
    .small-note {color: #9caecb; font-size: .9rem;}
    h2, h3 {color: #e4edff;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🏦 LoanIQ</h1>
  <p>Loan approval analytics • Applicant insights • Machine learning predictions</p>
</div>
""", unsafe_allow_html=True)

TARGET_CANDIDATES = ["Loan_Approved", "loan_approved", "Loan Approved", "loan approved", "Approved", "approved"]

def find_target(columns):
    for candidate in TARGET_CANDIDATES:
        if candidate in columns:
            return candidate
    return None

def load_data():
    uploaded = st.sidebar.file_uploader("Upload loan dataset (CSV)", type=["csv"])
    if uploaded is not None:
        return pd.read_csv(uploaded), uploaded.name
    for filename in ["loan_approval_data.csv", "data/loan_approval_data.csv"]:
        if Path(filename).exists():
            return pd.read_csv(filename), filename
    return None, None

def normalize_target(series):
    """Normalize common yes/no and 0/1 targets while preserving other labels."""
    s = series.copy()
    if s.dtype == bool:
        return s.astype(int), {0: "Not approved", 1: "Approved"}
    values = s.dropna().astype(str).str.strip().str.lower().unique().tolist()
    yes = {"yes", "approved", "approve", "true", "1", "accepted"}
    no = {"no", "rejected", "reject", "false", "0", "denied", "not approved"}
    if set(values).issubset(yes | no) and values:
        mapped = s.astype(str).str.strip().str.lower().map(lambda x: 1 if x in yes else 0)
        return mapped.astype(int), {0: "Not approved", 1: "Approved"}
    le = LabelEncoder()
    mapped = pd.Series(le.fit_transform(s.astype(str)), index=s.index)
    return mapped, {int(i): str(label) for i, label in enumerate(le.classes_)}

def make_preprocessor(X):
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    transformers = []
    if numeric:
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        transformers.append(("numeric", num_pipe, numeric))
    if categorical:
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        transformers.append(("categorical", cat_pipe, categorical))
    return ColumnTransformer(transformers, remainder="drop"), numeric, categorical

def train_models(data, target):
    clean = data.dropna(subset=[target]).copy()
    # Avoid identifier leakage and obvious record IDs.
    drop_cols = [c for c in clean.columns if c.lower() in {"applicant_id", "application_id", "id"}]
    X = clean.drop(columns=[target] + drop_cols, errors="ignore")
    y, label_map = normalize_target(clean[target])
    valid = y.notna()
    X, y = X.loc[valid], y.loc[valid]
    if X.empty or y.nunique() < 2 or len(y) < 8:
        raise ValueError("The dataset needs at least 8 complete rows and two target classes to train a model.")
    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )
    preprocessor, numeric, categorical = make_preprocessor(X_train)
    model_defs = {
        "Logistic Regression": LogisticRegression(max_iter=1500, class_weight="balanced"),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=max(1, min(5, len(X_train)))),
        "Naive Bayes": GaussianNB()
    }
    fitted, metric_rows, predictions = {}, [], {}
    for name, estimator in model_defs.items():
        pipe = Pipeline([("preprocess", preprocessor), ("model", estimator)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        fitted[name] = pipe
        predictions[name] = pred
        metric_rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, average="weighted", zero_division=0),
            "Recall": recall_score(y_test, pred, average="weighted", zero_division=0),
            "F1-score": f1_score(y_test, pred, average="weighted", zero_division=0)
        })
    return X, y, X_train, X_test, y_train, y_test, fitted, pd.DataFrame(metric_rows), predictions, label_map, numeric, categorical

data, source_name = load_data()

with st.sidebar:
    st.markdown("### Navigation")
    page = st.radio("Go to", ["Overview", "Explore Data", "Predict Approval", "Model Lab"], label_visibility="collapsed")
    st.markdown("---")
    st.caption("Project: ML-based Loan Approval Prediction")
    st.caption("Upload a CSV containing a `Loan_Approved` target column.")

if data is None:
    st.info("Upload your `loan_approval_data.csv` from the sidebar to open the dashboard.")
    st.markdown("### Expected dataset")
    st.write("The CSV should contain applicant information and a target column named `Loan_Approved` (or a similar approval label).")
    st.code("Applicant_Income, Credit_Score, Loan_Amount, Loan_Approved\n45000, 720, 150000, Yes\n28000, 590, 90000, No", language="csv")
    st.markdown("### Quick start")
    st.write("1. Put your dataset next to `app.py`, or upload it using the sidebar.")
    st.write("2. Install dependencies: `pip install -r requirements.txt`")
    st.write("3. Start the dashboard: `streamlit run app.py`")
    st.stop()

target = find_target(data.columns)
if target is None:
    st.error("I couldn't find the target column. Rename your approval outcome column to `Loan_Approved`.")
    st.write("Columns found:", data.columns.tolist())
    st.stop()

try:
    (X, y, X_train, X_test, y_train, y_test, models, metrics, predictions,
     label_map, numeric_cols, categorical_cols) = train_models(data, target)
except Exception as e:
    st.error(f"Could not train the models: {e}")
    st.write("Check that the target column contains at least two classes and enough rows.")
    st.stop()

best_model_name = metrics.sort_values("F1-score", ascending=False).iloc[0]["Model"]
best_model = models[best_model_name]
target_labels = data[target].dropna().astype(str).value_counts()
approved_mask = data[target].astype(str).str.strip().str.lower().isin(["yes", "approved", "approve", "true", "1", "accepted"])
approval_rate = approved_mask.mean() * 100 if approved_mask.any() or data[target].astype(str).str.lower().isin(["no","rejected","false","0","denied"]).any() else np.nan

if page == "Overview":
    st.subheader("Portfolio at a glance")
    total = len(data)
    unique_applicants = data["Applicant_ID"].nunique() if "Applicant_ID" in data.columns else total
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total applications", f"{total:,}")
    c2.metric("Applicant records", f"{unique_applicants:,}")
    c3.metric("Features", f"{max(0, data.shape[1]-1):,}")
    c4.metric("Best model (F1)", best_model_name, f"{metrics['F1-score'].max():.1%}")
    st.caption(f"Data source: {source_name} • Evaluation uses a held-out 20% test split (random_state=42).")
    left, right = st.columns(2)
    with left:
        st.markdown("#### Approval outcome distribution")
        counts = data[target].astype(str).value_counts().rename_axis("Outcome").reset_index(name="Applications")
        st.plotly_chart(px.pie(counts, names="Outcome", values="Applications", hole=.58,
                               color_discrete_sequence=["#38bdf8", "#818cf8", "#34d399", "#fb7185"]),
                        use_container_width=True)
    with right:
        numeric = data.select_dtypes(include=np.number).drop(columns=[target] if target in data.select_dtypes(include=np.number).columns else [], errors="ignore")
        if not numeric.empty:
            chosen = st.selectbox("Numeric feature", numeric.columns, key="overview_feature")
            st.plotly_chart(px.histogram(data, x=chosen, color=target, barmode="overlay", opacity=.75,
                                         title=f"{chosen} distribution by outcome"), use_container_width=True)
        else:
            st.info("No numeric columns available for a histogram.")
    st.markdown("#### Model leaderboard")
    st.dataframe(metrics.sort_values("F1-score", ascending=False).style.format({
        "Accuracy":"{:.1%}", "Precision":"{:.1%}", "Recall":"{:.1%}", "F1-score":"{:.1%}"
    }), use_container_width=True, hide_index=True)

elif page == "Explore Data":
    st.subheader("Explore the applicant dataset")
    f1, f2 = st.columns([1, 2])
    with f1:
        st.metric("Rows", f"{data.shape[0]:,}")
        st.metric("Columns", f"{data.shape[1]:,}")
        st.metric("Missing cells", f"{int(data.isna().sum().sum()):,}")
        selected_outcomes = st.multiselect("Filter outcomes", data[target].dropna().astype(str).unique().tolist(),
                                           default=data[target].dropna().astype(str).unique().tolist())
    filtered = data[data[target].astype(str).isin(selected_outcomes)]
    with f2:
        st.markdown("#### Outcome breakdown")
        outcome_counts = filtered[target].astype(str).value_counts().rename_axis("Outcome").reset_index(name="Count")
        st.plotly_chart(px.bar(outcome_counts, x="Outcome", y="Count", color="Outcome", text="Count"), use_container_width=True)
    numeric = filtered.select_dtypes(include=np.number).columns.tolist()
    if numeric:
        col = st.selectbox("Choose a numeric variable", numeric, key="explore_numeric")
        st.plotly_chart(px.histogram(filtered, x=col, color=target, marginal="box", barmode="overlay",
                                     title=f"{col} by outcome"), use_container_width=True)
    st.markdown("#### Dataset preview")
    st.dataframe(filtered.head(500), use_container_width=True, hide_index=True)
    with st.expander("Missing values by column"):
        missing = data.isna().sum().sort_values(ascending=False)
        st.dataframe(missing[missing > 0].rename("Missing values").to_frame(), use_container_width=True)

elif page == "Predict Approval":
    st.subheader("Applicant assessment")
    st.write("Enter the applicant's details below. The prediction is a model estimate, not a lending decision.")
    with st.form("prediction_form"):
        values = {}
        cols = st.columns(2)
        for i, col in enumerate(X.columns):
            with cols[i % 2]:
                series = X[col]
                if col in numeric_cols:
                    vals = pd.to_numeric(series, errors="coerce").dropna()
                    default = float(vals.median()) if not vals.empty else 0.0
                    minv = float(vals.min()) if not vals.empty else 0.0
                    maxv = float(vals.max()) if not vals.empty else max(default + 1, 1.0)
                    if minv == maxv:
                        maxv = minv + 1.0
                    values[col] = st.number_input(col.replace("_", " "), min_value=minv, max_value=maxv,
                                                  value=min(max(default, minv), maxv), key=f"pred_{col}")
                else:
                    options = series.dropna().astype(str).value_counts().index.tolist()
                    if not options:
                        options = ["Unknown"]
                    values[col] = st.selectbox(col.replace("_", " "), options, key=f"pred_{col}")
        submitted = st.form_submit_button("✨ Predict loan outcome", use_container_width=True)
    if submitted:
        applicant = pd.DataFrame([values], columns=X.columns)
        prediction = int(best_model.predict(applicant)[0])
        label = label_map.get(prediction, str(prediction))
        if label.lower() in {"1", "yes", "approved", "approve", "true", "accepted"}:
            st.success(f"Predicted outcome: {label}")
        else:
            st.warning(f"Predicted outcome: {label}")
        if hasattr(best_model, "predict_proba"):
            probabilities = best_model.predict_proba(applicant)[0]
            classes = best_model.named_steps["model"].classes_
            probability_df = pd.DataFrame({
                "Outcome": [label_map.get(int(c), str(c)) for c in classes],
                "Estimated probability": probabilities
            })
            st.plotly_chart(px.bar(probability_df, x="Outcome", y="Estimated probability", range_y=[0,1],
                                   text=probability_df["Estimated probability"].map(lambda x: f"{x:.1%}")),
                            use_container_width=True)
        st.caption(f"Model used: {best_model_name}. Predictions depend on the quality and representativeness of the uploaded data.")

elif page == "Model Lab":
    st.subheader("Compare machine learning models")
    st.write("All models use the same 80/20 train-test split and preprocessing pipeline.")
    st.plotly_chart(px.bar(metrics.melt(id_vars="Model", var_name="Metric", value_name="Score"),
                           x="Model", y="Score", color="Metric", barmode="group", range_y=[0,1],
                           title="Model performance comparison", text_auto=".1%"),
                    use_container_width=True)
    st.dataframe(metrics.sort_values("F1-score", ascending=False).style.format({
        "Accuracy":"{:.1%}", "Precision":"{:.1%}", "Recall":"{:.1%}", "F1-score":"{:.1%}"
    }), use_container_width=True, hide_index=True)
    chosen_model = st.selectbox("Confusion matrix for", list(models.keys()), index=list(models.keys()).index(best_model_name))
    pred = predictions[chosen_model]
    labels = sorted(pd.Series(y_test).unique().tolist())
    cm = confusion_matrix(y_test, pred, labels=labels)
    names = [label_map.get(int(x), str(x)) for x in labels]
    fig = go.Figure(data=go.Heatmap(z=cm, x=[f"Predicted: {n}" for n in names],
                                    y=[f"Actual: {n}" for n in names], text=cm, texttemplate="%{text}",
                                    colorscale="Blues"))
    fig.update_layout(title=f"{chosen_model} — confusion matrix", xaxis_title="", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)
    st.info("Weighted precision, recall and F1-score are used so the dashboard can handle either binary or multi-class labels.")

st.markdown("---")
st.markdown('<p class="small-note">LoanIQ • Educational machine-learning project. This tool is not financial advice and should not be used as the sole basis for real lending decisions.</p>', unsafe_allow_html=True)

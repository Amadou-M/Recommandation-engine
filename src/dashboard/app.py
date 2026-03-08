"""
Jour 4 — Dashboard Streamlit
Visualisation des recommandations en temps réel
"""
import os
import json
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

# ── CONFIG ─────────────────────────────────────────────────────────────
API_URL   = os.getenv("API_URL", "http://localhost:8000")
BASE_PATH = Path("data/delta")

st.set_page_config(
    page_title = "Recommendation Engine",
    page_icon  = "🛍️",
    layout     = "wide"
)

# ── STYLES ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #1e1e2e;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .stMetric label { font-size: 0.85rem; color: #888; }
</style>
""", unsafe_allow_html=True)

# ── HELPERS ────────────────────────────────────────────────────────────
@st.cache_data(ttl=30)
def get_health():
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.json()
    except Exception:
        return None

@st.cache_data(ttl=30)
def get_stats():
    try:
        r = requests.get(f"{API_URL}/stats", timeout=3)
        return r.json()
    except Exception:
        return None

@st.cache_data(ttl=10)
def get_recommendations(user_id: str, top_k: int = 10):
    try:
        r = requests.get(
            f"{API_URL}/recommendations/{user_id}",
            params={"top_k": top_k},
            timeout=5
        )
        return r.json()
    except Exception:
        return None

@st.cache_data(ttl=60)
def get_popular(top_k: int = 20):
    try:
        r = requests.get(
            f"{API_URL}/popular",
            params={"top_k": top_k},
            timeout=5
        )
        return r.json()
    except Exception:
        return None

@st.cache_data(ttl=300)
def load_user_sample():
    try:
        with open(BASE_PATH / "mappings/user_mapping.json") as f:
            labels = json.load(f)
        return labels[:200]
    except Exception:
        return []

# ── SIDEBAR ────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🛍️ Reco Engine")
    st.markdown("---")

    health = get_health()
    if health:
        st.success("✅ API connectée")
        redis_ok = health.get("redis") == "ok"
        st.info(f"Redis : {'✅' if redis_ok else '⚠️'} {health.get('redis')}")
        models = health.get("models", {})
        st.info(f"PyTorch : {'✅' if models.get('pytorch') == 'loaded' else '⚠️'}")
    else:
        st.error("❌ API non disponible")
        st.markdown("Lance : `uvicorn src.api.main:app --port 8000`")

    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["🏠 Accueil", "👤 Recommandations", "📊 Analytiques", "🔥 Populaires"]
    )

# ── PAGE : ACCUEIL ─────────────────────────────────────────────────────
if page == "🏠 Accueil":
    st.title("🛍️ Moteur de Recommandation E-commerce")
    st.markdown("**Amazon Beauty — ALS + PyTorch Embeddings**")
    st.markdown("---")

    stats = get_stats()
    if stats:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("👥 Utilisateurs",  f"{stats.get('n_users', 0):,}")
        col2.metric("📦 Produits",      f"{stats.get('n_items', 0):,}")
        col3.metric("🔑 Clés Redis",    f"{stats.get('redis_keys', 0):,}")
        col4.metric("🤖 Modèles actifs", "2")
    else:
        st.warning("API non disponible — lance l'API d'abord")

    st.markdown("---")
    st.markdown("### Architecture")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **Données**
        - 701,528 reviews
        - 112,565 produits
        - 631,986 utilisateurs
        - Delta Lake format
        """)
    with col2:
        st.markdown("""
        **Modèles**
        - ALS (RMSE = 2.54)
        - PyTorch Embeddings (RMSE = 1.72)
        - Cold start populaire
        - MLflow tracking
        """)
    with col3:
        st.markdown("""
        **Stack**
        - PySpark 3.5.1
        - Delta Lake 3.1.0
        - FastAPI + Redis
        - Streamlit dashboard
        """)

# ── PAGE : RECOMMANDATIONS ─────────────────────────────────────────────
elif page == "👤 Recommandations":
    st.title("👤 Recommandations Personnalisées")
    st.markdown("---")

    users = load_user_sample()

    col1, col2 = st.columns([3, 1])
    with col1:
        if users:
            user_id = st.selectbox("Sélectionner un utilisateur", users)
        else:
            user_id = st.text_input("User ID", value="AGKHLEW2SOWHNMFQIJGBECAF7INQ")
    with col2:
        top_k = st.slider("Top K", min_value=5, max_value=50, value=10)

    if st.button("🔍 Obtenir les recommandations", type="primary"):
        with st.spinner("Chargement..."):
            result = get_recommendations(user_id, top_k)

        if result:
            source_color = {
                "als"        : "🟢",
                "pytorch"    : "🔵",
                "cold_start" : "🟡"
            }
            emoji = source_color.get(result.get("source", ""), "⚪")
            st.success(f"{emoji} Source : **{result.get('source')}** — {result.get('count')} recommandations")

            items = result.get("items", [])
            if items:
                df = pd.DataFrame(items)
                df.index = df.index + 1
                df.columns = ["ASIN", "Score"]
                df["Score"] = df["Score"].round(4)

                col1, col2 = st.columns([2, 3])
                with col1:
                    st.dataframe(df, use_container_width=True)
                with col2:
                    fig = px.bar(
                        df.head(10),
                        x="Score", y="ASIN",
                        orientation="h",
                        title=f"Top {min(10, len(df))} recommandations",
                        color="Score",
                        color_continuous_scale="viridis"
                    )
                    fig.update_layout(
                        height=400,
                        yaxis={"categoryorder": "total ascending"}
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Impossible d'obtenir les recommandations")

# ── PAGE : ANALYTIQUES ─────────────────────────────────────────────────
elif page == "📊 Analytiques":
    st.title("📊 Analytiques & Performance")
    st.markdown("---")

    try:
        import mlflow
        mlflow.set_tracking_uri("./mlruns")
        client = mlflow.tracking.MlflowClient()

        st.subheader("Expériences MLflow")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**ALS — Grid Search**")
            als_exp = client.get_experiment_by_name("recommendation_als")
            if als_exp:
                runs = client.search_runs(
                    experiment_ids=[als_exp.experiment_id],
                    order_by=["metrics.rmse ASC"]
                )
                if runs:
                    data = [{
                        "Run"     : r.data.tags.get("mlflow.runName", r.info.run_id[:8]),
                        "RMSE"    : round(r.data.metrics.get("rmse", 0), 4),
                        "MAE"     : round(r.data.metrics.get("mae", 0), 4),
                        "Rank"    : r.data.params.get("rank", "?"),
                        "RegParam": r.data.params.get("regParam", "?"),
                    } for r in runs]
                    df_als = pd.DataFrame(data)
                    st.dataframe(df_als, use_container_width=True)

                    fig = px.bar(
                        df_als, x="Run", y="RMSE",
                        title="ALS RMSE par configuration",
                        color="RMSE", color_continuous_scale="reds"
                    )
                    st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**PyTorch Embeddings — Courbes d'apprentissage**")
            tf_exp = client.get_experiment_by_name("recommendation_tf_embeddings")
            if tf_exp:
                runs = client.search_runs(
                    experiment_ids=[tf_exp.experiment_id],
                    order_by=["metrics.val_rmse ASC"],
                    max_results=1
                )
                if runs:
                    run_id = runs[0].info.run_id
                    history = client.get_metric_history(run_id, "val_rmse")
                    train_h = client.get_metric_history(run_id, "train_rmse")

                    if history:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=[m.step for m in train_h],
                            y=[m.value for m in train_h],
                            name="Train RMSE", line=dict(color="#00b4d8")
                        ))
                        fig.add_trace(go.Scatter(
                            x=[m.step for m in history],
                            y=[m.value for m in history],
                            name="Val RMSE", line=dict(color="#f72585")
                        ))
                        fig.update_layout(
                            title="Courbes d'apprentissage PyTorch",
                            xaxis_title="Epoch",
                            yaxis_title="RMSE",
                            height=350
                        )
                        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("Comparaison des modèles")
        comparison = pd.DataFrame([
            {"Modèle": "ALS (best)",          "RMSE": 2.5457, "Type": "Collaborative Filtering"},
            {"Modèle": "PyTorch Embeddings",  "RMSE": 1.7241, "Type": "Deep Learning"},
        ])
        fig = px.bar(
            comparison, x="Modèle", y="RMSE",
            color="Type", title="RMSE par modèle (plus bas = meilleur)",
            color_discrete_map={
                "Collaborative Filtering": "#f72585",
                "Deep Learning"          : "#00b4d8"
            }
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.warning(f"MLflow non disponible : {e}")

# ── PAGE : POPULAIRES ──────────────────────────────────────────────────
elif page == "🔥 Populaires":
    st.title("🔥 Produits les plus Populaires")
    st.markdown("---")

    top_k = st.slider("Nombre de produits", 5, 50, 20)

    result = get_popular(top_k)
    if result:
        items = result.get("items", [])
        if items:
            df = pd.DataFrame(items)
            df.index = df.index + 1
            df.columns = ["ASIN", "Score"]

            col1, col2 = st.columns([2, 3])
            with col1:
                st.dataframe(df, use_container_width=True)
            with col2:
                fig = px.bar(
                    df.head(20),
                    x="Score", y="ASIN",
                    orientation="h",
                    title=f"Top {min(20, len(df))} produits populaires",
                    color="Score",
                    color_continuous_scale="oranges"
                )
                fig.update_layout(
                    height=500,
                    yaxis={"categoryorder": "total ascending"}
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("API non disponible ou cold start non chargé")
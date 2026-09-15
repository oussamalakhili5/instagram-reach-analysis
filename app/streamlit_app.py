"""Streamlit dashboard for the Instagram reach analysis project."""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
DATA_PATH = ROOT_DIR / "data" / "Instagram data.csv"
TARGET = "Impressions"
CLUSTER_FEATURES = ["Likes", "Saves", "Comments", "Shares", "Profile Visits", "Follows"]
NUMERIC_COLUMNS = [
    "Impressions",
    "From Home",
    "From Hashtags",
    "From Explore",
    "From Other",
    "Saves",
    "Comments",
    "Shares",
    "Likes",
    "Profile Visits",
    "Follows",
]


@st.cache_data
def load_data() -> tuple[pd.DataFrame, dict[str, int]]:
    """Load and clean the source data while retaining audit information."""
    raw = pd.read_csv(DATA_PATH, encoding="latin-1")
    audit = {
        "raw_rows": len(raw),
        "raw_columns": len(raw.columns),
        "missing_values": int(raw.isna().sum().sum()),
        "duplicates": int(raw.duplicated().sum()),
    }
    clean = raw.dropna().drop_duplicates().reset_index(drop=True)
    return clean, audit


@st.cache_resource
def train_analysis(data: pd.DataFrame) -> dict:
    """Fit the same clustering and regression pipeline used in the notebook."""
    cluster_scaler = StandardScaler()
    scaled = cluster_scaler.fit_transform(data[CLUSTER_FEATURES])
    cluster_model = KMeans(n_clusters=3, random_state=42, n_init=10)
    result = data.copy()
    result["Cluster"] = cluster_model.fit_predict(scaled)

    x_train, x_test, y_train, y_test = train_test_split(
        data[CLUSTER_FEATURES],
        data[TARGET],
        test_size=0.2,
        random_state=42,
    )
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    metrics = {
        "r2": r2_score(y_test, predictions),
        "mae": mean_absolute_error(y_test, predictions),
        "rmse": mean_squared_error(y_test, predictions) ** 0.5,
        "silhouette": silhouette_score(scaled, result["Cluster"]),
    }
    prediction_data = pd.DataFrame(
        {"Actual": y_test.to_numpy(), "Predicted": predictions}
    ).sort_values("Actual")
    importance = pd.DataFrame(
        {"Feature": CLUSTER_FEATURES, "Importance": model.feature_importances_}
    ).sort_values("Importance", ascending=False)
    return {
        "data": result,
        "model": model,
        "metrics": metrics,
        "predictions": prediction_data,
        "importance": importance,
    }


def format_number(value: float) -> str:
    return f"{value:,.0f}"


def show_overview(data: pd.DataFrame, audit: dict[str, int], analysis: dict) -> None:
    st.title("Analyse de la portée Instagram")
    st.markdown(
        "Une analyse interactive des facteurs associés à la portée des publications Instagram, "
        "combinant analyse descriptive, segmentation K-Means et régression Random Forest."
    )
    st.subheader("Vue d’ensemble du projet")
    cards = st.columns(4)
    cards[0].metric("Publications uniques", f"{len(data):,}")
    cards[1].metric("Impressions moyennes", format_number(data[TARGET].mean()))
    cards[2].metric("Impressions médianes", format_number(data[TARGET].median()))
    cards[3].metric("Variables numériques", len(NUMERIC_COLUMNS) - 1)

    st.subheader("Chaîne d’analyse")
    st.markdown(
        "**Données brutes → nettoyage → analyse exploratoire → clustering K-Means standardisé "
        "→ régression Random Forest → évaluation**"
    )
    st.info(
        f"La source contient {audit['raw_rows']} lignes et {audit['raw_columns']} colonnes. "
        f"{audit['duplicates']} doublons ont été supprimés ; aucune valeur manquante n’a été trouvée."
    )
    cluster_summary = (
        analysis["data"]
        .groupby("Cluster", as_index=False)
        .agg(Publications=(TARGET, "size"), Impressions_moyennes=(TARGET, "mean"))
    )
    st.dataframe(cluster_summary.style.format({"Impressions_moyennes": "{:,.0f}"}), hide_index=True)


def show_descriptive(data: pd.DataFrame) -> None:
    st.title("Analyse descriptive")
    tab_data, tab_distributions, tab_relationships = st.tabs(
        ["Données", "Distributions", "Relations"]
    )
    with tab_data:
        st.subheader("Aperçu des données")
        st.dataframe(data.head(10), use_container_width=True)
        left, right = st.columns(2)
        with left:
            st.write("Types de données")
            st.dataframe(data.dtypes.rename("dtype").to_frame(), use_container_width=True)
        with right:
            st.write("Statistiques descriptives")
            st.dataframe(data[NUMERIC_COLUMNS].describe().T, use_container_width=True)
        st.subheader("Matrice de corrélation")
        correlation = data[NUMERIC_COLUMNS].corr()
        fig = px.imshow(
            correlation,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            labels={"color": "Corrélation"},
        )
        fig.update_layout(height=650)
        st.plotly_chart(fig, use_container_width=True)
    with tab_distributions:
        selected = st.multiselect(
            "Variables à afficher",
            NUMERIC_COLUMNS,
            default=["Impressions", "Likes", "Saves", "Shares"],
        )
        if selected:
            long_data = data[selected].melt(var_name="Variable", value_name="Valeur")
            fig = px.histogram(
                long_data,
                x="Valeur",
                facet_col="Variable",
                facet_col_wrap=2,
                marginal="box",
                color="Variable",
            )
            fig.update_layout(showlegend=False, height=650, yaxis_title="Nombre de publications")
            st.plotly_chart(fig, use_container_width=True)
    with tab_relationships:
        x_axis = st.selectbox(
            "Comparer les impressions avec",
            [c for c in NUMERIC_COLUMNS if c != TARGET],
        )
        fig = px.scatter(
            data,
            x=x_axis,
            y=TARGET,
            hover_data=["Caption"],
            title=f"{x_axis} et {TARGET}",
            labels={x_axis: x_axis, TARGET: "Impressions"},
        )
        st.plotly_chart(fig, use_container_width=True)


def show_segmentation(analysis: dict) -> None:
    data = analysis["data"]
    st.title("Segmentation des publications")
    st.caption(
        "K-Means regroupe les publications à partir de variables d’engagement standardisées. "
        "Les identifiants des clusters sont algorithmiques et ne représentent pas une note de qualité."
    )
    summary = (
        data.groupby("Cluster")
        .agg(
            Publications=(TARGET, "size"),
            Impressions_moyennes=(TARGET, "mean"),
            Impressions_médianes=(TARGET, "median"),
            Likes_moyens=("Likes", "mean"),
            Sauvegardes_moyennes=("Saves", "mean"),
        )
        .reset_index()
    )
    summary["Description"] = summary["Impressions_moyennes"].rank(method="first").map(
        {1: "Portée faible", 2: "Portée intermédiaire", 3: "Portée élevée"}
    )
    st.dataframe(
        summary.style.format(
            {
                "Impressions_moyennes": "{:,.0f}",
                "Impressions_médianes": "{:,.0f}",
                "Likes_moyens": "{:,.1f}",
                "Sauvegardes_moyennes": "{:,.1f}",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )
    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            px.bar(summary, x="Cluster", y="Publications", text_auto=True, title="Publications par cluster"),
            use_container_width=True,
        )
    with right:
        st.plotly_chart(
            px.bar(
                summary,
                x="Cluster",
                y="Impressions_moyennes",
                text_auto=".0f",
                title="Impressions moyennes par cluster",
            ),
            use_container_width=True,
        )
    fig = px.scatter(
        data,
        x="Likes",
        y=TARGET,
        color=data["Cluster"].astype(str),
        hover_data=CLUSTER_FEATURES,
        title="Engagement et portée par cluster",
        labels={"color": "Cluster", "Likes": "Likes", TARGET: "Impressions"},
    )
    st.plotly_chart(fig, use_container_width=True)
    st.metric("Score de silhouette", f"{analysis['metrics']['silhouette']:.3f}")


def show_predictive(analysis: dict) -> None:
    st.title("Analyse prédictive")
    st.write(
        "Le modèle Random Forest Regressor prédit les **impressions** à partir des likes, "
        "sauvegardes, commentaires, partages, visites de profil et abonnements."
    )
    st.write("Variables utilisées par le modèle :")
    st.code(", ".join(CLUSTER_FEATURES))
    predictions = analysis["predictions"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=predictions["Actual"], y=predictions["Predicted"], mode="markers", name="Publications"))
    bounds = [predictions.min().min(), predictions.max().max()]
    fig.add_trace(go.Scatter(x=bounds, y=bounds, mode="lines", name="Prédiction parfaite"))
    fig.update_layout(xaxis_title="Impressions réelles", yaxis_title="Impressions prédites")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(predictions.style.format("{:,.0f}"), hide_index=True, use_container_width=True)


def show_performance(analysis: dict) -> None:
    st.title("Performance du modèle")
    metrics = analysis["metrics"]
    cards = st.columns(3)
    cards[0].metric("R²", f"{metrics['r2']:.3f}")
    cards[1].metric("MAE", f"{metrics['mae']:,.0f} impressions")
    cards[2].metric("RMSE", f"{metrics['rmse']:,.0f} impressions")
    st.caption("Les métriques sont calculées sur un échantillon de test fixe de 20 % (random_state=42).")
    importance = analysis["importance"]
    st.subheader("Importance des variables")
    fig = px.bar(
        importance.sort_values("Importance"),
        x="Importance",
        y="Feature",
        orientation="h",
        text_auto=".3f",
        title="Importance des variables selon Random Forest",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(importance.style.format({"Importance": "{:.3f}"}), hide_index=True)


def show_insights(data: pd.DataFrame, analysis: dict) -> None:
    st.title("Enseignements et recommandations")
    top_feature = analysis["importance"].iloc[0]["Feature"]
    st.markdown(
        f"- **Association :** `{top_feature}` possède l’importance la plus élevée dans le modèle "
        "Random Forest sur cet échantillon. Il s’agit d’une association, pas d’une preuve de causalité."
    )
    st.markdown(
        "- **Portée hétérogène :** K-Means distingue des publications à portée faible, "
        "intermédiaire et élevée ; le groupe à portée élevée contient peu de publications."
    )
    st.markdown(
        f"- **Portée typique :** la publication médiane reçoit {format_number(data[TARGET].median())} "
        f"impressions, contre {format_number(data[TARGET].mean())} en moyenne ; cet écart "
        "reflète une distribution asymétrique."
    )
    st.subheader("Prochaines étapes")
    st.markdown(
        "Utiliser le classement des variables pour prioriser les expérimentations autour de "
        "l’engagement et de la conversion du profil, puis les valider avec un jeu de données "
        "plus large et temporel. Le modèle ne doit pas être interprété comme un outil causal."
    )


def show_about(audit: dict[str, int]) -> None:
    st.title("À propos")
    st.write(
        "Ce projet portfolio reproduit les analyses du notebook associé dans une interface "
        "Streamlit interactive. Il utilise l’export Instagram fourni et recalcule toutes les "
        "métriques au démarrage de l’application."
    )
    st.write(f"Fichier source : `{DATA_PATH.relative_to(ROOT_DIR)}`")
    st.write(
        f"Audit des données brutes : {audit['raw_rows']} lignes, {audit['raw_columns']} colonnes, "
        f"{audit['duplicates']} doublons, {audit['missing_values']} valeurs manquantes."
    )


def main() -> None:
    st.set_page_config(page_title="Analyse de la portée Instagram", page_icon="📱", layout="wide")
    data, audit = load_data()
    analysis = train_analysis(data)
    st.sidebar.title("Analyse de la portée Instagram")
    page = st.sidebar.radio(
        "Navigation",
        [
            "🏠 Vue d’ensemble",
            "📊 Analyse descriptive",
            "🔍 Segmentation des publications",
            "🤖 Analyse prédictive",
            "📈 Performance du modèle",
            "💡 Enseignements et recommandations",
            "ℹ️ À propos",
        ],
    )
    if page == "🏠 Vue d’ensemble":
        show_overview(data, audit, analysis)
    elif page == "📊 Analyse descriptive":
        show_descriptive(data)
    elif page == "🔍 Segmentation des publications":
        show_segmentation(analysis)
    elif page == "🤖 Analyse prédictive":
        show_predictive(analysis)
    elif page == "📈 Performance du modèle":
        show_performance(analysis)
    elif page == "💡 Enseignements et recommandations":
        show_insights(data, analysis)
    else:
        show_about(audit)


if __name__ == "__main__":
    main()

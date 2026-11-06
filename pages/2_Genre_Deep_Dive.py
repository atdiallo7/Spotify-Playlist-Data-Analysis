"""
Streamlit page for Question 1, parts 6-10: machine learning and NLP views of
genre, and what they predict about clustering (Q2) and recommendations (Q3).

Reads data/processed/genre_tracks.csv.gz, written by the last cell of
FINAL.ipynb. The first load trains a random forest, so it takes ~30 seconds;
after that everything is cached.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from utils.genre_analysis import (
    feature_importance,
    genre_profiles,
)
from utils.genre_ml import (
    classifier_report,
    distinctive_title_words,
    genre_silhouette,
    genre_tree,
    model_importance,
    neighbour_purity,
    plot_confusion,
    plot_f1,
    plot_importance_comparison,
    plot_link_chart,
    plot_model_comparison,
    single_genre_sample,
    split,
    text_vs_audio_models,
    train_audio_classifier,
)

DATA_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "processed" / "genre_tracks.csv.gz"
)

CONFUSION_GENRES = [
    "black-metal", "death-metal", "grindcore", "heavy-metal", "metalcore",
    "techno", "minimal-techno", "detroit-techno", "chicago-house", "house",
    "classical", "ambient", "new-age", "piano", "sleep",
]
NLP_GENRES = [
    "classical", "sleep", "comedy", "metal", "country",
    "latin", "hip-hop", "study", "k-pop",
]


@st.cache_data
def load_tracks():
    return pd.read_csv(DATA_PATH)


@st.cache_resource(show_spinner="Training the genre classifier (first load only)...")
def run_models():
    """Everything expensive, computed once per server and shared."""
    genre_tracks = load_tracks()
    profiles = genre_profiles(genre_tracks)
    sample = single_genre_sample(genre_tracks)
    train_idx, test_idx = split(sample)
    # a smaller forest than the notebook's, to stay inside hosting memory
    rf = train_audio_classifier(sample, train_idx, min_samples_leaf=20)
    scores, f1, confusion, confused = classifier_report(rf, sample, test_idx)
    silhouettes = genre_silhouette(sample)
    _, purity = neighbour_purity(genre_tracks)

    link = (
        f1.merge(purity[["Genre", "Purity"]], on="Genre")
        .merge(silhouettes, on="Genre")
    )
    extremes = pd.concat([
        link.nlargest(8, "Purity"), link.nlargest(4, "F1"), link.nsmallest(3, "Purity"),
    ])["Genre"]
    link["Label"] = link["Genre"].where(link["Genre"].isin(extremes), "")

    return {
        "profiles": profiles,
        "scores": scores,
        "f1": f1,
        "confusion": confusion,
        "confused": confused,
        "perm": model_importance(rf, sample, test_idx),
        "eta": feature_importance(genre_tracks),
        "silhouettes": silhouettes,
        "purity": purity,
        "link": link,
        "text_models": text_vs_audio_models(sample, train_idx, test_idx, rf),
    }


st.set_page_config(page_title="Genre Deep Dive", page_icon=":microscope:", layout="wide")
st.title("Genre deep dive: machine learning and NLP")
st.caption(
    "Each view also predicts how the clustering and recommender behave: "
    "genres that look distinct here become clean clusters and in-genre recommendations."
)

if not DATA_PATH.exists():
    st.error("Run FINAL.ipynb first; its last cell writes the data this page needs.")
    st.stop()

genre_tracks = load_tracks()
r = run_models()

tabs = st.tabs([
    "6. Classifier", "7. What the model uses", "8. Genre families",
    "9. Track titles (NLP)", "10. Recommendations",
])

with tabs[0]:
    s = r["scores"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Genres", int(s["Genres"]))
    c2.metric("Top-1 accuracy", f"{s['Top-1 accuracy']:.0%}",
              f"random guess {s['Random guess']:.1%}", delta_color="off")
    c3.metric("Top-5 accuracy", f"{s['Top-5 accuracy']:.0%}")
    st.plotly_chart(plot_f1(r["f1"]), width="stretch")
    st.plotly_chart(plot_confusion(r["confusion"], CONFUSION_GENRES), width="stretch")
    st.subheader("Most confused pairs")
    st.dataframe(r["confused"].head(15).round(2), hide_index=True)

with tabs[1]:
    st.plotly_chart(plot_importance_comparison(r["perm"], r["eta"]), width="stretch")
    st.markdown(
        "Tempo barely differs between genres on average but the model still "
        "needs it, so it matters in combination. Liveness is weak by both measures."
    )

with tabs[2]:
    n_families = st.slider("Number of families", 3, 15, 8)
    tree_fig, families = genre_tree(r["profiles"], n_families=n_families)
    st.dataframe(families, hide_index=True)
    st.plotly_chart(tree_fig, width="stretch")
    sil = r["silhouettes"]
    st.metric("Average genre silhouette", f"{sil['Silhouette'].mean():.2f}")
    st.caption(
        "Negative means most songs sit closer to another genre than to their "
        "own, so clusters should be compared with families, not single genres."
    )

with tabs[3]:
    genres = st.multiselect(
        "Genres", sorted(genre_tracks["Genre"].unique()),
        default=[g for g in NLP_GENRES if g in set(genre_tracks["Genre"])],
    )
    if genres:
        st.dataframe(distinctive_title_words(genre_tracks, genres), hide_index=True)
    st.plotly_chart(plot_model_comparison(r["text_models"]), width="stretch")

with tabs[4]:
    st.plotly_chart(plot_link_chart(r["link"]), width="stretch")
    st.dataframe(r["purity"].round(3), hide_index=True)

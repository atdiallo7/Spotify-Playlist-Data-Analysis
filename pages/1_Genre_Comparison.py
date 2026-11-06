"""
Streamlit page for Question 1: genre similarities and differences.

Reads data/processed/genre_tracks.csv.gz, which the last cell of FINAL.ipynb
writes. Run the app from the repo root with:
    streamlit run app.py
"""

from pathlib import Path

import streamlit as st

import pandas as pd

from utils.genre_analysis import (
    feature_importance,
    genre_distances,
    genre_pairs_ranked,
    genre_profiles,
    genre_spread,
    nearest_genres,
    plot_feature_importance,
    plot_genre_compare,
    plot_genre_map,
    plot_profile_heatmap,
    plot_spread,
)

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "genre_tracks.csv.gz"

DEFAULT_GENRES = [
    "classical", "ambient", "jazz", "country", "pop", "r-n-b",
    "k-pop", "hip-hop", "rock", "latin", "edm", "metal",
]


@st.cache_data
def load():
    """Load the track-genre table and compute everything that does not
    depend on user input, once per session."""
    genre_tracks = pd.read_csv(DATA_PATH)
    profiles = genre_profiles(genre_tracks)
    return (
        profiles,
        genre_distances(profiles),
        feature_importance(genre_tracks),
        genre_spread(genre_tracks, min_tracks=100),
    )


st.set_page_config(page_title="Genre Comparison", page_icon=":musical_note:", layout="wide")
st.title("How do genres compare?")
st.caption(
    "All values are standardized: 0 is the average song, +1 is one standard "
    "deviation above it."
)

if not DATA_PATH.exists():
    st.error("Run FINAL.ipynb first; its last cell writes the data this page needs.")
    st.stop()

profiles, distances, importance, spread = load()
all_genres = sorted(profiles.index)

tab_profiles, tab_map, tab_compare, tab_features = st.tabs(
    ["Sound profiles", "Genre map", "Compare two genres", "What separates genres"]
)

with tab_profiles:
    picked = st.multiselect(
        "Genres to show",
        all_genres,
        default=[g for g in DEFAULT_GENRES if g in profiles.index],
    )
    if picked:
        st.plotly_chart(plot_profile_heatmap(profiles, picked), width="stretch")
    else:
        st.info("Pick at least one genre.")

with tab_map:
    st.plotly_chart(
        plot_genre_map(profiles, highlight=picked or None), width="stretch"
    )
    st.caption("Genres picked on the Sound profiles tab are highlighted.")

with tab_compare:
    col_a, col_b = st.columns(2)
    default_a = all_genres.index("hip-hop") if "hip-hop" in all_genres else 0
    default_b = all_genres.index("pop") if "pop" in all_genres else 1
    genre_a = col_a.selectbox("Genre A", all_genres, index=default_a)
    genre_b = col_b.selectbox("Genre B", all_genres, index=default_b)
    st.plotly_chart(plot_genre_compare(profiles, genre_a, genre_b), width="stretch")

    left, right = st.columns(2)
    left.subheader(f"Closest to {genre_a}")
    left.dataframe(nearest_genres(distances, genre_a).round(2), hide_index=True)
    right.subheader(f"Closest to {genre_b}")
    right.dataframe(nearest_genres(distances, genre_b).round(2), hide_index=True)

with tab_features:
    st.plotly_chart(plot_feature_importance(importance), width="stretch")
    st.plotly_chart(plot_spread(spread), width="stretch")

    ranked = genre_pairs_ranked(distances).round(2)
    left, right = st.columns(2)
    left.subheader("Most similar pairs")
    left.dataframe(ranked.head(10), hide_index=True)
    right.subheader("Most different pairs")
    right.dataframe(ranked.tail(10).iloc[::-1], hide_index=True)

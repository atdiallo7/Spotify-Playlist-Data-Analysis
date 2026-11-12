"""Streamlit demo of four optional upgrades to the team's Spotify recommender."""

import streamlit as st
from playlist_recommender import build_playlist, genre_column, load_songs
from playlist_recommender import MOOD_PROFILES, recommend_mood_activity
from playlist_recommender import recommend_hybrid
import re
from playlist_recommender import FEATURES


st.set_page_config(page_title="Spotify Smart Playlist", page_icon="🎵", layout="wide")
st.title("🎵 Spotify Smart Playlist — Experiment")
st.caption("Uses the team's existing K-Means clusters and seven prepared audio features.")

try:
    songs = load_songs()
except (FileNotFoundError, ValueError) as error:
    st.error(str(error))
    st.stop()

st.subheader("1. Pick up to three favorite songs")
if "favorite_uris" not in st.session_state:
    st.session_state.favorite_uris = []

query = st.text_input("Search by song or artist", placeholder="Try Starboy", key="playlist_search")
if len(query.strip()) >= 2:
    matching = songs[
        songs["Track Name"].str.contains(query.strip(), case=False, regex=False, na=False)
        | songs["Artist Name(s)"].str.contains(query.strip(), case=False, regex=False, na=False)
    ].head(50)
    if matching.empty:
        st.info("No matching songs found. Try another search.")
    else:
        choices = matching["Track URI"].tolist()
        labels = {
            row["Track URI"]: (
                f'{row["Track Name"]} — {row["Artist Name(s)"]} '
                f'[{str(row["Track URI"])[-6:]}]'
            )
            for _, row in matching.iterrows()
        }
        chosen = st.selectbox("Matching songs", choices, format_func=lambda uri: labels[uri])
        if st.button("Add favorite song"):
            if chosen in st.session_state.favorite_uris:
                st.info("You already added this song.")
            elif len(st.session_state.favorite_uris) >= 3:
                st.warning("You can choose up to three favorite songs.")
            else:
                st.session_state.favorite_uris.append(chosen)
                st.rerun()
else:
    st.caption("Type at least two characters to search the dataset.")

if st.session_state.favorite_uris:
    st.markdown("**Chosen songs:**")
    chosen_rows = songs.set_index("Track URI")
    for idx, uri in enumerate(st.session_state.favorite_uris):
        if uri in chosen_rows.index:
            record = chosen_rows.loc[uri]
            left, right = st.columns([5, 1])
            left.write(f'{idx + 1}. {record["Track Name"]} — {record["Artist Name(s)"]}')
            if right.button("Remove", key=f"remove_favorite_{idx}"):
                st.session_state.favorite_uris.pop(idx)
                st.rerun()

st.subheader("2. Choose how many recommendations")
count = st.select_slider("Number of songs", options=[5, 10, 15], value=5)

st.subheader("3. Optional genre filtering")
genre_available = genre_column(songs) is not None
same_genre = st.checkbox("Recommend from the same genre", disabled=not genre_available)
if not genre_available:
    st.info(
        "Genre filtering is not yet available: the processed dataset has no genre column. "
        "The team can add data/processed/song_genres.csv with 'Track URI' and 'Genre' columns."
    )

st.subheader("4. Optional audio preferences")
customize = st.checkbox("Customize energy, danceability, and mood")
energy = danceability = valence = None
if customize:
    energy = st.slider("Energy", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    danceability = st.slider("Danceability", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    valence = st.slider("Mood / valence (0 = darker, 1 = more positive)", 0.0, 1.0, 0.5, 0.05)
    st.caption("The sliders adjust the combined audio profile of your favorite songs.")

if st.button("Build My Playlist", type="primary", disabled=not bool(st.session_state.favorite_uris)):
    try:
        recommendations = build_playlist(
            songs,
            st.session_state.favorite_uris,
            count=count,
            same_genre=same_genre,
            energy=energy,
            danceability=danceability,
            valence=valence,
        )
    except ValueError as error:
        st.error(str(error))
    else:
        if recommendations.empty:
            st.warning("No songs matched these filters. Try fewer restrictions.")
        else:
            if len(recommendations) < count:
                st.info(f"Found {len(recommendations)} matches rather than {count}.")
            st.success(f"Found {len(recommendations)} songs!")
            st.dataframe(recommendations, hide_index=True, use_container_width=True)
            st.download_button(
                "Download playlist as CSV",
                data=recommendations.to_csv(index=False).encode("utf-8"),
                file_name="my_spotify_recommendations.csv",
                mime="text/csv",
            )



# Mood and Activity Recommendation

st.divider()
st.subheader("5. Mood & Activity Recommendations")

mood = st.selectbox(
    "Choose your mood or activity",
    list(MOOD_PROFILES.keys())
)

mood_count = st.select_slider(
    "How many songs?",
    options=[5, 10, 15],
    value=10,
    key="mood_count"
)

if st.button("Generate Mood Playlist"):
    st.session_state["mood_results"] = recommend_mood_activity(
        songs, mood, mood_count
    )
    st.session_state["mood_label"] = mood

if "mood_results" in st.session_state:
    results = st.session_state["mood_results"]

    st.subheader(
        f"{st.session_state['mood_label']} Playlist"
    )

    st.dataframe(
        results,
        hide_index=True,
        use_container_width=True
    )

    st.download_button(
        "Download Mood Playlist",
        data=results.to_csv(index=False),
        file_name="mood_playlist.csv",
        mime="text/csv"
    )


# Hybrid recommendation using song similarity and mood

st.divider()
st.subheader("6. Hybrid Music Recommendations")

st.write(
    "Combine your favorite songs with your mood or activity."
)

# Choose a mood or activity.
hybrid_mood = st.selectbox(
    "Choose your mood or activity",
    list(MOOD_PROFILES.keys()),
    key="hybrid_mood"
)

# Adjust how much importance to give each method.
similarity_percent = st.slider(
    "Importance of song similarity (%)",
    min_value=0,
    max_value=100,
    value=60,
    step=10
)

mood_percent = 100 - similarity_percent

st.write(
    f"Song similarity: {similarity_percent}% | "
    f"Mood match: {mood_percent}%"
)

# Use the favorite songs selected in Section 1.
favorite_uris = list(dict.fromkeys(
    st.session_state.get("favorite_uris", [])
))

if not favorite_uris:
    st.info("First select your favorite songs in Section 1.")

# Generate the hybrid recommendations.
if st.button("Generate Hybrid Playlist"):
    if not favorite_uris:
        st.warning("Please select at least one favorite song.")
    else:
        results = recommend_hybrid(
            songs,
            favorite_uris,
            hybrid_mood,
            count=10,
            similarity_weight=similarity_percent / 100
        )

        st.session_state["hybrid_results"] = results

# Display the recommended songs.
if "hybrid_results" in st.session_state:
    results = st.session_state["hybrid_results"]

    st.subheader("Your Hybrid Playlist")

    st.dataframe(
        results,
        hide_index=True,
        use_container_width=True
    )

    st.download_button(
        "Download Hybrid Playlist",
        data=results.to_csv(index=False),
        file_name="hybrid_playlist.csv",
        mime="text/csv"
    )
st.divider()
st.subheader("7. Listen to Recommended Songs")

# I added a Spotify player for recommended songs.
import streamlit.components.v1 as components

# Get our generated playlists.
playlists = {}

for name, key in [
    ("Mood Playlist", "mood_results"),
    ("Hybrid Playlist", "hybrid_results")
]:
    results = st.session_state.get(key)

    if results is not None and not results.empty:
        playlists[name] = results

if playlists:
    # Choose a playlist.
    playlist_name = st.selectbox(
        "Choose a playlist",
        list(playlists.keys())
    )

    results = playlists[playlist_name]

    # Choose a recommended song.
    song_index = st.selectbox(
        "Choose a song to listen to",
        range(len(results)),
        format_func=lambda i: results.iloc[i]["Track Name"]
    )

    selected = results.iloc[song_index]

    # Find the song's Spotify URI.
    if "Track URI" in results.columns:
        track_uri = selected["Track URI"]
    else:
        match = songs[
            (songs["Track Name"] == selected["Track Name"])
            & (songs["Artist Name(s)"] == selected["Artist Name(s)"])
        ]
        track_uri = (
            match.iloc[0]["Track URI"]
            if not match.empty else ""
        )

    # Extract the Spotify track ID.
    track_match = re.search(
        r"(?:spotify:track:|open\.spotify\.com/track/)"
        r"([A-Za-z0-9]{22})",
        str(track_uri)
    )

    if track_match:
        track_id = track_match.group(1)

        # Display Spotify's music player.
        components.iframe(
            f"https://open.spotify.com/embed/track/{track_id}",
            height=160
        )

        st.link_button(
            "Open in Spotify",
            f"https://open.spotify.com/track/{track_id}"
        )
    else:
        st.warning("Spotify player unavailable for this song.")

else:
    st.info("Generate a playlist first.")


# I added a chart to compare hybrid recommendation scores.

st.divider()
st.subheader("8. Recommendation Score Comparison")

hybrid_results = st.session_state.get("hybrid_results")

if hybrid_results is not None and not hybrid_results.empty:
    st.write("Compare similarity, mood, and hybrid scores.")

    chart_data = hybrid_results.head(5).copy()
    chart_data = chart_data.set_index("Track Name")

    st.bar_chart(
        chart_data[
            ["similarity_score", "mood_score", "hybrid_score"]
        ],
        stack=False,
        horizontal=True
    )

    st.write(
        "Higher similarity scores mean songs are closer to "
        "the selected favorites. Higher mood scores mean "
        "they better match the chosen activity."
    )

else:
    st.info("Generate a hybrid playlist to see the comparison.")


# I evaluated duplicates, artist variety, and audio diversity.

st.divider()
st.subheader("9. Recommendation Evaluation")


def evaluate_playlist(results):
    # Count repeated songs.
    unique = results.drop_duplicates(
        subset=["Track Name", "Artist Name(s)"]
    )
    duplicates = len(results) - len(unique)

    # Calculate artist variety.
    artist_variety = (
        results["Artist Name(s)"].nunique()
        / len(results) * 100
    )

    # Find audio features for recommended songs.
    keys = ["Track Name", "Artist Name(s)"]

    matched = unique[keys].merge(
        songs[keys + FEATURES].drop_duplicates(subset=keys),
        on=keys,
        how="inner"
    )

    # Measure audio diversity.
    if len(matched) > 1:
        audio_diversity = matched[FEATURES].std(
            ddof=0
        ).mean()
    else:
        audio_diversity = 0

    return {
        "Songs": len(results),
        "Duplicates": duplicates,
        "Artist variety (%)": round(artist_variety, 1),
        "Audio diversity": round(float(audio_diversity), 3)
    }


# Compare our recommendation methods.
evaluation = []

for name, key in [
    ("Mood & Activity", "mood_results"),
    ("Hybrid", "hybrid_results")
]:
    results = st.session_state.get(key)

    if results is not None and not results.empty:
        scores = evaluate_playlist(results)
        scores["Method"] = name
        evaluation.append(scores)

if evaluation:
    st.dataframe(
        evaluation,
        hide_index=True,
        use_container_width=True
    )
else:
    st.info("Generate playlists first to evaluate them.")

"""Four-tab Streamlit interface for the team's Spotify recommendation project.

Keep this beside playlist_recommender.py in the repository root.
Run: python -m streamlit run playlist_demo.py
"""

import re

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import altair as alt

from recommendation import  (
    FEATURES,
    MOOD_PROFILES,
    build_playlist,
    genre_column,
    load_songs,
    recommend_hybrid,
    recommend_mood_activity,
)


st.set_page_config(page_title="Spotify Smart Playlist", page_icon="🎵", layout="wide")
st.title("🎵 Spotify Smart Playlist")
st.caption("Personalized recommendations using the team's seven prepared audio features")


@st.cache_data(show_spinner="Loading Spotify songs...")
def get_songs():
    # Load the team's processed dataset.
    return load_songs()


try:
    songs = get_songs()
except (FileNotFoundError, ValueError) as error:
    st.error(str(error))
    st.stop()

if songs.empty:
    st.error("No usable songs were found in the dataset.")
    st.stop()

if "favorite_uris" not in st.session_state:
    st.session_state["favorite_uris"] = []


# Display a result table, CSV download, and Spotify player.
def show_playlist(results, prefix, filename):
    if results is None or results.empty:
        st.warning("No songs found. Try a different selection.")
        return

    st.success(f"Found {len(results)} songs!")
    st.dataframe(results, hide_index=True, use_container_width=True)
    st.download_button(
        "Download playlist as CSV",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        key=f"{prefix}_download",
    )

    with st.expander("🎧 Listen to a recommended song", expanded=False):
        selected_index = st.selectbox(
            "Choose a song",
            options=range(len(results)),
            format_func=lambda i: (
                f"{results.iloc[i]['Track Name']} — "
                f"{results.iloc[i]['Artist Name(s)']}"
            ),
            key=f"{prefix}_listen",
        )
        selected = results.iloc[selected_index]

        # Prefer exact track IDs when included in the results.
        if "Track URI" in results.columns:
            track_uri = selected["Track URI"]
        else:
            matching = songs[
                (songs["Track Name"] == selected["Track Name"])
                & (songs["Artist Name(s)"] == selected["Artist Name(s)"])
            ]
            track_uri = matching.iloc[0]["Track URI"] if not matching.empty else ""

        uri = str(track_uri).strip()
        matched = re.search(
            r"(?:spotify:track:|open\.spotify\.com/track/)([A-Za-z0-9]{22})",
            uri,
        )
        if matched:
            track_id = matched.group(1)
        elif re.fullmatch(r"[A-Za-z0-9]{22}", uri):
            track_id = uri
        else:
        
            track_id = None

        if track_id:
            components.iframe(
                f"https://open.spotify.com/embed/track/{track_id}",
                height=160,
            )
            st.link_button(
                "Open in Spotify",
                f"https://open.spotify.com/track/{track_id}",
            )
        else:
            st.info("Spotify playback is unavailable for this track.")


# Evaluate the variety and duplicates in recommended playlists.
def evaluate_playlist(results):
    keys = ["Track Name", "Artist Name(s)"]
    unique = results.drop_duplicates(subset=keys)
    duplicates = len(results) - len(unique)
    artist_variety = (
        100 * results["Artist Name(s)"].nunique() / len(results)
    )

    # Find the audio features for recommended songs.
    matched = unique[keys].merge(
        songs[keys + FEATURES].drop_duplicates(subset=keys),
        on=keys,
        how="inner",
    )
    features = matched[FEATURES].apply(
        pd.to_numeric, errors="coerce"
    ).dropna()

    diversity = (
        float(features.std(ddof=0).mean())
        if len(features) > 1 else 0.0
    )

    return {
        "Songs": len(results),
        "Duplicates": duplicates,
        "Artist variety (%)": round(artist_variety, 1),
        "Audio diversity": round(diversity, 3),
    }


# Organize the app into four tabs.
tab_similar, tab_mood, tab_hybrid, tab_evaluation = st.tabs(
    ["🎵 Similar Songs", "🌿 Mood & Activity",
     "🎚️ Hybrid", "📊 Evaluation"]
)


# TAB 1: SIMILAR SONGS
with tab_similar:
    st.subheader("1. Choose up to three favorite songs")

    query = st.text_input(
        "Search by song or artist",
        placeholder="Try Starboy",
        key="playlist_search"
    )

    if len(query.strip()) >= 2:
        matching = songs[
            songs["Track Name"].str.contains(
                query.strip(), case=False,
                regex=False, na=False
            )
            | songs["Artist Name(s)"].str.contains(
                query.strip(), case=False,
                regex=False, na=False
            )
        ].head(50)

        if matching.empty:
            st.info("No matching songs found.")
        else:
            choices = matching["Track URI"].tolist()

            labels = {
                row["Track URI"]: (
                    f'{row["Track Name"]} — '
                    f'{row["Artist Name(s)"]} '
                    f'[{str(row["Track URI"])[-6:]}]'
                )
                for _, row in matching.iterrows()
            }

            chosen = st.selectbox(
                "Matching songs",
                choices,
                format_func=lambda uri: labels[uri],
                key="matching_song"
            )

            if st.button(
                "Add favorite song",
                key="add_favorite"
            ):
                if chosen in st.session_state.favorite_uris:
                    st.info("You already added this song.")

                elif len(st.session_state.favorite_uris) >= 3:
                    st.warning(
                        "You can choose up to three favorite songs."
                    )

                else:
                    st.session_state.favorite_uris.append(chosen)
                    st.session_state.pop("similar_results", None)
                    st.session_state.pop("hybrid_results", None)
                    st.rerun()

    else:
        st.caption(
            "Type at least two characters to search the dataset."
        )

    # Display the selected favorite songs.
    if st.session_state.favorite_uris:
        st.write("**Chosen songs:**")

        chosen_rows = songs.set_index("Track URI")

        for index, uri in enumerate(
            st.session_state.favorite_uris
        ):
            if uri in chosen_rows.index:
                record = chosen_rows.loc[uri]
                left, right = st.columns([5, 1])

                left.write(
                    f'{index + 1}. {record["Track Name"]} — '
                    f'{record["Artist Name(s)"]}'
                )

                if right.button(
                    "Remove",
                    key=f"remove_favorite_{index}"
                ):
                    st.session_state.favorite_uris.pop(index)
                    st.session_state.pop("similar_results", None)
                    st.session_state.pop("hybrid_results", None)
                    st.rerun()

    # Choose the number of recommendations.
    st.subheader("2. Customize recommendations")

    count = st.select_slider(
        "Number of songs",
        options=[5, 10, 15],
        value=10,
        key="similar_count"
    )

    # Optional genre filtering.
    genre_available = genre_column(songs) is not None

    same_genre = st.checkbox(
        "Recommend from the same genre",
        disabled=not genre_available,
        key="same_genre"
    )

    if not genre_available:
        st.caption(
            "Genre filtering requires song-level genre labels. "
            "The genre heatmap alone does not provide these."
        )

    # Optional audio preferences.
    customize = st.checkbox(
        "Customize energy, danceability, and mood",
        key="customize"
    )

    energy = danceability = valence = None

    if customize:
        energy = st.slider(
            "Energy", 0.0, 1.0, 0.5, 0.05,
            key="similar_energy"
        )

        danceability = st.slider(
            "Danceability", 0.0, 1.0, 0.5, 0.05,
            key="similar_dance"
        )

        valence = st.slider(
            "Mood / valence", 0.0, 1.0, 0.5, 0.05,
            key="similar_valence"
        )

        st.caption(
            "These sliders adjust the combined audio profile. "
            "All seven features are used for similarity."
        )

    
   
    # Generate similar song recommendations.
    if st.button(
        "Build My Playlist",
        type="primary",
        key="build_similar"
    ):
        if not st.session_state.favorite_uris:
            st.warning("Please select a favorite song.")
        else:
            try:
                with st.spinner("Finding similar songs..."):
                    results = build_playlist(
                        songs,
                        st.session_state.favorite_uris,
                        count=count,
                        same_genre=same_genre,
                        energy=energy,
                        danceability=danceability,
                        valence=valence
                    )

                st.session_state["similar_results"] = results
                st.success(f"Generated {len(results)} recommendations!")

            except Exception as error:
                st.exception(error)

    # Display recommendations below the button.
    similar_results = st.session_state.get("similar_results")

    if similar_results is not None:
        st.subheader("Your Similar Songs Playlist")

        show_playlist(
            similar_results,
            "similar",
            "similar_playlist.csv"
        )



# TAB 2: MOOD AND ACTIVITY
with tab_mood:
    st.subheader("Mood & Activity Recommendations")

    st.write(
        "Find music for studying, working out, "
        "relaxing, and other activities."
    )

    mood = st.selectbox(
        "Choose your mood or activity",
        list(MOOD_PROFILES),
        key="mood"
    )

    mood_count = st.select_slider(
        "How many songs?",
        options=[5, 10, 15],
        value=10,
        key="mood_count"
    )

    st.caption(
        "This method uses rule-based audio feature scores."
    )

    if st.button(
        "Generate Mood Playlist",
        key="generate_mood"
    ):
        st.session_state["mood_results"] = (
            recommend_mood_activity(
                songs, mood, mood_count
            )
        )

        st.session_state["mood_label"] = mood

    mood_results = st.session_state.get(
        "mood_results"
    )

    if mood_results is not None:
        st.subheader(
            f"{st.session_state.get('mood_label', 'Mood')} Playlist"
        )

        show_playlist(
            mood_results,
            "mood",
            "mood_playlist.csv"
        )


# TAB 3: HYBRID RECOMMENDATIONS
with tab_hybrid:
    st.subheader("Hybrid Music Recommendations")

    st.write(
        "Combine your favorite songs with a mood or activity."
    )

    favorites = list(dict.fromkeys(
        st.session_state.favorite_uris
    ))

    if favorites:
        st.caption(
            f"{len(favorites)} favorite song(s) selected "
            "in the Similar Songs tab."
        )
    else:
        st.info(
            "First select your favorite songs "
            "in the Similar Songs tab."
        )

    hybrid_mood = st.selectbox(
        "Choose your mood or activity",
        list(MOOD_PROFILES),
        key="hybrid_mood"
    )

    # Adjust similarity and mood importance.
    similarity_percent = st.slider(
        "Importance of song similarity (%)",
        min_value=0,
        max_value=100,
        value=60,
        step=10,
        key="hybrid_weight"
    )

    st.write(
        f"Song similarity: **{similarity_percent}%** | "
        f"Mood match: **{100 - similarity_percent}%**"
    )

    hybrid_count = st.select_slider(
        "How many songs?",
        options=[5, 10, 15],
        value=10,
        key="hybrid_count"
    )

    if st.button(
        "Generate Hybrid Playlist",
        key="generate_hybrid"
    ):
        if not favorites:
            st.warning(
                "Choose a favorite song in the Similar Songs tab."
            )

        else:
            try:
                st.session_state["hybrid_results"] = (
                    recommend_hybrid(
                        songs,
                        favorites,
                        hybrid_mood,
                        count=hybrid_count,
                        similarity_weight=similarity_percent / 100
                    )
                )

                st.session_state["hybrid_label"] = hybrid_mood

            except ValueError as error:
                st.error(str(error))

    hybrid_results = st.session_state.get(
        "hybrid_results"
    )

    if hybrid_results is not None:
        st.subheader("Your Hybrid Playlist")

        show_playlist(
            hybrid_results,
            "hybrid",
            "hybrid_playlist.csv"
        )


# TAB 4: EVALUATION
with tab_evaluation:
    st.subheader("Recommendation Score Comparison")

    hybrid_results = st.session_state.get(
        "hybrid_results"
    )

    if hybrid_results is not None and not hybrid_results.empty:
        st.write(
            "Compare the top five hybrid recommendations "
            "and their similarity, mood, and hybrid scores."
        )

        chart_data = hybrid_results.head(5).set_index(
            "Track Name"
        )

        
        
        # Show playlist summary.
        st.subheader("🎧 Playlist Overview")

        col1, col2, col3 = st.columns(3)

        col1.metric("Songs", len(hybrid_results))
        col2.metric(
            "Unique Artist Entries",
            hybrid_results["Artist Name(s)"].nunique()
        )
        col3.metric(
            "Duplicates",
            hybrid_results.duplicated(
                ["Track Name", "Artist Name(s)"]
            ).sum()
        )

        st.divider()
        st.subheader("🎯 Recommendation Scorecard")

        # Prepare scores for the heatmap.
        scores = chart_data[
            ["similarity_score", "mood_score", "hybrid_score"]
        ].rename(columns={
            "similarity_score": "Similarity",
            "mood_score": "Mood",
            "hybrid_score": "Hybrid"
        }).reset_index()

        scores = scores.melt(
            "Track Name", var_name="Method", value_name="Score"
        )

       
        # Create colored boxes for each score.
        chart = alt.Chart(scores).mark_rect().encode(
            x=alt.X(
                "Method:N",
                title=None,
                sort=["Similarity", "Mood", "Hybrid"],
                axis=alt.Axis(labelAngle=0)
            ),
            y=alt.Y(
                "Track Name:N",
                title=None,
                axis=alt.Axis(labelLimit=250)
            ),
            color=alt.Color(
                "Method:N",
                scale=alt.Scale(
                    domain=["Similarity", "Mood", "Hybrid"],
                    range=["#2563EB", "#C2410C", "#7E22CE"]
                ),
                legend=None
            ),
            tooltip=["Track Name", "Method", "Score"]
        )

        # Display the score inside each box.
        labels = alt.Chart(scores).mark_text(
            color="white",
            fontSize=15
        ).encode(
            x=alt.X(
                "Method:N",
                sort=["Similarity", "Mood", "Hybrid"]
            ),
            y="Track Name:N",
            text=alt.Text("Score:Q", format=".0f")
        )

        # Show the heatmap in Streamlit.
        st.altair_chart(
            (chart + labels).properties(height=350),
            use_container_width=True
        )

        st.caption(
            "Similarity measures audio resemblance. "
            "Mood measures the activity match. "
            "Hybrid combines both scores."
        )

    else:
        st.info(
            "Generate a Hybrid Playlist to see the chart."
        )

    # Evaluate playlist diversity and duplicates.
    st.subheader("Recommendation Evaluation")

    evaluation = []

    for name, key in [
        ("Similar Songs", "similar_results"),
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

        st.caption(
            "Duplicates count repeated songs. "
            "Artist variety counts unique artist entries. "
            "Audio diversity measures variation across "
            "the seven audio features. Higher diversity "
            "does not always mean better recommendations."
        )

    else:
        st.info(
            "Generate playlists in the other tabs "
            "to see their evaluation."
        )

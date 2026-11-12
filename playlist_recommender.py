"""Optional multi-song playlist recommender for the Spotify team project.

Uses the processed audio features and clusters from the team's existing work.
The original recommendation.py and app.py are not modified.
"""

from pathlib import Path

import numpy as np
import pandas as pd


FEATURES = [
    "Danceability (prepared)",
    "Energy (prepared)",
    "Loudness (prepared)",
    "Speechiness (prepared)",
    "Acousticness (prepared)",
    "Instrumentalness (prepared)",
    "Valence (prepared)",
]

GENRE_COLUMNS = ("Genre", "genre", "track_genre", "playlist_genre")
REQUIRED_COLUMNS = ("Track URI", "Track Name", "Artist Name(s)", "Cluster", *FEATURES)


def genre_column(songs: pd.DataFrame) -> str | None:
    """Return the name of a genre column if the dataset includes one."""
    return next((column for column in GENRE_COLUMNS if column in songs.columns), None)


def load_songs(
    dataset_path: str | Path = "data/processed/songs_with_clusters.csv",
    genre_path: str | Path = "data/processed/song_genres.csv",
) -> pd.DataFrame:
    """Load the team's processed dataset; optionally add trustworthy genre labels."""
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Cannot find {dataset_path}. Run the team's clustering notebook "
            "to generate songs_with_clusters.csv first."
        )

    songs = pd.read_csv(dataset_path)
    missing = [column for column in REQUIRED_COLUMNS if column not in songs.columns]
    if missing:
        raise ValueError(f"The processed dataset is missing columns: {missing}")

    # The clustering CSV may not have genre. A separate lookup can supply it.
    genre_path = Path(genre_path)
    if genre_column(songs) is None and genre_path.exists():
        genre_map = pd.read_csv(genre_path)
        if not {"Track URI", "Genre"}.issubset(genre_map.columns):
            raise ValueError("song_genres.csv needs columns: Track URI, Genre")

        genre_map = genre_map.dropna(subset=["Track URI", "Genre"])
        # A track can have multiple genre tags; preserve each distinct tag.
        genre_map = (
            genre_map.groupby("Track URI", as_index=False)["Genre"]
            .agg(lambda tags: "; ".join(dict.fromkeys(str(tag).strip() for tag in tags if str(tag).strip())))
        )
        songs = songs.merge(genre_map, how="left", on="Track URI", validate="many_to_one")

    # Track URI is the team's unique song identifier.
    songs = songs.dropna(subset=["Track URI", "Track Name", "Artist Name(s)"])
    songs = songs.drop_duplicates(subset=["Track URI"]).copy()
    for column in FEATURES:
        songs[column] = pd.to_numeric(songs[column], errors="coerce")
    songs = songs.dropna(subset=FEATURES + ["Cluster"]).reset_index(drop=True)
    return songs


def _genre_tags(value: object) -> set[str]:
    """Split a song's genre labels without confusing genres with sound clusters."""
    if pd.isna(value):
        return set()
    return {tag.strip().casefold() for tag in str(value).replace("|", ";").replace(",", ";").split(";") if tag.strip()}


def build_playlist(
    songs: pd.DataFrame,
    chosen_uris: list[str],
    count: int = 5,
    same_genre: bool = False,
    energy: float | None = None,
    danceability: float | None = None,
    valence: float | None = None,
) -> pd.DataFrame:
    """Recommend songs nearest the combined audio profile of 1–3 selected tracks.

    Use the union of clusters of the selected songs as the candidate pool.
    User-chosen slider values (0–1) replace the corresponding centroid features.
    Songs already selected are excluded from the recommendations.
    """
    if count not in (5, 10, 15):
        raise ValueError("count must be 5, 10, or 15")
    chosen_uris = list(dict.fromkeys(chosen_uris))
    if not 1 <= len(chosen_uris) <= 3:
        raise ValueError("Choose between 1 and 3 songs")

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in songs.columns]
    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    clean = songs.drop_duplicates(subset=["Track URI"]).copy()
    clean[FEATURES] = clean[FEATURES].apply(pd.to_numeric, errors="coerce")
    clean = clean.dropna(subset=FEATURES + ["Cluster"])
    picked = clean[clean["Track URI"].isin(chosen_uris)]
    if len(picked) != len(chosen_uris):
        missing = set(chosen_uris) - set(picked["Track URI"])
        raise ValueError(f"Selected song(s) not found or missing audio features: {sorted(missing)}")

    # Multiple songs: take the average of their prepared audio features.
    target = picked[FEATURES].to_numpy(dtype=float).mean(axis=0)
    for column, value in (
        ("Energy (prepared)", energy),
        ("Danceability (prepared)", danceability),
        ("Valence (prepared)", valence),
    ):
        if value is not None:
            if not 0 <= value <= 1:
                raise ValueError("Audio sliders must be between 0 and 1")
            target[FEATURES.index(column)] = value

    # Include songs from any cluster represented by the chosen favorites.
    candidates = clean[clean["Cluster"].isin(picked["Cluster"].unique())].copy()
    candidates = candidates[~candidates["Track URI"].isin(chosen_uris)]

    if same_genre:
        column = genre_column(clean)
        if column is None:
            raise ValueError(
                "Genre labels are missing. Add a Genre column or "
                "data/processed/song_genres.csv with Track URI and Genre columns."
            )
        selected_genres = set().union(*picked[column].map(_genre_tags))
        if not selected_genres:
            raise ValueError("The selected song(s) have no genre labels")
        candidates = candidates[
            candidates[column].map(lambda genre: bool(_genre_tags(genre) & selected_genres))
        ].copy()

    # Euclidean distance across the seven scaled features.
    values = candidates[FEATURES].to_numpy(dtype=float)
    candidates["distance"] = np.linalg.norm(values - target, axis=1)
    candidates = candidates.sort_values("distance", ascending=True, kind="stable")
    candidates = candidates.drop_duplicates(subset=["Track Name", "Artist Name(s)"])
    output = ["Track Name", "Artist Name(s)", "distance", "Cluster"]
    genre = genre_column(candidates)
    if genre is not None:
        output.append(genre)
    return candidates[output].head(count).reset_index(drop=True)



# Mood and activity rules using all seven audio features
# Each feature has a preferred level and importance weight.

MOOD_PROFILES = {
    "Studying": [
        ("medium", 1), ("low", 3), ("low", 2),
        ("low", 3), ("high", 1), ("high", 3),
        ("medium", 1)
    ],
    "Working out": [
        ("high", 3), ("high", 3), ("high", 2),
        ("medium", 1), ("low", 1), ("low", 1),
        ("high", 2)
    ],
    "Relaxing": [
        ("low", 1), ("low", 3), ("low", 2),
        ("low", 2), ("high", 2), ("high", 2),
        ("medium", 1)
    ],
    "Driving": [
        ("high", 2), ("high", 3), ("medium", 1),
        ("medium", 1), ("low", 1), ("low", 1),
        ("high", 2)
    ],
    "Happy": [
        ("high", 2), ("high", 2), ("medium", 1),
        ("medium", 1), ("low", 1), ("low", 1),
        ("high", 4)
    ],
}


def recommend_mood_activity(songs, mood, count=10):
    # I used audio feature rules to recommend songs by mood.
    if mood not in MOOD_PROFILES:
        raise ValueError("Invalid mood or activity")

    data = songs.drop_duplicates(subset=["Track URI"]).copy()
    data[FEATURES] = data[FEATURES].apply(
        pd.to_numeric, errors="coerce"
    )
    data = data.dropna(subset=FEATURES)

    # Calculate a match score using all seven features.
    total_score = np.zeros(len(data))
    total_weight = 0
    contributions = []

    for feature, (level, weight) in zip(
        FEATURES, MOOD_PROFILES[mood]
    ):
        values = data[feature].to_numpy(dtype=float).clip(0, 1)

        if level == "high":
            match = values
        elif level == "low":
            match = 1 - values
        else:
            match = 1 - 2 * np.abs(values - 0.5)

        total_score += weight * match
        total_weight += weight
        contributions.append(weight * match)

    data["match_score"] = (
        100 * total_score / total_weight
    ).round(1)

    # Explain the two strongest matching features.
    best = np.argsort(
        np.column_stack(contributions), axis=1
    )[:, -2:][:, ::-1]

    data["Why it matches"] = [
        ", ".join(
            f"{MOOD_PROFILES[mood][i][0]} "
            f"{FEATURES[i].replace(' (prepared)', '').lower()}"
            for i in indices
        )
        for indices in best
    ]

    # Return the highest-scoring unique songs.
    data = data.sort_values(
        "match_score", ascending=False
    )
    data = data.drop_duplicates(
        subset=["Track Name", "Artist Name(s)"]
    )

    return data[
        ["Track Name", "Artist Name(s)",
         "match_score", "Why it matches"]
    ].head(count).reset_index(drop=True)


def recommend_hybrid(songs, favorite_uris, mood, count=10,
                     similarity_weight=0.6):
    # I combined song similarity and mood scores for recommendations.
    if mood not in MOOD_PROFILES:
        raise ValueError("Invalid mood or activity")

    data = songs.drop_duplicates(subset=["Track URI"]).copy()

    # Prepare the seven audio features.
    data[FEATURES] = data[FEATURES].apply(
        pd.to_numeric, errors="coerce"
    )
    data = data.dropna(subset=FEATURES)

    # Find the user's favorite songs.
    favorites = data[data["Track URI"].isin(favorite_uris)]


# I removed other copies of the songs I already selected.
selected_songs = favorites[
    ["Track Name", "Artist Name(s)"]
].drop_duplicates()

data = data.merge(
    selected_songs.assign(selected=True),
    on=["Track Name", "Artist Name(s)"],
    how="left"
)

data = data[data["selected"].isna()].drop(
    columns=["selected"]
)


    if favorites.empty:
        raise ValueError("Please select a favorite song")

    # Combine the audio features of favorite songs.
    favorite_profile = favorites[FEATURES].mean().to_numpy()
    audio_values = data[FEATURES].to_numpy()

    # Calculate audio similarity between 0 and 1.
    distances = np.sqrt(
        ((audio_values - favorite_profile) ** 2).sum(axis=1)
    )
    similarity = np.clip(
        1 - distances / np.sqrt(len(FEATURES)), 0, 1
    )

    # Calculate mood/activity match using our rules.
    mood_total = np.zeros(len(data))
    total_weight = 0

    for feature, (level, weight) in zip(
        FEATURES, MOOD_PROFILES[mood]
    ):
        values = data[feature].to_numpy().clip(0, 1)

        if level == "high":
            match = values
        elif level == "low":
            match = 1 - values
        else:
            match = 1 - 2 * np.abs(values - 0.5)

        mood_total += weight * match
        total_weight += weight

    mood_score = mood_total / total_weight

    # Combine both recommendation methods.
    data["similarity_score"] = (similarity * 100).round(1)
    data["mood_score"] = (mood_score * 100).round(1)

    data["hybrid_score"] = (
        similarity_weight * similarity
        + (1 - similarity_weight) * mood_score
    ) * 100

    # Remove favorite songs and return the best matches.
    data = data[~data["Track URI"].isin(favorite_uris)]

    data = data.sort_values(
        "hybrid_score", ascending=False
    ).drop_duplicates(
        subset=["Track Name", "Artist Name(s)"]
    )

    data["hybrid_score"] = data["hybrid_score"].round(1)

    return data[
        ["Track Name", "Artist Name(s)",
         "similarity_score", "mood_score", "hybrid_score"]
    ].head(count).reset_index(drop=True)


# I added a Spotify player to listen to recommended songs.

st.divider()
st.subheader("7. Listen to Recommended Songs")

# Get the playlists we already generated.
available_playlists = {}

for label, key in [
    ("Hybrid Playlist", "hybrid_results"),
    ("Mood Playlist", "mood_results")
]:
    results = st.session_state.get(key)

    if results is not None and not results.empty:
        available_playlists[label] = results

if not available_playlists:
    st.info("Generate a playlist first to listen to songs.")

else:
    # Choose which playlist to listen to.
    playlist_name = st.selectbox(
        "Choose your playlist",
        list(available_playlists.keys())
    )

    results = available_playlists[playlist_name]

    # Choose a recommended song.
    selected_index = st.selectbox(
        "Choose a song to listen to",
        range(len(results)),
        format_func=lambda i: (
            f"{results.iloc[i]['Track Name']} - "
            f"{results.iloc[i]['Artist Name(s)']}"
        )
    )

    selected_song = results.iloc[selected_index]

    # Find the selected song's Spotify Track URI.
    if "Track URI" in results.columns:
        track_uri = selected_song["Track URI"]
    else:
        matching = songs[
            (songs["Track Name"] == selected_song["Track Name"])
            & (songs["Artist Name(s)"] == selected_song["Artist Name(s)"])
        ]

        track_uri = (
            matching.iloc[0]["Track URI"]
            if not matching.empty else None
        )

    # Get the Spotify track ID from its URI or URL.
    track_id = None

    if track_uri is not None:
        uri = str(track_uri).split("?")[0].rstrip("/")

        if uri.startswith("spotify:track:"):
            track_id = uri.split(":")[-1]

        elif "open.spotify.com/track/" in uri:
            track_id = uri.split("/track/")[-1]

    # Display the Spotify player.
    if track_id and re.fullmatch(r"[A-Za-z0-9]{22}", track_id):
        spotify_url = f"https://open.spotify.com/track/{track_id}"
        embed_url = f"https://open.spotify.com/embed/track/{track_id}"

        st.link_button("Open in Spotify", spotify_url)

        if hasattr(st, "iframe"):
            st.iframe(embed_url, height=152)
        else:
            import streamlit.components.v1 as components
            components.iframe(embed_url, height=152)

    else:
        st.warning("Spotify player is unavailable for this song.")


# I evaluated the diversity and duplicates in my recommendations.

st.divider()
st.subheader("9. Recommendation Evaluation")

def evaluate_playlist(results):
    # Check for repeated songs.
    unique = results.drop_duplicates(
        subset=["Track Name", "Artist Name(s)"]
    )
    duplicates = len(results) - len(unique)

    # Check how many different artists are recommended.
    artist_variety = (
        results["Artist Name(s)"].nunique()
        / len(results) * 100
    )

    # Compare audio features across the recommended songs.
    keys = ["Track Name", "Artist Name(s)"]

    matched = unique[keys].merge(
        songs[keys + FEATURES].drop_duplicates(subset=keys),
        on=keys,
        how="inner"
    )

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


# Compare the playlists we have generated.
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

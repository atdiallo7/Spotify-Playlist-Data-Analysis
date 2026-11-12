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
    songs,
    chosen_uris,
    count=5,
    same_genre=False,
    energy=None,
    danceability=None,
    valence=None,
):
    # I recommend songs using the combined features of favorite songs.
    if count not in [5, 10, 15]:
        raise ValueError("Choose 5, 10, or 15 songs")

    chosen_uris = list(dict.fromkeys(chosen_uris))

    if not 1 <= len(chosen_uris) <= 3:
        raise ValueError("Choose 1 to 3 favorite songs")

    # Prepare the seven audio features.
    data = songs.drop_duplicates(subset=["Track URI"]).copy()
    data[FEATURES] = data[FEATURES].apply(
        pd.to_numeric, errors="coerce"
    )
    data = data.dropna(subset=FEATURES + ["Cluster"])

    # Find the selected favorite songs.
    favorites = data[data["Track URI"].isin(chosen_uris)]

    if len(favorites) != len(chosen_uris):
        raise ValueError("Some selected songs were not found")

    # Combine the audio features of favorite songs.
    favorite_profile = favorites[FEATURES].mean().to_numpy()

    # Apply optional audio preferences.
    preferences = {
        "Energy (prepared)": energy,
        "Danceability (prepared)": danceability,
        "Valence (prepared)": valence,
    }

    for feature, value in preferences.items():
        if value is not None:
            if not 0 <= value <= 1:
                raise ValueError("Audio preference must be between 0 and 1")
            favorite_profile[FEATURES.index(feature)] = value

    # Find songs from the selected songs' clusters.
    candidates = data[
        data["Cluster"].isin(favorites["Cluster"].unique())
    ].copy()

    # Remove selected favorites and duplicate recordings.
    selected_names = set(
        zip(favorites["Track Name"], favorites["Artist Name(s)"])
    )

    candidates = candidates[
        ~candidates["Track URI"].isin(chosen_uris)
    ].copy()

    candidates = candidates[
        ~pd.Series(
            list(zip(
                candidates["Track Name"],
                candidates["Artist Name(s)"]
            )),
            index=candidates.index
        ).isin(selected_names)
    ].copy()

    # Filter by genre if genre information is available.
    genre = genre_column(data)

    if same_genre:
        if genre is None:
            raise ValueError("Genre labels are not available")

        def get_genres(value):
            if pd.isna(value):
                return set()
            text = str(value).replace("|", ";").replace(",", ";")
            return {
                item.strip().lower()
                for item in text.split(";")
                if item.strip()
            }

        selected_genres = set()

        for value in favorites[genre]:
            selected_genres.update(get_genres(value))

        if not selected_genres:
            raise ValueError("Selected songs have no genre labels")

        candidates = candidates[
            candidates[genre].apply(
                lambda value: bool(
                    get_genres(value) & selected_genres
                )
            )
        ].copy()

    # Calculate Euclidean distance using seven audio features.
    audio_values = candidates[FEATURES].to_numpy(dtype=float)

    candidates["distance"] = np.sqrt(
        ((audio_values - favorite_profile) ** 2).sum(axis=1)
    )

    # Sort songs from closest to farthest.
    candidates = candidates.sort_values(
        "distance", ascending=True
    )

    candidates = candidates.drop_duplicates(
        subset=["Track Name", "Artist Name(s)"]
    )

    # Return recommended songs to the Streamlit app.
    columns = [
        "Track URI",
        "Track Name",
        "Artist Name(s)",
        "distance",
        "Cluster",
    ]

    if genre is not None:
        columns.append(genre)

    return candidates[columns].head(count).reset_index(drop=True)

    # I created mood profiles using all seven audio features.

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
    ]
}

def recommend_mood_activity(songs, mood, count=10):
    # I recommend songs using mood and activity rules.
    if mood not in MOOD_PROFILES:
        raise ValueError("Invalid mood or activity")

    data = songs.drop_duplicates(subset=["Track URI"]).copy()
    data[FEATURES] = data[FEATURES].apply(
        pd.to_numeric, errors="coerce"
    )
    data = data.dropna(subset=FEATURES)

    # Calculate scores using all seven audio features.
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

    # Explain why each song matches.
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

    # Remove duplicates and return the best songs.
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
    # I combined song similarity and mood scores.
    if mood not in MOOD_PROFILES:
        raise ValueError("Invalid mood or activity")

    if not 0 <= similarity_weight <= 1:
        raise ValueError("Similarity weight must be between 0 and 1")

    # Prepare the seven audio features.
    data = songs.drop_duplicates(subset=["Track URI"]).copy()
    data[FEATURES] = data[FEATURES].apply(
        pd.to_numeric, errors="coerce"
    )
    data = data.dropna(subset=FEATURES)

    # Find the selected favorite songs.
    favorites = data[data["Track URI"].isin(favorite_uris)]

    if favorites.empty:
        raise ValueError("Please select a favorite song")

    # Combine the features of selected songs.
    favorite_profile = favorites[FEATURES].mean().to_numpy()

    # Remove selected songs and other copies of them.
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

    # Calculate similarity with favorite songs.
    audio_values = data[FEATURES].to_numpy(dtype=float)

    distances = np.sqrt(
        ((audio_values - favorite_profile) ** 2).sum(axis=1)
    )

    similarity = np.clip(
        1 - distances / np.sqrt(len(FEATURES)), 0, 1
    )

    # Calculate mood and activity scores.
    mood_total = np.zeros(len(data))
    total_weight = 0

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

        mood_total += weight * match
        total_weight += weight

    mood_score = mood_total / total_weight

    # Combine similarity and mood using adjustable weights.
    data["similarity_score"] = (similarity * 100).round(1)
    data["mood_score"] = (mood_score * 100).round(1)

    data["hybrid_score"] = (
        similarity_weight * similarity
        + (1 - similarity_weight) * mood_score
    ) * 100

    # Rank songs and remove duplicates.
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


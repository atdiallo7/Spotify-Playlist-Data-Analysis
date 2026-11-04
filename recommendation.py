
import pandas as pd

# Load the clustered songs from Part 2
songs = pd.read_csv("data/processed/songs_with_clusters.csv")

# Audio features used to compare songs
features = [
    "Danceability (prepared)",
    "Energy (prepared)",
    "Loudness (prepared)",
    "Speechiness (prepared)",
    "Acousticness (prepared)",
    "Instrumentalness (prepared)",
    "Valence (prepared)"
]


def recommend_songs(song_name, artist_name):
    # Find the selected song
    selected = songs[
        (songs["Track Name"] == song_name)
        & (songs["Artist Name(s)"] == artist_name)
    ]

    if selected.empty:
        return "Song not found"

    selected_song = selected.iloc[0]
    cluster = selected_song["Cluster"]

    # Find other songs in the same cluster
    similar_songs = songs[songs["Cluster"] == cluster].copy()

    # Calculate distance using audio features
    similar_songs["distance"] = (
        (similar_songs[features] - selected_song[features]) ** 2
    ).sum(axis=1) ** 0.5

    # Remove the selected song and its copies
    similar_songs = similar_songs[
        ~(
            (similar_songs["Track Name"] == selected_song["Track Name"])
            & (similar_songs["Artist Name(s)"] == selected_song["Artist Name(s)"])
        )
    ]

    # Remove repeated songs by the same artist
    similar_songs = similar_songs.drop_duplicates(
        subset=["Track Name", "Artist Name(s)"]
    )

    # Return the five closest songs
    similar_songs = similar_songs.sort_values("distance")

    return similar_songs[
        ["Track Name", "Artist Name(s)", "distance"]
    ].head(5)


# Test the recommendation function
if __name__ == "__main__":
    print(recommend_songs("Starboy", "The Weeknd;Daft Punk"))

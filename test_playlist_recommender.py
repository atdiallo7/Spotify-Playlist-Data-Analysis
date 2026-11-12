"""Small tests that work without the original Spotify dataset."""

import unittest
import pandas as pd
from playlist_recommender import FEATURES, build_playlist, genre_column


def sample_data():
    names = ["Favorite A", "Favorite B", "Other Pop", "Other Rock", "Very Close", "Far", "Cluster 2", "Extra A", "Extra B", "Extra C", "Extra D", "Extra E", "Extra F", "Extra G", "Extra H", "Extra I", "Extra J"]
    rows = []
    for idx, name in enumerate(names):
        features = {feature: float((idx % 10) / 10) for feature in FEATURES}
        rows.append({
            "Track URI": f"spotify:track:{idx}",
            "Track Name": name,
            "Artist Name(s)": f"Artist {idx}",
            "Cluster": 0 if idx < 6 else 1,
            "Genre": "pop" if idx % 2 == 0 else "rock",
            **features,
        })
    return pd.DataFrame(rows)


class PlaylistTests(unittest.TestCase):
    def test_single_song_and_exclusion(self):
        songs = sample_data()
        result = build_playlist(songs, ["spotify:track:0"])
        self.assertEqual(len(result), 5)
        self.assertNotIn("Favorite A", result["Track Name"].tolist())
        self.assertTrue(result["distance"].is_monotonic_increasing)

    def test_multiple_songs_uses_multiple_clusters(self):
        songs = sample_data()
        result = build_playlist(songs, ["spotify:track:0", "spotify:track:6"], count=10)
        self.assertEqual(len(result), 10)
        self.assertTrue({0, 1}.issubset(set(result["Cluster"])))

    def test_genre_filter(self):
        songs = sample_data()
        result = build_playlist(songs, ["spotify:track:0"], same_genre=True)
        self.assertTrue(all(result["Genre"] == "pop"))
        self.assertNotIn("Favorite A", result["Track Name"].tolist())

    def test_slider_changes_ranking(self):
        songs = sample_data()
        original = build_playlist(songs, ["spotify:track:0"])
        adjusted = build_playlist(songs, ["spotify:track:0"], energy=1.0, valence=1.0)
        self.assertNotEqual(original["Track Name"].tolist(), adjusted["Track Name"].tolist())

    def test_genre_data_not_present(self):
        songs = sample_data().drop(columns="Genre")
        self.assertIsNone(genre_column(songs))
        with self.assertRaisesRegex(ValueError, "Genre labels are missing"):
            build_playlist(songs, ["spotify:track:0"], same_genre=True)


if __name__ == "__main__":
    unittest.main()

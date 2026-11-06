"""
Genre similarities and differences from Spotify audio features.

Everything here takes a "long" DataFrame with one row per track-genre pair
(columns: Track URI, Track Name, Genre, plus the audio features) and returns
either a tidy DataFrame or a Plotly figure, so the same calls work in the
notebook and in the Streamlit app.
"""

import numpy as np
import pandas as pd
import plotly.express as px

# Audio features only; popularity says nothing about how a song sounds
AUDIO_FEATURES = [
    "Danceability",
    "Energy",
    "Valence",
    "Acousticness",
    "Instrumentalness",
    "Speechiness",
    "Liveness",
    "Tempo",
    "Loudness",
]

# Shared look for every chart so they sit together cleanly in the app
TEMPLATE = "plotly_white"
DIVERGING = "RdBu_r"


def standardize(genre_tracks, features=AUDIO_FEATURES):
    """Z-score each feature across all tracks.

    Tempo and loudness are on very different scales than the 0-1 features,
    so without this they would dominate every distance.
    """
    out = genre_tracks.copy()
    values = out[features]
    out[features] = (values - values.mean()) / values.std()
    return out


def genre_profiles(genre_tracks, features=AUDIO_FEATURES, min_tracks=20):
    """Average standardized feature value per genre (one row per genre).

    A value of +1 means the genre sits one standard deviation above the
    average song on that feature.
    """
    z = standardize(genre_tracks, features)
    sizes = z["Genre"].value_counts()
    keep = sizes[sizes >= min_tracks].index
    profiles = z[z["Genre"].isin(keep)].groupby("Genre")[features].mean()
    profiles["Tracks"] = sizes[profiles.index]
    return profiles


def feature_importance(genre_tracks, features=AUDIO_FEATURES):
    """How much of each feature's variation is explained by genre (eta squared).

    0 means genre tells you nothing about the feature; 1 means songs inside a
    genre are identical on it. This is the one-way ANOVA effect size.
    """
    rows = []
    for feature in features:
        values = genre_tracks[[feature, "Genre"]].dropna()
        grand_mean = values[feature].mean()
        total = ((values[feature] - grand_mean) ** 2).sum()
        groups = values.groupby("Genre")[feature].agg(["mean", "size"])
        between = (groups["size"] * (groups["mean"] - grand_mean) ** 2).sum()
        rows.append({"Feature": feature, "Explained by Genre": between / total})
    return (
        pd.DataFrame(rows)
        .sort_values("Explained by Genre", ascending=False)
        .reset_index(drop=True)
    )


def genre_distances(profiles, features=AUDIO_FEATURES):
    """Euclidean distance between every pair of genre profiles (square matrix)."""
    points = profiles[features].to_numpy()
    diff = points[:, None, :] - points[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))
    return pd.DataFrame(dist, index=profiles.index, columns=profiles.index)


def genre_pairs_ranked(distances):
    """Every unique genre pair with its distance, closest first."""
    upper = np.triu(np.ones(distances.shape, dtype=bool), k=1)
    pairs = (
        distances.where(upper)
        .rename_axis(index="Genre A", columns="Genre B")
        .stack()
        .dropna()
        .rename("Distance")
        .reset_index()
    )
    return pairs.sort_values("Distance").reset_index(drop=True)


def nearest_genres(distances, genre, n=5):
    """The n genres that sound most like `genre`, closest first."""
    row = distances[genre].drop(genre).sort_values()
    return row.head(n).rename("Distance").reset_index().rename(
        columns={"index": "Genre"}
    )


def genre_spread(genre_tracks, features=AUDIO_FEATURES, min_tracks=20):
    """Average within-genre standard deviation of the standardized features.

    Low spread means the label describes a consistent sound; high spread
    means the genre is a grab bag.
    """
    z = standardize(genre_tracks, features)
    sizes = z["Genre"].value_counts()
    keep = sizes[sizes >= min_tracks].index
    spread = z[z["Genre"].isin(keep)].groupby("Genre")[features].std()
    return (
        spread.mean(axis=1)
        .rename("Spread")
        .sort_values()
        .reset_index()
    )


# ---------------------------------------------------------------- charts


def plot_feature_importance(importance):
    """Horizontal bar: which features differ most between genres."""
    fig = px.bar(
        importance,
        x="Explained by Genre",
        y="Feature",
        orientation="h",
        text_auto=".0%",
        template=TEMPLATE,
        title="Which audio features differ most between genres?",
    )
    fig.update_yaxes(autorange="reversed", title=None)
    fig.update_xaxes(tickformat=".0%", title="Share of variation explained by genre")
    fig.update_traces(marker_color="#1DB954")
    fig.update_layout(height=400)
    return fig


def plot_profile_heatmap(profiles, genres, features=AUDIO_FEATURES):
    """Heatmap of standardized feature means for the chosen genres.

    Red = above the average song, blue = below. Genres are ordered by energy
    so similar-sounding genres land near each other.
    """
    subset = profiles.loc[genres, features].sort_values("Energy")
    fig = px.imshow(
        subset,
        color_continuous_scale=DIVERGING,
        color_continuous_midpoint=0,
        zmin=-2,
        zmax=2,
        aspect="auto",
        text_auto=".1f",
        template=TEMPLATE,
        title="Genre sound profiles (std. devs from the average song)",
    )
    fig.update_layout(
        height=max(350, 32 * len(subset) + 120),
        coloraxis_colorbar=dict(title="z"),
    )
    fig.update_xaxes(side="top", title=None)
    fig.update_yaxes(title=None)
    return fig


def plot_genre_map(profiles, features=AUDIO_FEATURES, highlight=None):
    """2-D map of genres via PCA on their profiles; close points sound alike.

    Uses an SVD so no extra dependency is needed.
    """
    points = profiles[features].to_numpy()
    centered = points - points.mean(axis=0)
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vt[:2].T
    explained = (s ** 2 / (s ** 2).sum())[:2]

    # name each axis after the features that drive it most
    def axis_label(component, share):
        loadings = pd.Series(component, index=features)
        top = loadings.abs().sort_values(ascending=False).index[:2]
        parts = [f"{'+' if loadings[f] > 0 else '-'}{f}" for f in top]
        return f"{' / '.join(parts)} ({share:.0%})"

    plot_df = pd.DataFrame(
        {"PC1": coords[:, 0], "PC2": coords[:, 1], "Genre": profiles.index,
         "Tracks": profiles["Tracks"].to_numpy()}
    )
    plot_df["Group"] = "Other"
    if highlight:
        plot_df.loc[plot_df["Genre"].isin(highlight), "Group"] = "Selected"

    fig = px.scatter(
        plot_df,
        x="PC1",
        y="PC2",
        text=plot_df["Genre"].where(
            plot_df["Group"] == "Selected" if highlight else True, ""
        ),
        hover_name="Genre",
        hover_data={"Tracks": True, "PC1": False, "PC2": False, "Group": False},
        color="Group",
        color_discrete_map={"Selected": "#1DB954", "Other": "#B3B3B3"},
        template=TEMPLATE,
        title="Genre map: genres near each other sound alike",
    )
    fig.update_traces(textposition="top center", textfont_size=10, marker_size=9)
    fig.update_layout(
        height=600,
        showlegend=False,
        xaxis_title=axis_label(vt[0], explained[0]),
        yaxis_title=axis_label(vt[1], explained[1]),
    )
    return fig


def plot_genre_compare(profiles, genre_a, genre_b, features=AUDIO_FEATURES):
    """Side-by-side bars for two genres, sorted by how much they differ."""
    pair = profiles.loc[[genre_a, genre_b], features].T
    order = (pair[genre_a] - pair[genre_b]).abs().sort_values(ascending=False).index
    long = (
        pair.loc[order]
        .reset_index(names="Feature")
        .melt(id_vars="Feature", var_name="Genre", value_name="z")
    )
    fig = px.bar(
        long,
        x="z",
        y="Feature",
        color="Genre",
        barmode="group",
        orientation="h",
        template=TEMPLATE,
        color_discrete_sequence=["#1DB954", "#535353"],
        title=f"{genre_a} vs {genre_b} (biggest differences on top)",
    )
    fig.update_yaxes(autorange="reversed", title=None)
    fig.update_xaxes(title="Std. devs from the average song", zeroline=True)
    fig.update_layout(height=420, legend_title=None)
    return fig


def plot_spread(spread, n=10):
    """Most consistent and most varied genres, side by side."""
    both = pd.concat(
        [
            spread.head(n).assign(Type="Most consistent"),
            spread.tail(n).assign(Type="Most varied"),
        ]
    )
    fig = px.bar(
        both,
        x="Spread",
        y="Genre",
        color="Type",
        orientation="h",
        template=TEMPLATE,
        color_discrete_map={"Most consistent": "#1DB954", "Most varied": "#E8115B"},
        title="How consistent is each genre's sound?",
    )
    fig.update_yaxes(categoryorder="total ascending", title=None)
    fig.update_xaxes(title="Average within-genre spread (lower = more uniform)")
    fig.update_layout(height=550, legend_title=None)
    return fig

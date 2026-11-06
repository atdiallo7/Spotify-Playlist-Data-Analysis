"""
Deeper genre analysis with machine learning and NLP (Question 1, parts 6-10).

Each part answers a genre question AND predicts how the clustering (Q2) and
recommender (Q3) will behave:

6. Genre classifier: how predictable is a genre from sound alone, and which
   genres get confused? Confused genres will merge into the same cluster.
7. Permutation importance: which features the model actually relies on,
   compared with the simple one-feature-at-a-time view from part 1.
8. Genre family tree + silhouette: how many natural groups exist, and how
   well do genre labels form clusters at all?
9. NLP on track titles: do titles carry genre information audio misses?
10. Nearest-neighbour purity: if you recommend the 10 closest songs, how often
    do they stay in the seed song's genre?

All functions take the same long track-genre table as genre_analysis.py.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    silhouette_samples,
    top_k_accuracy_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

from utils.genre_analysis import AUDIO_FEATURES, TEMPLATE, standardize

SEED = 42
GREEN = "#1DB954"


# ------------------------------------------------------------ data prep


def single_genre_sample(genre_tracks, per_genre=300, min_tracks=100, seed=SEED):
    """One row per track, only tracks with exactly one genre, balanced.

    Tracks tagged with several genres would appear in both train and test
    under different labels, which leaks answers to the model, so they are
    left out. Sampling the same number per genre stops big genres from
    dominating accuracy.
    """
    counts = genre_tracks.groupby("Track URI")["Genre"].transform("size")
    single = genre_tracks[counts == 1]
    sizes = single["Genre"].value_counts()
    keep = sizes[sizes >= min_tracks].index
    single = single[single["Genre"].isin(keep)]
    # shuffle, then keep the first `per_genre` rows of each genre
    return (
        single.sample(frac=1, random_state=seed)
        .groupby("Genre")
        .head(per_genre)
        .reset_index(drop=True)
    )


def split(sample, seed=SEED):
    """Stratified 75/25 train-test split of the sample's row positions."""
    idx = np.arange(len(sample))
    return train_test_split(
        idx, test_size=0.25, stratify=sample["Genre"], random_state=seed
    )


# ------------------------------------------------- 6. genre classifier


def train_audio_classifier(sample, train_idx, features=AUDIO_FEATURES,
                           n_estimators=100, min_samples_leaf=10):
    """Random forest predicting genre from audio features.

    Every leaf stores a probability for each of the 100+ genres, so the
    forest gets large fast. min_samples_leaf=10 keeps it around 230 MB with
    almost no accuracy cost; the app uses 20 (~115 MB) to fit hosting limits.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        n_jobs=-1,
        random_state=SEED,
    )
    model.fit(sample.iloc[train_idx][features], sample.iloc[train_idx]["Genre"])
    return model


def classifier_report(model, sample, test_idx, features=AUDIO_FEATURES):
    """Overall scores, per-genre F1 and the most confused genre pairs."""
    X = sample.iloc[test_idx][features]
    y = sample.iloc[test_idx]["Genre"]
    proba = model.predict_proba(X)
    pred = model.classes_[proba.argmax(axis=1)]

    n_classes = len(model.classes_)
    scores = pd.Series({
        "Genres": n_classes,
        "Random guess": 1 / n_classes,
        "Top-1 accuracy": (pred == y).mean(),
        "Top-5 accuracy": top_k_accuracy_score(y, proba, k=5, labels=model.classes_),
    })

    per_genre = (
        pd.Series(
            f1_score(y, pred, labels=model.classes_, average=None),
            index=model.classes_,
            name="F1",
        )
        .sort_values(ascending=False)
        .rename_axis("Genre")
        .reset_index()
    )

    cm = pd.DataFrame(
        confusion_matrix(y, pred, labels=model.classes_, normalize="true"),
        index=model.classes_,
        columns=model.classes_,
    )
    off = cm.where(~np.eye(len(cm), dtype=bool))
    confused = (
        off.rename_axis(index="True genre", columns="Predicted as")
        .stack()
        .dropna()
        .rename("Share of true genre")
        .sort_values(ascending=False)
        .reset_index()
    )
    return scores, per_genre, cm, confused


def plot_f1(per_genre, n=12):
    """Easiest and hardest genres to recognise from sound."""
    both = pd.concat([
        per_genre.head(n).assign(Type="Easiest to recognise"),
        per_genre.tail(n).assign(Type="Hardest to recognise"),
    ])
    fig = px.bar(
        both, x="F1", y="Genre", color="Type", orientation="h",
        template=TEMPLATE,
        color_discrete_map={"Easiest to recognise": GREEN,
                            "Hardest to recognise": "#E8115B"},
        title="Which genres can a model recognise from sound alone?",
    )
    fig.update_yaxes(categoryorder="total ascending", title=None)
    fig.update_xaxes(range=[0, 1], title="F1 score (1 = always right)")
    fig.update_layout(height=600, legend_title=None)
    return fig


def plot_confusion(cm, genres):
    """Row-normalised confusion matrix for a readable subset of genres."""
    genres = [g for g in genres if g in cm.index]
    sub = cm.loc[genres, genres]
    fig = px.imshow(
        sub, color_continuous_scale="Greens", zmin=0, zmax=0.6,
        text_auto=".0%", aspect="auto", template=TEMPLATE,
        labels=dict(x="Predicted genre", y="True genre", color="Share"),
        title="Where the classifier gets confused (each row: where that genre's songs went)",
    )
    fig.update_xaxes(side="bottom", tickangle=-45)
    fig.update_layout(height=max(400, 34 * len(genres) + 200))
    return fig


# ------------------------------------------ 7. permutation importance


def model_importance(model, sample, test_idx, features=AUDIO_FEATURES):
    """Accuracy lost when each feature is shuffled (higher = model needs it)."""
    X = sample.iloc[test_idx][features]
    y = sample.iloc[test_idx]["Genre"]
    result = permutation_importance(
        model, X, y, n_repeats=5, random_state=SEED
    )
    return (
        pd.DataFrame({
            "Feature": features,
            "Accuracy drop": result.importances_mean,
            "Std": result.importances_std,
        })
        .sort_values("Accuracy drop", ascending=False)
        .reset_index(drop=True)
    )


def plot_importance_comparison(perm, eta):
    """Univariate (eta squared) vs model (permutation) importance, both ranked.

    Each is scaled to its own maximum so the shapes can be compared.
    """
    merged = perm.merge(eta, on="Feature")
    merged["Model (permutation)"] = merged["Accuracy drop"] / merged["Accuracy drop"].max()
    merged["One feature at a time (eta²)"] = (
        merged["Explained by Genre"] / merged["Explained by Genre"].max()
    )
    long = merged.melt(
        id_vars="Feature",
        value_vars=["Model (permutation)", "One feature at a time (eta²)"],
        var_name="Method", value_name="Relative importance",
    )
    fig = px.bar(
        long, x="Relative importance", y="Feature", color="Method",
        barmode="group", orientation="h", template=TEMPLATE,
        category_orders={"Feature": merged["Feature"].tolist()},
        color_discrete_sequence=[GREEN, "#535353"],
        title="What the model relies on vs what differs on average",
    )
    fig.update_yaxes(title=None)
    fig.update_layout(height=450, legend_title=None)
    return fig


# --------------------------------- 8. genre family tree and silhouette


def genre_tree(profiles, features=AUDIO_FEATURES, n_families=8):
    """Ward hierarchical clustering of genre profiles.

    Returns the dendrogram figure and a table of which family each genre
    lands in when the tree is cut into `n_families` groups.
    """
    X = profiles[features].to_numpy()
    labels = profiles.index.tolist()
    fig = ff.create_dendrogram(
        X, labels=labels, orientation="left",
        linkagefun=lambda x: linkage(x, method="ward"),
        color_threshold=None,
    )
    fig.update_layout(
        template=TEMPLATE,
        title="Genre family tree (genres joined lower down sound more alike)",
        height=max(600, 13 * len(labels)),
        showlegend=False,
    )
    fig.update_yaxes(tickfont_size=9)
    fig.update_xaxes(title="Ward distance")

    families = fcluster(linkage(X, method="ward"), n_families, criterion="maxclust")
    family_table = (
        pd.DataFrame({"Genre": labels, "Family": families})
        .groupby("Family")["Genre"]
        .agg(lambda g: ", ".join(sorted(g)))
        .reset_index()
    )
    family_table["Genres"] = family_table["Genre"].str.count(",") + 1
    return fig, family_table[["Family", "Genres", "Genre"]]


def genre_silhouette(sample, features=AUDIO_FEATURES, max_rows=15000):
    """Silhouette of each track using its genre label as the cluster.

    Positive = the track is closer to its own genre than to the nearest other
    genre. Averaged per genre, this says how cluster-like each label is,
    i.e. how likely clustering is to rediscover it.
    """
    z = standardize(sample, features)
    if len(z) > max_rows:
        z = z.sample(max_rows, random_state=SEED)
    s = silhouette_samples(z[features], z["Genre"])
    return (
        pd.DataFrame({"Genre": z["Genre"].to_numpy(), "Silhouette": s})
        .groupby("Genre")["Silhouette"].mean()
        .sort_values(ascending=False)
        .reset_index()
    )


# --------------------------------------------- 9. NLP on track titles


def title_vectorizer():
    """TF-IDF over words in track titles.

    Drops English stop words and words in under 5 titles; keeps one and two
    word phrases so "feat" or "remix" style tokens and short phrases show.
    """
    return TfidfVectorizer(
        stop_words="english",
        min_df=5,
        ngram_range=(1, 2),
        sublinear_tf=True,
        token_pattern=r"(?u)\b[^\W\d_]{2,}\b",  # letters only, no numbers
    )


# version tags and abbreviations that say nothing about genre
TITLE_NOISE = {"feat", "ft", "pt", "vol", "op", "remaster", "remastered",
               "version", "edit", "mix", "radio", "original", "bonus", "track"}


def distinctive_title_words(genre_tracks, genres, n=8, min_count=10):
    """Words that mark each genre's titles, by weighted log-odds.

    For each genre, compares how often a word appears in its titles vs all
    other genres' titles, using an informative Dirichlet prior (Monroe,
    Colaresi and Quinn, 2008). The z-score keeps rare words from winning on
    a lucky one or two uses, which plain frequency ratios and TF-IDF do not.
    """
    from sklearn.feature_extraction.text import CountVectorizer

    sub = genre_tracks[genre_tracks["Genre"].isin(genres)]
    vec = CountVectorizer(
        stop_words=list(
            TfidfVectorizer(stop_words="english").get_stop_words() | TITLE_NOISE
        ),
        min_df=min_count,
        binary=True,  # count titles containing the word, not repeats
        token_pattern=r"(?u)\b[^\W\d_]{2,}\b",
    )
    X = vec.fit_transform(sub["Track Name"].fillna("").astype(str))
    vocab = np.array(vec.get_feature_names_out())
    labels = sub["Genre"].to_numpy()

    total = np.asarray(X.sum(axis=0)).ravel().astype(float)
    prior = total / total.sum() * 500  # prior strength of 500 pseudo-words
    a0 = prior.sum()
    rows = []
    for genre in genres:
        mask = labels == genre
        if not mask.any():
            continue
        y_i = np.asarray(X[mask].sum(axis=0)).ravel().astype(float)
        y_j = total - y_i
        n_i, n_j = y_i.sum(), y_j.sum()
        delta = (np.log((y_i + prior) / (n_i + a0 - y_i - prior))
                 - np.log((y_j + prior) / (n_j + a0 - y_j - prior)))
        z = delta / np.sqrt(1 / (y_i + prior) + 1 / (y_j + prior))
        top = z.argsort()[::-1][:n]
        rows.append({"Genre": genre,
                     "Distinctive title words": ", ".join(vocab[top])})
    return pd.DataFrame(rows)


def text_vs_audio_models(sample, train_idx, test_idx, audio_model,
                         features=AUDIO_FEATURES):
    """Genre accuracy from titles only, audio only, and both combined.

    The combined model averages the two models' probabilities (late fusion),
    which is the same idea a hybrid recommender uses.
    """
    titles = sample["Track Name"].fillna("").astype(str)
    y = sample["Genre"]
    vec = title_vectorizer()
    X_train = vec.fit_transform(titles.iloc[train_idx])
    X_test = vec.transform(titles.iloc[test_idx])
    text_model = LogisticRegression(max_iter=2000, C=5)
    text_model.fit(X_train, y.iloc[train_idx])

    y_test = y.iloc[test_idx]
    p_text = text_model.predict_proba(X_test)
    p_audio = audio_model.predict_proba(sample.iloc[test_idx][features])
    # both models were fit on the same labels, so classes_ line up
    assert list(text_model.classes_) == list(audio_model.classes_)
    classes = audio_model.classes_
    rows = []
    for name, p in [("Titles only (TF-IDF)", p_text),
                    ("Audio only (random forest)", p_audio),
                    ("Titles + audio", (p_text + p_audio) / 2)]:
        rows.append({
            "Model": name,
            "Top-1 accuracy": (classes[p.argmax(axis=1)] == y_test).mean(),
            "Top-5 accuracy": top_k_accuracy_score(y_test, p, k=5, labels=classes),
        })
    return pd.DataFrame(rows)


def text_audio_agreement(genre_tracks, audio_distances):
    """Do genres with similar titles also sound similar?

    Builds a TF-IDF title vector per genre, measures cosine similarity
    between genres, and rank-correlates it with audio similarity.
    """
    genres = audio_distances.index
    docs = (
        genre_tracks[genre_tracks["Genre"].isin(genres)]
        .groupby("Genre")["Track Name"]
        .apply(lambda t: " ".join(t.dropna().astype(str)))
        .reindex(genres)
        .fillna("")
    )
    m = title_vectorizer().fit_transform(docs)
    text_sim = (m @ m.T).toarray()  # rows are L2-normalised, so this is cosine
    upper = np.triu_indices(len(genres), k=1)
    rho, p = spearmanr(text_sim[upper], -audio_distances.to_numpy()[upper])
    pairs = pd.DataFrame({
        "Genre A": genres[upper[0]],
        "Genre B": genres[upper[1]],
        "Title similarity": text_sim[upper],
        "Audio distance": audio_distances.to_numpy()[upper],
    })
    return rho, p, pairs


def plot_model_comparison(results):
    long = results.melt(id_vars="Model", var_name="Metric", value_name="Accuracy")
    fig = px.bar(
        long, x="Model", y="Accuracy", color="Metric", barmode="group",
        text_auto=".0%", template=TEMPLATE,
        color_discrete_sequence=[GREEN, "#535353"],
        title="Predicting genre: titles vs sound vs both",
    )
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    fig.update_xaxes(title=None)
    fig.update_layout(height=420, legend_title=None)
    return fig


# ------------------------------------ 10. nearest-neighbour genre purity


def neighbour_purity(genre_tracks, features=AUDIO_FEATURES, k=10,
                     max_rows=30000, min_tracks=100):
    """For each track, the share of its k nearest songs that share a genre.

    This is exactly what a 'more songs like this' recommender returns, so a
    genre's purity is how often recommendations stay inside that genre.
    """
    # one row per track with the full set of its genres
    tracks = (
        genre_tracks.groupby("Track URI")
        .agg(Genres=("Genre", set), **{f: (f, "first") for f in features})
        .reset_index()
    )
    z = standardize(tracks, features)[features].to_numpy()
    genres = tracks["Genres"].to_numpy()
    # search the full pool, but only score a sample of seed songs
    rng = np.random.default_rng(SEED)
    seeds = rng.choice(len(tracks), min(max_rows, len(tracks)), replace=False)
    _, idx = NearestNeighbors(n_neighbors=k + 1).fit(z).kneighbors(z[seeds])
    shared = np.array([
        np.mean([bool(genres[i] & genres[j]) for j in row[1:]])
        for i, row in zip(seeds, idx)
    ])
    tracks = tracks.iloc[seeds].copy()
    tracks["Purity"] = shared
    per_genre = (
        tracks.explode("Genres")
        .groupby("Genres")["Purity"].agg(["mean", "size"])
        .rename(columns={"mean": "Purity", "size": "Tracks"})
        .rename_axis("Genre")
        .reset_index()
    )
    # chance level: a random neighbour shares the genre at roughly its share
    per_genre["Chance"] = per_genre["Tracks"] / len(tracks)
    per_genre["Lift"] = per_genre["Purity"] / per_genre["Chance"]
    # judge genres by their full size, not their count in the seed sample
    sizes = genre_tracks["Genre"].value_counts()
    per_genre = per_genre[per_genre["Genre"].map(sizes) >= min_tracks]
    return shared.mean(), per_genre.sort_values("Purity", ascending=False)


def plot_link_chart(link):
    """Scatter: classifier F1 vs neighbour purity, sized by silhouette.

    Genres top-right are distinct in sound: they will form their own cluster
    (Q2) and keep recommendations in-genre (Q3).
    """
    fig = px.scatter(
        link, x="F1", y="Purity", hover_name="Genre", text="Label",
        color="Silhouette", color_continuous_scale="RdYlGn",
        color_continuous_midpoint=0, template=TEMPLATE,
        title="Distinct genres are easy to classify AND keep recommendations in-genre",
    )
    fig.update_traces(textposition="top center", textfont_size=9, marker_size=9)
    fig.update_xaxes(title="Classifier F1 (part 6)")
    fig.update_yaxes(title="Share of 10 nearest songs in same genre (part 10)",
                     tickformat=".0%")
    fig.update_layout(height=600)
    return fig

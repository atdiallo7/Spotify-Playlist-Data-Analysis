# Spotify-Playlist-Data-Analysis

Music Analysis on a Spotify user's playlist

**Project Topic Ideas (9/2)**  
Link to Word File

**Project Proposal Working Doc (9/10)**  
Link to Word File

**Project Background Research Doc (9/17)**  
[Background Research Document](https://indiana-my.sharepoint.com/:w:/r/personal/atdiallo_iu_edu/Documents/Background%20Research.docx?d=w63ae4f78c12e4371b16ec3ce6adc6a65&csf=1&web=1&e=C7wnLc)

## Project Overview

This project analyzes Spotify songs using audio features to identify similarities, group songs into clusters, and recommend songs based on user preferences.

The project has three main parts:

1. **Data Analysis:** Explore and prepare Spotify audio features.
2. **Song Clustering:** Use K-Means clustering to group songs with similar audio characteristics.
3. **Song Recommendation:** Recommend songs based on audio similarity, mood, and user preferences.

## Spotify Smart Playlist

We developed an interactive Streamlit application with four main features:

- **Similar Songs:** Select up to three favorite songs and generate 5, 10, or 15 similar song recommendations.
- **Mood & Activity:** Generate playlists for activities such as studying, working out, relaxing, and driving.
- **Hybrid Recommendations:** Combine song similarity and mood preferences using adjustable weights.
- **Evaluation:** Compare similarity, mood, and hybrid scores using an interactive heatmap. Evaluate duplicate songs, artist variety, and audio diversity.

The application also allows users to listen to recommended songs through Spotify and download playlists as CSV files.

### Recommendation Approach

Our recommendation system mainly uses **content-based filtering** based on seven Spotify audio features.

- **Similar Songs:** Uses K-Means clusters and Euclidean distance to recommend songs similar to the user's favorites.
- **Mood & Activity:** Uses rule-based scoring to recommend songs based on selected moods or activities.
- **Hybrid:** Combines similarity and mood scores using adjustable weights.

The system does not use collaborative filtering because it does not rely on other users' listening histories or ratings.
## Recommendation Methods

The recommendation system uses seven prepared Spotify audio features:

Danceability, Energy, Loudness, Speechiness, Acousticness, Instrumentalness, and Valence.

**Similar Songs** uses Euclidean distance and existing K-Means clusters to identify songs with similar audio characteristics.

**Mood & Activity** uses audio feature preferences to recommend songs for different activities.

**Hybrid Recommendations** combines similarity and mood scores to provide more personalized results.

## How to Run the Application

**1. Install the required libraries**

Open a terminal in the project folder and run:

`python -m pip install -r requirements.txt`

**2. Start the Streamlit application**

`python -m streamlit run app.py`

The application uses the processed dataset located at:

`data/processed/songs_with_clusters.csv`

## Evaluation

The Evaluation tab compares recommendation methods using similarity scores, mood scores, and hybrid scores.

It also measures playlist diversity, artist variety, and duplicate recommendations.

An interactive Altair heatmap helps users compare recommendation scores across songs.

## Technologies Used

Python, Pandas, NumPy, Scikit-learn, Streamlit, Altair, and K-Means clustering.

"""
Streamlit entry point for the group project.

Run locally with:
    streamlit run app.py
"""

import streamlit as st
from recommendation import recommend_songs, songs

from utils import __version__  # noqa: F401 -- confirms the utils package imports cleanly


def main():
    st.set_page_config(
        page_title="Group Project",  # TODO: replace with your app's real name
        page_icon=":bar_chart:",
        layout="wide",
    )

    st.title("Group Project")
    st.write(
        "This is the starting point for our Streamlit app. "
        "Replace this page with the app's actual content as the project develops."
    )

    st.header("Data")
    st.write("Load and display data here once we've decided on a source.")

    st.header("Analysis")
    st.write("Add charts, models, or other analysis here.")

    
    # Song recommendation section
    st.header("Song Recommendations")

    # Select a song
    song_list = sorted(songs["Track Name"].dropna().unique())
    song_name = st.selectbox("Select a song", song_list)

    # Select the artist
    artists = songs[songs["Track Name"] == song_name]
    artist_list = sorted(artists["Artist Name(s)"].dropna().unique())
    artist_name = st.selectbox("Select an artist", artist_list)

    # Show five similar songs
    if st.button("Get Recommendations"):
        results = recommend_songs(song_name, artist_name)

        if isinstance(results, str):
            st.warning(results)
        else:
            st.write("Recommended songs:")
            st.dataframe(results, hide_index=True)

            

    # Next steps for the project
    st.header("Next Steps")
    st.write(
        "We plan to combine everyone's work into the main app, "
        "improve the layout, and test the complete project. "
        "We may also add genre filtering to improve recommendations."
    )




if __name__ == "__main__":
    main()

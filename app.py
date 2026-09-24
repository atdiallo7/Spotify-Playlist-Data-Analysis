"""
Streamlit entry point for the group project.

Run locally with:
    streamlit run app.py
"""

import streamlit as st

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


if __name__ == "__main__":
    main()

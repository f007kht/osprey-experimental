"""Main Streamlit application entry point.

Module 1 - Osprey Backend: Document processing application.
"""

import streamlit as st

from app.ui.upload import render_upload
from app.ui.display import render_results
from app.ui.downloads import render_downloads
from app.ui.mongodb_config import render_mongodb_panel


def run():
    """Run the Streamlit application."""
    st.set_page_config(page_title="Module 1 - Osprey Backend", layout="wide")
    
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Upload", "Results", "Downloads", "MongoDB"])
    
    if page == "Upload":
        render_upload()
    elif page == "Results":
        render_results()
    elif page == "Downloads":
        render_downloads()
    else:
        render_mongodb_panel()


if __name__ == "__main__":
    run()


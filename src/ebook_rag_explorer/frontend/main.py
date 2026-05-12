"""Streamlit frontend for Ebook RAG Explorer."""

import streamlit as st

from ebook_rag_explorer.frontend.config import config
from ebook_rag_explorer.frontend.api_client import EbookRAGClient
from ebook_rag_explorer.frontend.styles import get_theme_css
from ebook_rag_explorer.frontend.components import (
    render_search_tab,
    render_books_tab,
    render_collections_tab,
    render_upload_tab,
)


def init_session_state():
    """Initialize session state variables."""
    if "theme" not in st.session_state:
        st.session_state.theme = "dark"

    if "api_url" not in st.session_state:
        st.session_state.api_url = config.default_api_url

    if "client" not in st.session_state:
        st.session_state.client = EbookRAGClient(st.session_state.api_url)

    if "collections" not in st.session_state:
        st.session_state.collections = []

    if "search_results" not in st.session_state:
        st.session_state.search_results = None

    if "search_query" not in st.session_state:
        st.session_state.search_query = ""


def get_client() -> EbookRAGClient:
    """Get or recreate the API client based on current URL."""
    client = st.session_state.client

    if client.base_url != st.session_state.api_url:
        client = EbookRAGClient(st.session_state.api_url)
        st.session_state.client = client

    return client


def check_api_health(client: EbookRAGClient) -> bool:
    """Check if the API is reachable.

    Args:
        client: The API client.

    Returns:
        True if healthy, False otherwise.
    """
    try:
        health = client.health_check()
        return health.get("status") == "healthy"
    except Exception:
        return False


def main():
    """Main entry point for the Streamlit app."""
    st.set_page_config(
        page_title=config.app_title,
        page_icon=config.app_icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_session_state()

    is_dark = st.session_state.theme == "dark"
    st.markdown(get_theme_css(is_dark), unsafe_allow_html=True)

    with st.sidebar:
        st.title("⚙️ Settings")

        st.radio(
            "🌙 Theme",
            options=["dark", "light"],
            index=0 if is_dark else 1,
            key="theme",
            format_func=lambda x: "🌙 Dark" if x == "dark" else "☀️ Light",
        )

        st.divider()

        new_url = st.text_input(
            "🔗 API URL",
            value=st.session_state.api_url,
            help="URL of the Ebook RAG Explorer backend API",
        )

        if new_url != st.session_state.api_url:
            st.session_state.api_url = new_url
            st.session_state.client = EbookRAGClient(new_url)
            st.rerun()

        st.divider()

        if st.button("🔄 Reconnect", use_container_width=True):
            st.rerun()

        client = get_client()
        if check_api_health(client):
            st.success("🟢 API Connected")
        else:
            st.error("🔴 API Unreachable")
            st.caption("Make sure the backend is running")

    st.title(f"{config.app_icon} {config.app_title}")

    client = get_client()

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Search",
        "📚 Books",
        "📁 Collections",
        "⬆️ Upload",
    ])

    with tab1:
        render_search_tab(client, is_dark)

    with tab2:
        render_books_tab(client, is_dark)

    with tab3:
        render_collections_tab(client, is_dark)

    with tab4:
        render_upload_tab(client)


if __name__ == "__main__":
    main()

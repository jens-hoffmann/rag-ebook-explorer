"""Collections component for viewing and managing collections."""

import streamlit as st
from ebook_rag_explorer.frontend.api_client import CollectionInfo, EbookRAGClient


def render_collections_tab(client: EbookRAGClient, is_dark: bool):
    """Render the collections management tab.

    Args:
        client: The API client instance.
        is_dark: Whether dark theme is active.
    """
    st.subheader("📁 Collections")

    col_title, col_refresh = st.columns([4, 1])

    with col_title:
        st.write("Manage your book collections")

    with col_refresh:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()

    try:
        collections = client.list_collections()
        st.session_state.collections = collections

        if not collections:
            st.info("📭 No collections found. Collections are created when you upload books with a collection name.")
            return

        st.divider()

        cols = st.columns(3)
        for i, collection in enumerate(collections):
            with cols[i % 3]:
                render_collection_card(client, collection, is_dark)

    except Exception as e:
        st.error(f"❌ Error loading collections: {str(e)}")


def render_collection_card(client: EbookRAGClient, collection: CollectionInfo, is_dark: bool):
    """Render a single collection card.

    Args:
        client: The API client instance.
        collection: The CollectionInfo to display.
        is_dark: Whether dark theme is active.
    """
    bg_color = "#1e2530" if is_dark else "#f0f2f6"
    border_color = "#2d3748" if is_dark else "#d1d5db"
    accent_color = "#4f8cff" if is_dark else "#1a73e8"

    st.markdown(f"""
    <div class="collection-card" style="background-color: {bg_color}; border: 1px solid {border_color};">
        <h3 style="margin: 0 0 8px 0; color: {accent_color};">📁 {collection.name}</h3>
        <div style="display: flex; justify-content: center; gap: 24px; margin: 16px 0;">
            <div>
                <div style="font-size: 2em; font-weight: bold; color: {accent_color};">{collection.book_count}</div>
                <div style="opacity: 0.7;">📚 Books</div>
            </div>
            <div>
                <div style="font-size: 2em; font-weight: bold; color: {accent_color};">{collection.chunk_count}</div>
                <div style="opacity: 0.7;">📄 Chunks</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    confirm_key = f"confirm_delete_coll_{collection.id}"
    delete_key = f"delete_coll_{collection.id}"

    if st.session_state.get(confirm_key, False):
        st.warning(f"Delete collection '{collection.name}'? This will also delete all books in it.")
        col_yes, col_no = st.columns(2)
        with col_yes:
            if st.button("Yes, delete", key=f"yes_coll_{collection.id}", type="primary"):
                try:
                    client.delete_collection(collection.id)
                    st.success(f"✅ Deleted collection: {collection.name}")
                    st.session_state[confirm_key] = False
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")

        with col_no:
            if st.button("Cancel", key=f"no_coll_{collection.id}"):
                st.session_state[confirm_key] = False
                st.rerun()
    else:
        if st.button("🗑️ Delete Collection", key=delete_key, use_container_width=True):
            st.session_state[confirm_key] = True
            st.rerun()

    st.divider()

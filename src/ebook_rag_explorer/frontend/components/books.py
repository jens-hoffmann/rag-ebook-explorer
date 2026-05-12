"""Books component for viewing and managing indexed books."""

import streamlit as st
from ebook_rag_explorer.frontend.api_client import BookInfo, EbookRAGClient


def render_books_tab(client: EbookRAGClient, is_dark: bool):
    """Render the books management tab.

    Args:
        client: The API client instance.
        is_dark: Whether dark theme is active.
    """
    st.subheader("📚 Indexed Books")

    collections = st.session_state.get("collections", [])

    col_filter, col_refresh = st.columns([4, 1])

    with col_filter:
        collection_options = ["All Collections"] + [c.name for c in collections]
        selected_filter = st.selectbox(
            "Filter by collection",
            options=collection_options,
            index=0,
            key="books_filter",
        )

    with col_refresh:
        st.write("")
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()

    collection_id = None
    if selected_filter != "All Collections":
        collection_id = selected_filter.lower().replace(" ", "_")

    try:
        books = client.list_books(collection_id=collection_id)

        if not books:
            st.info("📭 No books found. Upload a book to get started!")
            return

        st.divider()

        for book in books:
            render_book_card(client, book, is_dark)

    except Exception as e:
        st.error(f"❌ Error loading books: {str(e)}")


def render_book_card(client: EbookRAGClient, book: BookInfo, is_dark: bool):
    """Render a single book card with delete functionality.

    Args:
        client: The API client instance.
        book: The BookInfo to display.
        is_dark: Whether dark theme is active.
    """
    bg_color = "#1e2530" if is_dark else "#f0f2f6"
    border_color = "#2d3748" if is_dark else "#d1d5db"

    icon = "📄" if book.format.lower() == "pdf" else "📚"

    st.markdown(f"""
    <div style="background-color: {bg_color}; border: 1px solid {border_color}; 
                border-radius: 8px; padding: 16px; margin: 8px 0;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h4 style="margin: 0;">{icon} {book.title or book.id}</h4>
                <p style="margin: 4px 0 0 0; opacity: 0.8;">
                    <strong>ID:</strong> {book.id} | 
                    <strong>Format:</strong> {book.format.upper()} | 
                    <strong>Chunks:</strong> {book.chunk_count}
                    {f" | <strong>Collection:</strong> {book.collection_id}" if book.collection_id else ""}
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_del, col_space = st.columns([1, 4])

    with col_del:
        confirm_key = f"confirm_delete_{book.id}"
        delete_key = f"delete_{book.id}"

        if st.session_state.get(confirm_key, False):
            st.warning("Are you sure?")
            col_yes, col_no = st.columns(2)
            with col_yes:
                if st.button("Yes, delete", key=f"yes_{book.id}", type="primary"):
                    try:
                        client.delete_book(book.id)
                        st.success(f"✅ Deleted: {book.title or book.id}")
                        st.session_state[confirm_key] = False
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

            with col_no:
                if st.button("Cancel", key=f"no_{book.id}"):
                    st.session_state[confirm_key] = False
                    st.rerun()
        else:
            if st.button("🗑️ Delete", key=delete_key):
                st.session_state[confirm_key] = True
                st.rerun()

    st.divider()

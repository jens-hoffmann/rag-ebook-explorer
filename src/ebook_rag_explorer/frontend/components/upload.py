"""Upload component for indexing PDF/EPUB files."""

import streamlit as st
from ebook_rag_explorer.frontend.api_client import EbookRAGClient


def render_upload_tab(client: EbookRAGClient):
    """Render the file upload tab.

    Args:
        client: The API client instance.
    """
    st.subheader("📤 Upload a Book")

    collections = []
    try:
        collections = st.session_state.get("collections", [])
    except Exception:
        pass

    col1, col2 = st.columns([3, 1])

    with col1:
        collection_options = ["None"] + [c.name for c in collections]
        selected_collection = st.selectbox(
            "Collection (optional)",
            options=collection_options,
            index=0,
            help="Select a collection to add this book to",
        )

    with col2:
        st.write("")  # Spacing

    uploaded_file = st.file_uploader(
        "Drag and drop your PDF or EPUB file here",
        type=["pdf", "epub"],
        help="Supported formats: PDF, EPUB",
    )

    if uploaded_file is not None:
        file_extension = uploaded_file.name.split(".")[-1].lower()

        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.info(f"**File:** {uploaded_file.name}")
        with col_info2:
            st.info(f"**Size:** {uploaded_file.size / 1024:.1f} KB")

        collection_id = None
        if selected_collection != "None":
            collection_id = selected_collection.lower().replace(" ", "_")

        if st.button("🚀 Index Book", type="primary", use_container_width=True):
            with st.spinner("Indexing book..."):
                try:
                    file_bytes = uploaded_file.getvalue()
                    result = client.index_file(
                        file_bytes=file_bytes,
                        filename=uploaded_file.name,
                        collection_id=collection_id,
                    )

                    st.success(f"✅ Successfully indexed!")
                    st.write(f"**Title:** {result.title or 'N/A'}")
                    st.write(f"**Format:** {result.format.upper()}")
                    st.write(f"**Chunks indexed:** {result.chunks_indexed}")
                    if result.message:
                        st.write(f"**Message:** {result.message}")

                    st.session_state.upload_success = True

                except Exception as e:
                    st.error(f"❌ Error indexing book: {str(e)}")
                    st.session_state.upload_success = False

    st.divider()

    st.subheader("📋 Supported Formats")
    col_f1, col_f2 = st.columns(2)

    with col_f1:
        st.markdown("""
        <div style="padding: 16px; background-color: #1e2530; border-radius: 8px; text-align: center;">
            <h3 style="margin: 0;">📄 PDF</h3>
            <p style="margin: 8px 0 0 0; opacity: 0.8;">Portable Document Format</p>
        </div>
        """, unsafe_allow_html=True)

    with col_f2:
        st.markdown("""
        <div style="padding: 16px; background-color: #1e2530; border-radius: 8px; text-align: center;">
            <h3 style="margin: 0;">📚 EPUB</h3>
            <p style="margin: 8px 0 0 0; opacity: 0.8;">Electronic Publication</p>
        </div>
        """, unsafe_allow_html=True)

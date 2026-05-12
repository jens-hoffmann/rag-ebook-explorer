"""Search component with prominent source display."""

import streamlit as st
from ebook_rag_explorer.frontend.api_client import EbookRAGClient


def render_source_card(source, index: int, is_dark: bool):
    """Render a single source card.

    Args:
        source: The SourceDocument to display.
        index: The index number for this source.
        is_dark: Whether dark theme is active.
    """
    score_color = "#4f8cff" if is_dark else "#1a73e8"

    if source.score >= 0.8:
        score_label = "🟢 Excellent"
    elif source.score >= 0.6:
        score_label = "🟡 Good"
    elif source.score >= 0.4:
        score_label = "🟠 Moderate"
    else:
        score_label = "🔴 Low"

    with st.expander(f"📄 Source {index + 1} — {score_label} (Score: {source.score:.3f})", expanded=index < 2):
        st.markdown(f"""
        <div class="source-card">
            <div class="source-content">
                {source.content[:500]}{"..." if len(source.content) > 500 else ""}
            </div>
            <div class="source-metadata">
                <strong>Score:</strong> <span class="source-score" style="color: {score_color};">{source.score:.4f}</span>
                {f" | <strong>Page:</strong> {source.metadata.get('page', 'N/A')}" if source.metadata.get('page') else ""}
                {f" | <strong>Chapter:</strong> {source.metadata.get('chapter', 'N/A')}" if source.metadata.get('chapter') else ""}
                {f" | <strong>Source:</strong> {source.metadata.get('source', 'N/A')}" if source.metadata.get('source') else ""}
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_search_tab(client: EbookRAGClient, is_dark: bool):
    """Render the search tab with prominent sources display.

    Args:
        client: The API client instance.
        is_dark: Whether dark theme is active.
    """
    st.subheader("🔍 Ask a Question")

    collections = st.session_state.get("collections", [])

    col_query, col_filter = st.columns([4, 1])

    with col_query:
        query = st.text_input(
            "Search query",
            placeholder="e.g., What is machine learning?",
            help="Enter your question about the indexed books",
        )

    with col_filter:
        collection_options = ["All Collections"] + [c.name for c in collections]
        selected_filter = st.selectbox(
            "Filter by collection",
            options=collection_options,
            index=0,
        )

    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 4])

    with col_btn1:
        search_clicked = st.button("🔎 Search", type="primary", use_container_width=True)

    with col_btn2:
        clear_clicked = st.button("Clear", use_container_width=True)

    if clear_clicked:
        st.session_state.search_results = None
        st.session_state.search_query = ""

    if search_clicked and query:
        st.session_state.search_query = query

        collection_id = None
        if selected_filter != "All Collections":
            collection_id = selected_filter.lower().replace(" ", "_")

        with st.spinner("Searching..."):
            try:
                results = client.search(query=query, collection_id=collection_id)
                st.session_state.search_results = results
                st.session_state.search_error = None

            except Exception as e:
                st.session_state.search_results = None
                st.session_state.search_error = str(e)
                st.error(f"❌ Search error: {str(e)}")

    results = st.session_state.get("search_results")

    if results:
        st.divider()
        st.subheader(f"📊 Results for: \"{results.query}\"")

        col_stat1, col_stat2, col_stat3 = st.columns(3)
        with col_stat1:
            st.metric("Retrieved", results.retrieved_count)
        with col_stat2:
            st.metric("Reranked", results.reranked_count)
        with col_stat3:
            st.metric("Sources Shown", len(results.sources))

        if results.answer:
            st.markdown(f"""
            <div class="search-answer">
                <h4 style="margin-top: 0;">💡 Answer</h4>
                <p>{results.answer}</p>
            </div>
            """, unsafe_allow_html=True)

        st.divider()
        st.subheader("📚 Source Documents")

        if results.sources:
            for i, source in enumerate(results.sources):
                render_source_card(source, i, is_dark)
        else:
            st.info("No source documents found for this query.")

    elif st.session_state.get("search_error"):
        pass

    elif query and not search_clicked:
        st.info("👆 Press the Search button to find answers.")

    else:
        st.info("👆 Enter a question above to search through your indexed books.")

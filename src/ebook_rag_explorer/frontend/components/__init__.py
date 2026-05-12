"""Frontend components."""

from ebook_rag_explorer.frontend.components.search import render_search_tab
from ebook_rag_explorer.frontend.components.books import render_books_tab
from ebook_rag_explorer.frontend.components.collections import render_collections_tab
from ebook_rag_explorer.frontend.components.upload import render_upload_tab

__all__ = [
    "render_search_tab",
    "render_books_tab",
    "render_collections_tab",
    "render_upload_tab",
]

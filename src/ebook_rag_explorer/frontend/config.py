"""Frontend configuration."""

from dataclasses import dataclass


@dataclass
class FrontendConfig:
    """Configuration for the Streamlit frontend."""

    default_api_url: str = "http://localhost:8000"
    app_title: str = "📖 Ebook RAG Explorer"
    app_icon: str = "📚"

    retrieval_top_k: int = 20
    rerank_top_n: int = 5

    supported_formats: list[str] = None

    def __post_init__(self):
        self.supported_formats = ["pdf", "epub"]


config = FrontendConfig()

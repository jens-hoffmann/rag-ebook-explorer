"""CSS styles for dark and light themes."""

DARK_THEME = """
<style>
    /* Main background */
    .stApp {
        background-color: #0e1117;
        color: #c9d1d9;
    }

    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #2d3748;
    }

    /* Cards / Containers */
    [data-testid="stExpander"] {
        background-color: #1e2530;
        border: 1px solid #2d3748;
        border-radius: 8px;
    }

    /* Buttons - Primary */
    .stButton > button[kind="primary"] {
        background-color: #4f8cff;
        color: white;
        border: none;
    }

    /* Success / Info boxes */
    .stSuccess, .stInfo, .stWarning, .stError {
        border-radius: 8px;
    }

    /* Tabs */
    [data-testid="stTab"] {
        background-color: transparent;
        color: #c9d1d9;
    }

    [data-testid="stTab"][aria-selected="true"] {
        background-color: #1e2530;
        color: #4f8cff;
        border-bottom: 2px solid #4f8cff;
    }

    /* DataFrame */
    [data-testid="stDataFrame"] {
        background-color: #1e2530;
    }

    /* Table headers */
    thead th {
        background-color: #161b22 !important;
        color: #c9d1d9 !important;
    }

    /* Table rows */
    tbody tr {
        background-color: #1e2530;
        color: #c9d1d9;
    }

    tbody tr:hover {
        background-color: #2d3748;
    }

    /* File drop zone */
    [data-testid="stFileUploaderDropzone"] {
        background-color: #1e2530;
        border: 2px dashed #4f8cff;
        border-radius: 8px;
    }

    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #4f8cff;
    }

    /* Divider */
    hr {
        border-color: #2d3748;
    }

    /* Source cards */
    .source-card {
        background-color: #1e2530;
        border: 1px solid #2d3748;
        border-radius: 8px;
        padding: 16px;
        margin: 8px 0;
    }

    .source-score {
        color: #4f8cff;
        font-weight: bold;
    }

    .source-content {
        color: #c9d1d9;
        margin: 8px 0;
        padding: 8px;
        background-color: #0e1117;
        border-radius: 4px;
    }

    .source-metadata {
        color: #8b949e;
        font-size: 0.85em;
    }

    /* Collection cards */
    .collection-card {
        background-color: #1e2530;
        border: 1px solid #2d3748;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }

    /* Search answer */
    .search-answer {
        background-color: #1e2530;
        border-left: 4px solid #4f8cff;
        padding: 16px;
        margin: 16px 0;
        border-radius: 0 8px 8px 0;
    }

    /* Upload zone highlight */
    .upload-success {
        border-color: #34d399 !important;
    }
</style>
"""

LIGHT_THEME = """
<style>
    /* Main background */
    .stApp {
        background-color: #ffffff;
        color: #1f1f1f;
    }

    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #1f1f1f !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #f0f2f6;
        border-right: 1px solid #d1d5db;
    }

    /* Cards / Containers */
    [data-testid="stExpander"] {
        background-color: #f0f2f6;
        border: 1px solid #d1d5db;
        border-radius: 8px;
    }

    /* Success / Info boxes */
    .stSuccess, .stInfo, .stWarning, .stError {
        border-radius: 8px;
    }

    /* Tabs */
    [data-testid="stTab"] {
        background-color: transparent;
        color: #1f1f1f;
    }

    [data-testid="stTab"][aria-selected="true"] {
        background-color: #f0f2f6;
        color: #1a73e8;
        border-bottom: 2px solid #1a73e8;
    }

    /* DataFrame */
    [data-testid="stDataFrame"] {
        background-color: #ffffff;
    }

    /* Table headers */
    thead th {
        background-color: #f0f2f6 !important;
        color: #1f1f1f !important;
    }

    /* Table rows */
    tbody tr {
        background-color: #ffffff;
        color: #1f1f1f;
    }

    tbody tr:hover {
        background-color: #f0f2f6;
    }

    /* File drop zone */
    [data-testid="stFileUploaderDropzone"] {
        background-color: #f0f2f6;
        border: 2px dashed #1a73e8;
        border-radius: 8px;
    }

    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #1a73e8;
    }

    /* Divider */
    hr {
        border-color: #d1d5db;
    }

    /* Source cards */
    .source-card {
        background-color: #f0f2f6;
        border: 1px solid #d1d5db;
        border-radius: 8px;
        padding: 16px;
        margin: 8px 0;
    }

    .source-score {
        color: #1a73e8;
        font-weight: bold;
    }

    .source-content {
        color: #1f1f1f;
        margin: 8px 0;
        padding: 8px;
        background-color: #ffffff;
        border-radius: 4px;
    }

    .source-metadata {
        color: #5f6368;
        font-size: 0.85em;
    }

    /* Collection cards */
    .collection-card {
        background-color: #f0f2f6;
        border: 1px solid #d1d5db;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }

    /* Search answer */
    .search-answer {
        background-color: #f0f2f6;
        border-left: 4px solid #1a73e8;
        padding: 16px;
        margin: 16px 0;
        border-radius: 0 8px 8px 0;
    }

    /* Upload zone highlight */
    .upload-success {
        border-color: #188038 !important;
    }
</style>
"""


def get_theme_css(is_dark: bool) -> str:
    """Get the CSS for the specified theme.

    Args:
        is_dark: True for dark theme, False for light theme.

    Returns:
        CSS string to inject into the Streamlit app.
    """
    return DARK_THEME if is_dark else LIGHT_THEME

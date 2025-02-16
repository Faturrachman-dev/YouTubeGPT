from os import getenv
from modules.helpers import is_api_key_set, is_api_key_valid, get_available_models

import streamlit as st

from modules.helpers import get_default_config_value

GENERAL_ERROR_MESSAGE = "An unexpected error occurred. If you are a developer and run the app locally, you can view the logs to see details about the error."


def display_api_key_warning():
    """Checks whether an API key is provided and displays warning if not."""
    if not is_api_key_set():
        st.sidebar.warning(
            "Please provide your NVIDIA API key in the settings (sidebar) or as an environment variable (`NVIDIA_API_KEY`)."
        )
    elif "nvidia_api_key" in st.session_state and not is_api_key_valid(st.session_state.nvidia_api_key):
        st.sidebar.warning("NVIDIA API key seems to be invalid.")


def set_api_key_in_session_state():
    """If the env-var NVIDIA_API_KEY is set, its value is assigned to nvidia_api_key property in streamlit's session state.
    Otherwise an input field for the API key is displayed.
    """
    NVIDIA_API_KEY = getenv("NVIDIA_API_KEY")

    if not NVIDIA_API_KEY:
        st.sidebar.text_input(
            "Enter your NVIDIA API key",
            key="nvidia_api_key",
            type="password",
        )
    else:
        st.session_state.nvidia_api_key = NVIDIA_API_KEY


def is_temperature_and_top_p_altered() -> bool:
    """Check if temperature or top_p have been changed from defaults."""
    if st.session_state.temperature != get_default_config_value(
        "temperature"
    ) and st.session_state.top_p != get_default_config_value("top_p"):
        return True
    return False


def display_model_settings_sidebar():
    """Function for displaying the sidebar and adjusting settings.

    Every widget with a key is added to streamlit's session state and can be accessed in the application.
    For example here, the selectbox for model has the key 'model'.
    Thus the selected model can be accessed via st.session_state.model.
    """
    if "model" not in st.session_state:
        st.session_state.model = get_default_config_value("default_model.nvidia")

    # Always use NVIDIA
    st.session_state.api_provider = "nvidia"

    with st.sidebar:
        st.header("Model settings")
        model = st.selectbox(
            label="Select NVIDIA model",
            options=get_available_models(model_type="nvidia"),
            key="model",
            help=get_default_config_value("help_texts.model"),
        )

        st.slider(
            label="Adjust temperature",
            min_value=0.0,
            max_value=2.0,
            step=0.1,
            key="temperature",
            value=get_default_config_value("temperature"),
            help=get_default_config_value("help_texts.temperature"),
        )

        st.slider(
            label="Adjust Top P",
            min_value=0.0,
            max_value=1.0,
            step=0.1,
            key="top_p",
            value=get_default_config_value("top_p"),
            help=get_default_config_value("help_texts.top_p"),
        )

        if is_temperature_and_top_p_altered():
            st.warning(
                "It's generally recommended to alter temperature or top_p but not both."
            )


def display_link_to_repo(view: str = "main"):
    """Displays a link to the GitHub repository."""
    st.sidebar.write(
        f"[View the source code]({get_default_config_value(f'github_repo_links.{view}')})"
    )


def display_video_url_input(
    label: str = "Enter URL of the YouTube video:", disabled=False
):
    """Displays an input field for the URL of the YouTube video."""
    return st.text_input(
        label=label,
        key="url_input",
        disabled=disabled,
        help=get_default_config_value("help_texts.youtube_url"),
    )


def display_yt_video_container(video_title: str, channel: str, url: str):
    """Displays a YouTube video container with title and channel information."""
    st.subheader(
        f"'{video_title}' from {channel}.",
        divider="gray",
    )
    st.video(url)


def display_nav_menu():
    """Displays links to pages in sidebar."""
    st.sidebar.page_link(page="main.py", label="Home")
    st.sidebar.page_link(page="pages/summary.py", label="Summary")
    st.sidebar.page_link(page="pages/chat.py", label="Chat")
    st.sidebar.page_link(page="pages/library.py", label="Library")


def get_available_models(model_type: str, api_key: str = "") -> list:
    """Gets available NVIDIA models."""
    if model_type == "nvidia":
        return get_default_config_value("available_models.nvidia")
    return []

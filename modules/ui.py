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
    """Sets the NVIDIA API key in streamlit's session state."""
    if not is_api_key_set():
        st.sidebar.text_input(
            label="NVIDIA API Key",
            key="nvidia_api_key",
            type="password",
            placeholder="Enter your NVIDIA API key (nvapi-...)",
        )


def display_video_url_input(prefilled_url=None):
    """Displays the video URL input field.

    Args:
        prefilled_url: Optional URL to prefill the input field.
    Returns:
        The YouTube video URL.  // IMPORTANT:  It now ALWAYS returns the URL.
    """
    url = st.text_input(
        label="Enter YouTube Video URL:",
        placeholder="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        value=prefilled_url,
        key="video_url",
    )
    return url  # Always return the URL


def display_yt_video_container(video_title: str, channel: str, url: str):
    """Displays the YouTube video container."""
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
        available_models_dict = get_default_config_value("available_models")
        return available_models_dict.get("nvidia", []) if available_models_dict else []
    return []

def display_link_to_repo(page: str):
    if page == "main":
        st.sidebar.link_button(
            "View on GitHub",
            url=get_default_config_value("github_repo_links").get("main") if get_default_config_value("github_repo_links") else None,
        )
    elif page == "summary":
        st.sidebar.link_button(
            "View on GitHub",
            url=get_default_config_value("github_repo_links").get("summary") if get_default_config_value("github_repo_links") else None,
        )
    elif page == "chat":
        st.sidebar.link_button(
            "View on GitHub",
            url=get_default_config_value("github_repo_links").get("chat") if get_default_config_value("github_repo_links") else None,
        )

def display_chat_input():
    """
    Displays the chat input box.
    Returns:
        str: User's input.
    """
    return st.chat_input("Ask a question about this video")

def display_chat_messages(messages):
    """
    Displays the chat messages.
    Args:
        messages (list): List of chat messages.
    """
    for message in messages:
        if isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
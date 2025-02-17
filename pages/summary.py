import logging
from datetime import datetime as dt

import streamlit as st
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from modules.persistance import SQL_DB, LibraryEntry, Video

from modules.helpers import (
    get_default_config_value,
    is_api_key_set,
    is_api_key_valid,
    save_to_file,
)
from modules.persistance import LibraryEntry, Video, save_library_entry
from modules.summary import (
    TranscriptTooLongForModelException,
    get_transcript_summary,
)
from modules.ui import (
    GENERAL_ERROR_MESSAGE,
    display_api_key_warning,
    display_link_to_repo,
    display_model_settings_sidebar,
    display_nav_menu,
    display_video_url_input,
    display_yt_video_container,
    set_api_key_in_session_state,
)
from modules.youtube import (
    InvalidUrlException,
    NoTranscriptReceivedException,
    extract_youtube_video_id,
    fetch_youtube_transcript,
    get_video_metadata,
)

# --- SQLite stuff ---
SQL_DB.connect(reuse_if_open=True)
# create tables if they don't already exist
SQL_DB.create_tables([Video, LibraryEntry], safe=True)
# --- end ---

st.set_page_config("Summaries", layout="wide", initial_sidebar_state="auto")
display_api_key_warning()

# --- part of the sidebar which doesn't require an api key ---
display_nav_menu()
set_api_key_in_session_state()
display_link_to_repo("summary")
# --- end ---


@st.dialog(title="Transcript too long", width="large")
def display_dialog(message: str):
    st.warning(message)


def save_summary_to_lib():
    """Wrapper func for saving summaries to the library."""
    try:
        saved_video = Video.create(
            yt_video_id=extract_youtube_video_id(url_input),
            link=url_input,
            title=vid_metadata["name"],
            channel=vid_metadata["channel"],
            saved_on=dt.now(),
        )
        save_library_entry(
            entry_type="S",
            question_text=None,
            response_text=st.session_state.summary,
            video=saved_video,
        )
    except Exception as e:
        st.error("Saving failed! If you are a developer, see logs for details!")
        logging.error("Error when saving library entry: %s", e)
    else:
        st.success("Saved summary to library successfully!")


if is_api_key_set() and is_api_key_valid(st.session_state.nvidia_api_key):
    display_model_settings_sidebar()
    
    # define the columns
    col1, col2 = st.columns([0.4, 0.6], gap="large")

    with col1:
        url_input = display_video_url_input()
        custom_prompt = st.text_area(
            "Enter a custom prompt if you want:",
            key="custom_prompt_input",
            help=get_default_config_value("help_texts.custom_prompt"),
        )
        summarize_button = st.button("Summarize", key="summarize_button")
        if url_input != "":
            try:
                vid_metadata = get_video_metadata(url_input)
                display_yt_video_container(
                    video_title=vid_metadata["name"],
                    channel=vid_metadata["channel"],
                    url=url_input,
                )
            except InvalidUrlException as e:
                st.error(e.message)
                e.log_error()
            except Exception as e:
                logging.error("An unexpected error occurred %s", str(e))
                st.error(GENERAL_ERROR_MESSAGE)

    with col2:
        if summarize_button:
            try:
                transcript = fetch_youtube_transcript(url_input)

                # Initialize NVIDIA model
                llm = ChatNVIDIA(
                    api_key=st.session_state.nvidia_api_key,
                    model=st.session_state.model,
                    temperature=st.session_state.temperature,
                    top_p=st.session_state.top_p,
                )

                with st.spinner("Summarizing video :gear: Hang on..."):
                    if custom_prompt:
                        resp = get_transcript_summary(
                            transcript_text=transcript,
                            llm=llm,
                            custom_prompt=custom_prompt,
                        )
                    else:
                        resp = get_transcript_summary(
                            transcript_text=transcript, 
                            llm=llm
                        )
                    st.session_state.summary = resp
                st.markdown(st.session_state.summary)

                # button for saving summary to library
                st.button(label="Save summary to library", on_click=save_summary_to_lib)

            except TranscriptTooLongForModelException as e:
                st.error(str(e))
            except Exception as e:
                logging.error(
                    "An unexpected error occurred: %s", str(e), exc_info=True
                )
                st.error(GENERAL_ERROR_MESSAGE)

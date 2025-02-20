import os
print(f"NVIDIA_API_KEY from environment: {os.getenv('NVIDIA_API_KEY')}")

import logging
from datetime import datetime as dt

import streamlit as st
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from modules.persistance import SQL_DB, LibraryEntry, Video
from chromadb import HttpClient
from chromadb.config import Settings
from langchain_openai import OpenAIEmbeddings
from modules.rag import split_text_recursively, embed_excerpts

# ChromaDB settings for connecting to the Docker container
chroma_settings = Settings(
    chroma_api_impl="chromadb.api.fastapi.FastAPI",
    anonymized_telemetry=False
)

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
    summarize_with_refined_prompt,
)
from modules.ui import (
    GENERAL_ERROR_MESSAGE,
    display_api_key_warning,
    display_link_to_repo,
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

# --- API Key Handling (REVISED) ---
# Check environment variable FIRST
if os.getenv("NVIDIA_API_KEY"):
    # If it's in the environment, put it in the session state
    st.session_state.nvidia_api_key = os.getenv("NVIDIA_API_KEY")

display_api_key_warning()  # Now safe to call

# --- part of the sidebar which doesn't require an api key ---
display_nav_menu()
set_api_key_in_session_state()  # This will now handle the UI input
display_link_to_repo("summary")

# --- Check for pre-filled URL from the Chat page ---
if "video_url" in st.session_state:
    url = st.session_state.video_url
    # Clear the session state variable so it doesn't persist
    del st.session_state.video_url
    # auto-submit
    auto_submit = True

else:
    url = None
    auto_submit = False
# --- end check ---

# --- Model selection and settings (moved from sidebar) ---
if is_api_key_set() and is_api_key_valid():
    available_models_dict = get_default_config_value("available_models")
    available_models = (
        available_models_dict.get("nvidia", []) if available_models_dict else []
    )

    if "model" in st.session_state:
        try:
            default_index = available_models.index(st.session_state.model)
        except ValueError:
            default_index = 0
    else:
        default_index = 0

    st.selectbox(
        "Choose a model:",
        options=available_models,
        index=default_index,
        disabled=False,
        key="model",
        help=get_default_config_value("help_texts").get("model")
        if get_default_config_value("help_texts")
        else None,
    )

    default_temp = float(get_default_config_value("temperature"))
    if "temperature" in st.session_state:
        try:
            # Attempt to use the existing temperature, but ensure it's within bounds
            default_temp = max(0.0, min(2.0, float(st.session_state.temperature)))
        except (ValueError, TypeError):
            default_temp = float(get_default_config_value("temperature"))

    st.slider(
        "Temperature:",
        min_value=0.0,
        max_value=2.0,
        value=default_temp,  # Use calculated default
        step=0.1,
        key="temperature",
        help=get_default_config_value("help_texts").get("temperature")
        if get_default_config_value("help_texts")
        else None,
    )

    default_top_p = float(get_default_config_value("top_p"))
    if "top_p" in st.session_state:
        try:
            default_top_p = max(0.0, min(1.0, float(st.session_state.top_p)))
        except (ValueError, TypeError):
            default_top_p = float(get_default_config_value("top_p"))

    st.slider(
        "Top P:",
        min_value=0.0,
        max_value=1.0,
        value=default_top_p,  # Use calculated default
        step=0.01,
        key="top_p",
        help=get_default_config_value("help_texts").get("top_p")
        if get_default_config_value("help_texts")
        else None,
    )

else:
    # If API key is not valid, disable and show a warning
    st.selectbox("Choose a model:", options=["No models available"], disabled=True, key="model")
    st.slider("Temperature:", min_value=0.0, max_value=2.0, value=1.0, disabled=True, key="temperature")
    st.slider("Top P:", min_value=0.0, max_value=1.0, value=1.0, disabled=True, key="top_p")
    st.sidebar.warning("Please enter a valid NVIDIA API key to use the summarization features.")

# --- end model selection ---


@st.dialog(title="Transcript too long", width="large")
def display_dialog(message: str):
    st.warning(message)


def save_summary_to_lib():
    """Wrapper func for saving summaries to the library."""
    try:
        saved_video = Video.create(
            yt_video_id=extract_youtube_video_id(url),
            link=url,
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


# define the columns
col1, col2 = st.columns([0.4, 0.6], gap="large")

with col1:
    # --- FORM START ---
    with st.form(key="video_input_form"):
        url = display_video_url_input(url)  # Get the URL
        custom_prompt = st.text_area(
            "Enter a custom prompt if you want:",
            key="custom_prompt_input",
            help=get_default_config_value("help_texts").get("custom_prompt")
            if get_default_config_value("help_texts")
            else None,
        )
        submit_button = st.form_submit_button("Summarize")

        # Auto-submit if we have a URL from the chat page
        if auto_submit and url:
            submit_button = True  # Simulate button press
    # --- FORM END ---

    if url:
        try:
            vid_metadata = get_video_metadata(url)
            display_yt_video_container(
                video_title=vid_metadata["name"],
                channel=vid_metadata["channel"],
                url=url,
            )
        except InvalidUrlException as e:
            st.error(e.message)
            e.log_error()
        except Exception as e:
            logging.error("An unexpected error occurred %s", str(e))


with col2:
    # if summarize_button: # Removed this
    if submit_button: # Changed to check the form submission
        try:
            transcript = fetch_youtube_transcript(url)
            # --- Store video_id in SHARED session state ---
            video_id = extract_youtube_video_id(url)
            st.session_state.video_id = video_id

            # Initialize ChromaDB and create collection for the video
            try:
                chroma_client = HttpClient(host="localhost", port=8000, settings=chroma_settings)
                
                # Try to get existing collection or create new one
                try:
                    collection = chroma_client.get_collection(name=video_id)
                    logging.info(f"Using existing collection for video {video_id}")
                except Exception as collection_error:
                    logging.info(f"Creating new collection for video {video_id}")
                    collection = chroma_client.create_collection(name=video_id)
                
                # Split transcript into chunks and embed them
                logging.info("Splitting transcript into chunks...")
                transcript_chunks = split_text_recursively(
                    transcript_text=transcript,
                    chunk_size=512,
                    chunk_overlap=32,
                    len_func="characters"
                )
                logging.info(f"Created {len(transcript_chunks)} chunks")
                
                # Initialize NVIDIA model for embeddings
                nvidia_model = ChatNVIDIA(
                    model="meta/llama-3.1-405b-instruct",
                    api_key=os.getenv("NVIDIA_API_KEY"),
                    temperature=0.2,
                    top_p=0.7,
                    max_tokens=1024,
                )
                logging.info("Embedding chunks and storing in ChromaDB...")
                embed_excerpts(transcript_chunks, collection, nvidia_model, chunk_size=512)
                st.success("Successfully processed video transcript for chat functionality!")
                
            except Exception as e:
                logging.error("Error processing transcript for ChromaDB: %s", str(e), exc_info=True)
                st.error(f"Error processing video: {str(e)}")
                st.warning("There was an issue processing the video for chat functionality. Chat features may be limited.")

            # Initialize NVIDIA model
            llm = ChatNVIDIA(
                api_key=st.session_state.nvidia_api_key,
                model=st.session_state.model,
                temperature=st.session_state.temperature,
                top_p=st.session_state.top_p,
            )

            with st.spinner("Summarizing video :gear: Hang on..."):
                if custom_prompt:
                    resp = summarize_with_refined_prompt(
                        transcript_text=transcript,
                        llm=llm,
                        custom_prompt=custom_prompt,
                    )
                else:
                    resp = summarize_with_refined_prompt(
                        transcript_text=transcript,
                        llm=llm,
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
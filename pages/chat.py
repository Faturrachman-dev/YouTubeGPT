import logging
from datetime import datetime as dt

import chromadb
import randomname
import streamlit as st
from chromadb import Collection
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from modules.helpers import (
    get_available_models,
    get_default_config_value,
    is_api_key_set,
    is_api_key_valid,
    is_environment_prod,
    num_tokens_from_string,
    read_file,
)
from modules.persistance import (
    SQL_DB,
    LibraryEntry,
    Transcript,
    Video,
    delete_video,
    save_library_entry,
)
from modules.rag import (
    CHUNK_SIZE_TO_K_MAPPING,
    embed_excerpts,
    find_relevant_documents,
    generate_response,
    split_text_recursively,
)
from modules.transcription import download_mp3, generate_transcript
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

CHUNK_SIZE_FOR_UNPROCESSED_TRANSCRIPT = 512

st.set_page_config("Chat", layout="wide", initial_sidebar_state="auto")
display_api_key_warning()

# --- part of the sidebar which doesn't require an api key ---
display_nav_menu()
set_api_key_in_session_state()
display_link_to_repo("chat")
# --- end ---

# --- SQLite stuff ---
SQL_DB.connect(reuse_if_open=True)
# create tables if they don't already exist
SQL_DB.create_tables([Video, Transcript, LibraryEntry], safe=True)
# --- end ---

# --- Chroma ---
chroma_connection_established = False
chroma_settings = Settings(allow_reset=True, anonymized_telemetry=False)
collection: None | Collection = None
chroma_client = chromadb.Client(chroma_settings)
chroma_db: Chroma | None = None

# Check if a video has been processed
if "video_id" in st.session_state and st.session_state.video_id:
    try:
        collection = chroma_client.get_collection(name=st.session_state.video_id)
        chroma_db = Chroma(
            client=chroma_client,
            collection_name=st.session_state.video_id,
            embedding_function=OpenAIEmbeddings(
                api_key=st.session_state.openai_api_key,
                model=get_default_config_value("default_model.embeddings"),
            ),  # Use OpenAI embeddings
        )
        chroma_connection_established = True
    except Exception as e:
        logging.error(f"Error connecting to ChromaDB: {e}")
        st.error("Could not connect to Chroma. Please try again or process another video.")
        chroma_connection_established = False  # Ensure this is set to False on error
else:
    st.info(
        "No video selected. Please process a video on the 'Summary' page first."
    )
    chroma_connection_established = False


# --- Chat UI ---
if is_api_key_set() and is_api_key_valid(st.session_state.nvidia_api_key):
    display_model_settings_sidebar()

    prompt = st.chat_input("Ask a question about the video")
    if prompt:
        if chroma_connection_established:  # Only proceed if Chroma connection is good
            st.markdown(prompt)

            if collection:  # Check if the collection exists
                with st.spinner("Generating answer..."):
                    try:
                        relevant_docs = find_relevant_documents(
                            query=prompt,
                            db=chroma_db,
                            k=CHUNK_SIZE_TO_K_MAPPING.get(
                                collection.metadata["chunk_size"]
                            ),
                        )

                        # Use ChatNVIDIA
                        chat_llm = ChatNVIDIA(
                            api_key=st.session_state.nvidia_api_key,
                            model=st.session_state.model,
                            temperature=st.session_state.temperature,
                            top_p=st.session_state.top_p,
                        )

                        response = generate_response(
                            question=prompt,
                            llm=chat_llm,
                            relevant_docs=relevant_docs,
                        )
                        st.session_state.response = response

                        # Display response and save button
                        st.markdown(response)
                        st.button(
                            label="Save response to library",
                            on_click=save_response_to_lib,
                        )

                    except Exception as e:
                        logging.error(
                            "An unexpected error occurred: %s", str(e), exc_info=True
                        )
                        st.error(GENERAL_ERROR_MESSAGE)

            else: # collection is None
                st.info(
                    "No collection found for this video. Please process the video again."
                )

        else: # chroma_connection_established is False
            st.info(
                "No videos processed yet. Go to the 'Summary' page and enter a YouTube video URL."
            )

def save_response_to_lib():
    """Saves the generated response (question and answer) to the database."""
    try:
        video_id = extract_youtube_video_id(st.session_state.url_input)
        video_metadata = get_video_metadata(st.session_state.url_input)
        # save video to db if not already saved
        video, _ = Video.get_or_create(
            yt_video_id=video_id,
            defaults={
                "link": st.session_state.url_input,
                "title": video_metadata["name"],
                "channel": video_metadata["channel"],
                "saved_on": dt.now(),
            },
        )
        # save response to db
        library_entry = LibraryEntry.create(
            video=video,
            summary=st.session_state.response,
            question=st.session_state.user_prompt,
            type="chat",
            added_on=dt.now(),
        )
        st.success(
            f"Response to '{library_entry.question}' for video '{library_entry.video.title}' saved to library! :floppy_disk:"
        )
    except Exception as e:
        logging.error("An error occurred while saving to library: %s", e)
        st.error("Saving to library failed :cry: Please try again.")

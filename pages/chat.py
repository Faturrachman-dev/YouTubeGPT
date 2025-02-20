import logging
from datetime import datetime as dt

import chromadb
import randomname
import streamlit as st
from chromadb import HttpClient
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import AIMessage, HumanMessage

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
    NoDocumentsFoundException,
)
from modules.transcription import download_mp3, generate_transcript
from modules.ui import (
    GENERAL_ERROR_MESSAGE,
    display_api_key_warning,
    display_link_to_repo,
    display_nav_menu,
    display_video_url_input,
    display_yt_video_container,
    set_api_key_in_session_state,
    display_chat_input,
    display_chat_messages,
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
# Initialize chroma_connection_established OUTSIDE the try block
chroma_connection_established = False
# ChromaDB settings for connecting to the Docker container
chroma_settings = Settings(
    chroma_api_impl="chromadb.api.fastapi.FastAPI",
    anonymized_telemetry=False
)
collection: None | Collection = None

# --- Check for video_id BEFORE connecting to Chroma ---
if "video_id" in st.session_state and st.session_state.video_id:
    try:
        chroma_client = HttpClient(host="localhost", port=8000, settings=chroma_settings)
        collection = chroma_client.get_collection(st.session_state.video_id)
        chroma_connection_established = True
    except Exception as e:
        logging.error("An error occurred while connecting to Chroma: %s", str(e))
        chroma_connection_established = False  # Ensure this is set in case of any error
        st.error("Could not connect to Chroma.  Please ensure the video has been processed on the Summary page.")
else:
    st.info("Please summarize a video on the Summary page before using the Chat feature.")
    #  IMPORTANT:  Return early if no video_id.  Don't try to load models, etc.
    st.stop()

# --- UI Elements ---
# Model selection (only if API key is valid)
if is_api_key_set() and is_api_key_valid():
    available_models_dict = get_default_config_value("available_models")
    available_models = available_models_dict.get("nvidia", []) if available_models_dict else []

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
        help=get_default_config_value("help_texts").get("model") if get_default_config_value("help_texts") else None,
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
        value=default_temp,
        step=0.1,
        key="temperature",
        help=get_default_config_value("help_texts").get("temperature") if get_default_config_value("help_texts") else None,
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
        value=default_top_p,
        step=0.01,
        key="top_p",
        help=get_default_config_value("help_texts").get("top_p") if get_default_config_value("help_texts") else None,
    )
else:
    st.selectbox("Choose a model:", options=["No models available"], disabled=True, key="model")
    st.slider("Temperature:", min_value=0.0, max_value=2.0, value=1.0, disabled=True, key="temperature")
    st.slider("Top P:", min_value=0.0, max_value=1.0, value=1.0, disabled=True, key="top_p")
    if not is_api_key_set():
        st.sidebar.warning("Please enter a valid NVIDIA API key to use the chat features.")
    #  Don't show the "summarize a video" message if the API key is the problem.


# --- Main UI ---
# Only show chat input if Chroma connection is established
if chroma_connection_established:
    # Display excerpts from the video
    with st.status("Loading...", expanded=False) as status:
        if collection:
            st.write("Documents loaded successfully!")
            # Fetch all documents from the collection
            documents_from_chroma = collection.get()
            # Extract the 'documents' list which contains the text
            if documents_from_chroma:
                docs = documents_from_chroma.get('documents')
                if docs:
                    with st.expander("Show excerpts from video"):
                        st.write(docs)
                else:
                    st.warning("No documents found in the collection.")
            else:
                st.warning("Failed to retrieve documents from ChromaDB.")
        status.update(label="Load complete!", state="complete", expanded=False)

    prompt = st.chat_input("Ask a question about the video")
    if prompt:
        st.session_state.user_prompt = prompt  # Store the prompt
        if collection:  # Check if the collection exists
            try:
                with st.spinner("Generating answer..."):
                    # Find relevant documents
                    relevant_docs = find_relevant_documents(
                        query=prompt,
                        collection=collection,
                        embedding_model=get_default_config_value("default_model").get("embeddings") if get_default_config_value("default_model") else None,
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

# No `else` block needed here.  We've already handled the case where video_id is missing.

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

def chat_page():
    display_link_to_repo("chat")
    st.title("Chat with Video")

    if not is_api_key_set():
        st.error(
            "API Key not set. Please set your NVIDIA API key in the sidebar or through the environment variable `NVIDIA_API_KEY`."
        )
        return

    # Get video URL from user input
    url = st.text_input("Enter YouTube Video URL:", key="chat_url")

    if url:
        try:
            video_id = extract_youtube_video_id(url)
            if not video_id:
                raise InvalidUrlException(url)
            metadata = get_video_metadata(url)
            st.video(url)
            st.subheader(metadata["name"])
            st.caption(f"Channel: {metadata['channel']}")

            # --- CHROMA COLLECTION CHECK AND REDIRECT ---
            client = chromadb.HttpClient(host="localhost", port=8000)
            collection_name = video_id

            if collection_name not in [collection.name for collection in client.list_collections()]:
                st.warning("This video hasn't been processed yet. Redirecting to the Summary page...")
                # Set the URL in the session state for the Summary page
                st.session_state.video_url = url
                # Redirect to the Summary page
                st.switch_page("pages/summary.py") # Corrected line
                return
            # --- END CHROMA COLLECTION CHECK AND REDIRECT ---

            if "messages" not in st.session_state:
                st.session_state.messages = []

            display_chat_messages(st.session_state.messages)
            user_input = display_chat_input()

            if user_input:
                st.session_state.messages.append(HumanMessage(content=user_input))
                display_chat_messages(st.session_state.messages)

                with st.spinner("Generating response..."):
                    try:
                        response = generate_response(user_input, video_id)
                        st.session_state.messages.append(AIMessage(content=response))
                        display_chat_messages(st.session_state.messages)
                    except NoDocumentsFoundException:
                        st.error("No relevant documents found in the knowledge base.")
                    except Exception as e:
                        st.error(f"An error occurred: {e}")

        except InvalidUrlException:
            st.error("Invalid YouTube URL. Please enter a valid URL.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

chat_page()

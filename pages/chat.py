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

if is_api_key_set():
    if not is_api_key_valid(st.session_state.nvidia_api_key):
        st.error("NVIDIA API key is invalid.")
    else:
        display_model_settings_sidebar()

        # select-box for choosing the video
        videos = Video.select().order_by(Video.saved_on.desc())
        if videos.exists():
            selected_video = st.selectbox(
                label="Select a video",
                options=videos,
                format_func=lambda x: f"{x.title} ({x.channel})",
                placeholder="Select a video from the list",
                help=get_default_config_value("help_texts.selected_video"),
            )

            # delete button
            col1, col2 = st.columns(2)
            with col1:
                if st.button(
                    label="Delete video",
                    type="primary",
                    use_container_width=True,
                    disabled=not selected_video,
                ):
                    with st.spinner("Deleting video..."):
                        try:
                            delete_video(selected_video.yt_video_id)
                        except Exception as e:
                            logging.error("Error while deleting video: %s", e)
                            st.error("An error occurred while deleting the video.")
                        else:
                            st.cache_data.clear()
                            st.rerun()

            # get the Chroma collection for the selected video
            if selected_video:
                try:
                    chroma_client = chromadb.Client(settings=chroma_settings)
                    # try to get an existing collection
                    collection = chroma_client.get_collection(
                        name=selected_video.yt_video_id
                    )
                    chroma_connection_established = True
                except Exception as e:
                    logging.error(
                        "An error occurred while connecting to Chroma: %s", e
                    )
                    st.error(
                        "Could not connect to Chroma. Please try again or process another video."
                    )

        # --- user input ---
        url_input = display_video_url_input(disabled=chroma_connection_established)

        col1, col2 = st.columns([0.3, 0.7])
        with col1:
            if url_input and not chroma_connection_established:
                try:
                    # 1. get video metadata
                    vid_metadata = get_video_metadata(url_input)
                    display_yt_video_container(
                        video_title=vid_metadata["name"],
                        channel=vid_metadata["channel"],
                        url=url_input,
                    )

                    # 2. get transcript
                    original_transcript = fetch_youtube_transcript(url_input)

                    # 3. save video to db
                    saved_video = Video.create(
                        yt_video_id=extract_youtube_video_id(url_input),
                        link=url_input,
                        title=vid_metadata["name"],
                        channel=vid_metadata["channel"],
                        saved_on=dt.now(),
                    )

                    # 4. save transcript to db
                    saved_transcript = Transcript.create(
                        video=saved_video,
                        text=original_transcript,
                        processed_token_num=0,
                    )

                    # 5. split the transcript into chunks
                    transcript_excerpts = split_text_recursively(
                        transcript_text=original_transcript,
                        chunk_size=CHUNK_SIZE_FOR_UNPROCESSED_TRANSCRIPT,
                        chunk_overlap=32,
                        len_func="tokens",
                    )

                    # 6. embed the chunks and add them to the collection
                    embed_excerpts(
                        transcript_excerpts=transcript_excerpts,
                        collection=collection,
                        openai_embedding_model=OpenAIEmbeddings(
                            api_key=st.session_state.nvidia_api_key
                        ),
                        chunk_size=chunk_size,
                    )

                    # 7. update transcript in db
                    saved_transcript.processed_token_num = sum(
                        num_tokens_from_string(
                            string=e.page_content, model=st.session_state.model
                        )
                        for e in transcript_excerpts
                    )
                    saved_transcript.save()

                except InvalidUrlException as e:
                    st.error(e.message)
                except NoTranscriptReceivedException as e:
                    st.error(e.message)
                    e.log_error()
                except Exception as e:
                    logging.error(
                        "An unexpected error occurred: %s", str(e), exc_info=True
                    )
                    st.error(GENERAL_ERROR_MESSAGE)
                else:
                    st.success(
                        "The video has been processed! Please refresh the page and choose it in the select-box above."
                    )
                    st.rerun()

        with col2:
            if collection and collection.count() > 0:
                # Initialize vector store with OpenAI embeddings
                chroma_db = Chroma(
                    client=chroma_client,
                    collection_name=collection.name,
                    embedding_function=OpenAIEmbeddings(
                        api_key=st.session_state.nvidia_api_key
                    ),
                )

                with st.expander(label=":information_source: Tips and important notes"):
                    st.markdown(read_file(".assets/rag_quidelines.md"))

                prompt = st.chat_input(
                    placeholder="Ask a question or provide a topic covered in the video",
                )

                if prompt:
                    st.session_state.user_prompt = prompt
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

            else:
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

import logging
import math
import re
from typing import List

from langchain.chains.summarize import load_summarize_chain
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
import tiktoken  # Import tiktoken here

from modules.helpers import (
    count_tokens,
    get_default_config_value,
)

# Constants for chunking (removed fixed sizes)
# INITIAL_CHUNK_SIZE = 16000  # Removed
# OVERLAP_SIZE = 1000  # Removed

# Define NVIDIA_CONTEXT_WINDOWS here
NVIDIA_CONTEXT_WINDOWS = {
    "meta/llama-3.1-405b-instruct": 8192,
    "mixtral_8x7b": 32768,
    "nemotron_340b" : 4096
}

class TranscriptTooLongForModelException(Exception):
    """Exception raised when the transcript is too long for the model."""

    def __init__(self, message="Transcript is too long for the selected model."):
        self.message = message
        super().__init__(self.message)

    def log_error(self):
        """Logs the error message."""
        logging.error(self.message)


def get_transcript_summary(transcript_text: str, llm) -> str:
    """
    Generates a summary of a video transcript using a Refine prompt.

    Args:
        transcript_text (str): The full text of the video transcript.

    Returns:
        str: A concise summary of the video.
    """
    # This function is no longer used, but I'm keeping it here
    # in case you want to revert to a simpler summarization approach.
    # The actual summarization logic is now entirely within
    # summarize_with_refined_prompt.
    raise NotImplementedError("get_transcript_summary is not currently used.")


def split_text_into_chunks(text: str, model_name: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Splits the given text into chunks based on token count, with overlap.

    Args:
        text: The text to split.
        model_name: The name of the model (for tokenization).
        chunk_size: The *token* size of each chunk.
        overlap: The *token* overlap between chunks.

    Returns:
        A list of text chunks.
    """
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")  # Default

    tokens = encoding.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
        start += chunk_size - overlap
        # Prevent infinite loop in edge cases
        if start + overlap >= end and end < len(tokens):
            start = end

    return chunks


def summarize_with_refined_prompt(transcript_text: str, llm, custom_prompt: str = None) -> str:
    """
    Summarizes a transcript using a refined prompting strategy with map-reduce.  Handles long transcripts.

    Args:
        transcript_text: The transcript text.
        llm: The language model.
        custom_prompt:  An optional custom prompt.

    Returns:
        The summary.
    """

    # --- Adaptive Chunking ---
    model_name = llm.model  # CORRECTED model name access (again!)
    if model_name not in NVIDIA_CONTEXT_WINDOWS:
        raise ValueError(f"Model name '{model_name}' is not supported.")

    context_window = NVIDIA_CONTEXT_WINDOWS[model_name]
    # Use a smaller fraction to leave room for the prompt and summary
    initial_chunk_size = int(context_window * 0.4)  # 40% for chunks
    overlap_size = int(initial_chunk_size * 0.1)  # 10% overlap

    chunks = split_text_into_chunks(transcript_text, model_name, initial_chunk_size, overlap_size)
    logging.info(f"Initial chunk size: {initial_chunk_size}, overlap: {overlap_size}")
    logging.info(f"Number of chunks: {len(chunks)}")
    total_tokens = count_tokens(transcript_text, model_name) # Count tokens *once*
    logging.info(f"Total tokens in transcript: {total_tokens}")

    if total_tokens > context_window:
        raise TranscriptTooLongForModelException(
            f"Transcript is too long for the selected model ({model_name}).  Total tokens: {total_tokens}, context window: {context_window}."
        )
    # --- End Adaptive Chunking ---

    # --- Refined Prompting ---
    if custom_prompt:
        map_prompt_template = custom_prompt + "\n\n" + """
        "{text}"
        CONCISE SUMMARY:"""
    else:
        map_prompt_template = """Provide a detailed summary of the following text, extracting and emphasizing the key points, main arguments, and any supporting evidence or examples.  Aim for a comprehensive overview that captures the essence of the content, making it suitable for someone who hasn't read the original text. Include any important conclusions or findings.
        "{text}"
        DETAILED SUMMARY:"""

    map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])

    combine_prompt_template = """Combine the following summaries into a single, coherent, and comprehensive summary.  Ensure the final summary is well-organized, flows logically, and accurately reflects the main ideas and supporting details from the individual summaries. Eliminate redundancy and prioritize the most important information.
    "{text}"
    COMPREHENSIVE SUMMARY:"""
    combine_prompt = PromptTemplate(template=combine_prompt_template, input_variables=["text"])

    summary_chain = load_summarize_chain(
        llm,
        chain_type="map_reduce",
        map_prompt=map_prompt,
        combine_prompt=combine_prompt,
        verbose=False  # Set to True for debugging
    )

    try:
        # Convert chunks to Document objects
        docs = [Document(page_content=chunk) for chunk in chunks]
        summary = summary_chain.invoke(docs)  # Use invoke instead of run
        return summary["output_text"]  # Extract the summary text

    except Exception as e:
        raise TranscriptTooLongForModelException(f"Error during summarization: {e}")

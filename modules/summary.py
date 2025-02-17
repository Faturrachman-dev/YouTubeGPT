import logging
import math
import re
from typing import List

from langchain.chains.summarize import load_summarize_chain
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from modules.helpers import (
    count_tokens,
    get_default_config_value,
)

# Constants for chunking
INITIAL_CHUNK_SIZE = 16000  # Try a larger initial chunk size
OVERLAP_SIZE = 1000  # Keep some overlap

# Define NVIDIA_CONTEXT_WINDOWS here
NVIDIA_CONTEXT_WINDOWS = {
    "meta/llama-3.1-405b-instruct": 8192,  # Example value - ADJUST AS NEEDED!
    "mixtral_8x7b": 32768, # Example value - ADJUST AS NEEDED!
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


def get_transcript_summary(transcript_text: str, llm, custom_prompt: str = None):
    """Gets a summary of a transcript using the map-reduce method.
    DEPRECATED: Use summarize_with_refined_prompt instead.
    """
    logging.warning("get_transcript_summary is deprecated. Use summarize_with_refined_prompt instead.")
    prompt_template = """Write a concise summary of the following:
    "{text}"
    CONCISE SUMMARY:"""

    if custom_prompt:
        prompt_template = custom_prompt + '\n"""\n{text}\n"""\nCONCISE SUMMARY:'

    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
    summary_chain = load_summarize_chain(
        llm, chain_type="map_reduce", map_prompt=prompt, combine_prompt=prompt
    )
    return summary_chain.invoke(transcript_text)

def create_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Splits text into chunks with specified size and overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def summarize_with_refined_prompt(transcript_text: str, llm, custom_prompt: str = None):
    """Summarizes a transcript using a refined prompting strategy and chunking."""

    model_name = llm.model  # Get the model name from the llm object
    model_context_window = NVIDIA_CONTEXT_WINDOWS.get(model_name)
    if model_context_window is None:
        raise ValueError(f"Context window size not defined for model: {model_name}")

    # Check if the ENTIRE transcript fits within the model's context window
    num_tokens = count_tokens(transcript_text, model_name)
    logging.info(f"Number of tokens in transcript: {num_tokens}")

    if num_tokens > model_context_window:
        initial_chunk_size = model_context_window - OVERLAP_SIZE  # Ensure chunks fit
        overlap_size = OVERLAP_SIZE
    else:  # If it fits, just summarize it directly.  No chunking needed.
        initial_chunk_size = model_context_window
        overlap_size = 0

    chunks = create_chunks(transcript_text, initial_chunk_size, overlap_size)
    logging.info(f"Number of chunks created: {len(chunks)}")

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

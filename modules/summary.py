import logging

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from .helpers import num_tokens_from_string

# NVIDIA-specific context window sizes
NVIDIA_CONTEXT_WINDOWS = {
    "meta/llama-3.1-405b-instruct": 4096,
}

SYSTEM_PROMPT = """You are an expert in processing video transcripts according to user's request. 
You will receive a transcript of a video and a request from a user. Your task is to process the transcript according to the user's request.
If no specific request is provided, create a comprehensive summary of the video that captures its main points, key insights, and important details.
"""

DEFAULT_USER_PROMPT = """Please provide a comprehensive summary of the following video transcript. Include:
1. Main topic and key points
2. Important details and examples
3. Any conclusions or takeaways

Transcript:
{input}
"""

CUSTOM_USER_PROMPT = """Process the following video transcript according to this request:
{custom_prompt}

Transcript:
{input}
"""


class TranscriptTooLongForModelException(Exception):
    """Raised when the transcript is too long for the model's context window."""

    def __init__(self, transcript_length: int, model_name: str):
        self.message = (
            f"The transcript is too long ({transcript_length} tokens) for the model's context window "
            f"({NVIDIA_CONTEXT_WINDOWS.get(model_name, 4096)} tokens).\n\n"
            "Consider the following options:\n"
            "1. Choose another model with a larger context window.\n"
            "2. Use the 'Chat' feature to ask specific questions about the video. There you won't be limited by the number of tokens."
        )
        super().__init__(self.message)


def get_transcript_summary(
    transcript_text: str,
    llm: ChatNVIDIA,
    custom_prompt: str = None,
) -> str:
    """
    Generates a summary of a video transcript using NVIDIA NIM.

    Args:
        transcript_text (str): The transcript to summarize
        llm (ChatNVIDIA): The NVIDIA language model instance
        custom_prompt (str, optional): Custom prompt for specific summary requirements

    Returns:
        str: Generated summary

    Raises:
        TranscriptTooLongForModelException: If transcript length exceeds model's context window
    """
    # Check transcript length against model's context window
    context_window = NVIDIA_CONTEXT_WINDOWS.get(llm.model, 4096)
    transcript_tokens = num_tokens_from_string(transcript_text, llm.model)

    if transcript_tokens > context_window:
        raise TranscriptTooLongForModelException(transcript_tokens, llm.model)

    # Create prompt template based on whether custom prompt is provided
    if custom_prompt:
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("user", CUSTOM_USER_PROMPT),
        ])
        user_prompt = {"input": transcript_text, "custom_prompt": custom_prompt}
    else:
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("user", DEFAULT_USER_PROMPT),
        ])
        user_prompt = {"input": transcript_text}

    chain = prompt | llm | StrOutputParser()
    return chain.invoke(user_prompt)

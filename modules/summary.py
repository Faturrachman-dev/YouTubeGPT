import logging

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_openai import ChatOpenAI

from .helpers import num_tokens_from_string

SYSTEM_PROMPT = """You are an expert in processing video transcripts according to user's request. 
For example this could be summarization, question answering or providing key insights.
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("user", "{input}"),
    ]
)

# info about OpenAI's GPTs context windows: https://platform.openai.com/docs/models
CONTEXT_WINDOWS = {
    "gpt-3.5-turbo": {"total": 16385, "output": 4096},
    "gpt-4": {"total": 8192, "output": 4096},
    "gpt-4-turbo": {"total": 128000, "output": 4096},
    # https://community.openai.com/t/gpt-4o-max-tokens-output-response-length/748822
    "gpt-4o": {"total": 128000, "output": 4096},
    # https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/
    "gpt-4o-mini": {"total": 128000, "output": 16000},
}

# Placeholder for NVIDIA context windows.  REPLACE with actual values.
NVIDIA_CONTEXT_WINDOWS = {
    "meta/llama-3.1-405b-instruct": {"total": 4096, "output": 4096},  # Example!
}


class TranscriptTooLongForModelException(Exception):
    """Raised when the length of the transcript exceeds the context window of a language model."""

    def __init__(self, message, model_name: str):
        self.message = message
        self.model_name = model_name
        super().__init__(self.message)

    def log_error(self):
        # Assuming logging is configured globally
        logging.error("Transcript too long for %s.", self.model_name, exc_info=True)


def get_transcript_summary(transcript_text: str, llm: ChatNVIDIA | ChatOpenAI, **kwargs):
    """
    Generates a summary from a video transcript using a language model.

    Args:
        transcript_text (str): The full transcript text of the video.
        llm (ChatOpenAI): The language model instance to use for generating the summary.
        **kwargs: Optional keyword arguments.
            - custom_prompt (str): A custom prompt to replace the default summary request.

    Raises:
        TranscriptTooLongForModelException: If the transcript exceeds the model's context window.

    Returns:
        str: The summary/answer in markdown format.
    """

    user_prompt = f"""Based on the provided transcript of the video, create a summary that accurately captures the main topics and arguments. The summary should be in whole sentences and contain no more than 300 words.
        Additionaly, extract key insights from the video for contributing to better understanding, emphasizing the main points and providing actionable advise.
        Here is the transcript, delimited by ---
        ---
        {transcript_text}
        ---
        Answer in markdown format strictly adhering to this schema:

        ## <short title for the video, consisting of maximum five words>

        <your summary>

        ## Key insights

        <unnumbered list of key insights>
        """

    if "custom_prompt" in kwargs:
        user_prompt = f"""{kwargs['custom_prompt']}
            Here is the transcript, delimited by ---
            ---
            {transcript_text}
            ---
            """

    # if the number of tokens in the transcript (plus the number of tokens in the prompt) exeed the model's context window, an exception is raised
    if isinstance(llm, ChatNVIDIA):
        # Use NVIDIA context windows
        max_tokens = NVIDIA_CONTEXT_WINDOWS.get(llm.model, {}).get("total", 4096)  # Default to 4096 if not found
        model_name_for_tokens = llm.model # Use NVIDIA model name directly
    else:  # isinstance(llm, ChatOpenAI)
        max_tokens = CONTEXT_WINDOWS[llm.model_name]["total"] - num_tokens_from_string(
            string=user_prompt, model=llm.model_name
        )
        model_name_for_tokens = llm.model_name

    if num_tokens_from_string(string=transcript_text, model=model_name_for_tokens) > max_tokens:
        raise TranscriptTooLongForModelException(
            message=f"Your transcript exceeds the context window of the chosen model ({llm.model_name}), which is {max_tokens} tokens. "
            "Consider the following options:\n"
            "1. Choose another model with larger context window (such as gpt-4o).\n"
            "2. Use the 'Chat' feature to ask specific questions about the video. There you won't be limited by the number of tokens.\n\n"
            "You can get more information on context windows for different models in the [official OpenAI documentation about models](https://platform.openai.com/docs/models).",
            model_name=llm.model_name,
        )

    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"input": user_prompt})

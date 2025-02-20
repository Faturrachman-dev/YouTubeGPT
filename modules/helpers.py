import json
import logging
import os
import re
from pathlib import Path
from typing import List, Literal, Optional

import streamlit as st
import tiktoken
from dotenv import load_dotenv

load_dotenv()
# print(f"Loaded .env file: {load_dotenv()}")  # Removed debug print

# Load config
def load_config(file_path="config.json"):
    """Loads configuration from a JSON file."""
    # print(f"Attempting to load config from: {file_path}")  # Removed debug print
    try:
        with open(file_path, "r") as file:
            config = json.load(file)
            # print(f"Successfully loaded config: {config}")  # Removed debug print
            return config
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")  # Keep error print
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in config file: {e}")  # Keep error print
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}") # Keep error print
        return None

CONFIG = load_config()
# if CONFIG is None: # Removed debug prints
#     print("CONFIG is None.  Check for errors above.")
# else:
#     print(f"CONFIG keys: {CONFIG.keys()}")

def get_default_config_value(key: str) -> Optional[str]:
    """Retrieves a value from the config, handling potential KeyErrors."""
    if CONFIG is None:
        logging.warning("CONFIG is None. Cannot retrieve values.") # Keep warning
        return None
    try:
        # print(f"Attempting to retrieve key: {key}")  # Removed debug print
        value = CONFIG[key]
        # print(f"Retrieved value: {value}")  # Removed debug print
        return value
    except KeyError:
        logging.warning(f"Key '{key}' not found in config.json.") # Keep warning
        return None  # Or a suitable default value

def is_api_key_set() -> bool:
    """Checks whether the NVIDIA API key is set either in session state or as an environment variable."""
    return "nvidia_api_key" in st.session_state or bool(os.getenv("NVIDIA_API_KEY"))

def is_api_key_valid(api_key: str = None) -> bool:
    """
    Checks if the NVIDIA API key is valid.  If no key is provided, checks the session state.
    """
    if api_key is None:
        if "nvidia_api_key" in st.session_state:
            api_key = st.session_state.nvidia_api_key
        else:
            return False  # No key provided and not in session state

    if not api_key.startswith("nvapi-"):
        logging.error("Invalid NVIDIA API key format - must start with 'nvapi-'")
        return False
    elif len(api_key) < 20:  # Minimum length for NVIDIA keys
        logging.error("NVIDIA API key seems too short")
        return False

    logging.info("NVIDIA API key format validation successful") # Keep this one, it's helpful
    return True


def get_available_models(model_type: Literal["nvidia", "embeddings"], api_key: str = "") -> List[str]:
    """
    Retrieve available models from config based on the specified type.

    Args:
        model_type (Literal["nvidia", "embeddings"]): The type of models to retrieve
        api_key (str, optional): Not used for NVIDIA implementation. Defaults to "".

    Returns:
        List[str]: List of available model IDs for the specified type
    """
    return list(get_default_config_value(f"available_models.{model_type}"))


def extract_youtube_video_id(url: str):
    """
    Extracts the video ID from a given YouTube URL.

    The function supports various YouTube URL formats including standard watch URLs, short URLs, and embed URLs.

    Args:
        url (str): The YouTube URL from which the video ID is to be extracted.

    Returns:
        str or None: The extracted video ID as a string if the URL is valid and the video ID is found, otherwise None.

    Example:
        >>> extract_youtube_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        'dQw4w9WgXcQ'
    """
    # Regular expression for various YouTube URL formats
    youtube_regex = (
        r"(?:https?:\/\/)?(?:www\.)?"
        r"(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|"
        r"youtu\.be\/)([a-zA-Z0-9_-]{11})"
    )

    match = re.search(youtube_regex, url)
    if match:
        return match.group(1)  # The video ID is captured in group 1
    return None


def save_response_as_file(
    dir_name: str, filename: str, file_content, content_type: str = "text"
):
    """
    Saves given content to a file in the specified directory, formatted as either plain text, JSON, or Markdown.

    Args:
        dir_name (str): The directory where the file will be saved.
        filename (str): The name of the file without extension.
        file_content: The content to be saved. Can be a string for text or Markdown, or a dictionary/list for JSON.
        content_type (str): The type of content: "text" for plain text, "json" for JSON format, or "markdown" for Markdown format. Defaults to "text".

    The function creates the directory if it doesn't exist. It saves `file_content` in a file named `filename`
    within that directory, adding the appropriate extension (.txt for plain text, .json for JSON, .md for Markdown) based on `content_type`.
    """

    # Sanitize the filename by replacing slashes with underscores
    filename = filename.replace("/", "_").replace("\\", "_")

    # Create the directory if it does not exist
    os.makedirs(dir_name, exist_ok=True)

    # Adjust the filename extension based on the content type
    extensions = {"text": ".txt", "json": ".json", "markdown": ".md"}
    file_extension = extensions.get(content_type, ".txt")
    filename += file_extension

    # Construct the full path for the file
    file_path = os.path.join(dir_name, filename)

    # Write the content to the file, formatting it according to the content type
    with open(file_path, "w", encoding="utf-8") as file:
        if content_type == "json":
            json.dump(file_content, file, indent=4)
        else:
            file.write(file_content)

    # Log the full path of the saved file
    logging.info("File saved at: %s", file_path) # Keep this, it's useful


def get_preffered_languages():
    # TODO: return from configuration object or config.json
    return ["en-US", "en", "de"]


def num_tokens_from_string(string: str, model_name: str) -> int:
    """
    Returns the number of tokens in a text string.

    Args:
        string (str): The text string.
        model_name (str): The name of the encoding to use.

    Returns:
        int: The number of tokens in the text string.
    """
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.") # Keep warning
        encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string))
    return num_tokens


def read_file(file_path: str) -> str:
    """Read a file and return its contents as a string.

    Args:
        file_path (str): Path to the file to read

    Returns:
        str: Contents of the file
    """
    return Path(file_path).read_text(encoding='utf-8')


def is_environment_prod():
    if os.getenv("ENVIRONMENT") == "production":
        return True
    return False


def save_to_file(content: str, filepath: str) -> None:
    """
    Saves content to a file, creating directories if needed.

    Args:
        content (str): Content to save
        filepath (str): Path to save the file
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def count_tokens(text: str, model_name: str) -> int:
    """Counts the number of tokens in a string using tiktoken."""
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")  # Default
    return len(encoding.encode(text))

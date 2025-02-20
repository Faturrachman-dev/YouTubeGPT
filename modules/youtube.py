import json
import logging
import os
from typing import List

from requests import api
from requests.exceptions import RequestException
from youtube_transcript_api import (
    CouldNotRetrieveTranscript,
    Transcript,
    YouTubeTranscriptApi,
    NoTranscriptFound,
    TranscriptsDisabled,
    TranslationLanguageNotAvailable
)
from youtube_transcript_api.formatters import TextFormatter

from .helpers import (
    extract_youtube_video_id,
    get_preffered_languages,
)

OEMBED_PROVIDER = "https://noembed.com/embed"


class NoTranscriptReceivedException(Exception):
    def __init__(self, url: str):
        # message should be a user-friendly error message
        self.message = "Unfortunately, no transcript was found for this video. Therefore a summary can't be provided :slightly_frowning_face:"
        self.url = url
        super().__init__(self.message)

    def log_error(self):
        """Provides error context for developers."""
        logging.error("Could not find a transcript for %s", self.url)


class InvalidUrlException(Exception):
    def __init__(self, url, message="Invalid YouTube URL"):
        self.url = url
        self.message = message
        super().__init__(self.message)

    def log_error(self):
        """Provides error context for developers."""
        logging.error(f"{self.message}: {self.url}")


def get_video_metadata(url: str) -> dict:
    """
    Fetches metadata for a YouTube video using yt-dlp.  Requires yt-dlp to be installed.

    Args:
        url: The URL of the YouTube video.

    Returns:
        A dictionary containing the video's metadata, or None if an error occurs.
        The dictionary will have the keys "name" (video title) and "channel".

    Raises:
        InvalidUrlException: If the URL is invalid.
    """
    try:
        video_id = extract_youtube_video_id(url)
        if not video_id:
            raise InvalidUrlException(url)

        # Use yt-dlp to get video information.  This is more reliable than parsing HTML.
        # The command is: yt-dlp --dump-json --skip-download <video_url>
        command = ["yt-dlp", "--dump-json", "--skip-download", url]

        # Execute the command and capture the output
        import subprocess
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        video_info = result.stdout

        # Parse the JSON output
        import json
        video_info_json = json.loads(video_info)

        # Extract the title and channel
        video_title = video_info_json.get('title')
        channel_name = video_info_json.get('channel')

        if video_title and channel_name:
            return {"name": video_title, "channel": channel_name}
        else:
            logging.error("Could not retrieve video title and channel from yt-dlp output.")
            return {"name": "Unknown Title", "channel": "Unknown Channel"} # Return default values

    except subprocess.CalledProcessError as e:
        logging.error("Error running yt-dlp: %s", e)
        return {"name": "Unknown Title", "channel": "Unknown Channel"}  # Return default values on error
    except json.JSONDecodeError as e:
        logging.error("Error parsing yt-dlp output: %s", e)
        return {"name": "Unknown Title", "channel": "Unknown Channel"}  # Return default values on error
    except InvalidUrlException:
        raise
    except Exception as e:
        logging.error("An unexpected error occurred: %s", e)
        return {"name": "Unknown Title", "channel": "Unknown Channel"}  # Return default values on error


def fetch_youtube_transcript(url: str) -> str:
    """
    Fetches the transcript for a YouTube video.

    Args:
        url: The URL of the YouTube video.

    Returns:
        The transcript as a single string.

    Raises:
        InvalidUrlException: If the URL is not a valid YouTube URL.
        NoTranscriptReceivedException: If no transcript could be retrieved.
        Exception: For other errors during transcript retrieval.
    """
    try:
        video_id = extract_youtube_video_id(url)
        if not video_id:
            raise InvalidUrlException(url)

        preferred_languages = get_preffered_languages()
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        try:
            # Try to get a manually created transcript in a preferred language
            transcript = transcript_list.find_transcript(preferred_languages)
            transcript_text = " ".join([part["text"] for part in transcript.fetch()])
            return transcript_text

        except NoTranscriptFound:
            # If no manually created transcript is found, try auto-generated transcripts
            for transcript in transcript_list:
                if transcript.is_generated and transcript.language_code in preferred_languages:
                    transcript_text = " ".join([part["text"] for part in transcript.fetch()])
                    return transcript_text

                # If not in preferred languages, try to translate
                if transcript.is_translatable:
                    for language in preferred_languages:
                        try:
                            translated_transcript = transcript.translate(language)
                            transcript_text = " ".join([part["text"] for part in translated_transcript.fetch()])
                            return transcript_text
                        except TranslationLanguageNotAvailable:
                            logging.warning(f"Translation to {language} not available for video {video_id}")
                            continue # Try the next preferred language

            # If no usable transcript found, raise the exception
            raise NoTranscriptReceivedException(url)


        except Exception as e:
            logging.error("An error occurred while fetching the transcript: %s", e)
            raise

    except TranscriptsDisabled:
        raise NoTranscriptReceivedException(url)
    except InvalidUrlException:
        raise
    except Exception as e:
        logging.error("An unexpected error occurred: %s", e)
        raise


def analyze_transcripts(video_id: str):
    try:
        transcript_list: list[Transcript] = YouTubeTranscriptApi.list_transcripts(
            video_id
        )
    except Exception as e:
        print("An error occured when fetching transcripts: " + e)
        return
    else:
        for t in transcript_list:
            if t.is_generated:
                print(
                    f"found auto-generated transcript in {t.language} ({t.language_code})!"
                )
            else:
                print(f"found manual transcript in {t.language} ({t.language_code})!")

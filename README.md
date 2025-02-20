<p align="center">
  <img src=".assets/yt-summarizer-logo.png" alt="Logo" width="250">
</p>

<h1 align="center">YouTubeGPT- Your YouTube AI (NVIDIA Edition)</h1>

## Features :sparkles:

This modified version of YouTubeGPT lets you **summarize and chat (Q&A)** with YouTube videos using **NVIDIA's language models**. Its features include:

- **Provide a custom prompt for summaries** :writing_hand: [**VIEW DEMO**](https://youtu.be/rJqx3qvebws)
  - Tailor the summary to your needs by providing a custom prompt, or use the default summarization.
- **Get answers to questions about the video content** :question: [**VIEW DEMO**](https://youtu.be/rI8NogvHplE)
  - The application is optimized for question answering (Q&A).
- **Create your own library/knowledge base** :open_file_folder:
  - Summaries and answers can be saved to a library accessible on a separate page.
  - Summaries are also automatically saved in the `responses` directory within your project folder, organized by channel and video title.
- **Uses NVIDIA models for text generation** :robot:
  - **Currently available:** `meta/llama-3.1-405b-instruct`
  - NVIDIA models allow for summarizing longer videos.
- **Experiment with settings** :gear:
  - Adjust the temperature and top P of the model.
- **Choose UI theme** :paintbrush:
  - Go to the three dots in the upper right corner, select settings, and choose either light, dark, or a custom theme.

## Installation & Usage (NVIDIA-Only)

This version is configured to exclusively use NVIDIA's language models.  You will need an NVIDIA API key.  See [NVIDIA's instructions](https://developer.nvidia.com/) to get started.

**Prerequisites:**

*   **NVIDIA API Key:** Obtain your key from the NVIDIA developer website.
*   **Docker Desktop:**  Install Docker Desktop.  This is only used for running ChromaDB (the vector database).
*   **`yt-dlp`:** Install `yt-dlp` using pip: `pip install yt-dlp`
*   **Python 3.11:** This project is developed and tested with Python 3.11.

**Steps:**

1.  **Clone the Repository:**
    ```bash
    git clone <repository_url>  # Replace with the actual URL
    cd <repository_directory>
    ```

2.  **Set up Virtual Environment:**
    ```bash
    python -m venv .venv
    .\.venv\Scripts\Activate  # On Windows (PowerShell)
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set NVIDIA API Key (Using .env file):**
    *   Make sure your `.env` file in the root of your project directory contains the line:
        ```
        NVIDIA_API_KEY=your_nvidia_api_key
        ```
        Replace `your_nvidia_api_key` with your actual NVIDIA API key (no quotes). The provided code already uses `python-dotenv` to load environment variables from the `.env` file, so no further action is needed in the terminal.

5.  **Run ChromaDB (Docker):**
    *   Make sure Docker Desktop is running.
    *   In a *separate* PowerShell terminal, navigate to your project directory and run:
        ```powershell
        docker-compose up -d chromadb
        ```

6.  **Run the Application:**
    *   In the terminal where you activated the virtual environment, run:
        ```powershell
        streamlit run main.py
        ```

7.  **Access the Application:**
    *   The application should open automatically in your browser at `http://localhost:8501`.

**Important Notes:**

*   This version uses OpenAI embeddings for the vector database.  A future update might explore NVIDIA embedding models.
*   The `max_tokens` parameter is not directly supported by the NVIDIA Langchain integration.  It's handled implicitly.
*   Make sure your NVIDIA API key is kept secret.  Do not commit it to your repository.

## Technologies Used

- [YouTube Transcript API](https://github.com/jdepoix/youtube-transcript-api): For fetching transcripts.
- [yt-dlp](https://github.com/yt-dlp/yt-dlp):  For fetching video metadata.
- [LangChain](https://github.com/langchain-ai/langchain): For prompt creation, LLM interaction, and RAG.
- [langchain-nvidia-ai-endpoints](https://pypi.org/project/langchain-nvidia-ai-endpoints/):  For using NVIDIA's language models.
- [Streamlit](https://github.com/streamlit/streamlit): For the user interface.
- [ChromaDB](https://docs.trychroma.com/): As a vector store for embeddings (running in Docker).

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

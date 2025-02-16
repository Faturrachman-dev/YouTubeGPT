<p align="center">
  <img src=".assets/yt-summarizer-logo.png" alt="Logo" width="250">
</p>

<h1 align="center">YouTubeGPT- Your YouTube AI</h1>

## Features :sparkles:

YouTubeGPT lets you **summarize and chat (Q&A)** with YouTube videos using NVIDIA's language models. Its features include:

- **provide a custom prompt for summaries** :writing_hand: [**VIEW DEMO**](https://youtu.be/rJqx3qvebws)
  - you can tailor the summary to your needs by providing a custom prompt or just use the default summarization
- **get answers to questions about the video content** :question: [**VIEW DEMO**](https://youtu.be/rI8NogvHplE)
  - part of the application is designed and optimized specifically for question answering tasks (Q&A)
- **create your own library/knowledge base** :open_file_folder:
  - the summaries and answers can be saved to a library accessible at a separate page!
  - additionally, summaries can be automatically saved in the directory where you run the app. The summaries will be available under `<YT-channel-name>/<video-title>.md`
- **use NVIDIA models for text generation** :robot:
  - currently available: meta/llama-3.1-405b-instruct
  - by using NVIDIA models, you can summarize even longer videos and potentially get better responses
- **experiment with settings** :gear:
  - adjust the temperature and top P of the model
- **choose UI theme** :paintbrush:
  - go to the three dots in the upper right corner, select settings and choose either light, dark or my aesthetic custom theme

## Installation & usage

To run the app, you will first need to get an NVIDIA API-Key. This is very straightforward. Have a look at [NVIDIA's instructions](https://developer.nvidia.com/) to get started.

### Run with Docker (for ChromaDB only)

1. Ensure that your `.env` file contains the NVIDIA API key.
2. Adjust the path to save the summaries (l. 39 in [docker-compose.yml](docker-compose.yml))
3. Execute the following command to run ChromaDB:

```bash
# run chromadb
docker-compose up -d chromadb
```

The app will be accessible in the browser under <http://localhost:8501>.

### Development in virtual environment

```bash
# create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate
# install requirements
pip install -r requirements.txt
# you'll need an NVIDIA API key
export NVIDIA_API_KEY=<your-nvidia-api-key>
# run app
streamlit run main.py
```

The app will be accessible in the browser under <http://localhost:8501>.

## Technologies used

The project is built using some amazing libraries:

- The project uses [YouTube Transcript API](https://github.com/jdepoix/youtube-transcript-api) for fetching transcripts.
- [LangChain](https://github.com/langchain-ai/langchain) is used to create a prompt, submit it to an LLM and process its response.
- The UI is built using [Streamlit](https://github.com/streamlit/streamlit).
- [ChromaDB](https://docs.trychroma.com/) is used as a vector store for embeddings.

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

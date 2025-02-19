import logging
import uuid
from typing import List, Literal

from chromadb import Collection
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from modules.helpers import num_tokens_from_string


class NoDocumentsFoundException(Exception):
    """Exception raised when no relevant documents are found in the vector store."""
    pass


CHUNK_SIZE_FOR_UNPROCESSED_TRANSCRIPT = 512

# Mapping for chunk sizes to number of chunks to retrieve
CHUNK_SIZE_TO_K_MAPPING = {1024: 3, 512: 5, 256: 10, 128: 20}

# NVIDIA-specific context window sizes
NVIDIA_CONTEXT_WINDOWS = {
    "meta/llama-3.1-405b-instruct": 4096,
}

RAG_SYSTEM_PROMPT = """You are an expert in answering questions and providing information about a topic.
You will receive excerpts from a video transcript as context and a question or topic from a user.
Your task is to provide accurate, relevant information based on the context provided."""

RAG_USER_PROMPT = """Question/Topic: {question}

Relevant context from the video:
{context}

Please provide a detailed response based on the context above."""


def split_text_recursively(
    transcript_text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 32,
    len_func: Literal["tokens", "characters"] = "characters",
) -> List[Document]:
    """
    Splits text into chunks recursively.

    Args:
        transcript_text (str): Text to split
        chunk_size (int, optional): Size of each chunk. Defaults to 512.
        chunk_overlap (int, optional): Overlap between chunks. Defaults to 32.
        len_func (Literal["tokens", "characters"], optional): Function to measure length. Defaults to "characters".

    Returns:
        List[Document]: List of document chunks
    """
    # Always use character-based splitting for simplicity
    length_function = len

    # Create text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=length_function,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    # Split the text into documents
    return text_splitter.create_documents([transcript_text])


def embed_excerpts(
    transcript_excerpts: List[Document],
    collection: Collection,
    nvidia_model: ChatNVIDIA,
    chunk_size: int,
) -> None:
    """
    Embeds transcript excerpts and adds them to the collection.

    Args:
        transcript_excerpts (List[Document]): List of document chunks
        collection (Collection): ChromaDB collection
        nvidia_model (ChatNVIDIA): NVIDIA model for embeddings
        chunk_size (int): Size of chunks
    """
    try:
        # Get embeddings using NVIDIA model
        embeddings = []
        for doc in transcript_excerpts:
            response = nvidia_model.invoke([{"role": "user", "content": doc.page_content}])
            embeddings.append(response.content)  # Using the response content as embedding

        collection.add(
            ids=[str(uuid.uuid4()) for _ in transcript_excerpts],
            documents=[doc.page_content for doc in transcript_excerpts],
            embeddings=embeddings,
            metadatas=[{"chunk_size": chunk_size} for _ in transcript_excerpts],
        )
    except Exception as e:
        logging.error("Error embedding excerpts: %s", str(e))
        raise


def find_relevant_documents(
    query: str,
    db: Chroma,
    k: int = 4,
) -> List[Document]:
    """
    Finds relevant documents for a query using similarity search.

    Args:
        query (str): Query string
        db (Chroma): Vector store
        k (int, optional): Number of documents to retrieve. Defaults to 4.

    Returns:
        List[Document]: List of relevant documents
    """
    return db.similarity_search(query, k=k)


def format_docs_for_context(docs: List[Document]) -> str:
    """
    Formats documents into a string for context.

    Args:
        docs (List[Document]): List of documents

    Returns:
        str: Formatted context string
    """
    return "\n\n".join(doc.page_content for doc in docs)


def generate_response(
    question: str,
    llm: ChatNVIDIA,
    relevant_docs: List[Document],
) -> str:
    """
    Generates a response using RAG with NVIDIA NIM.

    Args:
        question (str): User's question
        llm (ChatNVIDIA): The NVIDIA language model instance
        relevant_docs (List[Document]): List of relevant document chunks
    
    Returns:
        str: Generated response
    """
    formatted_input = RAG_USER_PROMPT.format(
        question=question, 
        context=format_docs_for_context(relevant_docs)
    )
    
    # Check if input exceeds context window
    context_window = NVIDIA_CONTEXT_WINDOWS.get(llm.model, 4096)
    input_tokens = num_tokens_from_string(formatted_input, llm.model)
    
    if input_tokens > context_window:
        logging.warning(
            "Input exceeds model context window. Tokens: %d, Max: %d",
            input_tokens,
            context_window
        )
        raise ValueError(
            f"Input length ({input_tokens} tokens) exceeds model's context window ({context_window} tokens)"
        )
    
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM_PROMPT),
        ("user", formatted_input),
    ])
    
    rag_chain = rag_prompt | llm | StrOutputParser()
    return rag_chain.invoke({})

import os
import sys

import fitz
import numpy as np
import ollama
import redis
from redis.commands.search.query import Query

# Redis connection details
REDIS_HOST = "localhost"
REDIS_PORT = 6380
REDIS_DB = 0

# Vector search configuration
VECTOR_DIM = 768
INDEX_NAME = "embedding_index"
DOC_PREFIX = "doc:"
DISTANCE_METRIC = "COSINE"

# Chunking parameters
DEFAULT_CHUNK_SIZE = 400
DEFAULT_OVERLAP = 100

# Default Ollama model
DEFAULT_OLLAMA_MODEL = "deepseek-r1"

# Initialize Redis client
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)


def clear_redis_store():
    """Clears all data from the current Redis database."""
    print("Clearing existing Redis store...")
    try:
        redis_client.flushdb()
        print("Redis store cleared.")
    except redis.exceptions.ConnectionError as e:
        print(f"Error connecting to Redis: {e}")


def create_hnsw_index():
    """Creates an HNSW vector index in Redis if it doesn't exist."""
    try:
        redis_client.ft(INDEX_NAME).info()
        print(f"Index '{INDEX_NAME}' already exists.")
    except redis.exceptions.ResponseError:
        print(f"Creating index '{INDEX_NAME}'...")
        schema = (
            ("file", "TEXT"),
            ("page", "NUMERIC"),
            ("chunk", "TEXT"),
            (
                "embedding",
                "VECTOR",
                "HNSW",
                6,
                "DIM",
                VECTOR_DIM,
                "TYPE",
                "FLOAT32",
                "DISTANCE_METRIC",
                DISTANCE_METRIC,
            ),
        )
        try:
            redis_client.ft(INDEX_NAME).create_index(fields=schema, prefix=[DOC_PREFIX])
            print("Index created successfully.")
        except redis.exceptions.ConnectionError as e:
            print(f"Error connecting to Redis: {e}")
        except redis.exceptions.ResponseError as e:
            print(f"Error creating index: {e}")


def get_embedding(text: str, model: str = DEFAULT_OLLAMA_MODEL) -> list[float] | None:
    """Generates an embedding for the given text using the specified Ollama model."""
    try:
        response = ollama.embeddings(model=model, prompt=text)
        return response["embedding"]
    except ollama.OllamaAPIError as e:
        print(f"Error generating embedding with Ollama: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during embedding generation: {e}")
        return None


def store_embedding(file: str, page: int, chunk_index: int, chunk: str, embedding: list[float]):
    """Stores the text chunk and its embedding in Redis."""
    key = f"{DOC_PREFIX}:{file}_page_{page}_chunk_{chunk_index}"
    try:
        redis_client.hset(
            key,
            mapping={
                "file": file,
                "page": page,
                "chunk": chunk,
                "embedding": np.array(embedding, dtype=np.float32).tobytes(),  # Store as byte array
            },
        )
        print(f"Stored embedding for: {file} - Page {page} - Chunk {chunk_index}")
    except redis.exceptions.ConnectionError as e:
        print(f"Error connecting to Redis: {e}")
    except redis.exceptions.ResponseError as e:
        print(f"Error storing embedding in Redis for key '{key}': {e}")


def extract_text_from_pdf(pdf_path: str) -> list[tuple[int, str]]:
    """Extracts text content from each page of a PDF file.

    Args:
        pdf_path: The path to the PDF file.

    Returns:
        A list of tuples, where each tuple contains the page number (0-based) and the extracted text from that page.
    """
    try:
        doc = fitz.open(pdf_path)
        text_by_page = []
        for page_num, page in enumerate(doc):
            text_by_page.append((page_num, page.get_text()))
        return text_by_page
    except FileNotFoundError:
        print(f"Error: PDF file not found at '{pdf_path}'")
        return []
    except Exception as e:
        print(f"Error opening or reading PDF '{pdf_path}': {e}")
        return []


def split_text_into_chunks(
    text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP
) -> list[str]:
    """Splits a string of text into smaller chunks with a specified overlap.

    Args:
        text: The input string to be split.
        chunk_size: The desired maximum number of words per chunk.
        overlap: The number of overlapping words between consecutive chunks.

    Returns:
        A list of text chunks.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
    return chunks


def process_pdfs(
    model: str, data_dir: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP
):
    """Processes all PDF files in the specified directory, extracting text,
    chunking it, generating embeddings, and storing them in Redis.

    Args:
        model: The name of the Ollama model to use for embedding generation.
        data_dir: The path to the directory containing the PDF files.
        chunk_size: The desired size of text chunks.
        overlap: The number of overlapping words between chunks.
    """
    if not os.path.isdir(data_dir):
        print(f"Error: Directory not found at '{data_dir}'")
        return

    print(f"Processing PDF files in '{data_dir}' using model '{model}'...")
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(data_dir, file_name)
            text_by_page = extract_text_from_pdf(pdf_path)
            for page_num, text in text_by_page:
                chunks = split_text_into_chunks(text, chunk_size, overlap)
                for chunk_index, chunk in enumerate(chunks):
                    embedding = get_embedding(chunk, model=model)
                    if embedding is not None:
                        store_embedding(
                            file=file_name,
                            page=page_num,
                            chunk_index=chunk_index,
                            chunk=chunk,
                            embedding=embedding,
                        )
            print(f" -----> Processed {file_name}")


def main(ollama_model: str = DEFAULT_OLLAMA_MODEL):
    """Main function to clear Redis, create the index, and process PDF files."""
    clear_redis_store()
    create_hnsw_index()

    data_directory = "./data/"
    chunk_size = DEFAULT_CHUNK_SIZE
    overlap = DEFAULT_OVERLAP

    if not os.path.exists(data_directory):
        os.makedirs(data_directory)
        print(f"Created data directory: {data_directory}. Please place your PDF files there.")
    else:
        process_pdfs(ollama_model, data_directory, chunk_size, overlap)

    print("\n---Done processing PDFs---\n")


if __name__ == "__main__":
    model_to_use = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OLLAMA_MODEL
    main(model_to_use)
import json
import sys

import numpy as np
import redis
from ollama import Client as OllamaClient
from redis.commands.search.query import Query

# Redis connection details
REDIS_HOST = "localhost"
REDIS_PORT = 6380
REDIS_DECODE_RESPONSES = True

# Vector search configuration
INDEX_NAME = "embedding_index"
VECTOR_FIELD_NAME = "embedding"
VECTOR_DIM = 768
DISTANCE_METRIC = "COSINE"
DEFAULT_EMBEDDING_MODEL = "deepseek-r1"
DEFAULT_RAG_MODEL = "deepseek-r1"

# Initialize Redis and Ollama clients
redis_client = redis.StrictRedis(
    host=REDIS_HOST, port=REDIS_PORT, decode_responses=REDIS_DECODE_RESPONSES
)
ollama_client = OllamaClient()


def get_embedding(text: str, model: str = DEFAULT_EMBEDDING_MODEL) -> list[float]:
    """Generates an embedding for the given text using Ollama."""
    try:
        response = ollama_client.embeddings(model=model, prompt=text)
        return response["embedding"]
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return []


def search_embeddings(
    query: str, model: str = DEFAULT_EMBEDDING_MODEL, top_k: int = 3
) -> list[dict]:
    """Searches Redis for the top_k most similar embeddings to the query."""
    query_embedding = get_embedding(query, model=model)
    if not query_embedding:
        return []

    query_vector = np.array(query_embedding, dtype=np.float32).tobytes()

    try:
        q = (
            Query(f"*=>[KNN {top_k} @{VECTOR_FIELD_NAME} $vec AS vector_distance]")
            .sort_by("vector_distance")
            .return_fields("file", "page", "chunk", "vector_distance")
            .dialect(2)
        )

        results = redis_client.ft(INDEX_NAME).search(
            q, query_params={"vec": query_vector}
        )

        top_results = [
            {
                "file": doc.file,
                "page": doc.page,
                "chunk": doc.chunk,
                "similarity": float(doc.vector_distance),
            }
            for doc in results.docs
        ]

        if top_results:
            print("\n--- Search Results ---")
            for result in top_results:
                print(
                    f"---> File: {result['file']}, Page: {result['page']}, Chunk: {result['chunk']}, Similarity: {result['similarity']:.2f}"
                )

        return top_results

    except Exception as e:
        print(f"Search error: {e}")
        return []


def generate_rag_response(
    query: str, context_results: list[dict], model: str = DEFAULT_RAG_MODEL
) -> str:
    """Generates a response to the query using the provided context."""
    if not context_results:
        return "I don't know."

    context_str = "\n".join(
        [
            f"From {result.get('file', 'Unknown file')} (page {result.get('page', 'Unknown page')}, chunk {result.get('chunk', 'Unknown chunk')}) "
            f"with similarity {float(result.get('similarity', 0)):.2f}"
            for result in context_results
        ]
    )

    prompt = f"""You are a helpful AI assistant dedicated to resumé building.
Use the following context to answer the query as accurately as possible. If the context is
not relevant to the query, say 'I don't know'.

Context:
{context_str}

Query: {query}

Answer:"""

    try:
        response = ollama_client.chat(
            model=f"{model}:latest",
            messages=[{"role": "user", "content": prompt}],
            options={"num_predict": 256},
        )
        return response["message"]["content"].replace(",", "").replace("\n", "  ")
    except Exception as e:
        print(f"Error generating RAG response: {e}")
        return "I encountered an error while generating the response."


def interactive_search(default_model: str = DEFAULT_EMBEDDING_MODEL) -> str | None:
    """Interactive search interface."""
    print("\n🔍 RAG Search Interface")
    print("Type 'exit' to quit")

    while True:
        query = input("\nEnter your search query: ").strip()
        if query.lower() == "exit":
            break

        # Search for relevant embeddings
        context_results = search_embeddings(query, model=default_model)

        # Generate RAG response
        response = generate_rag_response(query, context_results, model=default_model)

        print("\n--- Response ---")
        print(response)
        return response  # Return the response for potential further use

    return None


if __name__ == "__main__":
    model_to_use = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_EMBEDDING_MODEL
    interactive_search(model_to_use)
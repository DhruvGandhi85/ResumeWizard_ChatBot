import redis
import json
import numpy as np
import ollama
from redis.commands.search.query import Query
import sys
import store_in_redis
import pandas as pd

redis_client = redis.StrictRedis(host="localhost", port=6380, decode_responses=True)

def search_embeddings(model, query, top_k=3):
    query_embedding = store_in_redis.get_embedding(model, query)
    query_vector = np.array(query_embedding, dtype=np.float32).tobytes()

    try:
        q = (
            Query(f"*=>[KNN {top_k} @embedding $vec AS vector_distance]")
            .sort_by("vector_distance")
            .return_fields("id", "filename", "page", "chunk", "vector_distance")
            .dialect(2))

        results = redis_client.ft("embedding_index").search(q, query_params={"vec": query_vector})
        top_matches = [{"filename": result.filename,"page": result.page,"chunk": result.chunk,"similarity": result.vector_distance,} 
                       for result in results.docs][:top_k]

        return top_matches

    except Exception as e:
        print(f"An error occurred during the search: {e}")
        return []


def generate_rag_response(chat_model, query, context_results):
    context_str = "\n".join(
        [f"From {result.get('filename', 'Unknown filename')} (page {result.get('page', 'Unknown page')}, chunk {result.get('chunk', 'Unknown chunk')}) "
            f"with similarity {float(result.get('similarity', 0)):.2f}"
            for result in context_results])

    print(f"context_str: {context_str}")

    prompt = f'As a dedicated AI assistant specializing in resume guidance, please utilize the provided contextual information to address the users query with precision. If the context lacks relevance to the query, respond with "I lack the necessary information to answer this effectively." \n Context: {context_str} \n Query: {query} \n Answer:'

    response = ollama.chat(
        model=f"{chat_model}:latest", 
        messages=[{"role": "user", "content": prompt}], 
        options={"num_predict": 256})

    return response

def create_recall_test_set(folder_path, num_samples=50):
    pdf_files = store_in_redis.find_pdfs_in_subfolders(folder_path)
    random_files = np.random.choice(pdf_files, size=num_samples, replace=False)
    recall_data = pd.DataFrame(columns=["query", "correct_filename"])
    for i in range(len(random_files)):
        file = random_files[i]
        trimmed_path = file.split("/")[-1]
        pdf_text = store_in_redis.convert_pdf_to_text(file)
        random_page_index = np.random.randint(0, len(pdf_text))
        words = pdf_text[random_page_index].split()
        if len(words) < 8:
            continue
        random_size = np.random.randint(4, 8)
        max_start = len(words) - random_size
        if max_start < 0:
            continue
        random_start = np.random.randint(0, max_start + 1)
        random_text = " ".join(words[random_start : random_start + random_size])
        recall_data.loc[i] = [random_text, trimmed_path]
    
    return recall_data

def compute_recall(model, ground_truth_df, top_k=5):
    success_count = 0
    total_queries = len(ground_truth_df)

    for index, row in ground_truth_df.iterrows():
        query = row["query"]
        correct_filename = row["correct_filename"]
        retrieved_docs = search_embeddings(model, query, top_k=top_k)
        found = False
        for doc in retrieved_docs:
            if (doc['filename'] == correct_filename):
                found = True
                break

        if found:
            success_count += 1
        print(f"Query: {query}, Found: {found}, Correct Filename: {correct_filename}")

    recall = success_count / total_queries
    print(f"Recall@{top_k} = {recall:.2%}")
    return recall


def interactive_search(embedding_model, chat_model, query=None):
    print("RAG Search Interface")
    print("Enter blank input to quit")

    while True:
        if query is None:
            query = input("\nEnter search query: ")
        else:
            query = query.strip()
        
        print(f"Query: {query}")
        if query is None or query == "":
            break

        context_results = search_embeddings(embedding_model, query)
        response = generate_rag_response(chat_model, query, context_results)
        response_content = response["message"]["content"]
        response_content = response_content.replace(",", "")
        response_content = response_content.replace("\n", "  ")

        print("\n Response: ")
        print(response_content)
        break

    return response_content
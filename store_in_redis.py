import ollama
import redis
import numpy as np
from redis.commands.search.query import Query
import os
from PyPDF2 import PdfReader
import sys

redis_client = redis.Redis(host='localhost', port=6380, db=0)

def wipe_redis():
    redis_client.flushdb()

def get_embedding_dimension(model: str, query: str = 'This is a test query.'):
    resp = ollama.embeddings(model=model, prompt=query)
    embedding = resp['embedding']
    return len(embedding)

def redis_hnsw_index(embedding_dimension: int):
    drop_index = f'FT.DROPINDEX index01 DD'
    try:
        redis_client.execute_command(drop_index)
    except redis.exceptions.ResponseError:
        pass
    # creates index 'index01' on hash with prefix 'doc:'
    # and schema with fields 'text' (TEXT) and 'embedding' (VECTOR)
    # with HNSW algorithm, 6 M connections, embedding_dimension dimensions, float32 type, and cosine distance metric
    create_index = f'''FT.CREATE index01
                    ON HASH PREFIX 1 doc: 
                    SCHEMA text TEXT embedding VECTOR 
                    HNSW 6 
                    DIM {embedding_dimension} 
                    TYPE FLOAT32 
                    DISTANCE_METRIC COSINE'''
    redis_client.execute_command(create_index)
    print('Created Redis HNSW index')

def get_embedding(model: str, query: str):
    resp = ollama.embeddings(model=model, prompt=query)
    return resp['embedding']

def find_pdfs_in_subfolders(root_dir):
    pdf_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for file in filenames:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(dirpath, file))
    return pdf_files

def convert_pdf_to_text(filepath):
    paginated_text = []
    try:
        with open(filepath, 'rb') as pdf_file:
            pdf_reader = PdfReader(pdf_file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                paginated_text.append(text)
    except Exception as e:
        print(f'Error reading PDF {filepath}: {e}')
    return paginated_text

def store_embedding(filename: str, page: str, chunk: str, embedding: list):
    key = f'doc::{filename}_page_{page}_chunk_{chunk}'
    redis_client.hset(key, mapping={'filename': filename, 'page': page, 'chunk': chunk,
                                    'embedding': np.array(embedding, dtype=np.float32).tobytes(),},)
    print(f'Stored embedding for: {chunk}')

def chunk_text_with_overlap(text, chunk_size=400, overlap=100):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i : i + chunk_size])
        chunks.append(chunk)
    return chunks

def process_pdfs(model, pdf_files, chunk_size, overlap):
    for pdf in pdf_files:
        print(f'Processing {pdf}')
        paginated_text = convert_pdf_to_text(pdf)
        if not paginated_text:
            print(f'No text found in {pdf}')
            continue
        for text in range(len(paginated_text)):
            contents = paginated_text[text]
            if contents == '':
                print(f'No text found in {pdf} on page {text}')
                continue
            chunks = chunk_text_with_overlap(contents, chunk_size, overlap)
            for chunk_index, chunk in enumerate(chunks):
                num_tokens = len(chunk.split())
                embedding = get_embedding(model, chunk)
                store_embedding(
                    filename=os.path.basename(pdf),
                    page=str(text),
                    chunk=chunk,
                    embedding=embedding,
                )
        print(f'Processed {pdf}')
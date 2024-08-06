import numpy as np
import faiss
import pickle

def load_embeddings(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data['embeddings'], data['chunks']

def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def search_similar_chunks(query_embedding, index, chunks, k=5):
    distances, indices = index.search(query_embedding.reshape(1, -1), k)
    results = [(chunks[i], distances[0][j]) for j, i in enumerate(indices[0])]
    return sorted(results, key=lambda x: x[1])

def truncate_text(text, word_limit=100):
    words = text.split()
    if len(words) <= word_limit:
        return text
    return ' '.join(words[:word_limit]) + '...'

# Example usage
if __name__ == "__main__":
    embeddings_file = "embeddings.pkl"
    embeddings, chunks = load_embeddings(embeddings_file)
    index = create_faiss_index(embeddings)
    
    query_embedding = embeddings[0]  # Using the first embedding as an example query
    results = search_similar_chunks(query_embedding, index, chunks)
    
    print("Similar chunks:")
    for chunk, distance in results:
        truncated_chunk = truncate_text(chunk)
        print(f"Distance: {distance:.4f}, Chunk: {truncated_chunk}")
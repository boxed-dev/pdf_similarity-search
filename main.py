from create_emdeddings import process_pdf
from process_emdeddings import load_embeddings, create_faiss_index, search_similar_chunks, truncate_text
from sentence_transformers import SentenceTransformer

def main():
    pdf_path = "your_pdf_file.pdf"
    embeddings_file = "embeddings.pkl"
    chunk_size = 1000  # Define chunk size here
    overlap = 200  # Define overlap here
    
    # Create embeddings if they don't exist
    try:
        embeddings, chunks = load_embeddings(embeddings_file)
        print("Loaded existing embeddings")
        print(f"Number of chunks: {len(chunks)}")
        print(f"Approximate chunk size: {chunk_size} words")
    except FileNotFoundError:
        print("Creating new embeddings")
        process_pdf(pdf_path, embeddings_file, chunk_size, overlap)
        embeddings, chunks = load_embeddings(embeddings_file)
    
    # Create FAISS index
    index = create_faiss_index(embeddings)
    
    # Load model for encoding queries
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Interactive query loop
    while True:
        query = input("Enter a query (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break
        
        query_embedding = model.encode([query])[0]
        results = search_similar_chunks(query_embedding, index, chunks)
        
        print("\nSimilar chunks (ranked by similarity):")
        for rank, (chunk, distance) in enumerate(results, 1):
            truncated_chunk = truncate_text(chunk)
            print(f"Rank: {rank}, Distance: {distance:.4f}")
            print(f"Text: {truncated_chunk}\n")
        
        if not results:
            print("No relevant results found. Try a different query.")
        print()

if __name__ == "__main__":
    main()
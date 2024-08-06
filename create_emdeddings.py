import PyPDF2
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text() + '\n'
    return text

def create_chunks(text, chunk_size=1000, overlap=200):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def create_embeddings(chunks, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks)
    return embeddings

def save_embeddings(embeddings, chunks, output_file):
    data = {
        'embeddings': embeddings,
        'chunks': chunks
    }
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)

def process_pdf(pdf_path, output_file, chunk_size=1000, overlap=200):
    text = extract_text_from_pdf(pdf_path)
    chunks = create_chunks(text, chunk_size, overlap)
    embeddings = create_embeddings(chunks)
    save_embeddings(embeddings, chunks, output_file)
    print(f"Embeddings saved to {output_file}")
    print(f"Number of chunks: {len(chunks)}")
    print(f"Approximate chunk size: {chunk_size} words")

if __name__ == "__main__":
    pdf_path = "your_pdf_file.pdf"
    output_file = "embeddings.pkl"
    process_pdf(pdf_path, output_file)
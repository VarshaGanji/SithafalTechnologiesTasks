import pdfplumber
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ''.join(page.extract_text() for page in pdf.pages)
    return text

# Chunk the text into smaller parts
def chunk_text(text, chunk_size=500):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# Example usage
pdf_path = "Sample_text.pdf"
text = extract_text_from_pdf(pdf_path)
chunks = chunk_text(text)
print(f"Number of Chunks: {len(chunks)}")
# Load a pre-trained Sentence Transformer
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for the text chunks
embeddings = model.encode(chunks)
print(f"Generated {len(embeddings)} embeddings of size {embeddings.shape[1]}")

def query_index(query, model, index, chunks, top_k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), k=top_k)
    return [chunks[i] for i in indices[0]]

# Example query
query = "Why are charts and graphs used?"
relevant_chunks = query_index(query, model, index, chunks)


# Load a lightweight LLM
generator = pipeline("text2text-generation", model="google/flan-t5-large")

# Generate a response using retrieved chunks
context = " ".join(relevant_chunks)
response = generator(f"Answer the query based on this context: {context}\n\nQuery: {query}")
print("Generated Response:", response[0]['generated_text'])


# RAG Implementation: NASCOP ART Guidelines
# Using Llama 3.1 Instruct (8B) and All Mini L6 v2

import os
import base64
import tempfile
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from typing import List, Dict, Any, Tuple

# PDF Processing
import PyPDF2
import fitz  # PyMuPDF
from PIL import Image
import io

# Text Processing
import re
import nltk
from nltk.tokenize import sent_tokenize

# Embedding and Vector Database
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import faiss

# LLM Integration
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from huggingface_hub import login

# Web Interface
import gradio as gr

# Download necessary NLTK data
nltk.download('punkt')

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configuration
class Config:
    # File paths
    pdf_path = "NASCOP_ART_guidelines.pdf"
    cache_dir = "cache"
    faiss_index_path = "faiss_index.bin"
    chunks_path = "chunks.csv"
    
    # Chunking parameters
    chunk_size = 512
    chunk_overlap = 50
    
    # Embedding model
    embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    
    # LLM model
    llm_model_name = "meta-llama/Llama-2-8b-hf"  # Will use this for non-commercial demo
    llama_model_name = "meta-llama/Llama-3.1-8B-Instruct"  # Actual model to use in production
    
    # Retrieval parameters
    top_k = 5
    
    # HuggingFace token (needed for some models)
    hf_token = ""  # Add your token here if needed

config = Config()

# Create cache directory if it doesn't exist
os.makedirs(config.cache_dir, exist_ok=True)

# Function to check if we need to process the PDF again
def check_cached_data():
    if os.path.exists(os.path.join(config.cache_dir, config.chunks_path)) and \
       os.path.exists(os.path.join(config.cache_dir, config.faiss_index_path)):
        return True
    return False

# 1. PDF Text Extraction
def extract_text_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract text and metadata from a PDF file.
    Returns a list of dictionaries, each containing the text and metadata for a page.
    """
    print(f"Extracting text from {pdf_path}...")
    
    # Open the PDF file
    doc = fitz.open(pdf_path)
    
    results = []
    
    # Process each page
    for page_num in tqdm(range(len(doc)), desc="Processing pages"):
        # Get the page
        page = doc[page_num]
        
        # Extract text
        text = page.get_text()
        
        # Extract images (optional, for future use)
        images = []
        image_list = page.get_images(full=True)
        
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            
            # Convert image bytes to a base64 string for storage or display
            image_b64 = base64.b64encode(image_bytes).decode('utf-8')
            images.append({
                'index': img_index,
                'base64': image_b64
            })
        
        # Store page data
        page_data = {
            'page_num': page_num + 1,
            'text': text,
            'images': images
        }
        
        results.append(page_data)
    
    doc.close()
    return results

# 2. Text Chunking
def clean_text(text: str) -> str:
    """Clean text by removing extra whitespace and special characters."""
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    # Remove any non-printable characters
    text = re.sub(r'[^\x20-\x7E\n]', '', text)
    return text.strip()

def chunk_text(pages: List[Dict[str, Any]], chunk_size: int = 512, overlap: int = 50) -> List[Dict[str, Any]]:
    """
    Split the text into chunks with overlap.
    Returns a list of dictionaries, each containing a chunk of text and its metadata.
    """
    print("Chunking text...")
    chunks = []
    
    for page_data in tqdm(pages, desc="Chunking pages"):
        page_num = page_data['page_num']
        text = clean_text(page_data['text'])
        
        # Skip if the page is empty
        if not text:
            continue
        
        # Split text into sentences to avoid cutting in the middle of a sentence
        sentences = sent_tokenize(text)
        
        current_chunk = ""
        
        for sentence in sentences:
            # If adding the next sentence would exceed chunk_size, save the current chunk and start a new one
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                chunks.append({
                    'text': current_chunk,
                    'page_num': page_num,
                    'chunk_id': len(chunks)
                })
                
                # Start new chunk with overlap from the previous chunk
                # Get the last few tokens for overlap
                words = current_chunk.split()
                if len(words) > overlap:
                    current_chunk = " ".join(words[-overlap:]) + " "
                else:
                    current_chunk = ""
            
            current_chunk += sentence + " "
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append({
                'text': current_chunk.strip(),
                'page_num': page_num,
                'chunk_id': len(chunks)
            })
    
    return chunks

# Save chunks to CSV for future use
def save_chunks_to_csv(chunks: List[Dict[str, Any]], output_path: str):
    """Save chunks to a CSV file."""
    df = pd.DataFrame(chunks)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(chunks)} chunks to {output_path}")
    return df

# Load chunks from CSV
def load_chunks_from_csv(input_path: str) -> pd.DataFrame:
    """Load chunks from a CSV file."""
    df = pd.DataFrame()
    if os.path.exists(input_path):
        df = pd.read_csv(input_path)
        print(f"Loaded {len(df)} chunks from {input_path}")
    return df

# 3. Generate Embeddings
class EmbeddingModel:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize the embedding model."""
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_size = self.model.get_sentence_embedding_dimension()
        print(f"Embedding size: {self.embedding_size}")
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(batch_texts)
            embeddings.append(batch_embeddings)
        
        return np.vstack(embeddings)

# 4. Vector Database with FAISS
class VectorDB:
    def __init__(self, embedding_size: int):
        """Initialize the vector database."""
        self.index = faiss.IndexFlatL2(embedding_size)
    
    def add_embeddings(self, embeddings: np.ndarray):
        """Add embeddings to the vector database."""
        # Ensure the embeddings are normalized for cosine similarity search
        faiss.normalize_L2(embeddings)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])  # Inner product for cosine similarity
        self.index.add(embeddings)
        print(f"Added {embeddings.shape[0]} embeddings to the vector database.")
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for the k most similar embeddings.
        Returns distances and indices.
        """
        # Ensure the query embedding is normalized
        faiss.normalize_L2(query_embedding)
        
        # Search for the k most similar embeddings
        scores, indices = self.index.search(query_embedding, k)
        return scores, indices
    
    def save_index(self, path: str):
        """Save the FAISS index to disk."""
        faiss.write_index(self.index, path)
        print(f"Saved FAISS index to {path}")
    
    @classmethod
    def load_index(cls, path: str, embedding_size: int):
        """Load the FAISS index from disk."""
        instance = cls(embedding_size)
        if os.path.exists(path):
            instance.index = faiss.read_index(path)
            print(f"Loaded FAISS index from {path}")
        return instance

# 5. RAG System
class RAGSystem:
    def __init__(self, config: Config):
        """Initialize the RAG system."""
        self.config = config
        self.setup()
    
    def setup(self):
        """Set up the RAG system."""
        # Check if we have cached data
        if check_cached_data():
            # Load chunks
            chunks_path = os.path.join(self.config.cache_dir, self.config.chunks_path)
            self.chunks_df = load_chunks_from_csv(chunks_path)
            
            # Initialize embedding model
            self.embedding_model = EmbeddingModel(self.config.embedding_model_name)
            
            # Load vector database
            faiss_path = os.path.join(self.config.cache_dir, self.config.faiss_index_path)
            self.vector_db = VectorDB.load_index(faiss_path, self.embedding_model.embedding_size)
            
        else:
            # Process PDF
            self.pages = extract_text_from_pdf(self.config.pdf_path)
            
            # Chunk text
            self.chunks = chunk_text(self.pages, self.config.chunk_size, self.config.chunk_overlap)
            
            # Save chunks
            chunks_path = os.path.join(self.config.cache_dir, self.config.chunks_path)
            self.chunks_df = save_chunks_to_csv(self.chunks, chunks_path)
            
            # Generate embeddings
            self.embedding_model = EmbeddingModel(self.config.embedding_model_name)
            self.embeddings = self.embedding_model.generate_embeddings(self.chunks_df['text'].tolist())
            
            # Create vector database
            self.vector_db = VectorDB(self.embedding_model.embedding_size)
            self.vector_db.add_embeddings(self.embeddings)
            
            # Save vector database
            faiss_path = os.path.join(self.config.cache_dir, self.config.faiss_index_path)
            self.vector_db.save_index(faiss_path)
        
        # Load LLM
        try:
            # Try to load the Llama 3.1 model first
            self.llm_tokenizer = AutoTokenizer.from_pretrained(
                self.config.llama_model_name, 
                use_auth_token=self.config.hf_token
            )
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                self.config.llama_model_name, 
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                use_auth_token=self.config.hf_token
            )
            self.llm_model_name = self.config.llama_model_name
            print(f"Loaded LLM: {self.config.llama_model_name}")
        except Exception as e:
            print(f"Error loading Llama 3.1: {e}")
            print(f"Falling back to {self.config.llm_model_name}")
            
            # Fall back to Llama 2
            self.llm_tokenizer = AutoTokenizer.from_pretrained(
                self.config.llm_model_name, 
                use_auth_token=self.config.hf_token
            )
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                self.config.llm_model_name, 
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                use_auth_token=self.config.hf_token
            )
            self.llm_model_name = self.config.llm_model_name
    
    def retrieve(self, query: str, k: int = None) -> List[Dict[str, Any]]:
        """Retrieve the k most relevant chunks for a query."""
        if k is None:
            k = self.config.top_k
        
        # Generate embedding for the query
        query_embedding = self.embedding_model.generate_embeddings([query])
        
        # Search for the most similar chunks
        scores, indices = self.vector_db.search(query_embedding, k)
        
        # Get the chunks
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx >= 0 and idx < len(self.chunks_df):  # Check if index is valid
                chunk = self.chunks_df.iloc[idx].to_dict()
                chunk['score'] = float(score)
                chunk['rank'] = i + 1
                results.append(chunk)
        
        return results
    
    def generate(self, query: str, context: str, max_tokens: int = 512) -> str:
        """Generate a response for a query using the LLM."""
        # Construct the prompt based on the model type
        if "llama-3" in self.llm_model_name.lower():
            # Llama 3.1 Instruct prompt format
            prompt = f"""<|begin_of_text|><|user|>
I'm looking for information from the NASCOP ART guidelines. Here's the relevant context:

{context}

Based on this context, please answer the following question:
{query}<|end_of_text|>
<|assistant|>"""
        else:
            # Llama 2 Chat prompt format
            prompt = f"""<s>[INST] <<SYS>>
You are an AI assistant that helps with questions about NASCOP ART guidelines. 
Answer questions based solely on the provided context.
Be concise, accurate, and helpful.
<</SYS>>

I'm looking for information from the NASCOP ART guidelines. Here's the relevant context:

{context}

Based on this context, please answer the following question:
{query} [/INST]
"""
        
        # Generate response
        inputs = self.llm_tokenizer(prompt, return_tensors="pt").to(device)
        
        generation_config = {
            "max_new_tokens": max_tokens,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
            "pad_token_id": self.llm_tokenizer.eos_token_id
        }
        
        with torch.no_grad():
            outputs = self.llm_model.generate(**inputs, **generation_config)
        
        response = self.llm_tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return response.strip()
    
    def query(self, query: str, k: int = None, max_tokens: int = 512) -> Dict[str, Any]:
        """
        Process a query end-to-end:
        1. Retrieve relevant chunks
        2. Generate a response
        3. Return the response and the retrieved chunks
        """
        # Retrieve relevant chunks
        chunks = self.retrieve(query, k)
        
        if not chunks:
            return {
                "answer": "I couldn't find any relevant information in the NASCOP ART guidelines.",
                "chunks": [],
                "sources": []
            }
        
        # Construct context from chunks
        context = "\n\n".join([f"Page {chunk['page_num']}: {chunk['text']}" for chunk in chunks])
        
        # Generate response
        answer = self.generate(query, context, max_tokens)
        
        # Construct sources
        sources = [f"Page {chunk['page_num']}" for chunk in chunks]
        
        return {
            "answer": answer,
            "chunks": chunks,
            "sources": sources
        }

# 6. Gradio Web Interface
def create_web_interface(rag_system: RAGSystem):
    """Create a web interface for the RAG system."""
    with gr.Blocks(title="NASCOP ART Guidelines RAG") as demo:
        gr.Markdown("# NASCOP ART Guidelines RAG System")
        gr.Markdown("Ask questions about NASCOP ART guidelines and get answers powered by Llama 3.1 and All Mini L6 v2.")
        
        with gr.Row():
            with gr.Column(scale=4):
                query_input = gr.Textbox(
                    label="Question",
                    placeholder="Ask a question about NASCOP ART guidelines...",
                    lines=2
                )
                
                with gr.Row():
                    submit_btn = gr.Button("Submit", variant="primary")
                    clear_btn = gr.Button("Clear")
                
                answer_output = gr.Textbox(
                    label="Answer",
                    lines=10,
                    interactive=False
                )
                
            with gr.Column(scale=2):
                gr.Markdown("### Retrieved Passages")
                passages_output = gr.Dataframe(
                    headers=["Page", "Passage", "Score"],
                    interactive=False
                )
        
        def process_query(query):
            if not query.strip():
                return "Please enter a question.", []
            
            result = rag_system.query(query)
            
            passages = []
            for chunk in result["chunks"]:
                # Truncate the text for display
                text = chunk["text"]
                if len(text) > 200:
                    text = text[:200] + "..."
                
                passages.append([
                    f"Page {chunk['page_num']}",
                    text,
                    f"{chunk['score']:.4f}"
                ])
            
            return result["answer"], passages
        
        def clear_outputs():
            return "", [], None
        
        submit_btn.click(
            process_query,
            inputs=[query_input],
            outputs=[answer_output, passages_output]
        )
        
        clear_btn.click(
            clear_outputs,
            inputs=[],
            outputs=[query_input, answer_output, passages_output]
        )
    
    return demo

# Main function to run the RAG system
def main():
    # Initialize the RAG system
    rag_system = RAGSystem(config)
    
    # Create and launch the web interface
    demo = create_web_interface(rag_system)
    demo.launch(share=True)

# Example usage in a notebook
if __name__ == "__main__":
    # If running in a notebook, you can use these examples instead of launching the web interface
    
    # Initialize the RAG system
    rag_system = RAGSystem(config)
    
    # Example queries
    example_queries = [
        "What are the first-line ART regimens recommended by NASCOP?",
        "How should I manage a patient with HIV and TB co-infection?",
        "What are the criteria for switching ART regimens?",
        "How should viral load monitoring be done?",
        "What are the recommendations for children living with HIV?"
    ]
    
    # Process each query
    for query in example_queries:
        print(f"\n\nQuery: {query}")
        result = rag_system.query(query)
        print(f"\nAnswer: {result['answer']}")
        print("\nSources:")
        for source in result['sources']:
            print(f"- {source}")

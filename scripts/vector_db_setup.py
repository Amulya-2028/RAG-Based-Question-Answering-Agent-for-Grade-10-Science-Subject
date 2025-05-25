import faiss
import pickle
import numpy as np

def build_faiss_index(embedding_file, index_file):
    # Load the embeddings and corresponding texts
    with open(embedding_file, 'rb') as f:
        texts, embeddings = pickle.load(f)
    
    # Convert embeddings to a numpy array (FAISS expects numpy arrays)
    embeddings = np.array(embeddings, dtype=np.float32)
    
    # Create a FAISS index for the embeddings
    dimension = embeddings.shape[1]  # The embedding dimension (e.g., 384 for MiniLM)
    index = faiss.IndexFlatL2(dimension)  # Using L2 distance for similarity search
    
    # Add the embeddings to the FAISS index
    index.add(embeddings)
    
    # Save the FAISS index to a file
    faiss.write_index(index, index_file)
    print(f"FAISS index saved to {index_file}")

if __name__ == "__main__":
    # Define the path for embeddings and FAISS index output
    embedding_file = r"path_of_index_texts.pkl"  # Path to your embeddings file
    index_file = r"path_of_faiss_index.bin"  # Path to save the FAISS index
    
    # Create the FAISS index
    build_faiss_index(embedding_file, index_file)

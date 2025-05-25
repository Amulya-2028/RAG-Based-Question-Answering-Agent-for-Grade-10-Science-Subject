import pickle
from sentence_transformers import SentenceTransformer

def generate_embeddings(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        texts = f.readlines()

    model = SentenceTransformer('all-MiniLM-L6-v2')  # Use a lightweight model for embedding
    embeddings = model.encode(texts, show_progress_bar=True)

    with open(output_file, 'wb') as f:
        pickle.dump((texts, embeddings), f)
    print(f"Embeddings saved to {output_file}")

if __name__ == "__main__":  # Corrected this line
    input_file = r"Path_of_ALL_IN_ONE_cleaned.txt"
    output_file = r"Path_of_index_texts.pkl"
    generate_embeddings(input_file, output_file)

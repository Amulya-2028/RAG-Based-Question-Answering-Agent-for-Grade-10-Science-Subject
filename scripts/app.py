import faiss
import numpy as np
import warnings
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

# Suppress all warnings (including specific warnings from transformers)
warnings.filterwarnings("ignore", category=UserWarning, message=".generation.")
warnings.filterwarnings("ignore", category=UserWarning, message=".attention mask.")

# Function to load an existing FAISS index
def load_faiss_index(faiss_index_path, text_file_path):
    """
    Load an existing FAISS index and associated text data.
    """
    # Load FAISS index
    index = faiss.read_index(faiss_index_path)
    print(f"Loaded FAISS index from {faiss_index_path}.")

    # Load text file corresponding to the FAISS index
    with open(text_file_path, 'r', encoding='utf-8') as file:
        texts = file.readlines()
    
    print(f"Loaded {len(texts)} lines from {text_file_path}.")
    return index, texts

# Function to retrieve top-k contexts using FAISS
def retrieve_context_faiss(question, index, texts, embedding_model, top_k=3):
    """
    Retrieves the top-k relevant contexts for the input question using FAISS.
    """
    # Generate the embedding for the question
    question_embedding = embedding_model.encode([question], convert_to_tensor=True).cpu().numpy()

    # Search the FAISS index for the nearest neighbors
    distances, indices = index.search(question_embedding, top_k)

    # Get the top-k relevant contexts
    relevant_contexts = [texts[idx] for idx in indices[0]]

    return relevant_contexts

# Load tokenizer and LLM
llama_model_name = "meta-llama/Llama-3.2-1B"  # Replace with your Llama model or any compatible model
tokenizer = AutoTokenizer.from_pretrained(llama_model_name)

# Manually set the pad_token to the eos_token
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(llama_model_name)

# Load embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Pretrained sentence transformer for embeddings

# Function to interact with the LLM
def generate_response_with_llm(question, context, tokenizer, model):
    """
    Generates a response from the LLM using the question and retrieved context.
    """
    # Combine the question and context into a single prompt
    prompt = f"Context: {' '.join(context)}\n\nQuestion: {question}\n\nAnswer:"

    # Tokenize the input
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024)

    # Generate a response
    outputs = model.generate(
        inputs.input_ids, 
        max_length=300,  # Increase max length to get a more complete answer
        num_return_sequences=1, 
        no_repeat_ngram_size=2, 
        early_stopping=True, 
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract the answer from the response
    if "Answer:" in response:
        answer_start = response.find("Answer:") + len("Answer:")
        answer = response[answer_start:].strip()
    else:
        # If no "Answer:" found, return the whole response or the relevant part
        answer = response.strip()

    return answer

# Streamlit App
def main():
    st.title("RAG Educational Q&A Tool")

    # Load the FAISS index and texts when the app starts
    faiss_index_path = r"path_to_faiss_index.bin"  # Path to your FAISS index file
    text_file_path = r"path_to_cleaned_t1.txt"  # Path to your text file

    index, texts = load_faiss_index(faiss_index_path, text_file_path)

    # User input for the doubt/question
    question = st.text_input("Ask your question:")

    if question:
        # Retrieve relevant context for the question
        context = retrieve_context_faiss(question, index, texts, embedding_model)

        # Generate the model's response based on the context
        response = generate_response_with_llm(question, context, tokenizer, model)

        # Display the results
        st.subheader("Relevant Context")
        for context_piece in context:
            st.write(context_piece.strip())  # Display each relevant context piece

         # Display the model's generated answer

        st.subheader("Final Answer")
        st.write(response)  # Display the final answer (same as generated response for now)

# Run the Streamlit app
if __name__ == "_main_":
    main()
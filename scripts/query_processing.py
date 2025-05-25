import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

# Function to load FAISS index and texts
def load_faiss_index(faiss_index_path, text_file_path):
    index = faiss.read_index(faiss_index_path)
    with open(text_file_path, 'r', encoding='utf-8') as file:
        texts = file.readlines()
    return index, texts

# Function to retrieve top-k contexts using FAISS
def retrieve_context_faiss(question, index, texts, embedding_model, top_k=3):
    question_embedding = embedding_model.encode([question], convert_to_tensor=True).cpu().numpy()
    _, indices = index.search(question_embedding, top_k)
    return [texts[idx].strip() for idx in indices[0] if idx < len(texts)]

# Function to generate a detailed response using LLM
def generate_response_with_llm(question, context, tokenizer, model):
    prompt = (
        f"Context: {' '.join(context)}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.get('attention_mask', None),
        max_length=200,
        num_return_sequences=1,
        temperature=0.7,
        do_sample=True
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.strip()

# Main execution
if __name__ == "_main_":
    faiss_index_path = r"path _of_faiss_index.bin"
    text_file_path = r"path_of_ALL_IN_ONE.txt"

    # Load FAISS index and text data
    index, texts = load_faiss_index(faiss_index_path, text_file_path)

    # Load SentenceTransformer embedding model
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Load LLM tokenizer and model
    llama_model_name = "meta-llama/Llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(llama_model_name)
    model = AutoModelForCausalLM.from_pretrained(llama_model_name)

    # Set pad token
    tokenizer.pad_token = tokenizer.eos_token

    # Input question
    question = input("Enter your question: ")
  

    # Retrieve relevant context
    context = retrieve_context_faiss(question, index, texts, embedding_model)

    # Generate and display the answer
    if not context:
        print("No relevant context found.")
    else:
        answer = generate_response_with_llm(question, context, tokenizer, model)
        print(answer)
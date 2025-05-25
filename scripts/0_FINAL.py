import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from transformers import pipeline


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
        max_length=250,
        num_return_sequences=1,
        temperature=0.7,
        do_sample=True
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.strip()

# Main execution
if __name__ == "__main__":
    faiss_index_path = r"path_of_faiss_index.bin"
    text_file_path = r"path_of_ALL_IN_ONE_cleaned.txt"

    # Load FAISS index and text data
    index, texts = load_faiss_index(faiss_index_path, text_file_path)

    # Load SentenceTransformer embedding model
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Load LLM tokenizer and model
    llama_model_name = "meta-llama/Llama-3.2-1B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(llama_model_name)
    model = AutoModelForCausalLM.from_pretrained(llama_model_name)

    # Set pad token
    tokenizer.pad_token = tokenizer.eos_token

    from transformers import pipeline

    llama_pipe = pipeline(
    "text-generation",
    model="meta-llama/Llama-3.2-1B-Instruct",
    torch_dtype="auto",  # Automatically use appropriate precision
    device_map="auto"    # Use GPU if available, otherwise CPU
)

    # Input question
    question = input("Enter your question: ")

    input_question = question.strip() + " in 100 words or less."

# Step 1: Use LLaMA for initial text generation
    llama_output = llama_pipe(
    input_question,
    max_length=250,          # Limit the output length
    temperature=0.7,         # Control randomness
    top_k=50,                # Use top-k sampling
    top_p=0.9,               # Use nucleus sampling
    repetition_penalty=1.2   # Avoid repetition
    )

# Extract generated text
    generated_text = llama_output[0]["generated_text"]

# Display the initial output from LLaMA
    print("\n\nOutput from LLaMA:")
    print(generated_text)
    # Retrieve relevant context
    context = retrieve_context_faiss(question, index, texts, embedding_model)

    # Generate and display the answer
    if not context:
        print("No relevant context found.")
    else:
        answer = generate_response_with_llm(question, context, tokenizer, model)
        print(answer)


# Initialize the pipelines
# llama_pipe = pipeline(
#     "text-generation",
#     model="meta-llama/Llama-3.2-1B-Instruct",
#     torch_dtype="auto",  # Automatically use appropriate precision
#     device_map="auto"    # Use GPU if available, otherwise CPU
# )

# # bart_pipe = pipeline("summarization", model="facebook/bart-large-cnn")

# # Input question for LLaMA model
# # user_question = input("\n\nEnter your question: ")
# input_question = question.strip() + " in 100 words or less."

# # Step 1: Use LLaMA for initial text generation
# llama_output = llama_pipe(
#     input_question,
#     max_length=250,          # Limit the output length
#     temperature=0.7,         # Control randomness
#     top_k=50,                # Use top-k sampling
#     top_p=0.9,               # Use nucleus sampling
#     repetition_penalty=1.2   # Avoid repetition
# )

# # Extract generated text
# generated_text = llama_output[0]["generated_text"]

# # Display the initial output from LLaMA
# print("\n\nOutput from LLaMA:")
# print(generated_text)

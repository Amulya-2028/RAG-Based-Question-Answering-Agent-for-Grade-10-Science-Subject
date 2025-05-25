from transformers import AutoTokenizer, AutoModelForCausalLM

def load_llama_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

if __name__ == "__main__":
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    tokenizer, model = load_llama_model(model_name)
    print("Model loaded successfully.")

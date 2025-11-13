import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
import streamlit as st

st.title("Ask Hugging Face")
st.subheader("Ask your questions to the Qwen Model")

def load_model():
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-3B-Instruct",
        device_map="mps",  # Usar GPU de Apple Silicon
        dtype=torch.float16,  # Reduce uso de memoria
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    return model

def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-3B-Instruct"
    )
    return tokenizer

def load_generator(model, tokenizer):
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )
    return generator

def load_messages(text_input):
    return [{"role": "user", "content": text_input}]

text_input = st.text_input("Enter your question: ")
ask_button = st.button("Ask")

if ask_button:
    st.spinner("Thinking...")
    try:
        model = load_model()
        tokenizer = load_tokenizer()
        generator = load_generator(model, tokenizer)
        messages = load_messages(text_input)
        response = generator(messages)
        st.write(response[0]['generated_text'][1]['content'])
    except Exception as e:
        st.error(f"Error: Answer not found")
        st.error(e)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
import streamlit as st

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Qwen AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS STYLES
# ============================================
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .stTextInput > div > div > input {
        font-size: 1.1rem;
    }
    .response-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #2196F3;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================
# CACHE FOR OPTIMIZATION
# ============================================
@st.cache_resource
def load_model():
    """Load the model once and keep it in cache"""
    with st.spinner("🔄 Loading model... (first time only)"):
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-3B-Instruct",
            device_map="mps",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
    return model

@st.cache_resource
def load_tokenizer():
    """Load the tokenizer once"""
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-3B-Instruct"
    )
    return tokenizer

def load_generator(_model, _tokenizer):
    """Create the generation pipeline"""
    generator = pipeline(
        "text-generation",
        model=_model,
        tokenizer=_tokenizer
    )
    return generator

# ============================================
# SIDEBAR WITH SETTINGS
# ============================================
with st.sidebar:
    st.header("⚙️ Settings")
    
    max_length = st.slider(
        "Maximum response length",
        min_value=50,
        max_value=512,
        value=500,
        step=50,
        help="Controls how much text the model generates"
    )
    
    temperature = st.slider(
        "Temperature",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="Higher temperature = more creative responses"
    )
    
    top_p = st.slider(
        "Top P (nucleus sampling)",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Controls response diversity"
    )
    
    st.divider()
    
    st.markdown("### 📊 Model Information")
    st.info("""
    **Model:** Qwen2.5-3B-Instruct  
    **Parameters:** 3 billion  
    """)

# ============================================
# MAIN HEADER
# ============================================
st.markdown('<h1 class="main-header">🤖 Qwen AI Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Ask your questions to the Qwen2.5-3B-Instruct model</p>', unsafe_allow_html=True)

# ============================================
# EXAMPLE QUESTIONS
# ============================================
with st.expander("💡 Example questions"):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        - What is artificial intelligence?
        - Explain the theory of relativity
        - Write a poem about the ocean
        """)
    with col2:
        st.markdown("""
        - How does machine learning work?
        - Give me tips to code better
        - Translate "Hello" to 5 languages
        """)

# ============================================
# INPUT AREA
# ============================================
col1, col2 = st.columns([5, 1])

with col1:
    text_input = st.text_input(
        "Your question:",
        placeholder="Type your question here...",
        label_visibility="collapsed"
    )

with col2:
    ask_button = st.button("🚀 Ask", use_container_width=True, type="primary")

# ============================================
# QUESTION PROCESSING
# ============================================
if ask_button and text_input:
    try:
        # Load model and tokenizer (only first time thanks to cache)
        model = load_model()
        tokenizer = load_tokenizer()
        generator = load_generator(model, tokenizer)
        
        # Prepare messages
        messages = [{"role": "user", "content": text_input}]
        
        # Generate response with spinner
        with st.spinner("🤔 Thinking..."):
            response = generator(
                messages,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True
            )
        
        # Extract answer
        answer = response[0]['generated_text'][1]['content']
        
        # Display current response
        st.markdown("### 💬 Response:")
        st.markdown(f'<div class="response-box">{answer}</div>', unsafe_allow_html=True)
        
        # Button to copy response
        # st.code(answer, language=None)
        
    except Exception as e:
        st.error("❌ Error generating response")
        with st.expander("View error details"):
            st.exception(e)

elif ask_button and not text_input:
    st.warning("⚠️ Please write a question before submitting")

# ============================================
# FOOTER
# ============================================
st.divider()
st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        Developed with ❤️ using Streamlit and Hugging Face Transformers
    </div>
""", unsafe_allow_html=True)
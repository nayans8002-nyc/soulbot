import streamlit as st
import ollama
import base64
import os

# --- CONFIGURATION ---
MODEL_NAME = "gemma3:4b"
IMAGE_FILE_NAME = "background.png"

st.set_page_config(page_title="Mental Health Chatbot 🌿")

# ------------------ OLLAMA CLIENT INITIALIZATION AND CHECK ------------------
def initialize_ollama():
    """Tries to connect to the Ollama server and sets the connection status."""
    st.session_state.setdefault('ollama_available', False)
    try:
        # 🟢 SYNTAX FIX APPLIED HERE: Use ollama.list() instead of ollama.client.list()
        # ollama.list() works directly on the main module for a simple check.
        ollama.list()
        st.session_state.ollama_available = True
    except Exception as e:
        st.session_state.ollama_available = False
        st.sidebar.error(f"❌ Ollama Connection Error: {e}")
        st.sidebar.caption("Please ensure the Ollama server is running and the model is pulled.")

# Initialize or re-check connection
if 'ollama_available' not in st.session_state:
    initialize_ollama()

# ------------------ LOAD BACKGROUND IMAGE ------------------
def get_base64(file_path):
    """Encodes a local file to Base64 for CSS background use."""
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        st.sidebar.warning(f"⚠ Background image '{file_path}' not found in the current directory.")
        st.sidebar.caption(f"Current Working Directory: {os.getcwd()}")
        return None
    except Exception as e:
        st.sidebar.error(f"❌ Unexpected error while reading image: {e}")
        return None

bin_str = get_base64(IMAGE_FILE_NAME)

# ------------------ APPLY BACKGROUND IF AVAILABLE ------------------
if bin_str:
    st.markdown(f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background: url("data:image/png;base64,{bin_str}") !important;
            background-size: cover !important;
            background-position: center !important;
            background-repeat: no-repeat !important;
        }}

        .main > div {{
            background: rgba(255, 255, 255, 0.7);
            border-radius: 10px;
            padding: 10px;
        }}
        
        [data-testid="stHeader"] {{
            background: rgba(0,0,0,0);
        }}
        </style>
    """, unsafe_allow_html=True)

# ------------------ CONVERSATION STATE ------------------
st.session_state.setdefault('conversation_history', [])

# ------------------ OLLAMA RESPONSE FUNCTIONS ------------------
def generate_response(user_input):
    """Generates a contextual response using the chat history."""
    if not st.session_state.ollama_available:
        return "Sorry, I can't connect to my AI brain right now. Please ensure Ollama is running."
        
    st.session_state['conversation_history'].append({"role": "user", "content": user_input})
    try:
        response = ollama.chat(model=MODEL_NAME, messages=st.session_state['conversation_history'])
        ai_response = response['message']['content']
        st.session_state['conversation_history'].append({"role": "assistant", "content": ai_response})
        return ai_response
    except Exception as e:
        return f"An error occurred during response generation: {e}"

def generate_affirmation():
    if not st.session_state.ollama_available:
        return None
    prompt = "Provide a single, positive, encouraging affirmation for someone who is feeling stressed or overwhelmed."
    response = ollama.chat(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}])
    return response['message']['content']

def generate_meditation_guide():
    if not st.session_state.ollama_available:
        return None
    prompt = "Provide a brief, 2-minute guided meditation script to help someone focus on their breath and reduce stress."
    response = ollama.chat(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}])
    return response['message']['content']

# ------------------ UI TITLE AND INFO ------------------
st.title("🌱 Soul Bot- AI for your emotional wellness💌")

if not st.session_state.ollama_available:
    st.warning("🚨 Chatbot is Offline: Ensure the Ollama server is running and the model "
              f"{MODEL_NAME}** is pulled to begin the conversation.")

# ------------------ DISPLAY CHAT HISTORY ------------------
for msg in st.session_state['conversation_history']:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])

# ------------------ USER INPUT ------------------
user_message = st.chat_input("How can I help you today? (e.g., 'I feel anxious')")

if user_message:
    with st.chat_message("user"):
        st.markdown(user_message)
        
    with st.spinner("Thinking..."):
        ai_response = generate_response(user_message)
        
    with st.chat_message("assistant"):
        st.markdown(ai_response)

# ------------------ BUTTONS ------------------
st.markdown("---")
st.subheader("Quick Support Tools")
col1, col2 = st.columns(2)

with col1:
    if st.button("🌟 Give me a Positive Affirmation", use_container_width=True):
        if st.session_state.ollama_available:
            with st.spinner("Generating affirmation..."):
                affirmation = generate_affirmation()
            with st.chat_message("assistant", avatar="✨"):
                st.markdown(f"Positive Affirmation:\n\n{affirmation}")
        else:
            st.warning("Ollama server is not connected.")

with col2:
    if st.button("🧘‍♂ Give me a Guided Meditation", use_container_width=True):
        if st.session_state.ollama_available:
            with st.spinner("Preparing meditation guide..."):
                meditation_guide = generate_meditation_guide()
            with st.chat_message("assistant", avatar="🧘"):
                st.markdown(f"Guided Meditation:\n\n{meditation_guide}")
        else:
            st.warning("Ollama server is not connected.")

# ------------------ DISCLAIMER ------------------
st.sidebar.markdown("---")
st.sidebar.info(
    "Disclaimer: This chatbot is for informational and support purposes only. "
    "It is not a substitute for professional mental health care, diagnosis, or treatment. "
    "If you are in crisis, please contact emergency services or a mental health professional."
)
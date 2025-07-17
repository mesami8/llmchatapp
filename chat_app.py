import streamlit as st
import requests
import json

# --- Page Configuration ---
st.set_page_config(
    page_title="LLM Chat",
    page_icon="👽",
    layout="wide",
    
    initial_sidebar_state="auto",
)

# --- Constants ---
OLLAMA_API_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "llama3:8b"

# --- Helper Functions ---

def get_installed_models():
    """
    Fetches the list of locally installed Ollama models.
    Returns a list of model names or an empty list if it fails.
    """
    try:
        response = requests.get("http://localhost:11434/api/tags")
        response.raise_for_status()
        models_data = response.json()
        return [model['name'] for model in models_data.get('models', [])]
    except requests.exceptions.RequestException as e:
        st.sidebar.error(f"Error fetching models: {e}")
        st.sidebar.warning("Could not connect to Ollama. Please ensure Ollama is running.")
        return []

def generate_response(model, messages):
    """
    Sends a request to the Ollama /api/chat endpoint and streams the response.
    Yields each content token as it is received.
    """
    try:
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
        }

        response = requests.post(
            OLLAMA_API_URL,
            json=payload,
            stream=True,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()

        full_response = ""
        for line in response.iter_lines():
            if line:
                try:
                    json_line = json.loads(line.decode('utf-8'))
                    if 'message' in json_line and 'content' in json_line['message']:
                        token = json_line['message']['content']
                        full_response += token
                        yield token
                except json.JSONDecodeError:
                    st.error("Failed to parse a response line from the model.")
                    continue

        # Save assistant response to session
        st.session_state.messages.append({"role": "assistant", "content": full_response})

    except requests.exceptions.ConnectionError:
        st.error("Connection Error: Could not connect to the Ollama API.")
    except requests.exceptions.RequestException as e:
        st.error(f"An API request error occurred: {e}")

# --- Main App ---
def main():
    # --- Session State Initialization ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "selected_history" in st.session_state:
        selected_index = st.session_state.selected_history
        st.session_state.messages = st.session_state.messages[selected_index:selected_index+2]
        del st.session_state["selected_history"]

    # --- Sidebar ---
    with st.sidebar:
        st.title("👽 LLM Chat")
        st.markdown("---")

        if st.button("Task no 1 ✔"):
            st.balloons()

        # Model Selection
        installed_models = get_installed_models()
        if installed_models:
            selected_model = st.selectbox(
                "Choose your model:",
                options=installed_models,
                index=installed_models.index(DEFAULT_MODEL) if DEFAULT_MODEL in installed_models else 0,
                help="Select the Ollama model to chat with."
            )
        else:
            selected_model = st.text_input("Enter model name:", value=DEFAULT_MODEL)
            st.markdown("_(Could not fetch model list. Enter model name manually)_")

        st.markdown("---")

        # Chat History
        st.markdown("### 📜 Chat History")
        if st.session_state.messages:
            for i, msg in enumerate(st.session_state.messages):
                if msg["role"] == "user":
                    label = msg["content"][:30] + ("..." if len(msg["content"]) > 30 else "")
                    if st.button(f" {label}", key=f"history_{i}"):
                        st.session_state.selected_history = i
                        st.rerun()
        else:
            st.caption("No conversation yet.")

        st.markdown("---")

        # Reset Conversation
        if st.button("🔄 Reset Conversation", use_container_width=True):
            st.session_state.messages = []
            st.success("Conversation has been reset.")
            st.rerun()

        st.markdown("---")
        st.info(
            "**How to use:**\n"
            "1. Make sure Ollama is running on your computer.\n"
            "2. Select a model from the dropdown.\n"
            "3. Type your message and press Enter."
        )

    # --- Main Chat Display ---
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --- Chat Input ---
    if prompt := st.chat_input("What would you like to ask?"):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response_generator = generate_response(
                selected_model,
                st.session_state.messages
            )
            st.write_stream(response_generator)

# --- Custom CSS ---
st.markdown("""
<style>
    [id="stBadge"] {
        font-size: 1.25rem;
        padding: 0.5rem 0.75rem;
        border-radius: 0.75rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Run App ---
if __name__ == "__main__":
    main()

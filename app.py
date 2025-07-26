import streamlit as st
import requests
import json
import os
from datetime import datetime
from pymongo import MongoClient
from bson.objectid import ObjectId
import hashlib

# --- Page Configuration ---
st.set_page_config(
    page_title="LLM Chat",
    page_icon="👽",
    layout="wide",
    initial_sidebar_state="auto",
)

# --- Constants ---
OLLAMA_API_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "llama3.2:1b"

# --- MongoDB Configuration ---
@st.cache_resource
def init_mongodb():
    """Initialize MongoDB connection"""
    try:
        # Get MongoDB URI from Streamlit secrets or environment variable
        try:
            mongo_uri = st.secrets.get("MONGODB_URI")
        except:
            mongo_uri = os.getenv("MONGODB_URI")
        
        if not mongo_uri:
            st.error("MongoDB URI not found. Please set MONGODB_URI in Streamlit secrets or environment variables.")
            return None, None
        
        client = MongoClient(mongo_uri)
        # Test connection
        client.admin.command('ping')
        db = client.get_database("llm_chat_app")
        return client, db
    except Exception as e:
        st.error(f"Failed to connect to MongoDB: {e}")
        return None, None

# --- Database Helper Functions ---
def get_user_id():
    """Generate a unique user ID based on session"""
    if "user_id" not in st.session_state:
        # Create a simple user identifier (in production, use proper authentication)
        session_info = f"{st.session_state.get('session_id', 'default')}"
        st.session_state.user_id = hashlib.md5(session_info.encode()).hexdigest()[:16]
    return st.session_state.user_id

def save_conversation_to_db(db, messages, model_used):
    """Save conversation to MongoDB"""
    if db is None:
        return None
    
    try:
        conversation = {
            "user_id": get_user_id(),
            "messages": messages,
            "model_used": model_used,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        result = db.conversations.insert_one(conversation)
        return str(result.inserted_id)
    except Exception as e:
        st.error(f"Error saving conversation: {e}")
        return None

def update_conversation_in_db(db, conversation_id, messages):
    """Update existing conversation in MongoDB"""
    if db is None or not conversation_id:
        return False
    
    try:
        db.conversations.update_one(
            {"_id": ObjectId(conversation_id)},
            {
                "$set": {
                    "messages": messages,
                    "updated_at": datetime.utcnow()
                }
            }
        )
        return True
    except Exception as e:
        st.error(f"Error updating conversation: {e}")
        return False

def load_conversation_history(db, limit=20):
    """Load conversation history from MongoDB"""
    if db is None:
        return []
    
    try:
        conversations = db.conversations.find(
            {"user_id": get_user_id()},
            {"_id": 1, "messages": 1, "model_used": 1, "created_at": 1}
        ).sort("created_at", -1).limit(limit)
        
        return list(conversations)
    except Exception as e:
        st.error(f"Error loading conversation history: {e}")
        return []

def load_conversation_by_id(db, conversation_id):
    """Load a specific conversation by ID"""
    if db is None:
        return None
    
    try:
        conversation = db.conversations.find_one(
            {"_id": ObjectId(conversation_id), "user_id": get_user_id()}
        )
        return conversation
    except Exception as e:
        st.error(f"Error loading conversation: {e}")
        return None

def delete_conversation(db, conversation_id):
    """Delete a conversation"""
    if db is None:
        return False
    
    try:
        result = db.conversations.delete_one(
            {"_id": ObjectId(conversation_id), "user_id": get_user_id()}
        )
        return result.deleted_count > 0
    except Exception as e:
        st.error(f"Error deleting conversation: {e}")
        return False

# --- Helper Functions ---
def get_installed_models():
    """
    Fetches the list of locally installed Ollama models.
    Returns a list of model names or an empty list if it fails.
    """
    try:
        # For deployment, you might want to use a remote Ollama instance
        try:
            ollama_url = st.secrets.get("OLLAMA_URL", "http://localhost:11434")
        except:
            ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        
        response = requests.get(f"{ollama_url}/api/tags", timeout=5)
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
        try:
            ollama_url = st.secrets.get("OLLAMA_URL", "http://localhost:11434")
        except:
            ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        
        api_url = f"{ollama_url}/api/chat"
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
        }

        response = requests.post(
            api_url,
            json=payload,
            stream=True,
            headers={"Content-Type": "application/json"},
            timeout=30
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
    # Initialize MongoDB
    client, db = init_mongodb()
    
    # --- Session State Initialization ---
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "current_conversation_id" not in st.session_state:
        st.session_state.current_conversation_id = None
    
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = DEFAULT_MODEL

    # Handle conversation loading
    if "load_conversation" in st.session_state:
        conversation_id = st.session_state.load_conversation
        conversation = load_conversation_by_id(db, conversation_id)
        if conversation:
            st.session_state.messages = conversation["messages"]
            st.session_state.current_conversation_id = conversation_id
            st.session_state.selected_model = conversation.get("model_used", DEFAULT_MODEL)
        del st.session_state["load_conversation"]
        st.rerun()

    # --- Sidebar ---
    with st.sidebar:
        st.title("👽 LLM Chat")
        st.markdown("---")

        if st.button("Task no 1 ✔"):
            st.balloons()

        # Model Selection
        installed_models = get_installed_models()
        if installed_models:
            model_index = 0
            if st.session_state.selected_model in installed_models:
                model_index = installed_models.index(st.session_state.selected_model)
            
            selected_model = st.selectbox(
                "Choose your model:",
                options=installed_models,
                index=model_index,
                help="Select the Ollama model to chat with."
            )
            st.session_state.selected_model = selected_model
        else:
            selected_model = st.text_input(
                "Enter model name:", 
                value=st.session_state.selected_model
            )
            st.session_state.selected_model = selected_model
            st.markdown("_(Could not fetch model list. Enter model name manually)_")

        st.markdown("---")

        # Save Current Conversation
        if st.session_state.messages and db is not None:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("💾 Save Chat", use_container_width=True):
                    if st.session_state.current_conversation_id:
                        success = update_conversation_in_db(
                            db, 
                            st.session_state.current_conversation_id, 
                            st.session_state.messages
                        )
                        if success:
                            st.success("Conversation updated!")
                    else:
                        conversation_id = save_conversation_to_db(
                            db, 
                            st.session_state.messages, 
                            st.session_state.selected_model
                        )
                        if conversation_id:
                            st.session_state.current_conversation_id = conversation_id
                            st.success("Conversation saved!")
            
            with col2:
                if st.button("🗑️ Delete", use_container_width=True):
                    if st.session_state.current_conversation_id:
                        success = delete_conversation(db, st.session_state.current_conversation_id)
                        if success:
                            st.session_state.messages = []
                            st.session_state.current_conversation_id = None
                            st.success("Conversation deleted!")
                            st.rerun()

        st.markdown("---")

        # Chat History
        st.markdown("### 📜 Chat History")
        if db is not None:
            conversations = load_conversation_history(db)
            if conversations:
                for conv in conversations:
                    # Get first user message as preview
                    preview = "New Conversation"
                    for msg in conv.get("messages", []):
                        if msg["role"] == "user":
                            preview = msg["content"][:40] + ("..." if len(msg["content"]) > 40 else "")
                            break
                    
                    # Format date
                    date_str = conv["created_at"].strftime("%m/%d %H:%M")
                    
                    if st.button(
                        f"💬 {preview}\n📅 {date_str}", 
                        key=f"load_{conv['_id']}",
                        use_container_width=True
                    ):
                        st.session_state.load_conversation = str(conv["_id"])
                        st.rerun()
            else:
                st.caption("No saved conversations yet.")
        else:
            st.caption("Database not connected.")

        st.markdown("---")

        # Reset Conversation
        if st.button("🔄 New Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.current_conversation_id = None
            st.success("Started new conversation.")
            st.rerun()

        st.markdown("---")
        
        # Connection Status
        if db is not None:
            st.success("🟢 Database Connected")
        else:
            st.error("🔴 Database Disconnected")
        
        st.info(
            "**How to use:**\n"
            "1. Make sure your Ollama instance is accessible.\n"
            "2. Select a model from the dropdown.\n"
            "3. Type your message and press Enter.\n"
            "4. Save important conversations for later."
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
                st.session_state.selected_model,
                st.session_state.messages
            )
            st.write_stream(response_generator)
        
        # Auto-save after each exchange if conversation exists
        if db is not None and st.session_state.current_conversation_id:
            update_conversation_in_db(
                db, 
                st.session_state.current_conversation_id, 
                st.session_state.messages
            )

# --- Custom CSS ---
st.markdown("""
<style>
    [id="stBadge"] {
        font-size: 1.25rem;
        padding: 0.5rem 0.75rem;
        border-radius: 0.75rem;
    }
    
    .stButton > button {
        width: 100%;
        text-align: left;
        white-space: pre-line;
    }
</style>
""", unsafe_allow_html=True)

# --- Run App ---
if __name__ == "__main__":
    main()
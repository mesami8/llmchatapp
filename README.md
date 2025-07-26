# LLM Chat – Streamlit + Ollama + MongoDB

A local **LLM chat interface** built with [Streamlit](https://streamlit.io/).  
It uses **Ollama** to run local LLM models and **MongoDB** to save and manage chat history.

---

## Features

- Chat with **local LLMs using Ollama**
- **Multiple model support** (auto-detects installed models)
- **Chat history** saved in MongoDB (per user session)
- Load, update, and delete previous conversations
- Simple **session-based user ID** (no auth required)

---

## Requirements

- Python 3.9+
- [Ollama](https://ollama.ai) installed and running
- MongoDB Atlas cluster or local MongoDB

---

## Installation

Clone this repository and install dependencies:

```bash
git clone <repo-url>
cd <repo-folder>
pip install -r requirements.txt

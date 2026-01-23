# 🌐 Skin Expert - Chat with Website RAG App

A Streamlit application that allows users to chat with the content of any website using Retrieval Augmented Generation (RAG).

## 🚀 Features
- **URL Ingestion**: Loads and processes content from any provided URL.
- **RAG Pipeline**: key features include local embeddings, ChromaDB vector storage, and Google Gemini LLM.
- **Expert Logic**: Custom "Dermatologist" prompt logic to provide safe and accurate skincare advice.
- **Chat Interface**: Interactive chat UI with session memory.

## 🛠️ Setup & Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/ZaynCodeHub/Capstone-3.git
   cd Capstone-3
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Setup**
   Create a `.env` file and add your Google API Key:
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```

4. **Run the App**
   ```bash
   streamlit run app.py
   ```

## 📂 Project Structure
- `app.py`: Main application code.
- `requirements.txt`: Project dependencies.
- `chroma_db/`: Local vector database (generated at runtime).

# 🌐 Chat with Website (RAG Application)

A powerful Streamlit application that allows users to chat with the content of any website using Retrieval Augmented Generation (RAG).

> **Note**: This application is currently configured with a **"Skin Expert" persona** (Dermatologist logic), originally designed for Jenpharm inquiries, but the core RAG architecture can be adapted for any domain.

## 🚀 Key Features
- **Instant Knowledge Ingestion**: Uses `UnstructuredURLLoader` to scrape and process text from any provided URL.
- **Advanced RAG Architecture**:
  - **Embeddings**: Uses local `HuggingFaceEmbeddings` (sentence-transformers/all-MiniLM-L6-v2) for cost-effective and fast vectorization.
  - **Vector Store**: Persists data locally using **ChromaDB**.
  - **LLM**: Powered by **Google Gemini** (`gemini-2.5-flash-lite-preview-09-2025`) for high-quality, reasoned responses.
- **Smart "Expert" Logic**:
  - **Strict Prompting**: Enforces factual accuracy (Temperature 0.0).
  - **Conflict Detection**: Logic to identify skincare product conflicts (e.g., Vitamin C vs. AHAs).
  - **Context-Aware**: Retrieves top-k (k=7) relevant chunks to answer queries accurately.
- **Interactive UI**: Built with Streamlit, featuring session memory and a chat-like interface.

## 🛠️ Tech Stack
- **Frontend**: Streamlit
- **Orchestration**: LangChain
- **LLM**: Google Generative AI (Gemini)
- **Database**: ChromaDB (Vector Store)
- **Utilities**: Python-dotenv, Unstructured

## ⚙️ Setup & Installation

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
   Create a `.env` file in the root directory and add your Google API Key:
   ```env
   GOOGLE_API_KEY=your_gemini_api_key_here
   ```

4. **Run the App**
   ```bash
   streamlit run app.py
   ```

## 📂 Project Structure
- `app.py`: The main application logic (UI, RAG pipeline, Persona definitions).
- `chroma_db/`: Directory where vector embeddings are stored (auto-generated).
- `requirements.txt`: Python package dependencies.

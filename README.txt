Your code performs RAG (Retrieval-Augmented Generation) using local tools. Here's what it does:
📂 1. Load Documents
	Reads .txt and .pdf files from the local directory /home/remlab/ps-bpt/PythonAIRAG/data
	Uses TextLoader and PDFMinerLoader to load content
	Splits text into chunks using CharacterTextSplitter
🧠 2. Create Embeddings
	Uses sentence-transformers/all-MiniLM-L6-v2 to convert chunks into embeddings
	Stores these embeddings in FAISS, a fast similarity search index
📚 3. Store Documents
	Stores chunks in InMemoryDocstore, which links embeddings to their original content
🧠 4. Initialize Local LLM
	Defines a custom LocalOllamaLLM class
	Connects to http://localhost:11434/api/generate to run the LLaMA3.2 3B model via Ollama
🔍 5. Run RAG Chain
	Uses RetrievalQA to:
	Retrieve relevant chunks from FAISS
	Feed them with the query to the LLM
	Return the final answer
❓ 6. Ask a Question
	Example: "What is SERIAL FLASH MEMORY?"
	The model returns an answer based on the retrieved context

                         ┌────────────────────────────┐
                         │   Local TXT / PDF Files    │
                         └────────────┬───────────────┘
                                      │
                            Load & Split into Chunks
                                      │
                         ┌────────────▼─────────────┐
                         │  HuggingFace Embeddings  │
                         │ (all-MiniLM-L6-v2 model) │
                         └────────────┬─────────────┘
                                      │
                                      ▼
                          ┌────────────────────────┐
                          │      FAISS Index       │◄─────────────┐
                          └────────┬───────────────┘              │
                                   │                              │
                          Similarity Search             InMemoryDocstore
                                   │                              │
                                   ▼                              │
                         ┌────────────────────────┐               │
                         │   Top Relevant Chunks   │──────────────┘
                         └────────────┬────────────┘
                                      │
                          Injected into Prompt
                                      │
                                      ▼
                        ┌──────────────────────────────┐
                        │   Local Ollama LLM (LLaMA3)   │
                        │ http://localhost:11434/api   │
                        └──────────────────────────────┘
                                      │
                                   Answer
                                      ▼
                        "What is SERIAL FLASH MEMORY?"


--------------------------------------------------------------------------------
Component           | Tool / Library                                            
--------------------------------------------------------------------------------
LLM:                Ollama (local model)
Embeddings:         HuggingFace (all-MiniLM-L6-v2)
Vector Store:       FAISS
Document Loader:    LangChain Community (TextLoader, PDFMinerLoader)
Text Splitter:      LangChain (CharacterTextSplitter)
Retrieval Chain:    LangChain’s RetrievalQA
Doc Storage:        In-memory (InMemoryDocstore)

                         ┌────────────────────────────┐
                         │   Local TXT / PDF Files    │
                         └────────────┬───────────────┘
                                      │
                            Load & Split into Chunks
                                      │
                         ┌────────────▼─────────────┐
                         │  HuggingFace Embeddings  │
                         │ (all-MiniLM-L6-v2 model) │
                         └────────────┬─────────────┘
                                      │
                                      ▼
                          ┌────────────────────────┐
                          │      FAISS Index       │◄─────────────┐
                          └────────┬───────────────┘              │
                                   │                              │
                          Similarity Search             InMemoryDocstore
                                   │                              │
                                   ▼                              │
                         ┌────────────────────────┐               │
                         │   Top Relevant Chunks   │──────────────┘
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌──────────────────────────────┐
                         │   Local Ollama LLM (LLaMA3)   │
                         │ http://localhost:11434/api   │
                         └──────────────────────────────┘
                                      │
                                   Answer
                                      ▼
                        "What is SERIAL FLASH MEMORY?"

                                   ───────────────────────────────

                                   **If No Relevant Content Found**

                         ┌────────────────────────────┐
                         │  FAISS Search returns low  │
                         │  similarity (irrelevant)   │
                         └────────────┬───────────────┘
                                      │
                          Skip to Fallback Logic:
                                      │
                         ┌────────────▼────────────┐
                         │   No Relevant Chunks     │
                         └────────────┬────────────┘
                                      │
                          Injected into Prompt
                                      │
                                      ▼
                        ┌──────────────────────────────┐
                        │   Local Ollama LLM (LLaMA3)   │
                        │ http://localhost:11434/api   │
                        └──────────────────────────────┘
                                      │
                                 Show Fallback Message:
                                      ▼
                      "Sorry, I couldn't find relevant information in the documents."
#-------------------------------------------------------------------------------------------------------------

                +----------------------------------------+
                |                Start                   |
                +----------------------------------------+
                            |
                            v
                +-----------------------------+
                |  Load Documents (all formats)  |
                +-----------------------------+
                            |
                            v
                +-----------------------------------+
                |  Split Documents into Chunks    |
                |  (Chunk size and overlap config)|
                +-----------------------------------+
                            |
                            v
                +-------------------------------------------+
                |  Generate Embeddings for Document Chunks |
                |  (Using HuggingFace Embedding model)     |
                +-------------------------------------------+
                            |
                            v
                +-------------------------------------------+
                |  Create FAISS Index for Similarity Search|
                +-------------------------------------------+
                            |
                            v
                +---------------------------------------------+
                |  Set Up In-memory Docstore for Retrieval   |
                +---------------------------------------------+
                            |
                            v
                +--------------------------------------+
                |  Load Local LLM (Ollama API)        |
                +--------------------------------------+
                            |
                            v
                +----------------------------------------------------+
                |  Build RAG Chain for Query Answering with Retrieval |
                +----------------------------------------------------+
                            |
                            v
                +-----------------------------------------------+
                |  Ask Question & Retrieve Answer from Documents |
                +-----------------------------------------------+
                            |
                            v
                +---------------------------+
                |        Output Answer       |
                +---------------------------+
                            |
                            v
                +---------------------------+
                |           End              |
                +---------------------------+

import logging
from flask import Flask, render_template, request, jsonify
import os
import requests
import numpy as np
from typing import List
from langchain_core.language_models import LLM
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import (
    TextLoader,
    PDFMinerLoader,
    CSVLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
)
from langchain.text_splitter import CharacterTextSplitter
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
import time

# Set up logging with INFO level
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# 🔹 Custom LLM class to call Ollama locally
class LocalOllamaLLM(LLM):
    model_name: str = "llama3.2:3b"  # Define the model name for Ollama
    url: str = "http://localhost:11434/api/generate"  # URL for accessing the local Ollama API

    def _call(self, prompt: str, stop: List[str] = None) -> str:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }
        headers = {"Content-Type": "application/json"}

        try:
            # Send request to Ollama API
            response = requests.post(self.url, json=payload, headers=headers, timeout=120)
            response.raise_for_status()  # Ensure no error occurred
            result = response.json()  # Extract the result from the response JSON
            return result.get("response", "No response from AI.")
        except requests.RequestException as e:
            return f"Error: {str(e)}"

    @property
    def _llm_type(self) -> str:
        return "ollama"

# 🔹 Load and split all supported documents
def load_documents(directory="/home/remlab/ps-bpt/PythonAIRAG/data"):
    """
    Loads documents from a specified directory and splits them into smaller chunks for processing.
    This function supports various formats, including .txt, .pdf, .csv, .docx, and .pptx.
    """
    all_docs = []
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    logging.info("Starting document loading and splitting.")

    # Loop through all files in the directory
    for file in os.listdir(directory):
        path = os.path.join(directory, file)

        try:
            logging.info(f"Processing file: {file}")
            
            if file.endswith(".txt"):
                loader = TextLoader(path)
            elif file.endswith(".pdf"):
                loader = PDFMinerLoader(path)
            elif file.endswith(".csv"):
                loader = CSVLoader(file_path=path)
            elif file.endswith(".docx") or file.endswith(".doc"):
                loader = UnstructuredWordDocumentLoader(path)
            elif file.endswith(".pptx"):
                loader = UnstructuredPowerPointLoader(path)
            else:
                logging.warning(f"Skipped unsupported file type: {file}")
                continue

            # Load the document and log the number of documents loaded
            docs = loader.load()
            logging.info(f"✅Loaded {len(docs)} document(s) from {file}")
            logging.info(f"Document preview: {docs[0].page_content[:200]}")  # Log first 200 characters for preview

            # Split the document into chunks
            chunks = splitter.split_documents(docs)
            logging.info(f"Split {len(docs)} document(s) into {len(chunks)} chunk(s)")

            # Add the chunks to the all_docs list
            all_docs.extend(chunks)

        except Exception as e:
            logging.error(f"Failed to process {file}: {e}")

    logging.info(f"Total number of chunks loaded: {len(all_docs)}")
    return all_docs

# 🔹 Main RAG pipeline function
def get_answer_from_documents(question: str):
    logging.info(f"Processing question: {question}")

    documents = load_documents()
    logging.info(f"Total number of chunks to process: {len(documents)}")

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    document_embeddings = [embedding_model.embed_query(doc.page_content) for doc in documents]
    
    logging.info(f"Generated embeddings for {len(document_embeddings)} chunks.")
    logging.debug(f"Embeddings preview: {document_embeddings[:5]}")  # Log the first 5 embeddings for preview

    dim = len(document_embeddings[0])
    index = faiss.IndexFlatL2(dim)
    embeddings_np = np.array(document_embeddings).astype('float32')
    index.add(embeddings_np)

    docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(documents)})
    index_to_docstore_id = {i: str(i) for i in range(len(documents))}
    
    vector_db = FAISS(
        embedding_function=embedding_model,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id
    )

    llm = LocalOllamaLLM()
    rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vector_db.as_retriever())
    
    logging.info("Starting RAG chain query processing.")
    result = rag_chain.invoke({"query": question})
    
    logging.info(f"Query processed. Result obtained: {result['result']}")
    return result["result"]

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")

@app.route("/documents", methods=["GET"])
def list_documents():
    try:
        docs = os.listdir("/home/remlab/ps-bpt/PythonAIRAG/data")
        return jsonify(docs)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/ask", methods=["POST"])
def ask():
    question = request.form.get("question")
    file = request.files.get("file")

    if not question.strip() and not file:
        return jsonify({"error": "Please enter a question or upload a file."}), 400

    start_time = time.time()

    # Ensure the directory /home/remlab/ps-bpt/PythonAIRAG/data exists
    upload_dir = '/home/remlab/ps-bpt/PythonAIRAG/data'
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
        logging.info(f"The directory was created at: {os.path.abspath(upload_dir)}")
    else:
        logging.info(f"The directory already exists at: {os.path.abspath(upload_dir)}")

    if file:
        try:
            file_path = os.path.join(upload_dir, file.filename)
            file.save(file_path)
            logging.info(f"File saved successfully at {file_path}")
        except Exception as e:
            logging.error(f"Error saving the file: {e}")
            return jsonify({"error": "Failed to save the uploaded file."}), 500

    # Process the question with the newly uploaded document
    answer = get_answer_from_documents(question)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    seconds, milliseconds = divmod(seconds * 1000, 1000)
    formatted_time = f"{int(minutes)}:{int(seconds)}:{int(milliseconds)}"
    
    return jsonify({"answer": answer, "time": formatted_time})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=1212, debug=True)
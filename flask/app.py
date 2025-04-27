import logging
from flask import Flask, render_template, request, jsonify
import os
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
import requests

# Set environment variable to enforce offline mode for HuggingFace
os.environ["HF_HUB_OFFLINE"] = "1"

# Set up logging with INFO level
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

class LocalOllamaLLM(LLM):
    """
    Custom Language Model (LLM) for interacting with the Ollama model via an API.
    """
    model_name: str = "gemma3:1b"
    url: str = "http://10.74.134.8:11434/api/generate"

    def _call(self, prompt: str, stop: List[str] = None) -> str:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }
        headers = {"Content-Type": "application/json"}

        try:
            response = requests.post(self.url, json=payload, headers=headers, timeout=120)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "No response from AI.")
        except requests.RequestException as e:
            return f"Error: {str(e)}"

    @property
    def _llm_type(self) -> str:
        return "ollama"

# 🔹 Load and split all supported documents
def load_documents(directory="/data/flask/data"):
    all_docs = []
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    logging.info("Starting document loading and splitting.")

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

            docs = loader.load()
            logging.info(f"✅ Loaded {len(docs)} document(s) from {file}")
            chunks = splitter.split_documents(docs)
            all_docs.extend(chunks)

        except Exception as e:
            logging.error(f"Failed to process {file}: {e}")

    logging.info(f"Total number of chunks loaded: {len(all_docs)}")
    return all_docs

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
logging.info("Embedding model loaded.")

def get_answer_from_documents(question: str, vector_db):
    logging.info(f"Processing question: {question}")

    llm = LocalOllamaLLM()
    rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vector_db.as_retriever())

    logging.info("Starting RAG chain query processing.")

    result = rag_chain.invoke({"query": question, "k": 5})
    retrieved_chunks = result.get("result", "")

    prompt = f"Answer the following question based on the provided documents:\n\n{retrieved_chunks}\n\nQuestion: {question}"
    answer = llm._call(prompt)

    logging.info(f"Query processed. Result obtained: {answer}")
    return answer

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")

@app.route("/documents", methods=["GET"])
def list_documents():
    try:
        docs = os.listdir("/data/flask/data")
        return jsonify(docs)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/delete", methods=["POST"])
def delete_document():
    # Get JSON data
    data = request.get_json()

    if not data or 'filename' not in data:
        return jsonify({"error": "Please provide the filename to delete."}), 400

    file_name = data['filename']

    try:
        file_path = os.path.join('/data/flask/data', file_name)

        if os.path.exists(file_path):
            os.remove(file_path)
            logging.info(f"Deleted file: {file_path}")
            return jsonify({"message": f"File '{file_name}' deleted successfully."})
        else:
            return jsonify({"error": f"File '{file_name}' not found."}), 404

    except Exception as e:
        logging.error(f"Error deleting file: {e}")
        return jsonify({"error": "Failed to delete the file."}), 500

@app.route("/restore", methods=["POST"])
def restore_document():
    data = request.get_json()
    filename = data.get("filename")

    if not filename:
        return jsonify({"error": "Please provide the filename to restore."}), 400

    try:
        # Define original path and trash path
        original_path = os.path.join('/data/flask/data', filename)
        trash_path = os.path.join('/data/flask/trash', filename)

        # Check if the file exists in the trash
        if os.path.exists(trash_path):
            # Move the file back to its original location
            shutil.move(trash_path, original_path)
            logging.info(f"Restored file: {filename}")
            return jsonify({"message": f"File '{filename}' restored successfully."})
        else:
            return jsonify({"error": f"File '{filename}' not found in trash."}), 404

    except Exception as e:
        logging.error(f"Error restoring file: {e}")
        return jsonify({"error": "Failed to restore the file."}), 500


@app.errorhandler(404)
def page_not_found(error):
    return jsonify({"error": "Page not found."}), 404

@app.errorhandler(500)
def internal_server_error(error):
    return jsonify({"error": "Internal server error."}), 500

@app.route("/ask", methods=["POST"])
def ask():
    question = request.form.get("question")
    file = request.files.get("file")

    if not question.strip() and not file:
        return jsonify({"error": "Please enter a question or upload a file."}), 400

    start_time = time.time()

    upload_dir = '/data/flask/data'
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
        logging.info(f"The directory was created at: {os.path.abspath(upload_dir)}")

    if file:
        try:
            file_path = os.path.join(upload_dir, file.filename)
            file.save(file_path)
            logging.info(f"File saved successfully at {file_path}")
        except Exception as e:
            logging.error(f"Error saving the file: {e}")
            return jsonify({"error": "Failed to save the uploaded file."}), 500

    documents = load_documents(upload_dir)
    document_embeddings = [embedding_model.embed_query(doc.page_content) for doc in documents]
    logging.info(f"Generated embeddings for {len(document_embeddings)} chunks.")

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

    answer = get_answer_from_documents(question, vector_db)

    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    seconds, milliseconds = divmod(seconds * 1000, 1000)
    formatted_time = f"{int(minutes)}:{int(seconds)}:{int(milliseconds)}"

    return jsonify({"answer": answer, "time": formatted_time})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=1212, debug=True)

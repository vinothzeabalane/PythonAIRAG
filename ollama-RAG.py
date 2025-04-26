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


# 🔹 Custom LLM class to call Ollama locally
class LocalOllamaLLM(LLM):
    """
    Custom Language Model class to interact with the Ollama model locally. 
    This class handles sending prompts to the Ollama model's API and receiving responses.
    """
    model_name: str = "llama3.2:3b"  # Define the model name for Ollama
    url: str = "http://localhost:11434/api/generate"  # URL for accessing the local Ollama API

    def _call(self, prompt: str, stop: List[str] = None) -> str:
        """
        Sends a prompt to the Ollama model and returns the response.
        
        Args:
            prompt (str): The text prompt to send to the model.
            stop (List[str], optional): List of stop sequences. Defaults to None.

        Returns:
            str: The response from the Ollama model or an error message if the request fails.
        """
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
            # Handle request errors
            return f"Error: {str(e)}"

    @property
    def _llm_type(self) -> str:
        return "ollama"


# 🔹 Load and split all supported documents
def load_documents(directory="/home/remlab/ps-bpt/PythonAIRAG/data"):
    """
    Loads documents from a specified directory and splits them into smaller chunks for processing.
    This function supports various formats, including .txt, .pdf, .csv, .docx, and .pptx.

    Args:
        directory (str): Path to the directory containing documents.

    Returns:
        list: A list of document chunks after splitting.
    """
    all_docs = []
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)  # Define chunk size and overlap

    # Loop through all files in the directory
    for file in os.listdir(directory):
        path = os.path.join(directory, file)

        try:
            # Load different file types based on extensions
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
                print(f"❌ Skipped unsupported file: {file}")
                continue

            # Load and split the document into chunks
            docs = loader.load()
            chunks = splitter.split_documents(docs)
            all_docs.extend(chunks)
            print(f"✅ Loaded and split: {file}")

        except Exception as e:
            # Handle any errors encountered during file processing
            print(f"⚠️ Failed to process {file}: {e}")

    return all_docs


# 🔹 Main RAG (Retrieval-Augmented Generation) pipeline
def main():
    """
    Main function that runs the Retrieval-Augmented Generation (RAG) pipeline to process documents,
    generate embeddings, and retrieve relevant information using a local LLM model.
    """
    # Step 1: Load and split all documents from the specified directory
    documents = load_documents()

    # Step 2: Initialize embedding model (using a pre-trained HuggingFace embedding model)
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Step 3: Generate embeddings for each document chunk
    document_embeddings = [embedding_model.embed_query(doc.page_content) for doc in documents]
    dim = len(document_embeddings[0])  # Get the dimension of the embeddings
    index = faiss.IndexFlatL2(dim)  # Initialize a FAISS index for similarity search

    # Step 4: Convert document embeddings to NumPy array and add them to the FAISS index
    embeddings_np = np.array(document_embeddings).astype('float32')
    index.add(embeddings_np)

    # Step 5: Create an in-memory docstore to store the documents for retrieval
    docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(documents)})
    index_to_docstore_id = {i: str(i) for i in range(len(documents))}

    # Step 6: Initialize FAISS vector store for document retrieval
    vector_db = FAISS(
        embedding_function=embedding_model,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id
    )

    # Step 7: Load the local LLM (Ollama)
    llm = LocalOllamaLLM()

    # Step 8: Set up the RetrievalQA chain to combine the retriever and LLM for query answering
    rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vector_db.as_retriever())

    # Step 9: Ask a question and retrieve the relevant answer from the documents
    question = "Summarize the Chewy machine details."
    result = rag_chain.invoke({"query": question})

    # Step 10: Print the result of the query
    print("\n📌 Question:", question)
    print("🤖 Answer:", result["result"])


if __name__ == "__main__":
    # Run the main RAG pipeline
    main()

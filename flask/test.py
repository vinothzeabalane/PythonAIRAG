

import os
os.environ["HF_HUB_OFFLINE"] = "1"  # force offline mode

from langchain.embeddings import HuggingFaceEmbeddings

embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

sentence = "Hello, world!"
embedding = embedder.embed_query(sentence)

print(f"Embedding for '{sentence}':")
print(embedding)

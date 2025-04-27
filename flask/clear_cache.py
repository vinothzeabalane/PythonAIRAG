import shutil
import os

# Clear HuggingFace cache
huggingface_cache = os.path.expanduser("~/.cache/huggingface")
if os.path.exists(huggingface_cache):
    shutil.rmtree(huggingface_cache)
    print("HuggingFace cache cleared.")

# Optionally clear FAISS index or other caches here

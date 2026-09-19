import os

# Set library verbosity configs before transformers/huggingface_hub are loaded
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")

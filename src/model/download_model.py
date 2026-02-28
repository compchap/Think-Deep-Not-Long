import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from dotenv import load_dotenv

load_dotenv()  # Load .env from project root
HF_TOKEN = os.getenv("HF_TOKEN")

# Define the project-local path

# PROJECT_MODEL_PATH = os.path.join(os.getcwd(), "models/deepseek-8b")
# MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

# PROJECT_MODEL_PATH = os.path.join(os.getcwd(), "models/qwen-4b")
# MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"

PROJECT_MODEL_PATH = os.path.join(os.getcwd(), "models/qwen")
MODEL_ID = "Qwen/Qwen3-0.6B"

print(f"🚀 Initializing download to: {PROJECT_MODEL_PATH}")

# Ensure the directory exists
os.makedirs(PROJECT_MODEL_PATH, exist_ok=True)

# Download Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID, 
    cache_dir=PROJECT_MODEL_PATH,
    token=HF_TOKEN
)

# Download Model
# dtype=torch.float16 is best for Mac M1 memory efficiency
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    dtype=torch.float16,  # Fixed: Use 'dtype' instead of deprecated 'torch_dtype' in transformers 5.2.0+
    cache_dir=PROJECT_MODEL_PATH,
    low_cpu_mem_usage=True,  # Optional: Helps with memory efficiency during download
    token=HF_TOKEN
)

print(f"✅ Model and Tokenizer successfully stored in {PROJECT_MODEL_PATH}")
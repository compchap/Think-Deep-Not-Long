import os
import sys
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from dotenv import load_dotenv

# Add project root to path
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _project_root)

from src.config.models import MODELS

load_dotenv()  # Load .env from project root
HF_TOKEN = os.getenv("HF_TOKEN")

def main():
    parser = argparse.ArgumentParser(
        description="Download model and tokenizer from Hugging Face"
    )
    parser.add_argument(
        "--model",
        default="qwen35",
        choices=list(MODELS.keys()),
        help="Model to download (short name from config)"
    )
    args = parser.parse_args()

    model_config = MODELS[args.model]
    model_id = model_config["model_id"]
    # Resolve cache_dir to absolute path (config uses relative paths)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cache_dir = os.path.join(project_root, model_config["cache_dir"])

    print(f"🚀 Initializing download for {args.model} to: {cache_dir}")

    os.makedirs(cache_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        token=HF_TOKEN
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16,
        cache_dir=cache_dir,
        low_cpu_mem_usage=True,
        token=HF_TOKEN
    )

    print(f"✅ Model and Tokenizer successfully stored in {cache_dir}")


if __name__ == "__main__":
    main()
from src.dtr_engine import DTREngine

# Model registry: short name -> {model_id, cache_dir}
MODELS = {
    "qwen4b": {
        "model_id": "Qwen/Qwen3-4B-Thinking-2507",
        "cache_dir": "models/qwen-4b",
    },
    "deepseek": {
        "model_id": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "cache_dir": "models/deepseek-8b",
    },
    "qwen.6b": {
        "model_id": "Qwen/Qwen3-0.6B",
        "cache_dir": "models/qwen",
    },
    # Add more models here
}

def main(model_name: str = "qwen", prompt: str = "Calculate 12 * 12: ", max_tokens: int = 25):
    if model_name not in MODELS:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list(MODELS.keys())}")
    
    config = MODELS[model_name]
    engine = DTREngine(**config)
    
    print(f"Using model: {config['model_id']}")
    print(f"Generating for: {prompt}")
    for word, is_deep, dtr in engine.generate_with_dtr(prompt, max_tokens=max_tokens):
        marker = "🧠" if is_deep else "  "
        print(f"{marker} '{word:50}' | DTR: {dtr:.2f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen", choices=list(MODELS.keys()), help="Model to use")
    parser.add_argument("--prompt", default="Calculate 12 * 12: ")
    parser.add_argument("--max-tokens", type=int, default=25)
    args = parser.parse_args()
    
    main(model_name=args.model, prompt=args.prompt, max_tokens=args.max_tokens)
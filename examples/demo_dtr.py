import sys
import os
import argparse

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.dtr_engine import DTREngine
from src.config.models import MODELS

def main(model_name: str = "qwen", prompt: str = "Calculate 12 * 12: ", max_tokens: int = 25):
    if model_name not in MODELS:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list(MODELS.keys())}")
    
    config = MODELS[args.model]
    engine = DTREngine(**config)
    
    print(f"Using model: {args.model}")
    print(f"Generating for: {prompt}")
    for word, is_deep, dtr in engine.generate_with_dtr(prompt, max_tokens=max_tokens):
        marker = "🧠" if is_deep else "  "
        print(f"{marker} '{word:50}' | DTR: {dtr:.2f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", 
        default="qwen", 
        choices=list(MODELS.keys()), 
        help="Model to use (must be downloaded first)")
    prompt = "Calculate 12 * 12: "
    # prompt = """ Circle 𝜔1 with radius 6 centered at point 𝐴 is internally tangent at point 𝐵 to circle 𝜔2 with radius 15. Points 𝐶 and 𝐷lie on 𝜔2 such that 𝐵𝐶 is a diameter of 𝜔2 and 𝐵𝐶 ⊥ 𝐴𝐷. The rectangle 𝐸𝐹𝐺 𝐻 is inscribed in 𝜔1 such that 𝐸𝐹 ⊥ 𝐵𝐶, 𝐶 is closer to 𝐺 𝐻 than to 𝐸𝐹, and 𝐷 is closer to 𝐹𝐺 than to 𝐸𝐻, as shown. Triangles △𝐷𝐺𝐹 and △𝐶 𝐻𝐺 have equal areas. The area of rectangle 𝐸𝐹𝐺 𝐻 is 𝑚 𝑛 , where 𝑚 and 𝑛 are relatively prime positive integers. Find 𝑚 + 𝑛."""

    parser.add_argument("--prompt", default=prompt)
    parser.add_argument("--max-tokens", type=int, default=25)
    args = parser.parse_args()
    
    main(model_name=args.model, prompt=args.prompt, max_tokens=args.max_tokens)
import torch
import os
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.model.model_utils import get_device

class DTREngine:
    def __init__(
        self,
        model_id: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        cache_dir: str | None = None,
        g: float = 0.5, # Settling threshold (used in Jensen-Shannon Divergence)
        rho: float = 0.85, # Depth fraction (used in late regime start calculation)
    ):
        self.device = get_device()
        print(f"Device: {self.device}")
        # Default cache_dir derived from model_id if not provided
        if cache_dir is None:
            model_name = model_id.split("/")[-1]  # e.g. DeepSeek-R1-Distill-Llama-8B
            cache_dir = os.path.abspath(f"models/{model_name}")
        
        self.model_id = model_id
        self.cache_dir = os.path.abspath(cache_dir)
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, 
            cache_dir=self.cache_dir,
            local_files_only=True   # Only use local files, no network calls
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            # Use float16 for Mac M1 memory efficiency. 
            # Keep the model in float16 for memory efficiency, but upcast later 
            # during generation to float32 for the numerically sensitive JSD math
            dtype=torch.float16, 
            cache_dir=self.cache_dir,
            local_files_only=True,   # Only use local files, no network calls
            low_cpu_mem_usage=True
        )
        self.model.to(self.device)
        # While RMSNorm and attention layers behave the same in train/eval mode, 
        # some models may have dropout or other layers that behave differently. 
        # Best practice is to call .eval() for inference.
        self.model.eval()
        
        # Print model and tokenizer configs
        print(f"Model config: {self.model.model}")
        self.print_tokenizer_info()
        
        self.g = g
        self.rho = rho
        self.L = self.model.config.num_hidden_layers # Total layers (L) [cite: 103]

    def print_tokenizer_info(self):
        """Print key tokenizer configs."""
        # info = {
        #     "name_or_path": self.tokenizer.name_or_path,
        #     "class": type(self.tokenizer).__name__,
        #     "vocab_size": self.tokenizer.vocab_size,
        #     "model_max_length": self.tokenizer.model_max_length,
        #     "is_fast": self.tokenizer.is_fast,
        #     "pad_token": self.tokenizer.pad_token,
        #     "pad_token_id": self.tokenizer.pad_token_id,
        #     "eos_token": self.tokenizer.eos_token,
        #     "eos_token_id": self.tokenizer.eos_token_id,
        #     "bos_token": self.tokenizer.bos_token,
        #     "bos_token_id": getattr(self.tokenizer, "bos_token_id", None),
        #     "unk_token": self.tokenizer.unk_token,
        #     "unk_token_id": getattr(self.tokenizer, "unk_token_id", None),
        # }
        # print("\nTokenizer Info:")
        # for k, v in info.items():
        #     print(f"  {k}: {v}")
        
        # For a config-like dump (attributes that are typically JSON-serializable)
        attrs = ["vocab_size", "model_max_length", "name_or_path", "pad_token", "eos_token", "bos_token", "unk_token"]
        for attr in attrs:
            if hasattr(self.tokenizer, attr):
                print(f"  {attr}: {getattr(self.tokenizer, attr)}")
        

    def calculate_jsd(self, p, q):
        """Eq 2: Jensen-Shannon Divergence [cite: 145]"""
        g = 0.5 # Default settling threshold - counts tokens that truly required sustained internal revision
        # if you set g too low (e.g., 0.25), the metric becomes "overly permissive". 
        # It starts counting simple, low-effort tokens as "thinking," 
        # which causes the correlation with accuracy to flatten out or disappear
        m = g * (p + q)

        # p_log = p.clamp(min=1e-10).log()
        # q_log = q.clamp(min=1e-10).log()
        m_log = m.clamp(min=1e-10).log()

        # Using kl_div for numerical stability on MPS
        return 0.5 * (F.kl_div(m_log, p, reduction='batchmean', log_target=False) + 
                      F.kl_div(m_log, q, reduction='batchmean', log_target=False))

    """
        h_final: normalized final hidden state.
        h_l: normalized intermediate hidden state.
        p_L, p_l: vocabulary distributions (Logit Lens).
            p_L is the final layer distribution, 
            p_l is the intermediate layer distribution.
        c_t: settling depth (first layer matching p_L).
        late_regime_start: start of late regime.
        is_deep: whether the token is a deep-thinking token.
        next_token: the next token to generate.
    """
    def generate_step(self, token_ids):
        with torch.no_grad():
            # Fixed: Pass output_hidden_states=True in forward call, not in from_pretrained
            outputs = self.model(token_ids, output_hidden_states=True)
            
        hidden_states = outputs.hidden_states  # Tuple of (embedding, layer1, ..., layerL)
        # Use hidden_states[-1] with same norm+lm_head pipeline as intermediate layers (Logit Lens consistency)
        h_final = self.model.model.norm(hidden_states[-1][:, -1, :])
        # Upcast to float32 for the numerically sensitive JSD math
        p_L = F.softmax(self.model.lm_head(h_final).to(torch.float32), dim=-1)  # Final layer distribution [cite: 108]
        
        # Algorithm 1: Find settling depth c_t [cite: 113, 155]
        # hidden_states[0] = embedding, hidden_states[1..L] = after each transformer layer
        c_t = self.L
        # Iterating through the layers to find the settling depth
        for l in range(1, self.L + 1):
            # Project intermediate hidden state to vocabulary space (Logit Lens) [cite: 105, 106]
            h_l = self.model.model.norm(hidden_states[l][:, -1, :])
            p_l = F.softmax(self.model.lm_head(h_l).to(torch.float32), dim=-1)
            
            if self.calculate_jsd(p_L, p_l).item() <= self.g:
                c_t = l
                break
        
        # Check if token is "deep-thinking" (settles in the late regime) [cite: 157, 161]
        late_regime_start = int(self.rho * self.L) 
        # print(f"late_regime_start = {late_regime_start}")
        is_deep = c_t >= late_regime_start # Late regime definition 
        
        next_token = torch.argmax(self.model.lm_head(h_final), dim=-1, keepdim=True)
        return next_token, is_deep

    def generate_with_dtr(self, prompt, max_tokens=50):
        """
        Generate tokens with DTR calculation (Algorithm 1).
        
        Args:
            prompt: Input text prompt
            max_tokens: Maximum tokens to generate
            
        Yields:
            Tuple of (token_str, is_deep, dtr) for each generated token
        """
        token_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        deep_tokens = 0
        
        for t in range(1, max_tokens + 1):
            next_token, is_deep = self.generate_step(token_ids)
            if is_deep: deep_tokens += 1
            
            token_ids = torch.cat([token_ids, next_token], dim=1)
            dtr = deep_tokens / t # Eq 6: Deep-Thinking Ratio [cite: 164]
            
            yield self.tokenizer.decode(next_token[0]), is_deep, dtr
            if next_token.item() == self.tokenizer.eos_token_id: break
    
    def estimate_dtr_from_prefix(self, prompt: str, prefix_length: int = 50) -> tuple[str, float, int]:
        """
        Generate a prefix and estimate DTR from it (for Think@n early stopping).
        
        Args:
            prompt: Input text prompt
            prefix_length: Number of tokens to generate for DTR estimation
            
        Returns:
            Tuple of (generated_text, prefix_dtr, tokens_generated)
        """
        token_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        deep_tokens = 0
        generated_tokens = []
        
        for t in range(1, prefix_length + 1):
            next_token, is_deep = self.generate_step(token_ids)
            if is_deep:
                deep_tokens += 1
            
            generated_tokens.append(next_token.item())
            token_ids = torch.cat([token_ids, next_token], dim=1)
            
            # Stop at EOS
            if next_token.item() == self.tokenizer.eos_token_id:
                break
        
        tokens_generated = len(generated_tokens)
        prefix_dtr = deep_tokens / tokens_generated if tokens_generated > 0 else 0.0
        generated_text = self.tokenizer.decode(generated_tokens)
        
        return generated_text, prefix_dtr, tokens_generated
    
    def generate_n_samples(
        self, 
        prompt: str, 
        n: int = 48, 
        prefix_length: int = 50, 
        max_tokens: int = 500,
        early_stop: bool = True,
        eta: float = 0.5
    ) -> list[dict]:
        """
        Generate n samples with early stopping based on prefix DTR (for Think@n).
        
        Args:
            prompt: Input text prompt
            n: Number of samples to generate
            prefix_length: Number of tokens for DTR estimation
            max_tokens: Maximum tokens per sample (full generation)
            early_stop: Whether to stop low-DTR samples early
            eta: Fraction of top samples to continue (e.g., 0.5 = top 50%)
            
        Returns:
            List of sample dicts with keys: 'text', 'dtr', 'tokens', 'full_generation'
        """
        samples = []
        
        # Phase 1: Generate prefixes for all n samples
        print(f"Generating {n} prefixes ({prefix_length} tokens each)...")
        for i in range(n):
            prefix_text, prefix_dtr, tokens_gen = self.estimate_dtr_from_prefix(prompt, prefix_length)
            samples.append({
                'text': prefix_text,
                'dtr': prefix_dtr,
                'tokens': tokens_gen,
                'full_generation': False,
                'sample_id': i
            })
        
        if not early_stop:
            # Continue all samples to completion
            print(f"Continuing all {n} samples to completion...")
            for sample in samples:
                full_text, final_dtr, total_tokens = self._continue_generation(
                    prompt, sample['text'], max_tokens
                )
                sample['text'] = full_text
                sample['dtr'] = final_dtr
                sample['tokens'] = total_tokens
                sample['full_generation'] = True
        else:
            # Phase 2: Rank by prefix DTR and continue only top eta%
            samples_sorted = sorted(samples, key=lambda x: x['dtr'], reverse=True)
            top_k = max(1, int(eta * n))
            
            print(f"Early stopping: continuing top {top_k}/{n} samples (η={eta})...")
            for i, sample in enumerate(samples_sorted):
                if i < top_k:
                    # Continue top samples
                    full_text, final_dtr, total_tokens = self._continue_generation(
                        prompt, sample['text'], max_tokens
                    )
                    sample['text'] = full_text
                    sample['dtr'] = final_dtr
                    sample['tokens'] = total_tokens
                    sample['full_generation'] = True
                else:
                    # Bottom samples stopped at prefix
                    sample['full_generation'] = False
        
        return samples
    
    def _continue_generation(self, prompt: str, prefix_text: str, max_tokens: int) -> tuple[str, float, int]:
        """
        Continue generation from a prefix to completion.
        
        Args:
            prompt: Original input prompt
            prefix_text: Already generated prefix text
            max_tokens: Maximum total tokens
            
        Returns:
            Tuple of (full_text, final_dtr, total_tokens)
        """
        # Re-encode prompt + prefix
        full_prompt = prompt + prefix_text
        token_ids = self.tokenizer.encode(full_prompt, return_tensors="pt").to(self.device)
        
        # Count deep tokens in what we generate from here
        deep_tokens = 0
        tokens_generated = 0
        generated_tokens = []
        
        # Continue generation
        for t in range(max_tokens):
            next_token, is_deep = self.generate_step(token_ids)
            if is_deep:
                deep_tokens += 1
            
            generated_tokens.append(next_token.item())
            tokens_generated += 1
            token_ids = torch.cat([token_ids, next_token], dim=1)
            
            # Stop at EOS
            if next_token.item() == self.tokenizer.eos_token_id:
                break
        
        # Calculate final DTR over the continued portion
        continuation_dtr = deep_tokens / tokens_generated if tokens_generated > 0 else 0.0
        
        # Decode only the new tokens
        continuation_text = self.tokenizer.decode(generated_tokens)
        full_text = prefix_text + continuation_text
        
        # Total tokens = prefix + continuation
        total_tokens = len(self.tokenizer.encode(full_prompt + continuation_text)) - len(self.tokenizer.encode(prompt))
        
        return full_text, continuation_dtr, total_tokens

    def main():
        """Entry point for running DTREngine from CLI or as a module."""
        import argparse

        parser = argparse.ArgumentParser(description="DTR Engine - Deep Thinking Ratio")
        parser.add_argument(
            "--model-id",
            default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            help="HuggingFace model ID",
        )
        parser.add_argument(
            "--cache-dir",
            default=None,
            help="Local cache directory for model (default: models/<model-name>)",
        )
        parser.add_argument(
            "--prompt",
            default="Calculate 12 * 12: ",
            help="Prompt to generate",
        )
        parser.add_argument(
            "--max-tokens",
            type=int,
            default=25,
            help="Max tokens to generate",
        )
        parser.add_argument("--g", type=float, default=0.5, help="Settling threshold")
        parser.add_argument("--rho", type=float, default=0.85, help="Depth fraction")
        args = parser.parse_args()

        engine = DTREngine(
            model_id=args.model_id,
            cache_dir=args.cache_dir,
            g=args.g,
            rho=args.rho,
        )
        print(f"Using model: {engine.model_id}, cache: {engine.cache_dir}")
        print(f"Generating for: {args.prompt}")
        for word, is_deep, dtr in engine.generate_with_dtr(args.prompt, max_tokens=args.max_tokens):
            marker = "<<deep-think-token>>" if is_deep else "  "
            print(f"{marker} '{word:10}' | DTR: {dtr:.2f}")


    if __name__ == "__main__":
        main()
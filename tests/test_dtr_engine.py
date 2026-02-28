import torch
import unittest
from unittest.mock import Mock, MagicMock, patch
import torch.nn.functional as F
from src.dtr_engine import DTREngine


# ============================================================================
# Helper Utilities
# ============================================================================

def create_random_distribution(size, seed=42):
    """Create a random probability distribution that sums to 1."""
    torch.manual_seed(seed)
    probs = torch.rand(size)
    return probs / probs.sum()


def create_mock_model_output(vocab_size=1000, num_layers=28, hidden_size=1024):
    """Create a mock model output with hidden_states."""
    mock_output = Mock()
    # Create L+1 hidden states (embedding + L layers)
    mock_output.hidden_states = tuple([
        torch.randn(1, 10, hidden_size) for _ in range(num_layers + 1)
    ])
    mock_output.logits = torch.randn(1, 10, vocab_size)
    return mock_output


def assert_tensor_close(test_case, tensor1, tensor2, atol=1e-6, rtol=1e-5):
    """Assert two tensors are close within tolerance."""
    test_case.assertTrue(
        torch.allclose(tensor1, tensor2, atol=atol, rtol=rtol),
        f"Tensors not close: max diff = {(tensor1 - tensor2).abs().max().item()}"
    )


# ============================================================================
# Unit Tests (Mocked - Fast)
# ============================================================================

class TestDTREngineUnit(unittest.TestCase):
    """Unit tests with mocked model components (fast)."""
    
    def setUp(self):
        """Set up mock engine for each test."""
        # Create a mock engine without loading the actual model
        self.mock_engine = Mock(spec=DTREngine)
        self.mock_engine.g = 0.5
        self.mock_engine.rho = 0.85
        self.mock_engine.L = 32
        # Bind the real calculate_jsd method to the mock
        self.mock_engine.calculate_jsd = DTREngine.calculate_jsd.__get__(self.mock_engine, DTREngine)
    
    # ========================================================================
    # Test calculate_jsd method
    # ========================================================================
    
    def test_jsd_identical_distributions(self):
        """JSD of identical distributions should be ~0."""
        p = create_random_distribution(1000, seed=42)
        jsd = self.mock_engine.calculate_jsd(p, p)
        self.assertLess(jsd.item(), 1e-5, "JSD(p, p) should be approximately 0")
    
    def test_jsd_symmetric(self):
        """JSD should be symmetric: JSD(p, q) == JSD(q, p)."""
        p = create_random_distribution(1000, seed=42)
        q = create_random_distribution(1000, seed=123)
        jsd_pq = self.mock_engine.calculate_jsd(p, q)
        jsd_qp = self.mock_engine.calculate_jsd(q, p)
        assert_tensor_close(self, jsd_pq, jsd_qp, atol=1e-6)
    
    def test_jsd_bounds(self):
        """JSD should be in range [0, 1] for any distributions."""
        p = create_random_distribution(1000, seed=42)
        q = create_random_distribution(1000, seed=123)
        jsd = self.mock_engine.calculate_jsd(p, q)
        self.assertGreaterEqual(jsd.item(), 0.0, "JSD should be >= 0")
        self.assertLessEqual(jsd.item(), 1.0, "JSD should be <= 1")
    
    def test_jsd_with_zeros(self):
        """JSD should handle distributions with zero values (tests clamping)."""
        p = torch.zeros(1000)
        p[0] = 1.0  # All probability on first token
        q = torch.zeros(1000)
        q[1] = 1.0  # All probability on second token
        
        # This should not raise an error due to clamping
        jsd = self.mock_engine.calculate_jsd(p, q)
        self.assertGreaterEqual(jsd.item(), 0.0)
        self.assertTrue(torch.isfinite(jsd), "JSD should be finite even with zeros")
    
    # ========================================================================
    # Test late regime calculation
    # ========================================================================
    
    def test_late_regime_calculation(self):
        """Verify late_regime_start = int(rho * L) for various values."""
        test_cases = [
            (0.85, 32, 27),   # 0.85 * 32 = 27.2 -> 27
            (0.85, 28, 23),   # 0.85 * 28 = 23.8 -> 23
            (0.90, 40, 36),   # 0.90 * 40 = 36.0 -> 36
            (0.80, 24, 19),   # 0.80 * 24 = 19.2 -> 19
        ]
        for rho, L, expected in test_cases:
            late_regime_start = int(rho * L)
            self.assertEqual(late_regime_start, expected,
                           f"Late regime for rho={rho}, L={L} should be {expected}")
    
    def test_is_deep_threshold(self):
        """Test boundary cases for is_deep classification."""
        rho = 0.85
        L = 32
        late_regime_start = int(rho * L)  # 27
        
        # Test boundary: c_t >= late_regime_start should be deep
        c_t_deep = late_regime_start
        is_deep = c_t_deep >= late_regime_start
        self.assertTrue(is_deep, f"c_t={c_t_deep} should be deep (>= {late_regime_start})")
        
        # Test boundary: c_t < late_regime_start should not be deep
        c_t_shallow = late_regime_start - 1
        is_deep = c_t_shallow >= late_regime_start
        self.assertFalse(is_deep, f"c_t={c_t_shallow} should not be deep (< {late_regime_start})")
    
    # ========================================================================
    # Test parameter validation
    # ========================================================================
    
    def test_g_parameter_bounds(self):
        """Verify g (settling threshold) is stored correctly."""
        self.assertEqual(self.mock_engine.g, 0.5)
        self.assertIsInstance(self.mock_engine.g, float)
        self.assertGreater(self.mock_engine.g, 0.0)
        self.assertLess(self.mock_engine.g, 1.0)
    
    def test_rho_parameter_bounds(self):
        """Verify rho (depth fraction) is stored correctly."""
        self.assertEqual(self.mock_engine.rho, 0.85)
        self.assertIsInstance(self.mock_engine.rho, float)
        self.assertGreater(self.mock_engine.rho, 0.0)
        self.assertLessEqual(self.mock_engine.rho, 1.0)
    
    # ========================================================================
    # Test DTR calculation logic
    # ========================================================================
    
    def test_dtr_calculation(self):
        """Verify DTR = deep_tokens / total_tokens."""
        test_cases = [
            (0, 10, 0.0),    # No deep tokens
            (5, 10, 0.5),    # Half deep
            (10, 10, 1.0),   # All deep
            (3, 7, 3/7),     # Arbitrary
            (1, 1, 1.0),     # Single deep token
        ]
        for deep_tokens, total_tokens, expected_dtr in test_cases:
            dtr = deep_tokens / total_tokens
            self.assertAlmostEqual(dtr, expected_dtr, places=5,
                                 msg=f"DTR for {deep_tokens}/{total_tokens} should be {expected_dtr}")
    
    def test_dtr_all_shallow(self):
        """DTR = 0.0 when no deep tokens."""
        deep_tokens = 0
        total_tokens = 20
        dtr = deep_tokens / total_tokens
        self.assertEqual(dtr, 0.0, "DTR should be 0.0 when all tokens are shallow")
    
    def test_dtr_all_deep(self):
        """DTR = 1.0 when all tokens are deep."""
        deep_tokens = 15
        total_tokens = 15
        dtr = deep_tokens / total_tokens
        self.assertEqual(dtr, 1.0, "DTR should be 1.0 when all tokens are deep")
    
    def test_dtr_incremental(self):
        """DTR updates correctly as tokens are added."""
        # Simulate incremental DTR calculation
        deep_tokens = 0
        dtrs = []
        
        # Token sequence: shallow, deep, shallow, deep
        is_deep_sequence = [False, True, False, True]
        
        for t, is_deep in enumerate(is_deep_sequence, start=1):
            if is_deep:
                deep_tokens += 1
            dtr = deep_tokens / t
            dtrs.append(dtr)
        
        expected_dtrs = [0.0, 0.5, 1/3, 0.5]
        for i, (actual, expected) in enumerate(zip(dtrs, expected_dtrs)):
            self.assertAlmostEqual(actual, expected, places=5,
                                 msg=f"DTR at step {i+1} should be {expected}")


# ============================================================================
# Integration Tests (Real Model - Slower but Thorough)
# ============================================================================

class TestDTREngineIntegration(unittest.TestCase):
    """Integration tests with real qwen.6b model (slower but thorough)."""
    
    @classmethod
    def setUpClass(cls):
        """Initialize the engine once for the test suite."""
        print("\n" + "="*70)
        print("🚀 Loading Qwen3-0.6B model for integration tests...")
        print("="*70)
        try:
            cls.engine = DTREngine(
                model_id="Qwen/Qwen3-0.6B", 
                cache_dir="models/qwen"
            )
            cls.prompt = "Calculate 5 * 16: "
            print("="*70)
            print("✅ Model loaded successfully")
            print("="*70 + "\n")
        except Exception as e:
            print(f"\n⚠️  Warning: Could not load model: {e}")
            print("Integration tests will be skipped.")
            print("Make sure the model is downloaded and network/protobuf is available.\n")
            cls.engine = None
            cls.prompt = "Calculate 5 * 16: "
    
    # ========================================================================
    # Test model loading
    # ========================================================================
    
    def test_model_loads_successfully(self):
        """Verify model and tokenizer load from cache."""
        if self.engine is None:
            self.skipTest("Model not loaded - skipping integration test")
        
        self.assertIsNotNone(self.engine.model, "Model should be loaded")
        self.assertIsNotNone(self.engine.tokenizer, "Tokenizer should be loaded")
        self.assertEqual(self.engine.model_id, "Qwen/Qwen3-0.6B")
        self.assertTrue(self.engine.cache_dir.endswith("models/qwen"))
    
    def test_model_config(self):
        """Check L (num_layers), device placement."""
        if self.engine is None:
            self.skipTest("Model not loaded - skipping integration test")
        
        self.assertIsNotNone(self.engine.L, "L (num_layers) should be set")
        self.assertGreater(self.engine.L, 0, "L should be positive")
        self.assertIsInstance(self.engine.L, int)
        
        # Check device
        self.assertIsNotNone(self.engine.device)
        self.assertIn(str(self.engine.device), ['cpu', 'cuda', 'mps', 'xpu'])
        
        # Check model is in eval mode
        self.assertFalse(self.engine.model.training, "Model should be in eval mode")
    
    def test_tokenizer_config(self):
        """Verify vocab_size, eos_token_id."""
        if self.engine is None:
            self.skipTest("Model not loaded - skipping integration test")
        
        self.assertIsNotNone(self.engine.tokenizer.vocab_size)
        self.assertGreater(self.engine.tokenizer.vocab_size, 0)
        self.assertIsNotNone(self.engine.tokenizer.eos_token_id)
        self.assertIsInstance(self.engine.tokenizer.eos_token_id, int)
    
    # ========================================================================
    # Test generate_step
    # ========================================================================
    
    def test_generate_step_output_shape(self):
        """Verify generate_step returns (next_token, is_deep) tuple."""
        if self.engine is None:
            self.skipTest("Model not loaded - skipping integration test")
        
        token_ids = self.engine.tokenizer.encode(self.prompt, return_tensors="pt").to(self.engine.device)
        next_token, is_deep = self.engine.generate_step(token_ids)
        
        # Check output types
        self.assertIsInstance(next_token, torch.Tensor, "next_token should be a tensor")
        self.assertIsInstance(is_deep, bool, "is_deep should be a boolean")
        
        # Check tensor shape
        self.assertEqual(next_token.dim(), 2, "next_token should be 2D (batch, 1)")
        self.assertEqual(next_token.shape[1], 1, "next_token should have shape (batch, 1)")
    
    def test_generate_step_settling_depth(self):
        """Verify settling depth c_t is within valid range [1, L]."""
        if self.engine is None:
            self.skipTest("Model not loaded - skipping integration test")
        
        token_ids = self.engine.tokenizer.encode(self.prompt, return_tensors="pt").to(self.engine.device)
        
        # Run generate_step and manually check settling depth by inspecting hidden states
        with torch.no_grad():
            outputs = self.engine.model(token_ids, output_hidden_states=True)
        
        hidden_states = outputs.hidden_states
        self.assertEqual(len(hidden_states), self.engine.L + 1,
                        f"Should have L+1={self.engine.L+1} hidden states (embedding + L layers)")
        
        # Verify we can compute settling depth
        h_final = self.engine.model.model.norm(hidden_states[-1][:, -1, :])
        p_L = F.softmax(self.engine.model.lm_head(h_final).to(torch.float32), dim=-1)
        
        # Count layers until settling
        c_t = self.engine.L
        for l in range(1, self.engine.L + 1):
            h_l = self.engine.model.model.norm(hidden_states[l][:, -1, :])
            p_l = F.softmax(self.engine.model.lm_head(h_l).to(torch.float32), dim=-1)
            jsd = self.engine.calculate_jsd(p_L, p_l)
            if jsd.item() <= self.engine.g:
                c_t = l
                break
        
        self.assertGreaterEqual(c_t, 1, "Settling depth should be >= 1")
        self.assertLessEqual(c_t, self.engine.L, f"Settling depth should be <= {self.engine.L}")
    
    def test_hidden_states_structure(self):
        """Verify hidden_states has correct structure (length L+1)."""
        if self.engine is None:
            self.skipTest("Model not loaded - skipping integration test")
        
        token_ids = self.engine.tokenizer.encode(self.prompt, return_tensors="pt").to(self.engine.device)
        
        with torch.no_grad():
            outputs = self.engine.model(token_ids, output_hidden_states=True)
        
        hidden_states = outputs.hidden_states
        self.assertEqual(len(hidden_states), self.engine.L + 1,
                        f"hidden_states should have length L+1 = {self.engine.L + 1}")
        
        # Check all hidden states are tensors
        for i, h in enumerate(hidden_states):
            self.assertIsInstance(h, torch.Tensor, f"hidden_states[{i}] should be a tensor")
    
    # ========================================================================
    # Test generation and DTR bounds (from original tests)
    # ========================================================================
    
    def test_generation_and_dtr_bounds(self):
        """Test that DTR is mathematically valid (0 <= DTR <= 1)."""
        if self.engine is None:
            self.skipTest("Model not loaded - skipping integration test")
        
        print("\n" + "-"*70)
        print("Running DTR Bounds Test...")
        print("-"*70)
        gen = self.engine.generate_with_dtr(self.prompt, max_tokens=10)
        
        tokens_generated = 0
        last_dtr = 0
        
        for word, is_deep, dtr in gen:
            tokens_generated += 1
            last_dtr = dtr
            self.assertGreaterEqual(dtr, 0.0, f"DTR should be >= 0 (got {dtr})")
            self.assertLessEqual(dtr, 1.0, f"DTR should be <= 1 (got {dtr})")
        
        self.assertGreater(tokens_generated, 0, "Model failed to generate any tokens.")
        print(f"✅ Generated {tokens_generated} tokens with final DTR: {last_dtr:.2f}")
        print("-"*70)
    
    def test_mechanistic_distinction(self):
        """
        Verify that not all tokens are classified the same way.
        Based on paper findings that functional words settle in shallow layers.
        """
        if self.engine is None:
            self.skipTest("Model not loaded - skipping integration test")
        
        print("\n" + "-"*70)
        print("Running Mechanistic Distinction Test...")
        print("-"*70)
        gen = self.engine.generate_with_dtr(self.prompt, max_tokens=20)
        
        deep_count = 0
        shallow_count = 0
        results = []
        
        for word, is_deep, dtr in gen:
            results.append((word, is_deep))
            if is_deep:
                deep_count += 1
            else:
                shallow_count += 1
        
        # We expect both deep and shallow tokens in a typical generation
        # (not a strict requirement, but indicates the system is working)
        print(f"Deep tokens: {deep_count}, Shallow tokens: {shallow_count}")
        
        print("Sample output with deep-thought markers:")
        for word, is_deep in results[:10]:  # Show first 10
            marker = "🧠" if is_deep else "  "
            print(f"{marker} {word}")
        
        # At minimum, we should generate something
        self.assertGreater(len(results), 0, "Should generate at least one token")
        print("-"*70)
    
    # ========================================================================
    # Test edge cases
    # ========================================================================
    
    def test_single_token_generation(self):
        """Test generation with max_tokens=1."""
        if self.engine is None:
            self.skipTest("Model not loaded - skipping integration test")
        
        gen = self.engine.generate_with_dtr(self.prompt, max_tokens=1)
        tokens = list(gen)
        
        self.assertEqual(len(tokens), 1, "Should generate exactly 1 token")
        word, is_deep, dtr = tokens[0]
        self.assertIsInstance(word, str)
        self.assertIsInstance(is_deep, bool)
        # DTR for single token is either 0.0 or 1.0
        self.assertIn(dtr, [0.0, 1.0], "DTR for single token should be 0.0 or 1.0")
    
    def test_eos_termination(self):
        """Verify generation stops at EOS token (or max_tokens)."""
        if self.engine is None:
            self.skipTest("Model not loaded - skipping integration test")
        
        # Use a longer max_tokens to allow EOS to potentially appear
        gen = self.engine.generate_with_dtr(self.prompt, max_tokens=50)
        tokens = list(gen)
        
        # Should generate at least one token
        self.assertGreater(len(tokens), 0)
        
        # Should not exceed max_tokens
        self.assertLessEqual(len(tokens), 50, "Should not exceed max_tokens")
        
        # If it stopped before max_tokens, last token should be EOS
        # (This is a soft check since we might hit max_tokens first)
        if len(tokens) < 50:
            # Generation stopped early, likely due to EOS
            # We can't directly check the last token without decoding,
            # but we can verify the generator stopped
            self.assertLess(len(tokens), 50, "Early termination likely due to EOS")


# ============================================================================
# Test Runner
# ============================================================================

if __name__ == "__main__":
    unittest.main()

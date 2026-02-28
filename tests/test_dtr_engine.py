import torch
import unittest
from dtr_engine import DTREngine

class TestDTREngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initialize the engine once for the test suite."""
        print("🚀 Loading DeepSeek-R1-8B on MPS...")
        cls.engine = DTREngine()
        cls.prompt = "Calculate 5 * 16: "

    def test_generation_and_dtr_bounds(self):
        """Test that DTR is mathematically valid (0 <= DTR <= 1)."""
        print("\nRunning Bounds Test...")
        gen = self.engine.generate_with_dtr(self.prompt, max_new_tokens=10)
        
        tokens_generated = 0
        last_dtr = 0
        
        for word, is_deep, dtr in gen:
            tokens_generated += 1
            last_dtr = dtr
            self.assertGreaterEqual(dtr, 0.0)
            self.assertLessEqual(dtr, 1.0)
        
        self.assertGreater(tokens_generated, 0, "Model failed to generate any tokens.")
        print(f"✅ Generated {tokens_generated} tokens with final DTR: {last_dtr:.2f}")

    def test_mechanistic_distinction(self):
        """
        Verify that early tokens (often templated) settle differently than later ones.
        Based on paper findings that functional words settle in shallow layers.
        """
        print("\nRunning Mechanistic Distinction Test...")
        gen = self.engine.generate_with_dtr(self.prompt, max_new_tokens=20)
        
        deep_token_found = False
        results = []
        
        for word, is_deep, dtr in gen:
            results.append((word, is_deep))
            if is_deep:
                deep_token_found = True
        
        # We expect at least one logic-heavy token in a math problem to be 'deep'
        self.assertTrue(deep_token_found, "Engine failed to identify any deep-thinking tokens in a math prompt.")
        
        print("Sample output with deep-thought markers:")
        for word, is_deep in results:
            marker = "🧠" if is_deep else "  "
            print(f"{marker} {word}")

if __name__ == "__main__":
    unittest.main()
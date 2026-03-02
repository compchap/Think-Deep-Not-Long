"""
Unit tests for Think@n implementation.

Tests answer extraction, aggregation methods, and ThinkAtN logic.
Uses mocked DTREngine for fast execution.
"""

import sys
import os
import unittest
from unittest.mock import Mock, MagicMock

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.answer_extraction import (
    extract_answer,
    normalize_answer,
    answers_match
)
from src.aggregation_methods import (
    majority_vote,
    cons_at_n,
    short_at_n,
    long_at_n,
    think_at_n,
    mean_at_n,
    calculate_cost,
    calculate_cost_with_prefix
)
from src.think_at_n import ThinkAtN


class TestAnswerExtraction(unittest.TestCase):
    """Test answer extraction from generated text."""
    
    def test_extract_boxed_notation(self):
        """Test extraction of LaTeX \\boxed{} answers."""
        text = "Therefore the answer is \\boxed{144}"
        self.assertEqual(extract_answer(text), "144")
        
        text = "We get \\boxed{12.5}"
        self.assertEqual(extract_answer(text), "12.5")
    
    def test_extract_answer_is_pattern(self):
        """Test extraction of 'answer is X' patterns."""
        text = "The answer is 42"
        self.assertEqual(extract_answer(text), "42")
        
        text = "Final answer: 100"
        self.assertEqual(extract_answer(text), "100")
    
    def test_extract_last_number(self):
        """Test fallback to last number in text."""
        text = "We calculate 5 + 3 = 8"
        self.assertEqual(extract_answer(text), "8")
        
        text = "First we get 10, then 20, finally 30"
        self.assertEqual(extract_answer(text), "30")
    
    def test_extract_no_answer(self):
        """Test behavior when no answer found."""
        text = "No numbers here"
        self.assertIsNone(extract_answer(text))
        
        text = ""
        self.assertIsNone(extract_answer(text))
    
    def test_normalize_answer(self):
        """Test answer normalization."""
        self.assertEqual(normalize_answer("144.0"), "144")
        self.assertEqual(normalize_answer("  12  "), "12")
        self.assertEqual(normalize_answer("ABC"), "abc")
        self.assertIsNone(normalize_answer(None))
    
    def test_answers_match(self):
        """Test answer comparison with normalization."""
        self.assertTrue(answers_match("144", "144.0"))
        self.assertTrue(answers_match("12", "12"))
        self.assertFalse(answers_match("12", "13"))
        self.assertFalse(answers_match(None, "12"))


class TestAggregationMethods(unittest.TestCase):
    """Test sample aggregation strategies."""
    
    def setUp(self):
        """Create sample data for testing."""
        # Mock samples with varying DTR, length, and answers
        self.samples = [
            {'text': 'Answer: \\boxed{144}', 'dtr': 0.8, 'tokens': 50, 'full_generation': True},
            {'text': 'Result is \\boxed{144}', 'dtr': 0.7, 'tokens': 60, 'full_generation': True},
            {'text': 'The answer is \\boxed{12}', 'dtr': 0.6, 'tokens': 40, 'full_generation': True},
            {'text': '\\boxed{144}', 'dtr': 0.5, 'tokens': 30, 'full_generation': True},
            {'text': 'I get \\boxed{144}', 'dtr': 0.4, 'tokens': 70, 'full_generation': True},
            {'text': 'Final: \\boxed{12}', 'dtr': 0.3, 'tokens': 35, 'full_generation': True},
        ]
    
    def test_majority_vote(self):
        """Test majority voting logic."""
        answers = ["144", "144", "144", "12", "12"]
        self.assertEqual(majority_vote(answers), "144")
        
        answers = ["12", "13", "14"]  # Tie -> returns first
        self.assertEqual(majority_vote(answers), "12")
        
        answers = [None, None, "144"]
        self.assertEqual(majority_vote(answers), "144")
    
    def test_cons_at_n(self):
        """Test Cons@n (vote over all samples)."""
        answer = cons_at_n(self.samples)
        self.assertEqual(answer, "144")  # 4 votes for 144 vs 2 for 12
    
    def test_think_at_n(self):
        """Test Think@n (vote over top DTR samples)."""
        # Top 50% by DTR: samples 0,1,2 (DTR: 0.8, 0.7, 0.6)
        # Answers: 144, 144, 12 -> majority 144
        answer = think_at_n(self.samples, eta=0.5)
        self.assertEqual(answer, "144")
        
        # Top 33% by DTR: samples 0,1 (DTR: 0.8, 0.7)
        # Answers: 144, 144 -> unanimous 144
        answer = think_at_n(self.samples, eta=0.33)
        self.assertEqual(answer, "144")
    
    def test_short_at_n(self):
        """Test Short@n (vote over shortest samples)."""
        # Shortest 50%: tokens 30,35,40 -> samples 3,5,2
        # Answers: 144, 12, 12 -> majority 12
        answer = short_at_n(self.samples, eta=0.5)
        self.assertEqual(answer, "12")
    
    def test_long_at_n(self):
        """Test Long@n (vote over longest samples)."""
        # Longest 50%: tokens 70,60,50 -> samples 4,1,0
        # Answers: 144, 144, 144 -> unanimous 144
        answer = long_at_n(self.samples, eta=0.5)
        self.assertEqual(answer, "144")
    
    def test_mean_at_n(self):
        """Test Mean@n (average accuracy)."""
        # Ground truth: 144
        # Correct: samples 0,1,3,4 = 4/6
        accuracy = mean_at_n(self.samples, ground_truth="144")
        self.assertAlmostEqual(accuracy, 4/6, places=2)
    
    def test_calculate_cost(self):
        """Test total cost calculation."""
        cost = calculate_cost(self.samples)
        expected = 50 + 60 + 40 + 30 + 70 + 35
        self.assertEqual(cost, expected)
    
    def test_calculate_cost_with_prefix(self):
        """Test Think@n cost with early stopping."""
        prefix_length = 20
        eta = 0.5
        
        # Cost = (prefix × n) + continuation for top eta%
        # prefix cost: 20 × 6 = 120
        # Top 3 samples (eta=0.5): 0,1,2 with tokens 50,60,40
        # Continuation: (50-20) + (60-20) + (40-20) = 90
        # Total: 120 + 90 = 210
        cost = calculate_cost_with_prefix(self.samples, prefix_length, eta)
        self.assertEqual(cost, 210)


class TestThinkAtN(unittest.TestCase):
    """Test ThinkAtN class logic."""
    
    def setUp(self):
        """Create mock DTREngine and ThinkAtN."""
        self.mock_engine = Mock()
        self.mock_engine.n = 12
        
        # Mock sample generation
        self.mock_samples = [
            {'text': f'Answer is \\boxed{{144}} sample {i}', 
             'dtr': 0.9 - i*0.1, 
             'tokens': 50 + i*5,
             'full_generation': True,
             'sample_id': i}
            for i in range(6)
        ]
        
        self.mock_engine.generate_n_samples = Mock(return_value=self.mock_samples)
        
        self.think = ThinkAtN(
            dtr_engine=self.mock_engine,
            n=12,
            eta=0.5,
            prefix_length=50
        )
    
    def test_initialization(self):
        """Test ThinkAtN initialization."""
        self.assertEqual(self.think.n, 12)
        self.assertEqual(self.think.eta, 0.5)
        self.assertEqual(self.think.prefix_length, 50)
        self.assertIsNotNone(self.think.engine)
    
    def test_rank_samples_by_dtr(self):
        """Test sample ranking by DTR."""
        ranked = self.think.rank_samples_by_dtr(self.mock_samples)
        
        # Check descending order
        dtrs = [s['dtr'] for s in ranked]
        self.assertEqual(dtrs, sorted(dtrs, reverse=True))
        
        # Check highest DTR is first
        self.assertEqual(ranked[0]['dtr'], 0.9)
    
    def test_select_top_samples(self):
        """Test top sample selection."""
        top = self.think.select_top_samples(self.mock_samples, eta=0.5)
        
        # Should select top 50% = 3 samples
        self.assertEqual(len(top), 3)
        
        # Check they are highest DTR
        self.assertEqual(top[0]['dtr'], 0.9)
        self.assertEqual(top[1]['dtr'], 0.8)
        self.assertEqual(top[2]['dtr'], 0.7)
    
    def test_get_sample_stats(self):
        """Test sample statistics calculation."""
        stats = self.think.get_sample_stats(self.mock_samples)
        
        self.assertEqual(stats['num_samples'], 6)
        self.assertEqual(stats['full_generations'], 6)
        self.assertEqual(stats['dtr']['max'], 0.9)
        self.assertEqual(stats['dtr']['min'], 0.4)
        self.assertGreater(stats['tokens']['total'], 0)
    
    def test_solve_returns_result_dict(self):
        """Test solve() returns properly formatted result."""
        result = self.think.solve(
            problem="Calculate 12 * 12",
            ground_truth="144"
        )
        
        # Check structure
        self.assertIn('samples', result)
        self.assertIn('think_at_n', result)
        self.assertIn('cons_at_n', result)
        self.assertIn('short_at_n', result)
        self.assertIn('long_at_n', result)
        
        # Check each method has required keys
        for method in ['think_at_n', 'cons_at_n', 'short_at_n', 'long_at_n']:
            self.assertIn('answer', result[method])
            self.assertIn('cost', result[method])
            self.assertIn('correct', result[method])
    
    def test_solve_accuracy_calculation(self):
        """Test accuracy is correctly calculated."""
        result = self.think.solve(
            problem="Test problem",
            ground_truth="144"
        )
        
        # All mock samples have answer 144, so should all be correct
        self.assertTrue(result['think_at_n']['correct'])
        self.assertTrue(result['cons_at_n']['correct'])


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def test_empty_samples(self):
        """Test behavior with empty sample list."""
        self.assertIsNone(cons_at_n([]))
        self.assertIsNone(think_at_n([]))
        self.assertEqual(calculate_cost([]), 0)
    
    def test_single_sample(self):
        """Test with single sample."""
        sample = [{'text': '\\boxed{42}', 'dtr': 0.5, 'tokens': 10}]
        answer = cons_at_n(sample)
        self.assertEqual(answer, "42")
    
    def test_all_same_dtr(self):
        """Test ranking when all samples have same DTR."""
        samples = [
            {'text': f'Answer {i}', 'dtr': 0.5, 'tokens': 10}
            for i in range(5)
        ]
        
        mock_engine = Mock()
        mock_engine.generate_n_samples = Mock(return_value=samples)
        think = ThinkAtN(mock_engine)
        
        ranked = think.rank_samples_by_dtr(samples)
        self.assertEqual(len(ranked), 5)


if __name__ == "__main__":
    unittest.main()

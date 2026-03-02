"""
Think@n Inference Scaling Implementation (Algorithm 2).

Implements test-time scaling via DTR-based sample selection and early stopping.
Achieves ~50% cost reduction while matching/exceeding self-consistency accuracy.
"""

from typing import Optional, List, Dict
from src.dtr_engine import DTREngine
from src.aggregation_methods import (
    think_at_n as think_at_n_vote,
    cons_at_n,
    short_at_n, 
    long_at_n,
    mean_at_n,
    calculate_cost,
    calculate_cost_with_prefix
)
from src.answer_extraction import extract_answer, answers_match


class ThinkAtN:
    """
    Think@n: Test-time scaling with DTR-based sample selection.
    
    Implements Algorithm 2 from "Think Deep, Not Just Long" paper.
    Generates multiple samples, ranks by prefix DTR, and aggregates via voting.
    
    Key features:
    - Early stopping of low-DTR samples (cost reduction)
    - Majority voting over high-DTR samples (accuracy)
    - Baseline comparisons (Cons@n, Short@n, Long@n)
    
    Usage:
        >>> engine = DTREngine(model_id="Qwen/Qwen3-0.6B")
        >>> think = ThinkAtN(engine)
        >>> result = think.solve("Calculate 12 * 12", ground_truth="144")
        >>> print(result['think_at_n']['answer'])
        '144'
    """
    
    def __init__(
        self,
        dtr_engine: DTREngine,
        n: int = 48,
        eta: float = 0.5,
        prefix_length: int = 50,
        max_tokens: int = 500
    ):
        """
        Initialize Think@n with a DTREngine.
        
        Args:
            dtr_engine: Initialized DTREngine for generation
            n: Number of samples per problem (default: 48 from paper)
            eta: Fraction of top samples to keep (default: 0.5 = 50%)
            prefix_length: Tokens for DTR estimation (default: 50 from paper)
            max_tokens: Max tokens per full generation (default: 500)
        """
        self.engine = dtr_engine
        self.n = n
        self.eta = eta
        self.prefix_length = prefix_length
        self.max_tokens = max_tokens
    
    def solve(
        self,
        problem: str,
        ground_truth: Optional[str] = None,
        n: Optional[int] = None,
        eta: Optional[float] = None,
        early_stop: bool = True
    ) -> Dict:
        """
        Solve a problem using Think@n and compare with baselines.
        
        Args:
            problem: Problem text/prompt
            ground_truth: Correct answer (optional, for accuracy calculation)
            n: Override default number of samples
            eta: Override default selection fraction
            early_stop: Whether to use early stopping (True = Think@n, False = full Cons@n)
            
        Returns:
            Dictionary with results for all methods:
            {
                'samples': List of generated samples,
                'think_at_n': {'answer': str, 'cost': int, 'accuracy': float},
                'cons_at_n': {'answer': str, 'cost': int, 'accuracy': float},
                'short_at_n': {'answer': str, 'cost': int, 'accuracy': float},
                'long_at_n': {'answer': str, 'cost': int, 'accuracy': float},
                'mean_at_n': {'accuracy': float}
            }
        """
        n = n or self.n
        eta = eta or self.eta
        
        print(f"\n{'='*70}")
        print(f"Problem: {problem[:80]}{'...' if len(problem) > 80 else ''}")
        print(f"{'='*70}")
        
        # Generate n samples with DTR computation
        samples = self.engine.generate_n_samples(
            prompt=problem,
            n=n,
            prefix_length=self.prefix_length,
            max_tokens=self.max_tokens,
            early_stop=early_stop,
            eta=eta
        )
        
        # Calculate answers using different aggregation methods
        think_answer = think_at_n_vote(samples, eta=eta)
        cons_answer = cons_at_n(samples)
        short_answer = short_at_n(samples, eta=eta)
        long_answer = long_at_n(samples, eta=eta)
        
        # Calculate costs
        if early_stop:
            think_cost = calculate_cost_with_prefix(samples, self.prefix_length, eta)
        else:
            think_cost = calculate_cost(samples)
        
        cons_cost = calculate_cost(samples)
        
        # Short@n cost: prefix overhead + shortest samples
        sorted_by_length = sorted(samples, key=lambda x: x.get('tokens', 0))
        k_short = max(1, int(eta * n))
        short_cost = (self.prefix_length * n) + sum(
            max(0, s['tokens'] - self.prefix_length) 
            for s in sorted_by_length[:k_short]
        )
        
        # Long@n cost: same as Cons@n (must generate all to find longest)
        long_cost = cons_cost
        
        # Calculate accuracies if ground truth provided
        think_correct = None
        cons_correct = None
        short_correct = None
        long_correct = None
        mean_accuracy = None
        
        if ground_truth is not None:
            think_correct = answers_match(think_answer, ground_truth)
            cons_correct = answers_match(cons_answer, ground_truth)
            short_correct = answers_match(short_answer, ground_truth)
            long_correct = answers_match(long_answer, ground_truth)
            mean_accuracy = mean_at_n(samples, ground_truth)
        
        # Build result dictionary
        result = {
            'problem': problem,
            'ground_truth': ground_truth,
            'samples': samples,
            'think_at_n': {
                'answer': think_answer,
                'cost': think_cost,
                'correct': think_correct,
                'method': 'Think@n (DTR-based)'
            },
            'cons_at_n': {
                'answer': cons_answer,
                'cost': cons_cost,
                'correct': cons_correct,
                'method': 'Cons@n (Self-Consistency)'
            },
            'short_at_n': {
                'answer': short_answer,
                'cost': short_cost,
                'correct': short_correct,
                'method': 'Short@n (Length-based)'
            },
            'long_at_n': {
                'answer': long_answer,
                'cost': long_cost,
                'correct': long_correct,
                'method': 'Long@n (Length-based)'
            },
            'mean_at_n': {
                'accuracy': mean_accuracy,
                'method': 'Mean@n (No aggregation)'
            }
        }
        
        # Print summary
        self._print_results(result)
        
        return result
    
    def _print_results(self, result: Dict):
        """Print formatted results for a problem."""
        print(f"\n{'-'*70}")
        print("RESULTS")
        print(f"{'-'*70}")
        
        methods = ['think_at_n', 'cons_at_n', 'short_at_n', 'long_at_n']
        
        print(f"{'Method':<25} {'Answer':<15} {'Cost':<10} {'Correct':<10}")
        print(f"{'-'*70}")
        
        for method in methods:
            data = result[method]
            answer = data['answer'] or 'N/A'
            cost = data['cost']
            correct = data['correct']
            
            # Format correct column
            if correct is None:
                correct_str = '-'
            else:
                correct_str = '✓' if correct else '✗'
            
            print(f"{data['method']:<25} {answer:<15} {cost:<10} {correct_str:<10}")
        
        # Mean@n (special case - no answer, just accuracy)
        if result['mean_at_n']['accuracy'] is not None:
            acc = result['mean_at_n']['accuracy']
            print(f"{result['mean_at_n']['method']:<25} {'N/A':<15} {'-':<10} {acc:.1%}")
        
        print(f"{'-'*70}")
        
        # Cost savings
        if result['think_at_n']['cost'] and result['cons_at_n']['cost']:
            savings = (1 - result['think_at_n']['cost'] / result['cons_at_n']['cost']) * 100
            print(f"Think@n cost savings vs Cons@n: {savings:.1f}%")
        
        if result['ground_truth']:
            print(f"Ground truth: {result['ground_truth']}")
        
        print(f"{'='*70}\n")
    
    def rank_samples_by_dtr(self, samples: List[Dict]) -> List[Dict]:
        """
        Rank samples by DTR in descending order.
        
        Args:
            samples: List of sample dicts with 'dtr' key
            
        Returns:
            Sorted list (highest DTR first)
        """
        return sorted(samples, key=lambda x: x.get('dtr', 0.0), reverse=True)
    
    def select_top_samples(self, samples: List[Dict], eta: Optional[float] = None) -> List[Dict]:
        """
        Select top η% samples by DTR.
        
        Args:
            samples: List of sample dicts
            eta: Fraction to keep (uses self.eta if None)
            
        Returns:
            Top η% samples
        """
        eta = eta or self.eta
        ranked = self.rank_samples_by_dtr(samples)
        k = max(1, int(eta * len(samples)))
        return ranked[:k]
    
    def get_sample_stats(self, samples: List[Dict]) -> Dict:
        """
        Get statistics about generated samples.
        
        Args:
            samples: List of sample dicts
            
        Returns:
            Dictionary with min/max/mean DTR and token counts
        """
        if not samples:
            return {}
        
        dtrs = [s.get('dtr', 0.0) for s in samples]
        tokens = [s.get('tokens', 0) for s in samples]
        full_gens = sum(1 for s in samples if s.get('full_generation', False))
        
        return {
            'num_samples': len(samples),
            'full_generations': full_gens,
            'prefix_only': len(samples) - full_gens,
            'dtr': {
                'min': min(dtrs),
                'max': max(dtrs),
                'mean': sum(dtrs) / len(dtrs)
            },
            'tokens': {
                'min': min(tokens),
                'max': max(tokens),
                'mean': sum(tokens) / len(tokens),
                'total': sum(tokens)
            }
        }

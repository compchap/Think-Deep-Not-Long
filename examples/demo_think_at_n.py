"""
Demo: Think@n Inference Scaling on Simple Math Problems.

Compares Think@n with baseline methods (Cons@n, Short@n, Long@n) on simple math problems.
Shows accuracy and cost reduction from DTR-based sample selection.

Usage:
    uv run examples/demo_think_at_n.py --model qwen.6b --n 12 --eta 0.5
"""

import sys
import os
import argparse

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.dtr_engine import DTREngine
from src.think_at_n import ThinkAtN
from src.config.models import MODELS

# Test problems: simple math with known answers
TEST_PROBLEMS = [
    {
        "problem": "Calculate 12 * 12: ",
        "answer": "144"
    },
    {
        "problem": "What is the square root of 144? ",
        "answer": "12"
    },
    {
        "problem": "If x + 5 = 13, what is x? ",
        "answer": "8"
    },
    {
        "problem": "Calculate 15 + 27: ",
        "answer": "42"
    },
    {
        "problem": "What is 100 divided by 4? ",
        "answer": "25"
    }
]

def main():
    parser = argparse.ArgumentParser(
        description="Demo: Think@n vs Baselines on Simple Math"
    )
    parser.add_argument(
        "--model",
        default="qwen.6b",
        choices=list(MODELS.keys()),
        help="Model to use (must be downloaded first)"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=12,
        help="Number of samples per problem (paper uses 48, we use 12 for faster demo)"
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.5,
        help="Fraction of top samples to keep (0.5 = top 50%%)"
    )
    parser.add_argument(
        "--prefix-length",
        type=int,
        default=50,
        help="Prefix length for DTR estimation (paper uses 50)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Max tokens per sample (paper uses natural stop/EOS; we cap at 100 for fast demo on simple math)"
    )
    parser.add_argument(
        "--num-problems",
        type=int,
        default=3,
        help="Number of problems to test (1-5)"
    )
    
    args = parser.parse_args()
    

    print("="*80)
    print("Think@n Demo: Test-Time Scaling with Deep-Thinking Ratio")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Model: {args.model}")
    print(f"  Samples per problem (n): {args.n}")
    print(f"  Top sample fraction (η): {args.eta}")
    print(f"  Prefix length: {args.prefix_length}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Problems to test: {args.num_problems}/{len(TEST_PROBLEMS)}")
    print()
    
    # Load model
    print("Loading model...")
    model_config = MODELS[args.model]
    engine = DTREngine(**model_config)
    
    # Initialize Think@n
    think = ThinkAtN(
        dtr_engine=engine,
        n=args.n,
        eta=args.eta,
        prefix_length=args.prefix_length,
        max_tokens=args.max_tokens
    )
    
    # Test problems
    problems = TEST_PROBLEMS[:args.num_problems]
    results = []
    
    for i, test_case in enumerate(problems, 1):
        print(f"\n{'#'*80}")
        print(f"Problem {i}/{len(problems)}")
        print(f"{'#'*80}")
        
        result = think.solve(
            problem=test_case["problem"],
            ground_truth=test_case["answer"],
            early_stop=True
        )
        results.append(result)
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY ACROSS ALL PROBLEMS")
    print("="*80)
    
    # Aggregate results
    methods = ['think_at_n', 'cons_at_n', 'short_at_n', 'long_at_n']
    summary = {method: {'correct': 0, 'total_cost': 0} for method in methods}
    
    for result in results:
        for method in methods:
            if result[method]['correct']:
                summary[method]['correct'] += 1
            summary[method]['total_cost'] += result[method]['cost']
    
    # Print summary table
    print(f"\n{'Method':<25} {'Accuracy':<15} {'Total Cost':<15} {'Avg Cost'}")
    print("-"*80)
    
    for method in methods:
        accuracy = summary[method]['correct'] / len(results)
        total_cost = summary[method]['total_cost']
        avg_cost = total_cost / len(results)
        method_name = results[0][method]['method']
        
        print(f"{method_name:<25} {accuracy:>6.1%}{'':<8} {total_cost:>10,}{'':<5} {avg_cost:>8,.0f}")
    
    # Cost comparison
    print("\n" + "-"*80)
    think_total = summary['think_at_n']['total_cost']
    cons_total = summary['cons_at_n']['total_cost']
    
    if cons_total > 0:
        savings = (1 - think_total / cons_total) * 100
        print(f"Think@n cost savings vs Cons@n: {savings:.1f}%")
        print(f"Think@n uses {think_total:,} tokens vs Cons@n {cons_total:,} tokens")
    
    # Accuracy comparison
    think_acc = summary['think_at_n']['correct'] / len(results)
    cons_acc = summary['cons_at_n']['correct'] / len(results)
    
    print(f"\nThink@n accuracy: {think_acc:.1%}")
    print(f"Cons@n accuracy: {cons_acc:.1%}")
    
    if think_acc >= cons_acc:
        print(f"✓ Think@n matches/exceeds Cons@n accuracy")
    else:
        print(f"⚠ Think@n accuracy below Cons@n (may improve with more samples)")
    
    print("\n" + "="*80)
    print("Demo completed!")
    print("="*80)


if __name__ == "__main__":
    main()

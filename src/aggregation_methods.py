"""
Aggregation methods for Think@n and baseline strategies.

Implements various sample aggregation strategies from the paper:
- Cons@n: Self-consistency (majority vote over all samples)
- Short@n: Vote over shortest samples
- Long@n: Vote over longest samples  
- Think@n: Vote over highest DTR samples
"""

from typing import List, Optional, Callable
from collections import Counter
from src.answer_extraction import extract_answer, normalize_answer


def majority_vote(answers: List[Optional[str]]) -> Optional[str]:
    """
    Perform majority voting over a list of answers.
    
    Args:
        answers: List of answer strings (may contain None)
        
    Returns:
        Most common answer, or None if no answers or tie
        
    Examples:
        >>> majority_vote(["144", "144", "12"])
        '144'
        >>> majority_vote(["12", "13", "14"])
        '12'  # Returns first in case of tie
    """
    # Filter out None values and normalize
    valid_answers = [normalize_answer(ans) for ans in answers if ans is not None]
    
    if not valid_answers:
        return None
    
    # Count occurrences
    counter = Counter(valid_answers)
    
    # Return most common (first in case of tie)
    most_common = counter.most_common(1)
    return most_common[0][0] if most_common else None


def cons_at_n(samples: List[dict], extract_fn: Callable = extract_answer) -> Optional[str]:
    """
    Consensus@n (Self-Consistency): Majority vote over all n samples.
    
    Baseline from paper Section 4 - uses all samples without filtering.
    
    Args:
        samples: List of sample dicts with 'text' key
        extract_fn: Function to extract answer from text
        
    Returns:
        Final answer via majority vote
    """
    answers = [extract_fn(sample['text']) for sample in samples]
    return majority_vote(answers)


def short_at_n(samples: List[dict], eta: float = 0.5, extract_fn: Callable = extract_answer) -> Optional[str]:
    """
    Short@n: Majority vote over shortest η% of samples.
    
    Baseline from paper - selects samples with minimum token count.
    
    Args:
        samples: List of sample dicts with 'text' and 'tokens' keys
        eta: Fraction of samples to keep (e.g., 0.5 = shortest 50%)
        extract_fn: Function to extract answer from text
        
    Returns:
        Final answer via majority vote on shortest samples
    """
    if not samples:
        return None
    
    # Sort by token count (ascending)
    sorted_samples = sorted(samples, key=lambda x: x.get('tokens', len(x['text'])))
    
    # Select shortest eta%
    k = max(1, int(eta * len(samples)))
    selected = sorted_samples[:k]
    
    answers = [extract_fn(sample['text']) for sample in selected]
    return majority_vote(answers)


def long_at_n(samples: List[dict], eta: float = 0.5, extract_fn: Callable = extract_answer) -> Optional[str]:
    """
    Long@n: Majority vote over longest η% of samples.
    
    Baseline from paper - selects samples with maximum token count.
    
    Args:
        samples: List of sample dicts with 'text' and 'tokens' keys
        eta: Fraction of samples to keep (e.g., 0.5 = longest 50%)
        extract_fn: Function to extract answer from text
        
    Returns:
        Final answer via majority vote on longest samples
    """
    if not samples:
        return None
    
    # Sort by token count (descending)
    sorted_samples = sorted(samples, key=lambda x: x.get('tokens', len(x['text'])), reverse=True)
    
    # Select longest eta%
    k = max(1, int(eta * len(samples)))
    selected = sorted_samples[:k]
    
    answers = [extract_fn(sample['text']) for sample in selected]
    return majority_vote(answers)


def think_at_n(samples: List[dict], eta: float = 0.5, extract_fn: Callable = extract_answer) -> Optional[str]:
    """
    Think@n: Majority vote over highest DTR η% of samples.
    
    Main algorithm from paper - selects samples with highest deep-thinking ratio.
    
    Args:
        samples: List of sample dicts with 'text' and 'dtr' keys
        eta: Fraction of samples to keep (e.g., 0.5 = top 50% by DTR)
        extract_fn: Function to extract answer from text
        
    Returns:
        Final answer via majority vote on highest DTR samples
    """
    if not samples:
        return None
    
    # Sort by DTR (descending)
    sorted_samples = sorted(samples, key=lambda x: x.get('dtr', 0.0), reverse=True)
    
    # Select top eta%
    k = max(1, int(eta * len(samples)))
    selected = sorted_samples[:k]
    
    answers = [extract_fn(sample['text']) for sample in selected]
    return majority_vote(answers)


def mean_at_n(samples: List[dict], ground_truth: str, extract_fn: Callable = extract_answer) -> float:
    """
    Mean@n: Average accuracy across all samples (no aggregation).
    
    Baseline from paper - measures average individual sample accuracy.
    
    Args:
        samples: List of sample dicts with 'text' key
        ground_truth: Correct answer for comparison
        extract_fn: Function to extract answer from text
        
    Returns:
        Accuracy as fraction of correct samples
    """
    if not samples:
        return 0.0
    
    correct = 0
    normalized_gt = normalize_answer(ground_truth)
    
    for sample in samples:
        answer = extract_fn(sample['text'])
        if normalize_answer(answer) == normalized_gt:
            correct += 1
    
    return correct / len(samples)


def calculate_cost(samples: List[dict]) -> int:
    """
    Calculate total token cost across samples.
    
    Sums the 'tokens' field from each sample.
    Used for cost comparison in paper experiments.
    
    Args:
        samples: List of sample dicts with 'tokens' key
        
    Returns:
        Total token count
    """
    return sum(sample.get('tokens', 0) for sample in samples)


def calculate_cost_with_prefix(
    samples: List[dict], 
    prefix_length: int, 
    eta: float = 0.5
) -> int:
    """
    Calculate Think@n cost with early stopping overhead.
    
    From paper Section 4:
    Cost = (prefix_length × n) + Σ|S_top_i| for top η×n samples
    
    Args:
        samples: List of sample dicts with 'tokens' and 'full_generation' keys
        prefix_length: Number of prefix tokens for DTR estimation
        eta: Fraction of samples continued to completion
        
    Returns:
        Total token cost with early stopping
    """
    n = len(samples)
    
    # Prefix cost: all samples generate prefix_length tokens
    prefix_cost = prefix_length * n
    
    # Continuation cost: only top eta% samples continue
    # Sort by DTR to identify top samples
    sorted_samples = sorted(samples, key=lambda x: x.get('dtr', 0.0), reverse=True)
    k = max(1, int(eta * n))
    top_samples = sorted_samples[:k]
    
    # Sum tokens from full generations (excluding prefix which is counted separately)
    continuation_cost = sum(
        max(0, sample.get('tokens', 0) - prefix_length) 
        for sample in top_samples 
        if sample.get('full_generation', False)
    )
    
    return prefix_cost + continuation_cost

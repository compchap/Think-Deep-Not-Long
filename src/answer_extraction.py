"""
Answer extraction utilities for Think@n.

Extracts numerical answers from model-generated text for evaluation.
Supports LaTeX \boxed{} notation and fallback patterns.
"""

import re
from typing import Optional


def extract_answer(text: str) -> Optional[str]:
    """
    Extract answer from generated text.
    
    Tries multiple strategies in order:
    1. LaTeX \boxed{...} notation (AIME style)
    2. "answer is X" patterns
    3. Last numerical value in text
    
    Args:
        text: Generated text containing the answer
        
    Returns:
        Extracted answer as string, or None if no answer found
        
    Examples:
        >>> extract_answer("Therefore \\boxed{144}")
        '144'
        >>> extract_answer("The answer is 12")
        '12'
        >>> extract_answer("We get x = 8")
        '8'
    """
    if not text:
        return None
    
    # Strategy 1: LaTeX \boxed{...}
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', text)
    if boxed_match:
        answer = boxed_match.group(1).strip()
        # Clean up common LaTeX artifacts
        answer = answer.replace('$', '').replace('\\', '').strip()
        return answer
    
    # Strategy 2: "answer is X" patterns (case insensitive)
    answer_patterns = [
        r'(?:the\s+)?answer\s+is\s+([+-]?\d+\.?\d*)',
        r'(?:final\s+)?answer:\s*([+-]?\d+\.?\d*)',
        r'equals?\s+([+-]?\d+\.?\d*)',
    ]
    
    for pattern in answer_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    # Strategy 3: Last numerical value (fallback)
    # Match integers and decimals, including negative numbers
    numbers = re.findall(r'[+-]?\d+\.?\d*', text)
    if numbers:
        return numbers[-1]
    
    return None


def normalize_answer(answer: Optional[str]) -> Optional[str]:
    """
    Normalize answer for comparison.
    
    - Converts to lowercase
    - Strips whitespace
    - Removes trailing .0 from decimals (144.0 -> 144)
    - Handles None gracefully
    
    Args:
        answer: Answer string to normalize
        
    Returns:
        Normalized answer or None
        
    Examples:
        >>> normalize_answer("144.0")
        '144'
        >>> normalize_answer("  12  ")
        '12'
    """
    if answer is None:
        return None
    
    answer = str(answer).strip().lower()
    
    # Remove trailing .0 from decimals
    if '.' in answer:
        try:
            num = float(answer)
            if num == int(num):
                answer = str(int(num))
        except ValueError:
            pass
    
    return answer


def answers_match(answer1: Optional[str], answer2: Optional[str]) -> bool:
    """
    Check if two answers are equivalent after normalization.
    
    Args:
        answer1: First answer
        answer2: Second answer
        
    Returns:
        True if answers match after normalization
        
    Examples:
        >>> answers_match("144", "144.0")
        True
        >>> answers_match("12", "13")
        False
    """
    norm1 = normalize_answer(answer1)
    norm2 = normalize_answer(answer2)
    
    if norm1 is None or norm2 is None:
        return False
    
    return norm1 == norm2

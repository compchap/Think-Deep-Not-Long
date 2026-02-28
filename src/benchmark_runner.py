import json
from dtr_engine import DTREngine

def run_mini_benchmark():
    engine = DTREngine()
    # Sample math problems
    test_cases = [
        {"q": "What is 15 * 15?", "a": "225"},
        {"q": "What is the square root of 144?", "a": "12"}
    ]
    
    for case in test_cases:
        full_text = ""
        is_deep_flags = []
        for word, is_deep, dtr in engine.generate_with_dtr(case['q']):
            full_text += word
            is_deep_flags.append(is_deep)
        
        correct = case['a'] in full_text
        final_dtr = sum(is_deep_flags) / len(is_deep_flags)
        print(f"Q: {case['q']} | Correct: {correct} | Final DTR: {final_dtr:.2f}")

if __name__ == "__main__":
    run_mini_benchmark()
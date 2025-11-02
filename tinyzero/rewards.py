import re
from typing import Dict, Any, Optional
import math

def extract_final_answer(text: str) -> Optional[float]:
    """
    Extract the final numerical answer from model output.
    Priority:
    1. <answer>X</answer> tags
    2. \\boxed{X}
    3. "final answer is X"
    4. Last number
    """
    # 1. Check for <answer> tags first (TinyZero format)
    answer_tag = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.IGNORECASE)
    if answer_tag:
        answer_str = answer_tag.group(1).replace(",", "").strip()
        # Extract number from the answer text
        numbers = re.findall(r'-?[\d,]+(?:\.\d+)?', answer_str)
        if numbers:
            try:
                return float(numbers[-1].replace(",", ""))
            except ValueError:
                pass
    
    # 2. LaTeX boxed answers
    boxed_match = re.search(r"\\boxed\{(.*?)\}", text)
    if boxed_match:
        answer_str = boxed_match.group(1).replace(",", "").strip()
        try:
            return float(answer_str)
        except ValueError:
            pass
    
    # 3. Explicit answer patterns
    patterns = [
        r"(?:final answer is|the answer is|result is|equals?)\s*:?\s*(-?[\d,]+(?:\.\d+)?)",
        r"=\s*(-?[\d,]+(?:\.\d+)?)\s*(?:[\.\?!]|$)",
    ]
    
    for pattern in reversed(patterns):
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            answer_str = matches[-1].replace(",", "").strip()
            try:
                return float(answer_str)
            except ValueError:
                continue
    
    # 4. Fallback: last number
    numbers = re.findall(r'-?[\d,]+(?:\.\d+)?', text)
    if numbers:
        last_num_str = numbers[-1].replace(",", "").strip()
        try:
            return float(last_num_str)
        except ValueError:
            pass
    
    return None


def check_proper_format(text: str) -> bool:
    """
    Check if response uses the proper TinyZero format:
    <think>reasoning</think>
    <answer>answer</answer>
    
    Returns True if BOTH tags are present.
    """
    has_think = bool(re.search(r"<think>.*?</think>", text, re.IGNORECASE | re.DOTALL))
    has_answer = bool(re.search(r"<answer>.*?</answer>", text, re.IGNORECASE | re.DOTALL))
    
    return has_think and has_answer


def check_multiplication_cot(text: str, num1: int, num2: int, correct_answer: int) -> bool:
    """
    Check if multiplication shows reasoning.
    Looks for:
    - Proper format (<think> tags)
    - Step-by-step explanation
    - Mathematical operations shown
    """
    # Check for proper format first
    if not check_proper_format(text):
        return False
    
    # Extract content from <think> tags
    think_match = re.search(r"<think>(.*?)</think>", text, re.IGNORECASE | re.DOTALL)
    if not think_match:
        return False
    
    think_content = think_match.group(1)
    
    # Check for multiplication statement or repeated addition
    # Pattern 1: num1 * num2 = answer
    pattern1 = rf"{num1}\s*[\*×x]\s*{num2}\s*=\s*{correct_answer}"
    if re.search(pattern1, think_content):
        return True
    
    # Pattern 2: num1 times num2
    pattern2 = rf"{num1}\s*times\s*{num2}\s*(?:is|=)\s*{correct_answer}"
    if re.search(pattern2, think_content, re.IGNORECASE):
        return True
    
    # Pattern 3: Repeated addition (e.g., 8+8+8 for 8*3)
    addition_pattern = rf'\b{num1}\b.*?\+'
    additions = len(re.findall(addition_pattern, think_content[:500]))
    if additions >= max(1, num2 - 2):
        return True
    
    return False


def check_countdown_reasoning(text: str, numbers: list, target: int) -> bool:
    """
    Check if countdown solution shows proper reasoning.
    Must have:
    - <think> and <answer> tags
    - Uses original numbers
    - Shows calculations step-by-step
    """
    # Check for proper format
    if not check_proper_format(text):
        return False
    
    # Extract content from <think> tags
    think_match = re.search(r"<think>(.*?)</think>", text, re.IGNORECASE | re.DOTALL)
    if not think_match:
        return False
    
    think_content = think_match.group(1)
    
    # Count how many original numbers appear in reasoning
    numbers_used = sum(1 for num in numbers if str(num) in think_content)
    
    # Check for operations
    has_operations = any(op in think_content for op in ['+', '-', '×', '*', '÷', '/'])
    
    # Check for calculation steps (equals signs)
    equals_count = think_content.count('=')
    has_steps = equals_count >= 2  # At least 2 calculation steps
    
    # Good reasoning = uses numbers + operations + multiple steps
    return numbers_used >= 2 and has_operations and has_steps


def compute_reward_with_partial_credit(
    generated_text: str,
    problem: Dict[str, Any],
    tolerance: float = 0.01,
    check_reasoning: bool = True
) -> float:
    """
    Compute reward with PARTIAL CREDIT for reasoning AND format.
    
    Reward Scheme (NEW - checks format!):
    - 1.0: Correct answer + proper format + good reasoning
    - 0.8: Correct answer + proper format, weak reasoning
    - 0.6: Correct answer + good reasoning, wrong format
    - 0.5: Correct answer, no format, no reasoning
    - 0.3: Wrong answer + proper format + reasoning attempt
    - 0.0: Wrong answer, no format, no reasoning
    """
    predicted_answer_num = extract_final_answer(generated_text)
    
    if predicted_answer_num is None:
        return 0.0
    
    final_answer_correct = False
    has_reasoning = False
    has_proper_format = check_proper_format(generated_text)
    
    # --- Check Final Answer ---
    if problem['task'] == 'multiplication':
        correct_answer = problem['answer']
        if math.isclose(predicted_answer_num, correct_answer, abs_tol=0.01):
            final_answer_correct = True
    
    elif problem['task'] == 'countdown':
        target = problem['target']
        if target != 0 and math.isclose(predicted_answer_num, target, rel_tol=tolerance):
            final_answer_correct = True
        elif target == 0 and math.isclose(predicted_answer_num, target, abs_tol=tolerance):
            final_answer_correct = True
    
    # --- Check for Reasoning (within proper format) ---
    if check_reasoning:
        if problem['task'] == 'multiplication':
            has_reasoning = check_multiplication_cot(
                generated_text,
                problem['num1'],
                problem['num2'],
                problem['answer']
            )
        elif problem['task'] == 'countdown':
            has_reasoning = check_countdown_reasoning(
                generated_text,
                problem['numbers'],
                problem['target']
            )
    
    # --- Apply Partial Credit ---
    if not check_reasoning:
        # Binary reward (for V* computation)
        return 1.0 if final_answer_correct else 0.0
    
    # Training rewards with format consideration
    if final_answer_correct:
        if has_proper_format and has_reasoning:
            return 1.0  # Perfect!
        elif has_proper_format:
            return 0.8  # Good format, weak reasoning
        elif has_reasoning:
            return 0.6  # Good reasoning, wrong format
        else:
            return 0.5  # Just correct answer
    else:
        # Wrong answer
        if has_proper_format and has_reasoning:
            return 0.3  # Good attempt with proper format
        else:
            return 0.0  # Nothing good


def compute_reward(
    generated_text: str,
    problem: Dict[str, Any],
    tolerance: float = 0.01,
    require_cot: bool = False
) -> float:
    """
    Binary reward function - for V* computation.
    No partial credit - just 1.0 or 0.0.
    """
    return compute_reward_with_partial_credit(
        generated_text, problem, tolerance, check_reasoning=False
    )


# Export functions
__all__ = [
    'compute_reward',
    'compute_reward_with_partial_credit',
    'extract_final_answer',
    'check_multiplication_cot',
    'check_countdown_reasoning',
    'check_proper_format'
]
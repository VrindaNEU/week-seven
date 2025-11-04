print("="*60)
print("🔍 REWARDS MODULE LOADED - VERSION 3.0 WITH PROPER WEBSHOP REWARDS")
print("="*60)
"""
Unified reward system for TinyZero and RAGEN
FIXED: Proper WebShop reward computation based on trajectory quality
"""
import re
from typing import Dict, Any, Optional, Union, List
import math


# ============================================================
# MATH REWARDS (TinyZero Original) - UNCHANGED
# ============================================================

def extract_final_answer(text: str) -> Optional[float]:
    """Extract numerical answer from model output."""
    answer_tag = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.IGNORECASE)
    if answer_tag:
        answer_str = answer_tag.group(1).replace(",", "").strip()
        numbers = re.findall(r'-?[\d,]+(?:\.\d+)?', answer_str)
        if numbers:
            try:
                return float(numbers[-1].replace(",", ""))
            except ValueError:
                pass
    
    boxed_match = re.search(r"\\boxed\{(.*?)\}", text)
    if boxed_match:
        answer_str = boxed_match.group(1).replace(",", "").strip()
        try:
            return float(answer_str)
        except ValueError:
            pass
    
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
    
    numbers = re.findall(r'-?[\d,]+(?:\.\d+)?', text)
    if numbers:
        last_num_str = numbers[-1].replace(",", "").strip()
        try:
            return float(last_num_str)
        except ValueError:
            pass
    
    return None


def compute_math_reward(
    generated_text: str,
    problem: Dict[str, Any],
    tolerance: float = 0.01,
    check_reasoning: bool = False
) -> float:
    """Compute reward for math problems."""
    predicted_answer_num = extract_final_answer(generated_text)
    
    if predicted_answer_num is None:
        return 0.0
    
    if problem.get('task') == 'multiplication':
        correct_answer = problem['answer']
        if math.isclose(predicted_answer_num, correct_answer, abs_tol=0.01):
            return 1.0
    
    elif problem.get('task') == 'countdown':
        target = problem['target']
        if target != 0 and math.isclose(predicted_answer_num, target, rel_tol=tolerance):
            return 1.0
        elif target == 0 and math.isclose(predicted_answer_num, target, abs_tol=tolerance):
            return 1.0
    
    return 0.0


# ============================================================
# WEBSHOP REWARDS (RAGEN Extension) - COMPLETELY REWRITTEN
# ============================================================

def parse_webshop_action(action: str) -> tuple:
    """
    Parse a WebShop action to extract type and argument.
    
    Returns:
        (action_type, argument) tuple
        Examples:
        - "search[blue headphones]" → ("search", "blue headphones")
        - "click[B09QKP7XQL]" → ("click", "B09QKP7XQL")
        - "buy now" → ("buy", "")
    """
    action = str(action).strip().lower()
    
    # Try bracket format: action[argument]
    match = re.match(r'(\w+)\[(.*?)\]', action)
    if match:
        return (match.group(1), match.group(2))
    
    # Simple commands
    if 'buy' in action:
        return ("buy", "")
    if 'back' in action:
        return ("back", "")
    
    return ("unknown", action)


def is_valid_webshop_action(action: str) -> bool:
    """
    Check if action follows WebShop format.
    Valid actions:
    - search[query]
    - click[product_id]
    - buy now
    - back
    """
    action = str(action).strip().lower()
    
    # Check for valid formats
    valid_patterns = [
        r'^search\[.+\]$',       # search[something]
        r'^click\[[\w-]+\]$',    # click[product_id]
        r'^buy(\s+now)?$',       # buy or buy now
        r'^back$',               # back
    ]
    
    return any(re.match(pattern, action) for pattern in valid_patterns)


def compute_webshop_reward(
    trajectory: Dict[str, Any],
    task: Dict[str, Any]
) -> float:
    """
    Compute WebShop reward based on:
    1. Final outcome (did agent buy correct item?)
    2. Action quality (are actions well-formed?)
    3. Efficiency (fewer steps = better)
    
    Reward breakdown:
    - 1.0: Successfully bought correct item
    - 0.5-0.8: Bought item but wrong attributes
    - 0.3-0.5: Made progress (searched, clicked valid products)
    - 0.1-0.3: Valid action format but no progress
    - 0.0: Invalid actions or no attempt
    """
    if not trajectory or not isinstance(trajectory, dict):
        return 0.0
    
    actions = trajectory.get('actions', [])
    if not actions:
        return 0.0
    
    env_reward = float(trajectory.get('total_reward', 0.0))
    
    # CASE 1: Perfect success (WebShop gives reward=1.0 for correct purchase)
    if env_reward >= 0.9:
        # Efficiency bonus: fewer steps is better
        num_turns = trajectory.get('num_turns', len(actions))
        efficiency = max(0.0, 1.0 - (num_turns - 3) * 0.05)  # Penalty after 3 steps
        return min(1.0, 0.9 + efficiency * 0.1)
    
    # CASE 2: Partial success (WebShop sometimes gives partial rewards)
    if env_reward > 0.0:
        return 0.5 + env_reward * 0.4  # Scale 0.1-0.9 env reward to 0.5-0.8
    
    # CASE 3: No environment reward - evaluate action quality
    reward = 0.0
    
    # Analyze action sequence
    action_types = []
    valid_count = 0
    
    for action in actions:
        action_type, arg = parse_webshop_action(action)
        action_types.append(action_type)
        
        if is_valid_webshop_action(action):
            valid_count += 1
    
    # Reward 1: Valid action format (teaches syntax)
    if valid_count > 0:
        format_quality = valid_count / len(actions)
        reward += 0.15 * format_quality
    
    # Reward 2: Logical action sequence (search → click → buy)
    has_search = 'search' in action_types
    has_click = 'click' in action_types
    has_buy = 'buy' in action_types
    
    if has_search:
        reward += 0.15  # Good: agent tried to search
        
        # Check if search has meaningful query
        for action in actions:
            if action.startswith('search[') and len(action) > 10:
                # Non-trivial search query
                reward += 0.10
                break
    
    if has_search and has_click:
        reward += 0.15  # Better: search then click
    
    if has_search and has_click and has_buy:
        reward += 0.15  # Best sequence even if wrong product
    
    # Reward 3: Avoid repetition (shows learning)
    unique_actions = len(set(actions))
    if len(actions) > 1:
        diversity = unique_actions / len(actions)
        reward += 0.10 * diversity
    
    # Penalty: Extremely long or short actions (likely garbage)
    avg_len = sum(len(str(a)) for a in actions) / len(actions)
    if avg_len < 5 or avg_len > 100:
        reward *= 0.5  # Reduce reward for weird lengths
    
    return min(reward, 0.8)  # Cap at 0.8 without actual success


# ============================================================
# UNIFIED REWARD FUNCTION (Auto-Detect)
# ============================================================

def compute_reward(
    generated_output: Union[str, Dict[str, Any]],
    problem: Dict[str, Any],
    tolerance: float = 0.01,
    require_cot: bool = False
) -> float:
    """
    Unified reward function - auto-detects environment type.
    
    Supports:
    - TinyZero math tasks (text input)
    - RAGEN WebShop tasks (trajectory dict input)
    
    Args:
        generated_output: Either text string (math) or trajectory dict (WebShop)
        problem: Problem/task specification
        tolerance: Tolerance for numerical answers
        require_cot: Whether to require chain-of-thought (math only)
    
    Returns:
        Reward value (0.0 to 1.0)
    """
    # Auto-detect based on input type
    if isinstance(generated_output, dict):
        # Dictionary with trajectory info → WebShop
        if 'actions' in generated_output or 'total_reward' in generated_output:
            return compute_webshop_reward(generated_output, problem)
    
    # String output → Math task
    if isinstance(generated_output, str):
        return compute_math_reward(generated_output, problem, tolerance, require_cot)
    
    # Unknown type
    print(f"⚠️  Unknown output type for reward: {type(generated_output)}")
    return 0.0


def compute_reward_with_partial_credit(
    generated_text: str,
    problem: Dict[str, Any],
    tolerance: float = 0.01,
    check_reasoning: bool = True
) -> float:
    """
    Compute reward with PARTIAL CREDIT for reasoning.
    (For TinyZero math tasks only - kept for compatibility)
    """
    # For now, just use binary rewards
    return compute_math_reward(generated_text, problem, tolerance, False)


# ============================================================
# TESTING / DEBUGGING HELPERS
# ============================================================

def debug_webshop_reward(trajectory: Dict[str, Any], task: Dict[str, Any]) -> None:
    """Print detailed reward breakdown for debugging."""
    print("\n" + "="*60)
    print("REWARD DEBUG")
    print("="*60)
    
    actions = trajectory.get('actions', [])
    env_reward = trajectory.get('total_reward', 0.0)
    
    print(f"Environment Reward: {env_reward:.3f}")
    print(f"Number of actions: {len(actions)}")
    print(f"\nActions:")
    for i, action in enumerate(actions):
        action_type, arg = parse_webshop_action(action)
        is_valid = is_valid_webshop_action(action)
        print(f"  {i+1}. {action[:60]}")
        print(f"     → Type: {action_type}, Valid: {is_valid}")
    
    computed_reward = compute_webshop_reward(trajectory, task)
    print(f"\nComputed Reward: {computed_reward:.3f}")
    print("="*60 + "\n")


# Export functions
__all__ = [
    'compute_reward',
    'compute_reward_with_partial_credit',
    'compute_math_reward',
    'compute_webshop_reward',
    'extract_final_answer',
    'is_valid_webshop_action',
    'parse_webshop_action',
    'debug_webshop_reward',
]
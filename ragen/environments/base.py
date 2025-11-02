"""
Base class for multi-turn environments
This defines the interface all environments must follow
"""
from typing import Dict, Tuple, Any, Optional
from abc import ABC, abstractmethod


class MultiTurnEnvironment(ABC):
    """
    Base class for multi-turn RL environments.
    
    Think of this like a video game interface:
    - reset() = start new game
    - step() = take action, see what happens
    - is_done() = check if game is over
    """
    
    def __init__(self, config: Dict):
        """
        Initialize environment
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.max_turns = config.get('max_turns', 5)
        self.current_turn = 0
        self.history = []  # Store all (state, action, reward) tuples
    
    @abstractmethod
    def reset(self, task_data: Dict) -> str:
        """
        Start a new episode.
        
        Args:
            task_data: Task information (like "find beige pillows under $30")
        
        Returns:
            Initial observation (what the agent sees first)
        
        Example:
            task_data = {'instruction': 'Find beige pillows under $30'}
            observation = env.reset(task_data)
            # observation = "WebShop Search Page\n[search box]"
        """
        raise NotImplementedError("Each environment must implement reset()")
    
    @abstractmethod
    def step(self, action: str) -> Tuple[str, float, bool, Dict]:
        """
        Execute one action in the environment.
        
        Args:
            action: Action string from the agent (e.g., "search[pillows]")
        
        Returns:
            observation: What agent sees after action
            reward: Reward for this step (usually 0 until end)
            done: Whether episode is finished
            info: Extra information dictionary
        
        Example:
            obs, reward, done, info = env.step("search[beige pillows]")
            # obs = "Search Results\n1. Beige Pillow Set..."
            # reward = 0.0 (not done yet)
            # done = False
            # info = {'turn': 1, 'action_valid': True}
        """
        raise NotImplementedError("Each environment must implement step()")
    
    @abstractmethod
    def compute_reward(self, trajectory: list) -> float:
        """
        Calculate final reward for entire trajectory.
        
        Args:
            trajectory: List of (state, action, reward) tuples
        
        Returns:
            Total reward (usually 0.0 or 1.0 for success/failure)
        """
        raise NotImplementedError("Each environment must implement compute_reward()")
    
    def is_done(self) -> bool:
        """
        Check if episode should end.
        
        Returns:
            True if max turns reached or task completed
        """
        return self.current_turn >= self.max_turns
    
    def get_trajectory(self) -> list:
        """
        Get the full trajectory so far.
        
        Returns:
            List of (state, action, reward) tuples
        """
        return self.history.copy()
    
    def add_to_history(self, state: str, action: str, reward: float):
        """
        Add one step to history.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
        """
        self.history.append({
            'state': state,
            'action': action,
            'reward': reward,
            'turn': self.current_turn
        })
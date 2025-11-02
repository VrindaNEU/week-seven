"""
Real WebShop environment wrapper for RAGEN
Uses the actual WebShop framework
"""
import sys
from pathlib import Path
from typing import Dict, Tuple

# Add WebShop to Python path
WEBSHOP_PATH = Path(__file__).parent.parent.parent.parent / "WebShop"
sys.path.insert(0, str(WEBSHOP_PATH))

from web_agent_site.envs import WebAgentTextEnv
from .base import MultiTurnEnvironment


class WebShopEnvironment(MultiTurnEnvironment):
    """
    Wrapper around the REAL WebShop environment.
    
    This uses the actual Princeton WebShop benchmark with:
    - Real product database (1000 products)
    - Real search functionality
    - Real attribute matching
    - Official WebShop tasks
    """
    
    def __init__(self, config: Dict):
        """Initialize real WebShop environment"""
        super().__init__(config)
        
        # WebShop configuration
        self.max_turns = config.get('environment', {}).get('max_turns', 10)
        num_products = config.get('environment', {}).get('num_products', 100)
        
        print(f"Initializing REAL WebShop Environment...")
        print(f"  Max turns: {self.max_turns}")
        print(f"  Product count: {num_products}")
        
        # Create real WebShop environment
        self.env = WebAgentTextEnv(
            observation_mode='text',  # Text-based observations
            num_products=num_products,
        )
        
        # Episode state
        self.current_instruction = None
        self.session = None
        
        print("✓ REAL WebShop Environment initialized!")
    
    def reset(self, task_data: Dict) -> str:
        """
        Start new WebShop task.
        
        Args:
            task_data: Can contain 'session' for specific task
        
        Returns:
            Initial observation from WebShop
        """
        self.current_turn = 0
        self.history = []
        
        # Get session ID if provided
        self.session = task_data.get('session', None)
        
        # Reset WebShop environment
        obs_tuple = self.env.reset(session=self.session)
        
        # WebShop returns (observation, info) tuple
        if isinstance(obs_tuple, tuple):
            obs = obs_tuple[0]
        else:
            obs = obs_tuple
        
        # Extract instruction from observation
        if '[SEP]' in obs:
            parts = obs.split('[SEP]')
            if len(parts) >= 3:
                self.current_instruction = parts[2].strip()
        
        return obs
    
    def step(self, action: str) -> Tuple[str, float, bool, Dict]:
        """
        Execute action in real WebShop.
        
        Args:
            action: Action string from agent
        
        Returns:
            observation, reward, done, info
        """
        self.current_turn += 1
        
        # Execute action in real WebShop
        try:
            result = self.env.step(action)
            
            # WebShop returns (obs, reward, done, info)
            if len(result) == 4:
                obs, reward, done, info = result
            else:
                obs, reward, done = result
                info = {}
                
        except Exception as e:
            print(f"WebShop step error: {e}")
            obs = f"Error: Invalid action '{action}'"
            reward = 0.0
            done = True
            info = {'error': str(e)}
        
        # Add turn limit check
        if self.current_turn >= self.max_turns and not done:
            done = True
            info['timeout'] = True
        
        # Store in history
        self.add_to_history(obs, action, reward)
        
        return obs, reward, done, info
    
    def compute_reward(self, trajectory: list) -> float:
        """
        Calculate reward from trajectory.
        WebShop provides rewards automatically.
        """
        if not trajectory:
            return 0.0
        
        total = sum(step.get('reward', 0.0) for step in trajectory)
        return float(total)
    
    def render_text(self, state: str) -> str:
        """Render state as text"""
        return state
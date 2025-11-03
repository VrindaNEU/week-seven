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
        
        try:
            # Reset WebShop environment
            obs_tuple = self.env.reset(session=self.session)
            
            # WebShop returns (observation, info) tuple
            if isinstance(obs_tuple, tuple):
                obs = obs_tuple[0]
            else:
                obs = obs_tuple
            
            # Validate observation
            if obs is None or not isinstance(obs, str):
                print(f"⚠️  Invalid observation from reset: {obs}")
                obs = "Welcome to WebShop! Please search for products."
            
            # Extract instruction from observation
            if '[SEP]' in obs:
                parts = obs.split('[SEP]')
                if len(parts) >= 3:
                    self.current_instruction = parts[2].strip()
            
            return obs
            
        except Exception as e:
            print(f"⚠️  Error in reset: {e}")
            import traceback
            traceback.print_exc()
            return "Welcome to WebShop! Please search for products."
    
    def step(self, action: str) -> Tuple[str, float, bool, Dict]:
        """
        Execute action in real WebShop.
        
        Args:
            action: Action string from agent
        
        Returns:
            observation, reward, done, info
        """
        self.current_turn += 1
        
        # Initialize info dict FIRST to prevent NoneType errors
        info = {
            'turn': self.current_turn,
            'max_turns': self.max_turns,
            'timeout': False,
            'success': False,
            'error': None
        }
        
        # Validate action
        if action is None or (isinstance(action, str) and not action.strip()):
            print(f"⚠️  Invalid action at turn {self.current_turn}: {repr(action)}")
            info['error'] = 'invalid_action'
            obs = "Invalid action. Please try again."
            return obs, 0.0, True, info
        
        # Execute action in real WebShop
        try:
            result = self.env.step(action)
            
            # WebShop should return (obs, reward, done, info)
            if isinstance(result, tuple):
                if len(result) == 4:
                    obs, reward, done, raw_info = result
                elif len(result) == 3:
                    obs, reward, done = result
                    raw_info = {}
                else:
                    raise ValueError(f"Unexpected result length: {len(result)}")
            else:
                # Unexpected return type
                print(f"⚠️  WebShop returned non-tuple: {type(result)}")
                obs = str(result) if result else "Error: Invalid response"
                reward = 0.0
                done = True
                raw_info = {}
            
            # Validate observation
            if obs is None:
                print(f"⚠️  WebShop returned None observation")
                obs = "No observation available."
                info['error'] = 'none_observation'
            
            # Update info with WebShop's info (safely)
            if raw_info and isinstance(raw_info, dict):
                info.update(raw_info)
            
            # Convert reward to float
            reward = float(reward) if reward is not None else 0.0
            
            # Convert done to bool
            done = bool(done) if done is not None else False
                
        except Exception as e:
            print(f"⚠️  WebShop step error at turn {self.current_turn}: {e}")
            import traceback
            traceback.print_exc()
            
            obs = f"Error occurred: {str(e)}"
            reward = 0.0
            done = True
            info['error'] = str(e)
        
        # Check for timeout (after getting results from env)
        if self.current_turn >= self.max_turns and not done:
            done = True
            info['timeout'] = True
        
        # Check for success (reward > 0 typically means success in WebShop)
        if reward > 0:
            info['success'] = True
        
        # Store in history
        try:
            self.add_to_history(obs, action, reward)
        except Exception as e:
            print(f"⚠️  Error adding to history: {e}")
        
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
        if state is None:
            return ""
        return str(state)
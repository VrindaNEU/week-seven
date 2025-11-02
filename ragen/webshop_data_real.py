"""
Real WebShop dataset loader
Loads tasks from the actual WebShop benchmark
"""
import json
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from typing import Dict

# Path to WebShop data
WEBSHOP_DATA_PATH = Path(__file__).parent.parent.parent / "WebShop" / "data"


class WebShopDataset(Dataset):
    """
    Real WebShop dataset from the official benchmark.
    """
    
    def __init__(self, split: str = 'train', max_samples: int = None):
        """
        Load real WebShop tasks.
        
        Args:
            split: 'train' or 'test'  
            max_samples: Limit dataset size
        """
        print(f"Loading REAL WebShop dataset (split={split})...")
        
        # Load task file
        if split == 'train':
            task_file = WEBSHOP_DATA_PATH / "items_ins_v2_1000.json"
        else:
            task_file = WEBSHOP_DATA_PATH / "items_human_ins.json"
        
        print(f"  Loading from: {task_file}")
        
        with open(task_file, 'r') as f:
            raw_data = json.load(f)
        
        # Parse tasks - handle both dict and list formats
        self.data = []
        
        if isinstance(raw_data, dict):
            # Format: {session_id: task_data}
            for session_id, task_info in raw_data.items():
                # Handle nested structure
                if isinstance(task_info, dict):
                    instruction = task_info.get('instruction_text', task_info.get('instruction', ''))
                    goal = task_info.get('goal', {})
                elif isinstance(task_info, list):
                    # Some entries might be lists
                    instruction = str(task_info[0]) if task_info else ''
                    goal = {}
                else:
                    instruction = str(task_info)
                    goal = {}
                
                self.data.append({
                    'session': session_id,
                    'instruction': instruction,
                    'goal': goal
                })
                
        elif isinstance(raw_data, list):
            # Format: list of tasks
            for i, item in enumerate(raw_data):
                if isinstance(item, dict):
                    instruction = item.get('instruction', item.get('instruction_text', ''))
                    session = item.get('session', f'task_{i}')
                    goal = item.get('goal', {})
                elif isinstance(item, str):
                    instruction = item
                    session = f'task_{i}'
                    goal = {}
                else:
                    instruction = str(item)
                    session = f'task_{i}'
                    goal = {}
                
                self.data.append({
                    'session': session,
                    'instruction': instruction,
                    'goal': goal
                })
        
        # Limit size
        if max_samples and max_samples < len(self.data):
            self.data = self.data[:max_samples]
        
        print(f"✓ Loaded {len(self.data)} REAL WebShop tasks")
        
        # Show example
        if self.data:
            example_instruction = self.data[0]['instruction']
            if len(example_instruction) > 100:
                example_instruction = example_instruction[:100] + "..."
            print(f"\n  Example: {example_instruction}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        return self.data[idx]


def create_webshop_dataloaders(config: Dict) -> tuple:
    """
    Create dataloaders for REAL WebShop.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        train_loader, eval_loader
    """
    train_size = config.get('data', {}).get('train_size', 500)
    eval_size = config.get('data', {}).get('eval_size', 50)
    batch_size = config.get('apo', {}).get('batch_size', 1)
    
    # Create datasets
    train_dataset = WebShopDataset(split='train', max_samples=train_size)
    eval_dataset = WebShopDataset(split='test', max_samples=eval_size)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: x
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: x
    )
    
    return train_loader, eval_loader
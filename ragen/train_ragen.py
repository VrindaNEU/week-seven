"""
Training script for RAGEN (Multi-turn agent training)
Based on TinyZero's train.py but adapted for multi-turn environments
"""
import argparse
import yaml
import torch
import sys
from pathlib import Path
from tqdm import tqdm
import time
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from tinyzero.models import PolicyModel, ReferenceModel
from tinyzero.utils import set_seed, save_checkpoint, load_checkpoint, AverageMeter
from ragen.agent_trainer import MultiTurnAPOTrainer
from ragen.webshop_data_real import create_webshop_dataloaders
from ragen.environments import WebShopEnvironment

#  Action sanitizer import (required by WebShop env to normalize actions)
try:
    from ragen.action_sanitizer import sanitize
except Exception:
    # Fallback (no-op) if sanitizer module isn't present; training will still run
    def sanitize(text, fallback_query=None):
        return (text or "").strip()


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train RAGEN with A*PO')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/ragen_webshop.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs/ragen',
        help='Directory to save outputs'
    )
    parser.add_argument(
        '--eval_only',
        action='store_true',
        help='Only run evaluation'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Run in debug mode (fewer steps, faster)'
    )
    parser.add_argument(
        '--skip-initial-eval',
        action='store_true',
        help='Skip initial evaluation (faster startup)'
    )
    return parser.parse_args()


class RAGENTrainer:
    """RAGEN trainer - extends TinyZero training for multi-turn agents"""
    
    def __init__(self, config: dict, output_dir: str):
        """
        Initialize RAGEN trainer
        
        Args:
            config: Configuration dictionary
            output_dir: Directory to save outputs
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log file
        self.log_file = self.output_dir / 'training.log'
        self.metrics_file = self.output_dir / 'metrics.json'
        
        # Initialize metrics storage
        self.metrics_history = {
            'train_loss': [],
            'train_reward': [],
            'train_success_rate': [],
            'train_avg_turns': [],
            'eval_success_rate': [],
            'steps': []
        }
        
        # Set seed
        set_seed(config.get('seed', 42))
        
        # Create dataloaders
        print("Creating WebShop dataloaders...")
        self.train_loader, self.eval_loader = create_webshop_dataloaders(config)
        
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Eval samples: {len(self.eval_loader.dataset)}")
        
        # Initialize models
        print("Loading models...")
        self.policy = PolicyModel(
            config['model']['name'],
            device=config['model']['device']
        )
        self.ref_model = ReferenceModel(
            config['model']['ref_model'],
            device=config['model']['device']
        )
        print("Models loaded successfully!")
        
        # Initialize environment
        print("Initializing WebShop environment...")
        self.environment = WebShopEnvironment(config)
        print("Environment ready!")
        
        # Initialize multi-turn trainer
        self.apo_trainer = MultiTurnAPOTrainer(
            self.policy,
            self.ref_model,
            config,
            self.environment
        )
        
        # Training state
        self.global_step = 0
        self.best_success_rate = 0.0
        self.skip_initial_eval = False  # Will be set by main()
        
        # Meters for tracking
        self.loss_meter = AverageMeter()
        self.reward_meter = AverageMeter()
    
    def log(self, message: str):
        """Log message to file and console"""
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        
        with open(self.log_file, 'a') as f:
            f.write(log_message + '\n')
    
    def save_metrics(self):
        """Save metrics history to JSON"""
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
    
    def train_epoch(self, epoch: int):
        """Train for one epoch"""
        self.policy.train()
        
        pbar = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            desc=f"Epoch {epoch}"
        )
        
        for batch_idx, batch in pbar:
            # Check if we've reached max steps
            if self.global_step >= self.config['training']['max_steps']:
                break
            
            # Training step (MULTI-TURN!)
            try:
                loss, metrics = self.apo_trainer.train_step_multiturn(batch)
                
                # Update meters
                self.loss_meter.update(loss)
                self.reward_meter.update(metrics['avg_reward'])
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{self.loss_meter.avg:.4f}",
                    'reward': f"{self.reward_meter.avg:.3f}",
                    'success': f"{metrics['success_rate']:.2%}",
                    'step': self.global_step
                })
                
            except Exception as e:
                self.log(f"Error in training step: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            # Logging
            if self.global_step % self.config['logging']['log_every'] == 0:
                self.log(
                    f"Step {self.global_step}: "
                    f"Loss={self.loss_meter.avg:.4f}, "
                    f"Reward={self.reward_meter.avg:.3f}, "
                    f"Success={metrics['success_rate']:.2%}"
                )
                
                # Store metrics
                self.metrics_history['train_loss'].append(self.loss_meter.avg)
                self.metrics_history['train_reward'].append(self.reward_meter.avg)
                self.metrics_history['train_success_rate'].append(metrics['success_rate'])
                self.metrics_history['train_avg_turns'].append(metrics.get('avg_turns', 0))
                self.metrics_history['steps'].append(self.global_step)
                
                # Reset meters
                self.loss_meter.reset()
                self.reward_meter.reset()
            
            # Evaluation (only if eval_every is reasonable)
            if (self.global_step > 0 and
                self.global_step % self.config['training']['eval_every'] == 0 and 
                self.config['training']['eval_every'] < 100):
                self.log(f"\n{'='*50}")
                self.log(f"Running evaluation at step {self.global_step}...")
                
                eval_success = self.evaluate()
                
                # Store eval metrics
                self.metrics_history['eval_success_rate'].append(eval_success)
                
                # Log results
                self.log(f"Eval - Success Rate: {eval_success:.2%}")
                
                # Save best model
                if eval_success > self.best_success_rate:
                    self.best_success_rate = eval_success
                    self.log(f"New best success rate: {self.best_success_rate:.2%}")
                    save_checkpoint(
                        self.apo_trainer.policy.model,
                        self.apo_trainer.optimizer,
                        self.global_step,
                        self.output_dir / 'best_model.pt',
                        success_rate=self.best_success_rate
                    )
                
                self.log(f"{'='*50}\n")
                
                # Back to training mode
                self.policy.train()
            
            # Checkpointing (only if save_every is reasonable)
            if (self.global_step > 0 and
                self.global_step % self.config['training']['save_every'] == 0 and
                self.config['training']['save_every'] < 100):
                checkpoint_path = self.output_dir / f'checkpoint_{self.global_step}.pt'
                save_checkpoint(
                    self.apo_trainer.policy.model,
                    self.apo_trainer.optimizer,
                    self.global_step,
                    checkpoint_path
                )
                self.log(f"Saved checkpoint to {checkpoint_path}")
            
            # Save metrics periodically
            if self.global_step % self.config['logging']['log_every'] == 0:
                self.save_metrics()
            
            self.global_step += 1
    
    def evaluate(self) -> float:
        """
        Evaluate on eval set
        
        Returns:
            Success rate
        """
        self.policy.eval()
        
        successes = 0
        total = 0
        
        print(f"\n🔍 Evaluating on {len(self.eval_loader.dataset)} tasks...")
        
        # Pull commonly used knobs from config
        model_cfg = self.config.get('model', {})
        sampling_cfg = self.config.get('sampling', {})
        use_min_new_tokens = sampling_cfg.get('use_min_new_tokens', False)
        min_new_tokens = sampling_cfg.get('min_new_tokens', 10)
        gen_max_len = model_cfg.get('max_length', 512)

        for batch in tqdm(self.eval_loader, desc="Evaluating", leave=False):
            for task in batch:
                # Run one episode
                state = self.environment.reset(task)
                done = False
                episode_reward = 0.0
                
                for turn in range(self.config['environment']['max_turns']):
                    # 1) Render state -> text observation
                    if hasattr(self.environment, "render_text"):
                        obs_text = self.environment.render_text(state)
                    else:
                        obs_text = str(state)

                    # 2) Generate raw action text
                    with torch.no_grad():
                        gen_kwargs = dict(
                            max_new_tokens=max(16, min_new_tokens if use_min_new_tokens else 32),
                            temperature=0.7,
                            do_sample=True,
                        )

                        action_raw = self.policy.generate([obs_text], **gen_kwargs)[0]
                    
                    # 3)  Sanitize into a valid WebShop action
                    fallback_query = task.get("instruction", task.get("prompt", ""))
                    action = sanitize(action_raw, fallback_query)

                    # Optional: log first few eval transitions
                    if total < 3 and turn == 0:
                        print(f"[Eval] Obs: {obs_text[:100]!r}")
                        print(f"[Eval] Raw: {action_raw!r}  ->  Sanitized: {action!r}")

                    # 4) Step the environment
                    next_state, reward, done, info = self.environment.step(action)
                    episode_reward += reward
                    state = next_state
                    
                    if done:
                        break
                
                # Check success
                if episode_reward > 0.5:
                    successes += 1
                total += 1
        
        success_rate = successes / max(total, 1)
        print(f"✓ Evaluation complete: {successes}/{total} = {success_rate:.2%}")
        
        return success_rate
    
    def train(self):
        """Main training loop"""
        self.log("Starting RAGEN training...")
        self.log(f"Config: {self.config}")
        
        # Run initial evaluation (SKIP in debug mode only if requested)
        if not self.skip_initial_eval:
            self.log("Running initial evaluation...")
            initial_success = self.evaluate()
            self.log(f"Initial success rate: {initial_success:.2%}")
        else:
            self.log("⚡ Skipping initial evaluation (debug/flag)")
            initial_success = 0.0
        
        # Training loop
        for epoch in range(self.config['training']['num_epochs']):
            self.log(f"\n{'='*60}")
            self.log(f"Starting Epoch {epoch + 1}/{self.config['training']['num_epochs']}")
            self.log(f"{'='*60}")
            
            self.train_epoch(epoch)
            
            # Check if we've reached max steps
            if self.global_step >= self.config['training']['max_steps']:
                self.log("Reached maximum training steps")
                break
        
        # Final evaluation (always run, even in debug)
        self.log("\n" + "="*60)
        self.log("Running final evaluation...")
        final_success = self.evaluate()
        
        self.log("\n" + "="*60)
        self.log("TRAINING COMPLETE!")
        self.log(f"Initial success rate: {initial_success:.2%}")
        self.log(f"Final success rate: {final_success:.2%}")
        self.log(f"Best success rate: {self.best_success_rate:.2%}")
        
        if not self.skip_initial_eval:
            self.log(f"Improvement: {(final_success - initial_success):+.2%}")
        
        self.log("="*60)
        
        # Save final model
        save_checkpoint(
            self.apo_trainer.policy.model,
            self.apo_trainer.optimizer,
            self.global_step,
            self.output_dir / 'final_model.pt',
            success_rate=final_success
        )
        
        # Save final metrics
        self.save_metrics()
        
        return final_success


def main():
    """Main entry point"""
    args = parse_args()
    
    # Load config
    print(f"Loading config from {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Debug mode - ULTRA FAST settings for local testing
    if args.debug:
        print("⚡ Running in DEBUG mode - ULTRA FAST settings")
        config['training']['max_steps'] = 10           # Just 10 steps!
        config['training']['eval_every'] = 999         # Don't eval during training
        config['training']['save_every'] = 999         # Don't save checkpoints
        config['data']['train_size'] = 4               # Tiny dataset
        config['data']['eval_size'] = 2                # Tiny eval set
        config['environment']['max_turns'] = 2         # Only 2 turns per episode
        config['apo']['v_star_samples'] = 1            # Single V* sample
        config['apo']['adaptive_vstar'] = False        # Disable adaptive
        config['model']['max_length'] = 256            # Shorter sequences
        config['model']['sft_max_length'] = 512        # Shorter sequences
    
    # Create trainer
    trainer = RAGENTrainer(config, args.output_dir)
    
    # Set skip initial eval flag (respect CLI flags; do NOT override later)
    trainer.skip_initial_eval = args.skip_initial_eval or args.debug
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.global_step = load_checkpoint(
            trainer.apo_trainer.policy.model,
            trainer.apo_trainer.optimizer,
            args.resume
        )
        print(f"Resumed from step {trainer.global_step}")
    
    # Run evaluation only if specified
    if args.eval_only:
        print("Running evaluation only...")
        success = trainer.evaluate()
        print(f"Success Rate: {success:.2%}")
        return
    
    # Train
    trainer.train()


if __name__ == '__main__':
    main()

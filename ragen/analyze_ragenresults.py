"""
Analyze RAGEN training results
"""
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_metrics(results_dir: str = './ragen_results'):
    """Load metrics from training"""
    metrics_file = Path(results_dir) / 'metrics.json'
    
    if not metrics_file.exists():
        print(f"❌ Metrics file not found: {metrics_file}")
        return None
    
    with open(metrics_file) as f:
        metrics = json.load(f)
    
    return metrics

def plot_training_curves(metrics):
    """Plot training curves"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    steps = metrics['steps']
    
    # Plot 1: Success Rate
    ax = axes[0, 0]
    if metrics['train_success_rate']:
        ax.plot(steps, metrics['train_success_rate'], 'b-', label='Train', alpha=0.7)
    if metrics['eval_success_rate']:
        # Eval might have fewer points
        eval_steps = steps[::len(steps)//len(metrics['eval_success_rate'])][:len(metrics['eval_success_rate'])]
        ax.plot(eval_steps, metrics['eval_success_rate'], 'r-', label='Eval', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Success Rate')
    ax.set_title('Success Rate Over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Average Reward
    ax = axes[0, 1]
    ax.plot(steps, metrics['train_reward'], 'g-', alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('Average Reward')
    ax.set_title('Average Reward Over Training')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Loss
    ax = axes[1, 0]
    ax.plot(steps, metrics['train_loss'], 'purple', alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Average Turns
    ax = axes[1, 1]
    ax.plot(steps, metrics['train_avg_turns'], 'orange', alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('Average Turns')
    ax.set_title('Average Turns Per Episode')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ragen_training_curves.png', dpi=300, bbox_inches='tight')
    print("✓ Saved training curves to ragen_training_curves.png")
    plt.show()

def print_summary(metrics):
    """Print training summary"""
    print("\n" + "="*60)
    print("RAGEN TRAINING SUMMARY")
    print("="*60)
    
    if metrics['train_success_rate']:
        initial_success = metrics['train_success_rate'][0]
        final_success = metrics['train_success_rate'][-1]
        print(f"Initial Success Rate: {initial_success:.2%}")
        print(f"Final Success Rate:   {final_success:.2%}")
        print(f"Improvement:          {(final_success - initial_success):+.2%}")
    
    if metrics['eval_success_rate']:
        best_eval = max(metrics['eval_success_rate'])
        print(f"\nBest Eval Success:    {best_eval:.2%}")
    
    if metrics['train_avg_turns']:
        avg_turns = sum(metrics['train_avg_turns']) / len(metrics['train_avg_turns'])
        print(f"\nAverage Turns/Episode: {avg_turns:.1f}")
    
    total_steps = len(metrics['steps'])
    print(f"Total Training Steps:  {total_steps}")
    
    print("="*60)

def main():
    """Main analysis"""
    print("Loading RAGEN results...")
    
    metrics = load_metrics()
    
    if metrics is None:
        print("\n❌ No metrics found. Make sure to download results first:")
        print("modal volume get ragen-outputs /outputs ./ragen_results")
        return
    
    print("✓ Metrics loaded")
    
    # Print summary
    print_summary(metrics)
    
    # Plot curves
    print("\nGenerating plots...")
    plot_training_curves(metrics)
    
    print("\n✅ Analysis complete!")

if __name__ == '__main__':
    main()
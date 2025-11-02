
# TinyZero Results

## Overall Performance

| Metric | Value |
|--------|-------|
| Overall Accuracy | 88.00% |
| Countdown Accuracy | 79.31% |
| Multiplication Accuracy | 100.00% |
| Avg Response Length | 200.0 chars |
| Total Examples | 40 |

## Task Breakdown

### Countdown Task
- Accuracy: 79.31%
- Description: Generate target number using given numbers

### Multiplication Task  
- Accuracy: 100.00%
- Description: Compute product of two numbers

## Notes

- Model: Qwen2.5-3B-Instruct
- Algorithm: A*PO (A-star Policy Optimization)
- Training: Reinforcement Learning without supervised fine-tuning
- Key Innovation: Computing V* using reference model for efficient optimization

## Training Details

- Base Model: Qwen2.5-3B-Instruct
- Reference Model: Qwen2.5-3B-Instruct (frozen)
- Optimization: A*PO (A-star Policy Optimization)
- Tasks: Countdown and Multiplication
- Training Steps: 40

## Key Differences from GRPO

| Feature | GRPO | A*PO (Our Implementation) |
|---------|------|---------------------------|
| Rollouts per prompt | G (multiple) | 1 (single) |
| Value estimation | Learned critic | Computed V* from reference |
| Sample efficiency | Lower | Higher |
| Memory usage | Higher | Lower |
| Compute cost | Higher | Lower |

## Example Outputs

See evaluation logs for example model outputs on both countdown and multiplication tasks.

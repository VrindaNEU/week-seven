# RAGEN with A*-PO: Optimizing Multi-Turn Reasoning and Self-Evolving LLM Agents

## Overview

This project combines **RAGEN** (multi-turn, reasoning-based agent training) with **A*-PO** (A-star Policy Optimization) to create a novel agent training system for complex, multi-step tasks. The system demonstrates how LLM agents can learn sophisticated behaviors through reinforcement learning on the WebShop environment.

---
### Presentation Deck
**[Presentation Deck - PDF version](https://github.com/nikhilp0799/neu-self-improve-ai/blob/2f53e17c1088dac23c2a91d2254e1d24d9e1578c/week_06/RAGEN_with_A*-PO_PPT.pdf)**

**[Presentation Deck - Google Slides link](https://docs.google.com/presentation/d/1Pq03qLqbD_pofNJh-acGk7tSwl_ocN9TynirUGOES9s/edit?usp=sharing)**

For detailed slides, methodology, and results visualization, see the presentation deck above.
---

## What We Built

### 1. Successful Integration: RAGEN + A*-PO

We built a novel agent training system from scratch that combines:
- **RAGEN's multi-turn, reasoning-based rollouts**: Enables agents to perform complex, sequential decision-making tasks
- **A*-PO's two-stage, critic-free optimizer**: Eliminates the need for expensive critic networks by estimating optimal values (V*) directly from reference model samples

**Key Innovation**: This hybrid approach allows agents to learn multi-step tasks through reinforcement learning without requiring separate value function training.

---

## Results & Performance

### Current Performance Metrics

| Metric | RAGEN (A*-PO, ours) |
|--------|---------------------|
| **Success Rate (%)** | 0 |
| **Loss** | 1.77 → -0.49 |
| **Average Reward** | 0.40 |
| **Training Steps to Converge** | 100 |
| **Evaluation Step** | 50 |

### Key Observations

1. **Training Stability**: The agent successfully learned from the environment, as evidenced by loss reduction from 1.77 to -0.49, indicating effective policy improvement.

2. **Reward Learning**: Average reward of 0.40 shows the agent is learning to navigate the environment, though success rate remains at 0%.

### Current Limitations & Next Steps

**Success Rate Improvement (Next Week Goal)**: While the agent demonstrates learning through reward signals and loss reduction, the success rate remains at 0%. Improving success rate is a primary focus for next week's submission. Potential approaches include:

- Refining V* estimation quality
- Adjusting reward shaping
- Improving multi-turn action sequences
- Enhancing exploration strategies

---

## System Architecture

```
┌─────────────────────────────────────────────────┐
│         RAGEN + A*-PO Training Pipeline          │
├─────────────────────────────────────────────────┤
│                                                  │
│  Stage 1: Offline V* Estimation                  │
│    ├─> Reference model sampling                 │
│    ├─> Multi-turn trajectory generation         │
│    ├─> Reward computation                       │
│    └─> V* database construction                 │
│                                                  │
│  Stage 2: A*-PO Policy Optimization             │
│    ├─> Policy rollout generation                │
│    ├─> Advantage computation (R - V*)          │
│    ├─> Weighted loss updates                    │
│    └─> Policy improvement                       │
│                                                  │
└─────────────────────────────────────────────────┘
```

---

## Project Structure

```
week_06/
├── ragen/
│   ├── agent_trainer.py          # Multi-turn A*-PO trainer
│   ├── train_ragen.py            # Main training script
│   ├── environments/
│   │   └── webshop.py            # WebShop environment wrapper
│       └── base.py 
│   └── webshop_data_real.py      # WebShop data loaders
│     └── README_RAGEN.md               # This file
│
├── tinyzero/
│   ├── apo_trainer.py            # Core A*-PO algorithm
│   ├── models.py                 # Policy & reference models
│   └── vstar_cache.py            # V* caching system
│
├── configs/
│   └── ragen_webshop.yaml        # Training configuration
│
├── modal_train_ragen.py          # Modal deployment script
```

---

## Setup & Usage

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (H100 recommended)
- Java 21+ (for WebShop environment)
- Modal account (for cloud training)

### Installation

```bash
cd week_06

# Install dependencies
pip install -r requirements.txt

```

### Training

```bash

# Modal training (recommended)
modal run modal_train_ragen.py
```

---

## Configuration

Key configuration parameters in `configs/ragen_webshop.yaml`:

```yaml
apo:
  beta: 0.3                        # V* temperature
  v_star_samples: 2                # Samples for V* estimation
  adaptive_vstar: true              # Enable adaptive sampling
  learning_rate: 3e-7               # Conservative learning rate

environment:
  max_turns: 10                     # Maximum turns per episode
  num_products: 100                 # Number of products in environment

training:
  max_steps: 150                    # Training steps
  eval_every: 50                    # Evaluation frequency
```

---

## Future Work & Next Steps

### 1. Iterative V* Refinement

**Problem**: Single offline V* computation may become outdated as the policy improves.

**Solution**: Instead of a single offline step, re-calculate V* values halfway through training using the new, smarter policy. This gives the agent a more accurate target to aim for as it evolves.

**Expected Impact**: Improved policy performance by maintaining alignment between V* estimates and current policy capabilities.

### 2. Adaptive Sampling (StarPO-S Idea)

**Problem**: Offline V* samples may not cover high-variance, uncertain regions effectively.

**Solution**: Integrate trajectory filtering from the RAGEN paper into Stage 1. By focusing N offline samples on high-variance, uncertain prompts, we can build a more robust V* database from the start.

**Expected Impact**: Better V* quality through strategic sampling, leading to improved policy optimization.

---

## Key Insights

1. **Multi-Turn Learning**: The combination of RAGEN's multi-turn structure with A*-PO's efficient optimization enables learning complex sequential tasks.

2. **V* Quality is Critical**: The success of A*-PO depends heavily on accurate V* estimates. Poor V* quality directly limits policy performance.

3. **Critic-Free Advantage**: Eliminating the need for separate critic networks makes training more efficient, but places greater importance on offline V* computation.

4. **Self-Evolving Agents**: The system demonstrates how agents can improve through iterative RL training without external supervision.

---

## Team
**Built for INFO 7375 - Self-Improving AI Systems**
**Week 07 - RAGEN + A*-PO Integration**

- **[Nikhil Pandey]** 002775062
- **[Anvitha Hiriadka]** 002472965
- **[Ahsan Zafar Syed]** 002801441
- **[Vrinda Shinde]** 002290028
- **[Praneeth Reddy]** 002089375

---
## References

- [RAGEN](https://arxiv.org/pdf/2504.20073)
- [A*-PO**](https://arxiv.org/pdf/2505.20686)
- [WebShop](https://arxiv.org/pdf/2207.01206)
- [DeepSeek-R1 Paper](https://arxiv.org/pdf/2501.12948)
- [Modal Documentation](https://modal.com/docs)
- [Course Materials](https://course-website-link)
---
## Acknowledgments

- **Professor [Suhabe Bugrara]** for guidance on model selection and optimization strategies
- **DeepSeek Research** for the R1 paper and inspiration
- **Qwen Team** at Alibaba for the excellent base models
- **Modal Labs** for accessible cloud GPU infrastructure
- **Course TAs** for feedback and support


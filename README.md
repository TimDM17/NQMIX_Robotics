# Multi-Agent Reinforcement Learning Algorithm Comparison

Comparing state-of-the-art MARL algorithm on the MaMuJoCo Humanoid environment.
More environments coming soon.

## Algorithms

- **NQMIX** - Non-monotonic value function factorization
- **FACMAC** - Continuous action actor-critic (coming coon)

## Environment

**MaMuJoCo Humanoid "9|8" (Gymnasium Robotics)**
- 2 agents: Upper body (9 actions) + Lower body (8 actions)
- Cooperative task: Bipedal locomotion
- Partial observability per agent
- Continuous action space: [-0.4, 0.4]

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Training
```bash
# NQMIX (default)
python scripts/train.py --config configs/nqmix_humanoid.yaml

# Custom seed
python scripts/train.py --config configs/nqmix_humanoid.yaml --seed 123
```

### Evaluation
```bash
python scripts/evaluate.py --checkpoint results/nqmix_humanoid/best.pth
```

## Results

| Algorithm | Avg Reward | Training Time | Parameters |
|-----------|------------|---------------|------------|
| NQMIX     | TBD        | TBD           | TBD        |
| QMIX      | TBD        | TBD           | TBD        |
| FACMAC    | TBD        | TBD           | TBD        |

## Project Structure
## Project Structure
```
marl-comparison/
├── configs/           # Algorithm configurations
├── src/
│   ├── agents/        # Algorithm implementations
│   ├── networks/      # Neural network modules
│   ├── training/      # Training and evaluation
│   └── utils/         # Logging and metrics
├── scripts/           # Entry points
└── results/           # Checkpoints and logs
```

## References

- **NQMIX**: Chen(2020) - Non-monotonic Value Function Factorization for Deep Multi-Agent Reinforcement Learning

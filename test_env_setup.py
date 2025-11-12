import gymnasium as gym
from gymnasium_robotics import mamujoco_v1


print("="*70)
print("Testing MaMuJoCo Humanoid Environment")
print("="*70)

# Create environment
env = mamujoco_v1.parallel_env("Humanoid", "9|8")

print(f"\n✓ Environment created successfully")
print(f" Agents: {env.possible_agents}")
print(f" Agent 0 (upper Body): {env.action_spaces['agent_0'].shape[0]} actions")
print(f" Agent 1 (lower Body): {env.action_spaces['agent_1'].shape[0]} actions")
print(f" Agent 0 obs dim: {env.observation_spaces['agent_0'].shape[0]}")
print(f" Agent 1 obs dim: {env.observation_spaces['agent_1'].shape[0]}")

print("\n" + "="*70)

# Get dimensions for NQMIX
obs_dims = [
    env.observation_spaces[agent].shape[0]
    for agent in env.possible_agents
]
actions_dims = [
    env.action_spaces[agent].shape[0]
    for agent in env.possible_agents
]
state_dim = sum(obs_dims)

print(f"\nGlobal state dim: {state_dim}")
print("\n" + "=" * 70)

# Test random episode
print(f"\n Running 10 random steps...")
obs, info = env.reset()

for step in range(10):
    actions = {agent: env.action_spaces[agent].sample() for agent in env.agents}
    obs, rewards, terminated, truncated, info = env.step(actions)
    print(f" Step {step + 1}: reward = {sum(rewards.values())}")

env.close()
print("\n✓ Environment test completed!")
print("=" * 70)
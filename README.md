# EDA-Physical-Design-Router-via-Reinforcement-Learning
Reinforcement learning code to finish the routing works on a 2D grid world, as well as to visualize the result of the routing paths.

## File list
- routing_gym.py: RL environment for routing on a 2D grid world.
- PPO_structure.py: RL agent implemented using PPO Algorithm.
- Net.py: Neural Networks of the PPO agent.
- IL_expert_alternate: Generate expert policies using A* algorithm. 
- BC_train.py: Pre-training code using Behavior Cloning. Training runs on GPU by default if CUDA is available, can also train on CPU but much slower.
- PPO_train.py: Training code. Training runs on GPU by default if CUDA is available, can also train on CPU but much slower.
- PPO_test.py: Testing code. Results can be visualized on a grid map.

## Requirements
- OpenAI Gym 0.26.2
- Pytorch 1.10.0+cu113
- Numpy 1.22.4
- Matplotlib 3.5.3

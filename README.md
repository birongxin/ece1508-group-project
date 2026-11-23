# ece1508-group-project
ece1508 Reinforcement Learning group project

Update #1:
Before you run this make sure you download the july, august and september data for 2025 from here: https://open.toronto.ca/dataset/bike-share-toronto-ridership-data/ and save them in the folder data. Please run the data.py before env.py which will create the processed datasets and I've also included a sample walkthrough of the environment in env.py for usage.

Update #2: 
Please download stable_baselines3 before running PPO training and testing modules. 
1. Added train_ppo.py as PPO training module
2. Added test_ppo.py as PPO testing module
3. Running order: data.py -> env.py -> train_ppo.py -> test_ppo.py. The PPO implementation was tested with short training runs. We probably need to tune the training hyperparameter to evaluate policy convergence in the future. 

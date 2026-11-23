from pathlib import Path
import pandas as pd
from stable_baselines3 import PPO
from env import BikeShareEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
stations = pd.read_csv(BASE_DIR / "data/processed/stations.csv")
od = pd.read_parquet(BASE_DIR / "data/processed/od_demand.parquet")

env = BikeShareEnv(stations=stations, od=od)
model = PPO.load(str(BASE_DIR / "models/bike_ppo"),env=env)

# def make_env():
#     return BikeShareEnv(
#         stations=stations,
#         od=od,
#         truck_capacity=30,
#         episode_hours=24,
#         seed=1508
#     )

# env = DummyVecEnv([make_env])

# # Load training normalization statistics
# env = VecNormalize.load(
#     str(MODEL_DIR / "vecnormalize.pkl"),
#     env
# )
# env.training = False
# env.norm_reward = False
# model = PPO.load(
#     str(MODEL_DIR / "bike_ppo"),
#     env=env
# )

def test_policy(policy, seed=2025):
    env = BikeShareEnv(stations=stations, od=od)
    obs, info = env.reset(seed=seed)
    env.action_space.seed(seed)
    total_reward = 0
    done = False
    while not done:
        if policy == "PPO":
            action, _ = model.predict(obs, deterministic=False)
        elif policy == "Random":
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

    completed = env.total_completed_trips
    unmet = env.total_unmet_requests

    print(f"\n--- {policy} ---")
    print(f"Total Reward: {total_reward}")
    print(f"Completed Trips: {completed}")
    print(f"Unmet Requests: {unmet}")
    print(f"Service Rate: {completed / (completed + unmet):.2%}")
    print(f"Distance: {env.total_distance_driven} km")

    env.close()


test_policy("PPO")
test_policy("Random")
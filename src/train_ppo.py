from pathlib import Path
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from env import BikeShareEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
LOG_DIR = BASE_DIR / "logs"

stations = pd.read_csv(DATA_DIR / "stations.csv")
od = pd.read_parquet(DATA_DIR / "od_demand.parquet")

env = BikeShareEnv(stations=stations, od=od)

# def make_env():
#     return Monitor(
#         BikeShareEnv(
#             stations=stations,
#             od=od,
#             truck_capacity=30,
#             episode_hours=24,
#             seed=1508
#         )
#     )

# env = DummyVecEnv([make_env])

# env = VecNormalize(
#     env,
#     norm_obs=False,
#     norm_reward=True,
#     clip_reward=10.0
# )


check_env(env)
env = Monitor(env)

# Initialize the PPO agent
model = PPO(
    # These parameters can be adjusted later accordingly
    policy="MlpPolicy",
    env=env,
    learning_rate=3e-4,
    n_steps=512,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    verbose=1,
    tensorboard_log=str(LOG_DIR),
    seed=1508
)
# Train PPO and Save
model.learn(total_timesteps=2048)
model.save(str(MODEL_DIR / "bike_ppo"))
#env.save(str(MODEL_DIR / "vecnormalize.pkl"))
env.close()


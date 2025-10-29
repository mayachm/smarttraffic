import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
import numpy as np
import traci
from sumolib import checkBinary

# SUMO binary and config
sumoBinary = checkBinary("sumo-gui")
sumoConfig = "simple.sumocfg"

# All TLS IDs in the 4x4 grid
tls_ids = [
    'A0','A1','A2','A3',
    'B0','B1','B2','B3',
    'C0','C1','C2','C3',
    'D0','D1','D2','D3'
]

# Multi-TLS environment (single SUMO instance)
class MultiTLSEnv(gym.Env):
    def __init__(self, tls_ids):
        super(MultiTLSEnv, self).__init__()
        self.tls_ids = tls_ids

        # Start SUMO once
        traci.start([sumoBinary, "-c", sumoConfig])

        # Prepare per-TLS observation & action spaces
        self.lanes_list = [traci.trafficlight.getControlledLanes(tls) for tls in tls_ids]
        self.num_phases = [len(traci.trafficlight.getCompleteRedYellowGreenDefinition(tls)[0].phases) for tls in tls_ids]

        # Observation = concatenated lanes vehicle counts
        self.observation_space = gym.spaces.Box(
            low=0,
            high=100,
            shape=(sum(len(lanes) for lanes in self.lanes_list),),
            dtype=np.float32
        )

        # Action = discrete for each TLS
        self.action_space = gym.spaces.MultiDiscrete(self.num_phases)

    def reset(self, seed=None, options=None):
        traci.load(["-c", sumoConfig])  # restart simulation
        obs = []
        for lanes in self.lanes_list:
            obs.extend([traci.lane.getLastStepVehicleNumber(lane) for lane in lanes])
        return np.array(obs, dtype=np.float32), {}

    def step(self, actions):
        # Apply actions to all TLS
        for idx, tls in enumerate(self.tls_ids):
            traci.trafficlight.setPhase(tls, int(actions[idx]))

        traci.simulationStep()

        # Collect observations & reward
        obs = []
        reward = 0
        done = False
        for lanes in self.lanes_list:
            lane_vehicles = [traci.lane.getLastStepVehicleNumber(lane) for lane in lanes]
            obs.extend(lane_vehicles)
            reward -= sum(lane_vehicles)  # minimize congestion
            done = done or traci.simulation.getMinExpectedNumber() <= 0

        return np.array(obs, dtype=np.float32), reward, done, False, {}

    def close(self):
        traci.close()

# ----------------- Training -----------------
env = MultiTLSEnv(tls_ids)
env = VecMonitor(DummyVecEnv([lambda: env]))

model = PPO("MlpPolicy", env, verbose=1, device="cpu")
model.learn(total_timesteps=10000)  # quick test; increase for real training
model.save("ppo_multi_4x4")

env.close()
print("Training finished, model saved as ppo_multi_4x4.zip")

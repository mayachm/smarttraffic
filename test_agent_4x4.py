import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np
import traci
from sumolib import checkBinary
import matplotlib.pyplot as plt

# ---------------- SUMO paths ----------------
sumoBinary = checkBinary("sumo-gui")  # ensures SUMO GUI is found
sumoConfig = "simple.sumocfg"

# All traffic light IDs in the 4x4 grid
tls_ids = [
    'A0','A1','A2','A3',
    'B0','B1','B2','B3',
    'C0','C1','C2','C3',
    'D0','D1','D2','D3'
]

# ---------------- Multi-TLS Environment ----------------
class MultiTLSEnv(gym.Env):
    def __init__(self, sumoBinary, sumoConfig, tls_ids):
        super(MultiTLSEnv, self).__init__()
        self.tls_ids = tls_ids
        self.sumoBinary = sumoBinary
        self.sumoConfig = sumoConfig

        # ✅ Start SUMO GUI once
        traci.start([self.sumoBinary, "-c", self.sumoConfig], label="main")

        # Prepare per-TLS observation & action spaces
        self.lanes_list = [traci.trafficlight.getControlledLanes(tls) for tls in tls_ids]
        self.num_phases = [len(traci.trafficlight.getCompleteRedYellowGreenDefinition(tls)[0].phases) for tls in tls_ids]

        # Observation = concatenated lane vehicle counts
        self.observation_space = gym.spaces.Box(
            low=0,
            high=100,
            shape=(sum(len(lanes) for lanes in self.lanes_list),),
            dtype=np.float32
        )

        # Action = discrete phase for each TLS
        self.action_space = gym.spaces.MultiDiscrete(self.num_phases)

    def reset(self, seed=None, options=None):
        # Restart simulation
        traci.load(["-c", self.sumoConfig])
        obs = []
        for lanes in self.lanes_list:
            obs.extend([traci.lane.getLastStepVehicleNumber(lane) for lane in lanes])
        return np.array(obs, dtype=np.float32), {}

    def step(self, actions):
        # Apply actions
        for idx, tls in enumerate(self.tls_ids):
            traci.trafficlight.setPhase(tls, int(actions[idx]))

        # Advance simulation
        traci.simulationStep()

        # Collect observations & reward
        obs = []
        reward = 0
        done = False
        for lanes in self.lanes_list:
            lane_vehicles = [traci.lane.getLastStepVehicleNumber(lane) for lane in lanes]
            obs.extend(lane_vehicles)
            reward -= sum(lane_vehicles)  # reward = negative congestion
        done = traci.simulation.getMinExpectedNumber() <= 0
        return np.array(obs, dtype=np.float32), reward, done, False, {}

    def close(self):
        traci.close()
        print("SUMO closed.")

# ---------------- Main ----------------
if __name__ == "__main__":
    # Load trained model
    model = PPO.load("ppo_multi_4x4")

    # Initialize environment
    env = MultiTLSEnv(sumoBinary, sumoConfig, tls_ids)
    obs, info = env.reset()

    done = False
    step = 0
    rewards_list = []  # collect rewards

    try:
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _, info = env.step(action)
            step += 1
            rewards_list.append(reward)
            print(f"Step {step}, reward {reward}")

    finally:
        env.close()
        print("Simulation ended.")

# ---------------- Plot rewards ----------------
plt.figure(figsize=(10,5))
plt.plot(rewards_list, label="Reward per step")
plt.xlabel("Step")
plt.ylabel("Reward (negative congestion)")
plt.title("Traffic Control Rewards Over Time")
plt.legend()
plt.grid(True)
plt.show()

import os
import traci
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class SumoTrafficEnv(gym.Env):
    def __init__(self, sumoBinary, sumoConfig, tls_id):
        super(SumoTrafficEnv, self).__init__()
        self.tls_id = tls_id

        # ✅ Start SUMO before calling any traci functions
        sumo_cmd = ["sumo-gui", "-c", sumoConfig]
        traci.start(sumo_cmd, label=f"sim_{tls_id}")  # unique label per TLS

        # ✅ Now we can safely get info from TraCI
        self.lanes = traci.trafficlight.getControlledLanes(tls_id)

        # Define observation and action space
        self.observation_space = spaces.Box(low=0, high=1, shape=(len(self.lanes),), dtype=np.float32)
        # ✅ Get how many phases this traffic light has
        num_phases = traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)[0].phases
        self.action_space = spaces.Discrete(len(num_phases))

    def reset(self, seed=None, options=None):
        traci.load(["-c", "simple.sumocfg", "--start"])
        obs = np.zeros(len(self.lanes), dtype=np.float32)
        return obs, {}

    def step(self, action):
        traci.simulationStep()
        obs = np.zeros(len(self.lanes), dtype=np.float32)
        reward = 0
        done = False
        info = {}
        return obs, reward, done, False, info

    def close(self):
      for env in self.envs.values():
         try:
          env.close()  # each SumoTrafficEnv
         except:
          pass


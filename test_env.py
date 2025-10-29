import gymnasium as gym
from gymnasium import spaces

import numpy as np
import traci

class SumoTrafficEnv(gym.Env):
    def __init__(self, sumoBinary, sumoConfig, tls_id):
        super(SumoTrafficEnv, self).__init__()
        self.sumoBinary = sumoBinary
        self.sumoConfig = sumoConfig
        self.tls_id = tls_id

        # Placeholder, real values set in reset()
        self.lanes = None
        self.phase_count = None

        # Observation / action space placeholders
        self.observation_space = None
        self.action_space = None

        self.max_steps = 200
        self.current_step = 0

    def reset(self):
        traci.start([self.sumoBinary, "-c", self.sumoConfig])
        self.current_step = 0

        # Now we can safely access TraCI
        self.lanes = traci.trafficlight.getControlledLanes(self.tls_id)
        self.phase_count = len(traci.trafficlight.getCompleteRedYellowGreenDefinition(self.tls_id)[0].phases)

        self.observation_space = spaces.Box(low=0, high=100, shape=(len(self.lanes),), dtype=np.float32)
        self.action_space = spaces.Discrete(self.phase_count)

        obs = self._get_observation()
        return obs

    def _get_observation(self):
        obs = [traci.lane.getLastStepVehicleNumber(lane) for lane in self.lanes]
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        traci.trafficlight.setPhase(self.tls_id, action)
        traci.simulationStep()
        self.current_step += 1

        obs = self._get_observation()
        reward = -np.sum(obs)
        done = self.current_step >= self.max_steps
        info = {}
        return obs, reward, done, info

    def close(self):
        traci.close()

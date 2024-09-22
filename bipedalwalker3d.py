import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np
import time

class BipedalWalker3DEnv(gym.Env):
    def __init__(self, render_mode=None):
        super(BipedalWalker3DEnv, self).__init__()
        self.render_mode = render_mode
        self.jointIds = []
        self.paramIds = []
        
        self.physicsClientId = p.connect(p.GUI if render_mode == "gui" else p.DIRECT)
  
        self.robotId = None
        self.required_joint_indexes = [0, 1, 2, 7, 8, 9]  

        self.render()

        self.hull_height_limit = 0.2
        self.roll_limit = 0.3
        self.pitch_limit = 0.5

        self.num_timesteps = 1000

        self.vel_rew_weight = 1        
        self.pos_rew_weight = 2        
        self.frnt_rew_weight = 3       
        self.sgtl_rew_weight = 4       
        self.healthy_reward = 5        

        low = np.array([0, 0, -10, -90, -90, -5, -5, -5, -5, -5, -5]).astype(np.float32)
        high = np.array([1, 1, 10, 90, 90, 5, 5, 5, 5, 5, 5]).astype(np.float32)

        self.action_space = spaces.Box(low=np.array([-1, -1, -1, -1, -1, -1]).astype(np.float32),
                                        high=np.array([1, 1, 1, 1, 1, 1]).astype(np.float32))
        self.observation_space = spaces.Box(low, high)
        self.required_joint_indexes = [0, 1, 2, 7, 8, 9]  

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)

        target_positions = list(action)
        p.setJointMotorControlArray(self.robotId, self.required_joint_indexes, p.POSITION_CONTROL, targetPositions=target_positions)

        p.setJointMotorControlArray(self.robotId, self.required_joint_indexes, p.TORQUE_CONTROL, forces=[5] * len(self.required_joint_indexes))

        observation = self._get_obs()
        reward = self.reward()
        
        done = self.check_done_condition()

        return observation, reward, done, {}

    def render(self):
        if self.render_mode == "gui":
            p.setGravity(0, 0, -10)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            self.planeId = p.loadURDF("plane.urdf")
            self.startPos = [0, 0, 0.30542796636118535]
            self.startOrientation = p.getQuaternionFromEuler([0, 0, 1.5])

            try:
                self.robotId = p.loadURDF("/home/nihat/Projects/BipedalWalker/urdf/Bipedal_Simple.urdf", self.startPos, self.startOrientation)
                if self.robotId is None:
                    raise RuntimeError("Failed to load the robot URDF.")
                else:
                    print("Robot loaded successfully with ID:", self.robotId)  # Debug statement
            except Exception as e:
                print("Error loading robot URDF:", e)
                raise RuntimeError("Failed to load the robot URDF.")

            for i in range(14):
                p.changeDynamics(self.robotId, i, lateralFriction=10.0, spinningFriction=10.0, rollingFriction=10.0)

            for i in range(p.getNumJoints(self.robotId)): 
                jointInfo = p.getJointInfo(self.robotId, i)
                if (jointInfo[2] == p.JOINT_PRISMATIC or jointInfo[2] == p.JOINT_REVOLUTE): 
                    self.jointIds.append(i)
                    self.paramIds.append(p.addUserDebugParameter(jointInfo[1].decode("utf-8"), -1, 1, 0))

            self.initial_joint_values = p.getJointStates(self.robotId, self.required_joint_indexes)

    def _get_obs(self):
        if self.robotId is None:
            raise RuntimeError("Robot is not initialized. Ensure that render() has been called successfully.")
        
        observation = np.array([]).astype(np.float32)
        for i in range(len(self.required_joint_indexes)): 
            currState = float(p.getJointState(self.robotId, self.required_joint_indexes[i])[0])
            observation = np.append(observation, currState)

        cartesian_pos_or = p.getBasePositionAndOrientation(self.robotId)
        height = np.array([cartesian_pos_or[0][2]]).astype(np.float32)
        roll = np.array([cartesian_pos_or[1][0]]).astype(np.float32)
        pitch = np.array([cartesian_pos_or[1][2]]).astype(np.float32)
        velocity = p.getBaseVelocity(self.robotId)
        x_velocity = np.array([velocity[0][0]]).astype(np.float32)         
        x_pos = np.array([cartesian_pos_or[0][0]]).astype(np.float32)

        observation = np.append(observation, [height, roll, pitch, x_velocity, x_pos])
        return observation.astype(np.float32)

    def reward(self):
        obs = self._get_obs()
        height = obs[6].astype(np.float32)
        roll = obs[7].astype(np.float32)
        pitch = obs[8].astype(np.float32)
        x_velocity = obs[9].astype(np.float32)
        x_pos = obs[10].astype(np.float32)

        # Penalize for falling
        if height < self.hull_height_limit or abs(roll) > self.roll_limit:
            self.reset() 
            return -20 

        velocity_reward = x_velocity*self.vel_rew_weight        # Reward for velocity in x axis  
        position_reward = x_pos*self.pos_rew_weight             # Reward for x distance from origin
        frontal_angle_reward = roll*self.frnt_rew_weight        # Reward for hull frontal angle 
        sagittal_angle_reward = pitch*self.sgtl_rew_weight      # Reward for hull sagittal angle
        
        reward = position_reward + velocity_reward + frontal_angle_reward + sagittal_angle_reward + self.healthy_reward
        
        return reward

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed) 

        if self.robotId is None:
            self.render()

        p.resetBasePositionAndOrientation(self.robotId, self.startPos, self.startOrientation)

        for joint in range(len(self.required_joint_indexes)):
            p.resetJointState(self.robotId, self.required_joint_indexes[joint], 0.1)  # Small non-zero value

        observation = self._get_obs()
        
        return observation, {} 

    def close(self):
        p.disconnect()

    def check_done_condition(self):
        obs = self._get_obs()
        height = obs[6].astype(np.float32)
        roll = obs[7].astype(np.float32)
        pitch = obs[8].astype(np.float32)

        if height < self.hull_height_limit or abs(roll) > self.roll_limit or abs(pitch) > self.pitch_limit:
            return True

        self.num_timesteps -= 1
        if self.num_timesteps <= 0:
            return True

        return False


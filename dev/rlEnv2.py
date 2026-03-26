import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import time
import torch
from stable_baselines3 import PPO

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

# ==============================
# ENVIRONMENT
# ==============================
class MultiRobotEnv(gym.Env):
    def __init__(self, render=False):
        super(MultiRobotEnv, self).__init__()

        if p.isConnected():
            p.disconnect()

        self.render = render

        if self.render:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Action space: forward, left, right, stop
        self.action_space = spaces.Discrete(4)

        # Observation space:
        # [robot_x, robot_y, goal_x, goal_y, human_x, human_y]
        self.observation_space = spaces.Box(low=-10, high=10, shape=(6,), dtype=np.float32)

        # self.reset()

    def reset(self, seed=None, options=None):
        print("Environment reset: loading objects...")

        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(1/120)

        p.loadURDF("plane.urdf")

        # Robots
        self.robot1 = p.loadURDF("r2d2.urdf", [-4, -4, 0.1])
        # self.robot2 = p.loadURDF("r2d2.urdf", [2, 2, 0.1])

        # p.changeDynamics(self.robot2, -1, mass=0)

        # Human (moving obstacle)
        self.human = p.loadURDF("sphere2.urdf", [1, 1, 0.5])

        # Goal
        self.goal = np.array([3, 3])

        self.t = 0
        self.prev_dist = self._distance_to_goal()

        return self._get_obs(), {}

    def step(self, action):
        self._apply_action_vel(self.robot1, action)

        # Move human
        self.t += 0.03
        hx = np.sin(self.t)
        hy = np.cos(self.t)
        p.resetBasePositionAndOrientation(self.human, [hx, hy, 0.5], [0,0,0,1])

        p.stepSimulation()

        if self.render:
            print(f"Goal: {self._distance_to_goal():.2f}, Human: {self._distance_to_human():.2f}")
            robot_pose = p.getBasePositionAndOrientation(self.robot1)
            print(robot_pose[0])

        obs = self._get_obs()
        reward = self._compute_reward()
        done = self._is_done()

        return obs, reward, done, False, {}

    # ==============================
    # HELPERS
    # ==============================

    def _get_obs(self):
        r_pos, _ = p.getBasePositionAndOrientation(self.robot1)
        h_pos, _ = p.getBasePositionAndOrientation(self.human)
        # r2_pos, _ = p.getBasePositionAndOrientation(self.robot2)

        dist_human = np.linalg.norm(np.array(r_pos[:2]) - np.array(h_pos[:2]))
        # dist_robot = np.linalg.norm(np.array(r_pos[:2]) - np.array(r2_pos[:2]))

        return np.array([
            r_pos[0], r_pos[1],
            self.goal[0], self.goal[1],
            h_pos[0], h_pos[1]
            # dist_robot
        ], dtype=np.float32)

    def _apply_action_pos(self, robot, action): # based on setting location coords directly
        pos, orien = p.getBasePositionAndOrientation(robot)
        x, y = pos[0], pos[1]

        step = 0.02

        if action == 0:  # forward
            
            if (abs(x) > 50 or abs(y) > 50) and self.render:
                print("OUT OF BOUNDS:", x, y)
            else:
                dir = self.goal - np.array([x, y])
                dir = dir / (np.linalg.norm(dir) + 1e-6)
                x += dir[0] * step
                y += dir[1] * step
        
        elif action == 1:  # left
            y += step

        elif action == 2:  # right
            y -= step
        
        elif action == 3:  # stop
            pass

        p.resetBasePositionAndOrientation(robot, [x, y, 0.1], orien)

    def _apply_action_vel(self, robot, action):
        speed = 0.6   # smaller = slower
        alpha = 0.8

        pos, _ = p.getBasePositionAndOrientation(robot)
        x, y = pos[0], pos[1]

        vel, _ = p.getBaseVelocity(robot)
        vx_old, vy_old = vel[0], vel[1]

        vx_new, vy_new = 0, 0

        if action == 0:  # forward
            vx_new, vy_new = speed, 0

        elif action == 1:  # left
            vx_new, vy_new = -speed, 0

        elif action == 2:  # right
            vx_new, vy_new = 0, speed

        elif action == 3:  # stop
            vx_new, vy_new = 0, -speed

        # 🔥 SMOOTHING (key fix)
        vx = alpha * vx_old + (1 - alpha) * vx_new
        vy = alpha * vy_old + (1 - alpha) * vy_new

        p.resetBaseVelocity(robot, linearVelocity=[vx, vy, 0])

    def _distance_to_goal(self):
        r_pos, _ = p.getBasePositionAndOrientation(self.robot1)
        return np.linalg.norm(np.array(r_pos[:2]) - self.goal)
    
    def _distance_to_human(self):
        r_pos, _ = p.getBasePositionAndOrientation(self.robot1)
        h_pos, _ = p.getBasePositionAndOrientation(self.human)
        return np.linalg.norm(np.array(r_pos[:2]) - np.array(h_pos[:2]))

    def _compute_reward(self):
        reward = 0

        r_pos, _ = p.getBasePositionAndOrientation(self.robot1)
        h_pos, _ = p.getBasePositionAndOrientation(self.human)
        # r2_pos, _ = p.getBasePositionAndOrientation(self.robot2)

        dist_goal = np.linalg.norm(np.array(r_pos[:2]) - self.goal)
        dist_human = np.linalg.norm(np.array(r_pos[:2]) - np.array(h_pos[:2]))
        # dist_robot = np.linalg.norm(np.array(r_pos[:2]) - np.array(r2_pos[:2]))

        # Goal attraction (weaker)
        reward += -0.2 * dist_goal

        # Success reward
        if dist_goal < 0.1:
            reward += 100

        # STRONG collision penalty
        if dist_human < 0.2:
            reward -= 80

        # VERY STRONG avoidance shaping
        if dist_human < 0.8:
            reward -= 3 * (0.8 - dist_human)

        # Time penalty
        reward -= 0.01

        # Moving toward goal
        if dist_goal < self.prev_dist:
            reward += 0.5

        self.prev_dist = dist_goal

        return reward

    def _is_done(self):
        return self._distance_to_goal() < 0.1

toBeTrained = True

# ==============================
# TRAINING
# ==============================
if toBeTrained:
    env = MultiRobotEnv(render=False)
    obs, _ = env.reset()

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=200000)

    model.save("ppo_multi_robot")

    p.disconnect()

# ==============================
# TESTING (VISUAL)
# ==============================

else:
    model = PPO.load("ppo_multi_robot")

test_env = MultiRobotEnv(render=True)
obs, _ = test_env.reset()

print("//////////////// Starting Sim /////////////////")

for _ in range(10000):
    action, _ = model.predict(obs)
    obs, reward, done, _, _ = test_env.step(action)
    time.sleep(1/120)

    if done:
        print("//////////////// Goal reached! ////////////////")
        break
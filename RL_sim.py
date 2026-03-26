import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import time
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

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
        # [robot_x, robot_y, goal_x, goal_y, dist_human, dist_other]
        self.observation_space = spaces.Box(low=-10, high=10, shape=(5,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        print("Environment reset: loading objects...")

        p.resetSimulation()
        p.setGravity(0.0, 0.0, 0.0)
        p.setTimeStep(1/60)
        p.loadURDF("plane.urdf")
        
        # Robots
        self.robot1 = p.loadURDF("r2d2.urdf", [-4, -4, 0.1])
        # self.robot2 = p.loadURDF("r2d2.urdf", [2, 2, 0.1])

        # p.changeDynamics(self.robot2, -1, mass=0)

        # Human (moving obstacle)
        self.human = p.loadURDF("sphere2.urdf", [1, 1, 0.5])

        # Goal
        self.goal = np.array([3, 3])
        goal_location = [3.0, 3.0, 0.0]

        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=0.1,  # adjust size as needed
            rgbaColor=[1, 0, 0, 1] # Red color (R, G, B, Alpha)
        )

        # Create a multi-body with no collision shape (baseCollisionShapeIndex=-1)
        # This ensures it is a static, non-physical marker
        marker_id = p.createMultiBody(
            baseMass=0, # zero mass makes it static/immovable
            baseCollisionShapeIndex=-1, # no collision shape
            baseVisualShapeIndex=visual_shape_id,
            basePosition=goal_location
        )

        self.t = 0
        self.prev_dist = self._distance_to_goal()

        return self._get_obs(), {}

    def step(self, action):
        self._apply_action_vel(self.robot1, action)

        # Move human
        self.t += 0.02
        hx = np.sin(self.t)
        hy = np.cos(self.t)
        p.resetBasePositionAndOrientation(self.human, [hx, hy, 0.5], [0,0,0,1])

        p.stepSimulation()

        if self.render:
            print(f"Dist Goal: {self._distance_to_goal():.2f}") #, Dist Human: {self._distance_to_human():.2f}")
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
            dist_human
            # dist_robot
        ], dtype=np.float32)

    def _apply_action(self, robot, action):
        pos, orien = p.getBasePositionAndOrientation(robot)
        x, y = pos[0], pos[1]

        step = 0.01

        if action == 0:  # forward
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
        vx, vy = 0, 0
        speed = 0.4   # smaller = slower

        pos, _ = p.getBasePositionAndOrientation(robot)
        x, y = pos[0], pos[1]

        if action == 0:  # forward
            direction = self.goal - np.array([x, y])
            direction = direction / (np.linalg.norm(direction) + 1e-6)
            vx = direction[0] * speed
            vy = direction[1] * speed

        elif action == 1:  # left
            vy = speed

        elif action == 2:  # right
            vy = -speed

        elif action == 3:  # stop
            vx, vy = 0, 0

        p.resetBaseVelocity(robot, linearVelocity=[vx, vy, 0])

    def _distance_to_goal(self):
        r_pos, _ = p.getBasePositionAndOrientation(self.robot1)
        return np.linalg.norm(np.array(r_pos[:2]) - self.goal)
    
    # def _distance_to_human(self):
    #     r1_pos, _ = p.getBasePositionAndOrientation(self.robot1)
    #     r2_pos, _ = p.getBasePositionAndOrientation(self.robot2)
    #     return np.linalg.norm(np.array(r1_pos[:2]) - np.array(r2_pos[:2]))

    def _compute_reward(self):
        reward = 0

        r_pos, _ = p.getBasePositionAndOrientation(self.robot1)
        h_pos, _ = p.getBasePositionAndOrientation(self.human)
        # r2_pos, _ = p.getBasePositionAndOrientation(self.robot2)

        dist_goal = np.linalg.norm(np.array(r_pos[:2]) - self.goal)
        dist_human = np.linalg.norm(np.array(r_pos[:2]) - np.array(h_pos[:2]))
        # dist_robot = np.linalg.norm(np.array(r_pos[:2]) - np.array(r2_pos[:2]))

        reward += -0.1 * dist_goal

        # Goal reward
        if dist_goal < 0.3:
            reward += 20

        # Collision penalty
        if dist_human < 0.2: # or dist_robot < 0.2:
            reward -= 20

        # Too close to human
        # if dist_human < 0.7:
        #     reward -= 5
        reward -= 2 * (1 / (dist_human + 0.1))

        # Time penalty
        reward -= 0.05

        # Moving toward goal
        if dist_goal < self.prev_dist:
            reward += 1

        self.prev_dist = dist_goal

        return reward

    def _is_done(self):
        return self._distance_to_goal() < 0.1

toBeTrained = True

# ==============================
# TRAINING
# ==============================
if toBeTrained:
    env = Monitor(MultiRobotEnv(render=False))
    obs, _ = env.reset()

    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_logs/")
    model.learn(total_timesteps=300000)

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

# log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "/home/jay/Project_IISc_Intern/output.mp4")

time.sleep(10)

# log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "/home/jay/Project_IISc_Intern/output.mp4")

for _ in range(5000):
    action, _ = model.predict(obs)
    obs, reward, done, _, _ = test_env.step(action)
    time.sleep(1/60)

    if done:
        print("//////////////// Goal reached! ////////////////")
        # p.stopStateLogging(log_id)
        # p.disconnect()
        break
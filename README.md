# RL-Based Multi-Robot Human-Aware Navigation

This repository contains a simulation project developed as part of the research proposal:

**"Modelling of a Reinforcement Learning-based Multi-Robot Navigation and Control in Human-Aware Inspection Operations"**


## 📌 Overview

This project demonstrates a centralized Reinforcement Learning (RL) approach for controlling multiple robots in a shared environment with human presence. The system is designed to enable safe, efficient, and coordinated navigation while accounting for human-aware constraints.

The intial implementation uses **Proximal Policy Optimization (PPO)** to train a shared policy for robot in a simulated environment.

![Python](https://img.shields.io/badge/Python-3.12.3-blue)
![RL](https://img.shields.io/badge/RL-PPO-green)
![Simulation](https://img.shields.io/badge/Simulation-PyBullet-orange)


## 🎯 Key Features

- Multi-robot navigation in a shared environment
- Human-aware behavior (collision avoidance, safe distancing)
- Centralized RL policy
- Simulation using PyBullet
- Training using Stable Baselines3


<!-- ## 🧠 Methodology (Brief)

The system follows a centralized reinforcement learning framework:

- **State Space:**
  Robot positions, goal locations, distances to humans and other robots

- **Action Space:**
  Discrete actions (move forward, turn, stop)

- **Reward Design:**
  - Positive reward for reaching goals
  - Penalty for collisions
  - Penalty for unsafe proximity to humans
  - Time penalty for inefficient paths

--- -->

## ⚙️ Tech Stack

- Python
- PyBullet (Simulation Environment)
- Stable Baselines3 (RL Framework)
- PPO (Reinforcement Learning Algorithm)


## 🎥 Demo Video

👉 [Click here to watch the demo](https://drive.google.com/file/d/1t6WHoGGrdnXT0gKtgFRtGzPcOEKZGOSs/view?usp=sharing)


## 📊 Results (Preliminary)

The trained policy demonstrates:

- Goal-directed navigation
- Reduced collision behavior over training
- Safe interaction with moving human agents


## 🚀 Future Work

- Integration with ROS 2 and Gazebo
- More complex human behavior modeling
- Improved coordination between multiple robots
- Scalability to larger robot teams

<!-- ---

## 📂 Repository Structure -->
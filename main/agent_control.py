from __future__ import division, print_function, absolute_import
import pickle                                                                                                     
import sys
import numpy as np
import random
import gym
import time
import tensorflow as tf
import argparse
import pickle
import json
import ray
from agents.sac import SACAgent
from agents.utils import modify_obs
import matplotlib.pyplot as plt
from ray.rllib.agents.sac import SACTrainer, DEFAULT_CONFIG
import gym_minigrid 
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

def parse_args():
    """
    parse argument from commandline
    """
    parser = argparse.ArgumentParser("Agent Control Experiment Parser")
    parser.add_argument("--env-name", type=str, default="MiniGrid-TrapMazeS11N5-v0", help="minigrid environment to load")
    parser.add_argument("--mode", type=str, default='rgb_array') # 'human','None' or 'rgb_array'
    parser.add_argument("--load-dir", type=str, default="", help="where models should be loaded")
    parser.add_argument("--input-file", type=str, default="", help="where expert data should be loaded")
    parser.add_argument("--agent", type=str, default="DAgger", help="agent type")
    parser.add_argument("--test", action="store_true", default=False, help="collect the data from replay")
    parser.add_argument("--iter", type=int, default=100000, help="total number of training iterations")
    parser.add_argument("--save-rate", type=int, default=50, help="save model every x episodes")
    parser.add_argument("--gpu", action="store_true", default=False, help="use gpus")
    return parser.parse_args()


def test(agent, display_mode='human', episode_cnt = 100):    
    # ray.init()
    rew_avg = []
    rew = 0
    env = gym.make(agent.env)
    env.render(display_mode)
    obs = env.reset()
    act = agent.action(obs)
    obs, reward, done, info = env.step(act)
    rew += reward
    while episode_cnt:
        env.render(display_mode)
        act = agent.action(obs)
        # print(act)
        obs, reward, done, info = env.step(act)
        time.sleep(0.1)
        rew += reward
        if done:
            # rew += reward
            print("Accumulative reward: {}".format(rew))
            rew_avg.append(rew)
            obs = env.reset()
            rew = 0
            episode_cnt -=1
            # break
    print("mean reward over {} episodes: {}".format(episode_cnt, np.mean(rew_avg)))



if __name__ == "__main__":
    ray.init(num_cpus=32, num_gpus=2, dashboard_host='0.0.0.0')
    arglist = parse_args()

    agent = SACAgent(
                name="Soft Actor Critic Agent",
                environment=arglist.env_name,
                training_iterations=arglist.iter,
                checkpoint_path=arglist.load_dir,
                gpu=arglist.gpu
            )

    if arglist.test is True:
        assert arglist.load_dir != '', "Checkpoint path required for testing."
        test(agent)
    else:
        agent.train()


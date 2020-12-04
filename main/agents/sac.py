from agents import Agent
import random as r
import ray
from ray.rllib.agents.sac import SACTrainer, DEFAULT_CONFIG
from ray import tune

class SACAgent(Agent):
    def __init__(self, name, environment,training_iterations=10000, checkpoint_path=None):
        self.name = name
        self.env = environment
        self.config = DEFAULT_CONFIG
        self.iterations = training_iterations
        self.trainer = SACTrainer(env = self.env)
        # load model
        if checkpoint_path != '':
            self.trainer.restore(checkpoint_path)
        
    def action(self, obs):
        act = self.trainer.compute_action(obs)
        return act

    def train(self, save_iter = 100):
        for it in range(self.iterations):
            self.trainer.train()
            if it % save_iter == 0:
                checkpoint = self.trainer.save()
                print("checkpoint saved at", checkpoint)

    

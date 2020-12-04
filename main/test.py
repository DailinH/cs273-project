# from ray import tune
import ray
import gym
import gym_minigrid
from ray.rllib.agents.a3c.a2c import A2CTrainer
from ray.rllib.agents.DRIL import DRILTrainer, DEFAULT_CONFIG
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet_v2 import FullyConnectedNetwork

class CustomModel(TFModelV2):
    """Example of a custom model that just delegates to a fc-net."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        super(CustomModel, self).__init__(obs_space, action_space, num_outputs,
                                          model_config, name)
        self.model = FullyConnectedNetwork(obs_space, action_space,
                                           num_outputs, model_config, name)
        self.register_variables(self.model.variables())

    def forward(self, input_dict, state, seq_lens):
        return self.model.forward(input_dict, state, seq_lens)

    def value_function(self):
        return self.model.value_function()
        



ray.init()
ModelCatalog.register_custom_model("my_model", CustomModel)

config = {
    "eager": True,
    "env_config":{},
    "log_level": "DEBUG",
    "microbatch_size":10,
    "sample_batch_size": 15
    # "model":{ 
    #     "custom_model": "my_model"
    # },
    # "min_iter_time_s":10,ps
    # "vf_share_layers": True
    }

def a2cTraining(env='MiniGrid-TrapMazeS9N5-v0', config=config, iterations = 10000):
    trainer = A2CTrainer(env=env, config = config)
    for it in range(iterations):
        print("it {}".format(it))
        print(trainer.train())

def sacTraining(env='MiniGrid-TrapMazeS9N5-v0', config=config, iterations = 10000):
    trainer = SACTrainer(env=env)
    for it in range(iterations):
        print(trainer.train())

def DRILTraining(env='MiniGrid-TrapMazeS11N5-v0', config=DEFAULT_CONFIG, iterations = 10000):
    config['log_level'] = 'DEBUG'
    trainer = DRILTrainer(env=env, config = config)
    for it in range(iterations):
        print(trainer.train())

#  MiniGrid-TrapMazeS9N5-v0
#
# a2cTraining()

DRILTraining()

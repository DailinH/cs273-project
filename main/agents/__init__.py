class Agent(object):
    def __init__(self, name, model, environment, trainer_config, training_iterations):
        raise NotImplemented()

    def action(self, obs):
        raise NotImplemented()

    def train(self, environment, trainer_config, training_iterations):
        raise NotImplemented() 
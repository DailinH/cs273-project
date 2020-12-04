import random as r
import numpy as np

def weighted_rand_choice(weight):
    values = weight.values()
    k = r.uniform(0,sum(values))
    tot = 0
    act = None
    for i, v in enumerate(values):
        tot += v
        if tot > k:
            act = i
            return act
    print("k {}, values {}, act {}".format(k, values, act))
    # print(act)
    return act

def modify_obs(obs, arr=True):
    """
    modify the observations, making it easier to handle
    convert img representation to 1 digit / pixel
    """
    agent_pos = obs['agent_pos']
    obs['image'][agent_pos[0]][agent_pos[1]] = 0 # add extra agent position
    img = np.ndarray.flatten(np.asarray([[0.299*j[0]+0.587*j[1]+0.114*j[2] for j in i] for i in obs['image']])) # grayscale convertion
    m_obs = np.array(img)
    m_obs = np.append(m_obs, obs['direction'])
    if arr==False:
        m_obs = np.array_str(m_obs)
    return m_obs
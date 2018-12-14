import numpy as np


def calcReward(obs):
    ## write your own reward
    pos = obs['trackPos']
    angle = obs['angle']

    sp = np.array(obs['speedX'])
    spy = np.array(obs['speedY'])
    reward = sp * np.cos(1.0 * obs['angle']) - np.abs(1.0 * sp * np.sin(obs['angle'])) - 2 * sp * np.abs(obs['trackPos'] * np.sin(obs['angle'])) - spy * np.cos(obs['angle'])
    """
    reward_2 = 0
    if pos < 0 :
        reward_2 = (pos - 0) * 10
    elif pos > 0.6:
        reward_2 = (0.6 - pos) * 10
    if np.abs(pos-0.3) < 0.3:
        if np.abs(angle) < 0.1:
            reward = sp*(200*(np.abs(angle)-0.1)**2 + 2)
        else:
            reward = sp*(np.cos(angle*2) - np.abs(np.sin(angle)))*2
    
    """
    return reward

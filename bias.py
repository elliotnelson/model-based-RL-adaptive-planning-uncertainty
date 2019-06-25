import math
import numpy as np

from replay_buffer import normalize, obs_stdev, delta_obs_stdev


def bias_data(obs, actions, rewards, obs_next, dones):

    # assert CartPole before calling this

    actions_biased = []

    for i in range(obs.shape[0]):

        if obs[i][0]<0.:
            actions_biased.append(1.-actions[i])
        else:
            actions_biased.append(actions[i])

    actions_biased = np.array(actions_biased)

    return obs, actions_biased, rewards, obs_next, dones


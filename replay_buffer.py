import math
import random
import numpy as np
import tensorflow as tf

from baselines.common.segment_tree import SumSegmentTree, MinSegmentTree

# this is a modified & extended copy of baselines/deepq/replay_buffer.py


class ReplayBuffer(object):
    def __init__(self, size):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done, t_horizon, p_transition, p_future): # modified
        data = (obs_t, action, reward, obs_tp1, done, t_horizon, p_transition, p_future) # modified

# these should be the same
#        print(self._next_idx)
#        print(len(self._storage))

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            print('Should not get here. Should always append.')
            exit()
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) # % self._maxsize # ELN: modified so that we never loop back, but always keep _next_idx at end of the list

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done, _, _, _ = data # modified
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)

    # ** IF I EDIT THIS, CONSIDER add_episode() in PER CLASS BELOW
    # ** currently this is just copied from PrioritizedReplayBuffer(), w/ priorities and horizon args removed
    # ** ideally, this should be wrapped like PER's add() wraps ER's add()
    def add_episode(self, obs, actions, rewards, dones, obs_last):
        n = len(obs)
        obs.append(obs_last)
        while len(self._storage)+n>self._maxsize:
            if len(self._storage)==0:
                print('Not enough room for latest whole episode in replay buffer.')
                exit()
            for idx in range(0,self._maxsize):
                if self._storage[idx][4]==1.0: # end of 1st episode in buffer
                    break
            del self._storage[0:idx+1] # delete old episode from the buffer
            self._next_idx -= idx+1 # backtrack
            assert self._next_idx == len(self._storage)
        idx0 = len(self._storage) # shoudl be < self._maxsize-n
        for i in range(n): # add the earliest transitions first
            self.add(obs[i], actions[i], rewards[i], obs[i+1], float(dones[i]), 1, 0, 0)
        idxes = list(range(idx0,idx0+n))
        return


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size, alpha, gamma):
        """Create Prioritized Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        alpha: float
            how much prioritization is used
            (0 - no prioritization, 1 - full prioritization)
        See Also
        --------
        ReplayBuffer.__init__
        """
        super(PrioritizedReplayBuffer, self).__init__(size)
        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

        self.gamma = gamma # added, for discounting td error priorities inherited from successor states (in update_priorities)

    def add(self, *args, **kwargs): # currently, should only be called by add_episode()
        """See ReplayBuffer.store_effect"""
        idx = self._next_idx
        super().add(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha # ELN: this initializes the added transition's priority to _max_priority; this determines the (alpha-dependent) distribution that will be sampled from)
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        p_total = self._it_sum.sum(0, len(self._storage) - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, beta):
        """Sample a batch of experiences.
        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        beta: float
            To what degree to use importance weights
            (0 - no corrections, 1 - full correction)
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        weights: np.array
            Array of shape (batch_size,) and dtype np.float32
            denoting importance weight of each sampled transition
        idxes: np.array
            Array of shape (batch_size,) and dtype np.int32
            idexes in buffer of sampled experiences
        """
        assert beta >= 0 # ELN: modified from > to >=

        idxes = self._sample_proportional(batch_size) # sample from the priority-weighted distribution

        print('buffer length')
        print(len(self._storage))
        print('idxes sampled at alpha=0')
        print(idxes)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)

        for idx in idxes: # apply importance sampling weights
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)
        encoded_sample = self._encode_sample(idxes)
        return tuple(list(encoded_sample) + [weights, idxes])

    def update_priorities(self, idxes, priorities, idx_sampled=None):
        """Update priorities of sampled transitions.
        sets priority of transition at index idxes[i] in buffer
        to priorities[i].
        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        idxes_updated = idxes # we'll append to this list the earlier transitions whose priorities depend on idxes priority updates

        for idx, priority in zip(idxes, priorities):

            # assert priority > 0 # enforced on self._it_sum and self._it_min below, instead of here
            assert 0 <= idx < len(self._storage)

            # ELN: replace the old priority in the element of self._storage, and update the transition's priority_future
            data = self._storage[idx]
            o, a, r, o2, d, t_horizon, priority_old, priority_future = data
            self._storage[idx] = o, a, r, o2, d, t_horizon, priority, priority_future + abs(priority)-abs(priority_old)

            # ELN: update the priority_future's for preceding transitions in same episode
            n = 1
            # recall t_horizon defined above
            while n<=t_horizon and idx-n>=0: # until we get back to an earlier transition in same episode  whose t_horizon does not include the (future) transition idx
                data = self._storage[idx-n] # recall idx=0 is earliest/oldest transition; we're moving that way, back in time
                o, a, r, o2, done, t_horizon, p, priority_future = data
                if done: break # we got back to the last transition of another episode
                priority_future += (self.gamma ** n) * (abs(priority) - abs(priority_old)) # recall priority & priority_old are defined above, for transition idx
                self._storage[idx-n] = o, a, r, o2, done, t_horizon, p, priority_future 
                if idx-n not in idxes_updated: # add to list of updated transitions
                    idxes_updated.append(idx-n)                
                n += 1 

        idx_max = max(idxes)
        assert idx_max==max(idxes_updated)
        for idx in idxes_updated:

            if idx==idx_max:
                assert self._storage[idx][4]
            #    print('done=True for this idx')
            else:
                assert not self._storage[idx][4]

            #if idx==idx_sampled:
            #    print('This transition was sampled for obs_init')
            #print('idx, td error, future, priority')
            #print(idx)
            #print(self._storage[idx][-2])
            #print(self._storage[idx][-1])

            priority = abs(self._storage[idx][-1]) # abs() may be redundant here, if already positive
            #print(priority ** self._alpha)

            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)

        return

    # ** IF I EDIT THIS, CONSIDER add_episode() in ReplayBuffer() CLASS ABOVE
    def add_episode(self, obs, actions, rewards, dones, obs_last, priorities, horizons):
# note: I think that env.step() assumes 'action' has different shapes in discrete vs. continuous env's, so these should be treated differently to ensure same shape in replay_buffer in both cases
#                    if isinstance(envs[0].action_space, Discrete):
#                        action = np.expand_dims(action, axis=1) ## I'm not sure this is correct for multiple discrete dimensions
#                    else:
#                        action = np.squeeze(action, axis=0) ## this works for HalfCheetah, which has action_dims=6
        n = len(obs)
        obs.append(obs_last)
        while len(self._storage)+n>self._maxsize:
            # delete earliest episode(s)
#            print('ok, delete the 1st episode (make sure this is ok with rest of code here)')
#            print(len(self._storage))
            if len(self._storage)==0:
                print('Not enough room for latest whole episode in replay buffer.')
                exit()
            for idx in range(0,self._maxsize):
                if self._storage[idx][4]==1.0: # end of 1st episode in buffer
                    break
            del self._storage[0:idx+1] # delete old episode from the buffer
            self._next_idx -= idx+1 # backtrack
#            print('deleting this many entries: ' + str(idx+1))
            assert self._next_idx == len(self._storage)
#            print('_next_idx = ' + str(self._next_idx))
#            print('buffer length = ' + str(len(self._storage)))
#            print('do we have room now?') 
        idx0 = len(self._storage) # shoudl be < self._maxsize-n
        for i in range(n): # add the earliest transitions first
            self.add(obs[i], actions[i], rewards[i], obs[i+1], float(dones[i]), horizons[i], 0, 0) # initialize priorities at zero, update below
#            print(len(self._storage))
#            print(obs[i])
#        print(self._storage[-1])
#        assert self._storage[idx0][0][0]==obs[0][0] # check that the 1st obs is in the expected position in the buffer
#        print('length of episode = ' + str(n))
        idxes = list(range(idx0,idx0+n))
#        print('idxes')
#        print(idxes)
        if priorities is not None:
            self.update_priorities(idxes, priorities)
#        for i in range(n):
#            print(self._storage[idx0+i])
        return

    def idxes_episode(self, idx):

        done = False
        i = 0
        while not done:
            done = self._storage[idx+i][4]
            i += 1
        idx_max = idx + i-1

        done = False
        i = 1
        while not done and idx-i>=0:
            done = self._storage[idx-i][4]
            i += 1
        idx_min = max(idx - (i-2), 0)

        return list(range(idx_min, idx_max+1))

    def update_horizons(self, fixed_horizon=None, idxes=None, horizons=None):

        if fixed_horizon is not None:
            for idx in range(len(self._storage)):
                # this does not work; item assignment not supported ## self._storage[idx][-3] = fixed_horizon
                print('Oops')
                exit()
        else:
            print('UPDATE update_horizons()...')

        return


# normalize the data in a replay buffer object from: github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py

def obs_mean(replay_buffer):

    buf = replay_buffer._storage
    buf = list(zip(*buf))
    obs = np.array(list(buf[0]))

    return np.mean(obs, axis=0)

def obs_stdev(replay_buffer):

    buf = replay_buffer._storage
    buf = list(zip(*buf))
    obs = np.array(list(buf[0]))

    return np.std(obs, axis=0)

# compute the standard deviation of delta_obs = obs_next - obs
def delta_obs_stdev(replay_buffer):

    buf = replay_buffer._storage
    buf = list(zip(*buf))
    obs = np.array(list(buf[0]))
    obs_next = np.array(list(buf[3]))
    delta_obs = obs_next - obs

    return np.std(delta_obs, axis=0)

# not currently using this method
# checked that this is working correctly
def normalize(replay_buffer):

    buf = replay_buffer._storage
    size = len(buf)

    buf = list(zip(*buf))    

    obs = np.array(list(buf[0]))
    obs_next = np.array(list(buf[3]))

    obs -= obs.mean(axis=0)
    obs /= np.std(obs, axis=0)

    obs_next -= obs_next.mean(axis=0)
    obs_next /= np.std(obs_next, axis=0)

    for i in range(size):
        print('Ideally the next line would be robust to added elements to buffer entries, instead of listing 4, 5, 6, 7...')
        replay_buffer._storage[i] = (obs[i], replay_buffer._storage[i][1], replay_buffer._storage[i][2], obs_next[i], replay_buffer._storage[i][4], replay_buffer._storage[i][5], replay_buffer._storage[i][6], replay_buffer._storage[i][7])

    return



#### SCRAP NOTES ####

# THIS IS EQUIVALENT TO THE NUMPY CODE IN normalize() #
#
#    obs_mean = sum(obs)/size
#    obs_next_mean = sum(obs_next)/size
#
#    obs2, obs_next2 = [], []
#    for o in obs:
#        obs2.append(o - obs_mean)
#    for oo in obs_next:
#        obs_next2.append(oo - obs_next_mean)
#
#    obs_var = np.zeros(obs[0].shape)
#    obs_next_var = np.zeros(obs[0].shape)
#    for o in obs2:
#        obs_var += np.square(o)/size
#    for oo in obs_next2:
#        obs_next_var += np.square(oo)/size
#
#    obs_std = np.sqrt(obs_var)
#    obs_next_std = np.sqrt(obs_next_var)
#
#    obs3, obs_next3 = [], []
#    for o in obs2:
#        obs3.append(o/obs_std)
#    for oo in obs_next2:
#        obs_next3.append(oo/obs_next_std)


import time
import os
import sys
import argparse

#sys.path.append("/home/e5nelson/.conda/envs/ml_eln2/lib/python3.6/site-packages/") # already included in sys.path
#sys.path.insert(0, "/home/e5nelson/.conda/envs/ml_eln2/lib/python3.6/site-packages/") # already included in sys.path

# import gym from ml_eln2, not ml; recall that it's the subdirectory gym/gym that we needed to put directly in site-packages as'gym'
sys.path.remove('/usr/local/anaconda3/envs/ml/lib/python3.6/site-packages')
import gym
sys.path.append('/usr/local/anaconda3/envs/ml/lib/python3.6/site-packages') # note: I created copy env, ml_eln

sys.path.append("/home/e5nelson/.local/lib/python3.5/site-packages")
sys.path.append("/usr/local/anaconda3/lib/python3.6/site-packages") # for mpi4py
#sys.path.append("/usr/local/anaconda3/lib/python3.7/site-packages")
sys.path.append("~/rl/venv/lib/python3.7/site-packages") # when using MBP
sys.path.append("/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages") # when using MBP
sys.path.append("./baselines")
sys.path.append("/home/e5nelson/rl/spinningup") # AI-Vector
sys.path.append("~/.mujoco")

import argparse
import numpy as np
import tensorflow as tf
import matplotlib

import tensorflow_probability as tfp
import mujoco_py

from gym.spaces import Box, Discrete

import baselines
from baselines import logger
from baselines.common import explained_variance
from baselines.common.vec_env.vec_env import VecEnv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv

from replay_buffer import ReplayBuffer, PrioritizedReplayBuffer 
from replay_buffer import normalize, obs_stdev, delta_obs_stdev

# I'm not using this because I'm saving *.py files to log dir instead of using convert_json
#import spinup ## note that as of April 2019, spinup works with ml_eln virtual env, but not ml env
#from spinup.utils.logx import EpochLogger
#from spinup.utils.serialization_utils import convert_json

from tmodels import use_env_model, PredictiveModel, a2c_transition_data, fnn, fnn2

from log_data import log_list_to_file, float_to_str

from render import render

# modified from vpg.py


# compute a scalar norm of gradients, from list returned by compute_gradients()
def gradients_norm(grad_var_list, order=None):

    # policy gradient norm
    grads_pi = list(zip(*grad_var_list))[0] # stackoverflow.com/questions/8081545/how-to-convert-list-of-tuples-to-multiple-lists
    ### grads_pi = np.asarray(grads_pi_batch).flatten() # list all the grads in a vector
    grads_pi_norms = [] # a list of norms of different blocks of the policy network's variables
    for j in range(len(grads_pi)):
        grads_pi_norms.append(np.linalg.norm(grads_pi[j], ord=order))
    grad_pi_norm = np.linalg.norm(np.asarray(grads_pi_norms))

    return grad_pi_norm

def kl(probs1, probs2):
    p1 = tf.distributions.Categorical(probs=probs1)
    p2 = tf.distributions.Categorical(probs=probs2)
    return tf.distributions.kl_divergence(p1, p2)

# given (r1,r2,...r_t), generate list of discounted returns, bootstrapped from V(s_t) as last element
def returns_nstep(rewards, val_nstep, gamma, terminal=None, done_probs_cum=None): # terminal and done_probs are alternatives to each other, the former corresponds to done_probs_cum=(False,...,False,terminal)

    n = len(rewards)
    r = rewards.copy()
    if done_probs_cum is not None:
        assert len(done_probs_cum)==n
        terminal = done_probs_cum[-1]
        for i in range(1,n): # r[0] unchanged, b/c even if done[0]=True, we still get r[0] reward
            r[i] *= done_probs_cum[i-1]
    else:
        assert terminal is not None
    returns_disc_batch = np.zeros(n)
    returns_disc_batch[n-1] = r[n-1] + gamma * (1 - terminal) * val_nstep # bootstrap from V(s_t)=0 if s_t=terminal
    for i in range(1,n):
        returns_disc_batch[(n-1)-i] = r[(n-1)-i] + gamma * returns_disc_batch[(n-1)-i+1]
    return returns_disc_batch

def cum_sum_discounted(reward_list, gamma):

    T = len(reward_list)
    discount_list = []

    # list of gamma^t discount coefficients
    ## probably better to do this w/ numpy array, as in pg_cartpole.py
    for t in range(T):
        discount = gamma**t
        discount_list.append(discount)

    # numpy arrays: easier to multiply element-wise with *
    discount_list = np.array(discount_list)
    G_list = np.zeros_like(reward_list, dtype=np.float32)

    for t in range(T):
        G = sum(reward_list[t:] * discount_list[t:])
        G_list[t] = G 

    return G_list

# policy and value network shared up until linear output layer; matches OpenAI Baselines
def fnn_shared(obs, action_dims, hidden_sizes=(64,64)):

    l2 = 0e-5
    reg = tf.contrib.layers.l2_regularizer(scale=l2, scope='shared')

    x = obs
    for num_neurons in hidden_sizes:
        x = tf.layers.dense(x, units=num_neurons, activation=tf.nn.tanh, kernel_regularizer=reg)

    return x

def fnn_policy(obs_features, action_dims, hidden_sizes=(64,64)): # hidden_sizes=(64,64)

    l2_pi = 0e-5
    reg = tf.contrib.layers.l2_regularizer(scale=l2_pi, scope='policy')

    x = obs_features
    for num_neurons in hidden_sizes:
        x = tf.layers.dense(x, units=num_neurons, activation=tf.nn.tanh, kernel_regularizer=reg)
    # initialize weights at zero to maximize initial exploration: kernel_initializer=tf.zeros_initializer()
    x = tf.layers.dense(x, units=action_dims, activation=None, kernel_initializer=tf.zeros_initializer(), kernel_regularizer=reg)
    return x  # = logits

#def policy_gaussian(obs, action_dims, network_mean, network_logstd):
#    mu = network_mean(obs, action_dims) # tensor, with dims for batch size and action_dims
#    log_std = network_logstd(obs, action_dims)
#    dist = tfp.distributions.Normal(loc=mu, scale=tf.exp(log_std))
#    actions = tf.squeeze(dist.sample([1])) # samples across the batch dimension as well
#    actions_logprobs = tf.reduce_sum(dist.log_prob(actions), axis=1) # probability = product of probs for each action dimension
#    return actions, actions_logprobs, log_std 

def fnn_value(obs_features, hidden_sizes=(64,64)): # hidden_sizes=(64,64)

    # b_initializer = tf.random_normal_initializer(mean=100)

    l2_value = 0e-5
    reg = tf.contrib.layers.l2_regularizer(scale=l2_value, scope='value')

    x = obs_features
    for num_neurons in hidden_sizes:
        x = tf.layers.dense(x, units=num_neurons, activation=tf.nn.tanh, kernel_regularizer=reg)
    x = tf.layers.dense(x, units=1, activation=None, kernel_regularizer=reg) # bias_initializer=b_initializer
    return x  # = V(obs)


class A2C():

    def __init__(self,
                 env_name,
                 sess,
                 nn_policy=fnn_policy,
                 nn_value=fnn_value,
                 nn_shared=None,
                 n_envs=1,
                 gamma=0.99,
                 lr=1e-2, lr_decay_rate=2e-7, lr_min_to_max=1e-3, # checked: lr_decay_rate=1.25e-8 matches the schedule in OpenAI baselines a2c.py (given their total_timesteps = 80M?); 1.25e-6 anneals to zero in ~1M timesteps; lr_decay_rate=1e-5 works for cartpole
                 # for CartPole, can use e.g. lr=1e-2, lr_decay_rate=1e-5
                 alpha=0.99, epsilon=1e-5, # RMSProp parameters
                 nsteps=200, # if this exceeds episode length, update will be at end of episode
                 n_iters=1e50,
                 v_coef=0.5,
                 entropy_coef=0.01,
                 entropy_coef_continuous_control=1e-4,
                 max_grad_norm=None):

        if n_envs>1:
            print('Need to make replay_buffer organized appropriately according to episodes for n_envs>1. Probably need to figure out how to use add_episodes() without screwing up model obs learning.')
            exit()

        self.sess = sess

        self.envs = []
        for e in range(n_envs):
            self.envs.append(gym.make(env_name))
        env = gym.make(env_name) ## this is just used to define a few things
        self.env = env ## references in env_model

        obs_dims = env.observation_space.shape[0]
        if isinstance(env.action_space, Discrete):
            discrete_actions = True
        else:
            discrete_actions = False
            assert isinstance(env.action_space, Box)
            entropy_coef = entropy_coef_continuous_control # overrides default entropy_coef value
    
        if discrete_actions:
            action_dims = env.action_space.n
        else:
            action_dims = env.action_space.shape[0]
            assert len(env.action_space.shape)==1  ## if this is violated, I should understand what the other tuple components are
    
        if discrete_actions:
            self.actions_ph = tf.placeholder(tf.int32, [None,]) ## COMMA? # batch of actions taken
            self.actions_one_hot = tf.one_hot(self.actions_ph, action_dims) # turn each action into a one-hot vector
        else: # continuous
            self.actions_ph = tf.placeholder(tf.float32, [None, action_dims])
    
        self.obs_ph = tf.placeholder(tf.float32, [None, obs_dims])
    
        if nn_shared is not None:
            with tf.variable_scope('shared'):
                obs_features = nn_shared(obs_ph, action_dims)
        else:
            obs_features = self.obs_ph
    
        with tf.variable_scope('policy'):
            if discrete_actions:
                actions_logits = nn_policy(obs_features, action_dims)
                self.actions_probs = tf.nn.softmax(actions_logits) ## Confusing name; not the exp of actions_logprobs below...
                pi_entropies = -tf.reduce_sum(tf.nn.softmax(actions_logits) * tf.nn.log_softmax(actions_logits), axis=1)
                self.pi_entropy = tf.reduce_mean(pi_entropies)
                self.entropy_loss = -self.pi_entropy
                ## tf.summary.scalar('S_pi', pi_entropy)
                self.actions_out = tf.squeeze(tf.multinomial(logits=actions_logits, num_samples=1))  # draws an action from the distribution specified by logits
                # self.actions_out = tf.squeeze(tf.random.categorical(logits=actions_logits, num_samples=1))
                actions_logprobs = tf.reduce_sum(self.actions_one_hot * tf.nn.log_softmax(actions_logits), axis=1) # multiply across the action_dims dimension to pick out (at each timestep or batch element) the log-probability of the action that was taken 
            else:
                network_mean = nn_policy 
                mu = network_mean(obs_features, action_dims) # tensor, with dims for batch size and action_dims
                #network_logstd = nn_policy
                #log_std = network_logstd(obs_features, action_dims)
                log_std = tf.get_variable('log_std', initializer=-0.5*np.ones(action_dims, dtype=np.float32)) # from github.com/openai/spinningup/blob/master/spinup/algos/vpg/core.py
                pi_dist = tfp.distributions.Normal(loc=mu, scale=tf.exp(log_std)) # probability distribution(s) pi(actions|obs_ph)
                self.actions_out = pi_dist.sample()
    ##            #self.actions_out = tf.squeeze(pi_dist.sample(), axis=[-1,-2]) # samples across the batch dimension as well
    ##            self.actions_out = tf.squeeze(pi_dist.sample(), axis=-1) # samples across the batch dimension as well
    ##            self.actions_out = pi_dist.sample(sample_shape=tf.constant([None,action_dims])) # samples across the batch dimension as well
                actions_logprobs = tf.reduce_sum(pi_dist.log_prob(self.actions_ph), axis=1) 
                # self.actions_out, actions_logprobs, log_std = policy_gaussian(obs_features, action_dims, network_mean=network_mean, network_std=network_std)
                self.pi_entropy = tf.reduce_sum(log_std) + 0.5*(1+1.83788) # from formula for differential entropy for Gaussian; note that log of matrix determinant of diagonal covariance -> sum over components of log_std (https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Differential_entropy)
                self.entropy_loss = -self.pi_entropy # sum over batch
                self.entropy_loss = -self.pi_entropy # sum over batch
    
        with tf.variable_scope('value'):
            ## obs_value_ph = tf.placeholder(tf.float32, [None, obs_dims])
            self.values = nn_value(obs_features)
    
        # variable collections
        pi_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='policy')
        val_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='value')
        shared_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='shared')
        params = tf.trainable_variables('policy')+tf.trainable_variables('value')+tf.trainable_variables('shared')
    
        # policy loss
        self.returns_disc_ph = tf.placeholder(tf.float32, [None,]) ## COMMA? # shape: 'None' stands for batch size
        self.pi_loss_0 = -tf.reduce_mean(tf.stop_gradient(self.returns_disc_ph - self.values) * actions_logprobs) # normalized dot-product of returns and log-probabilities, over timesteps (or batch)
        self.pi_loss = self.pi_loss_0 + tf.losses.get_regularization_loss(scope='policy')
    
        # value loss
        self.v_loss_0 = tf.losses.mean_squared_error(labels=self.returns_disc_ph, predictions=tf.squeeze(self.values)) # can add weights=n_envs*nsteps; the weights (coefficient) changes the 'mean' to a 'sum' over timesteps and workers, consistent with the 'sum' in pi_loss 
        # (alternate:) v_loss = tf.reduce_mean((values - ...)**2)
        self.v_loss = self.v_loss_0 + tf.losses.get_regularization_loss(scope='value')

        self.entropy_coef = entropy_coef
 
        # total loss
        self.loss = self.pi_loss + v_coef*self.v_loss + self.entropy_coef*self.entropy_loss
        self.loss += tf.losses.get_regularization_loss(scope='shared')

        self.lr0 = lr
        self.lr = lr
        self.lr_decay_rate = lr_decay_rate
        self.lr_min_to_max = lr_min_to_max 
        self.lr_ph = tf.placeholder(tf.float32, [])
        optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr_ph, decay=alpha, epsilon=epsilon)
        # optimizer = tf.train.AdamOptimizer(learning_rate=lr_ph)
        # optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, clip_norm=1e10)
        # clip the gradients; adapted from baselines a2c.py
        grads = tf.gradients(self.loss, params)
        if max_grad_norm is not None:
            grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        self.train_op = optimizer.apply_gradients(grads)
        # train_op = optimizer.minimize(self.loss)  # shorter, if we don't clip gradients
        self.grads_vars = optimizer.compute_gradients(self.loss, var_list=pi_vars+val_vars+shared_vars)
        self.grads_vars_pi = optimizer.compute_gradients(self.loss, var_list=pi_vars+shared_vars) # just the policy network's gradients
        self.grad_v = optimizer.compute_gradients(self.v_loss_0, var_list=val_vars+shared_vars)
        self.grad_pi = optimizer.compute_gradients(self.pi_loss_0, var_list=pi_vars+shared_vars)
        ## .. = optimizer.compute_gradients(self.loss, var_list=val_vars+shared_vars)0i
        ## grads = tf.zeros(...) # couldn't get to work, for minimize's grad_loss; what shape should this tensor have?    

        ### restructure this? with A2C.train(), or by defining these params outside of A2C()
        self.n_envs = n_envs
        self.n_iters = n_iters
        self.nsteps = nsteps
        self.gamma = gamma

        self.total_timesteps = 0
        self.total_timesteps_sim = 0

    def run(self, obs_init, nsteps, n_envs, n_episodes=None, replay_buffer=None, render=False, envs=None):

        obs_batch, actions_batch, rewards_batch, dones_batch = [[] for _ in range(n_envs)],  [[] for _ in range(n_envs)],  [[] for _ in range(n_envs)],  [[] for _ in range(n_envs)]

####        reward, done = [0]*n_envs, [False]*n_envs # initialize

        if envs is None:
            envs = self.envs
        else:
            assert len(envs)==n_envs

        for e in range(n_envs): # *** GPU-OPTIMIZED? ***
##        envs_stepped = 0
##        tf.while_loop(envs_stepped<env.num_envs, # loop in parallel over A2C workers
##            run,
##            envs_stepped += 1
##            loop_vars = ? ... [i]
            
            obss, actions, rewards, dones = [], [], [], []
            obs = obs_init[e]
            obs_last = obs_init
            n_eps = 0
            for i in range(nsteps): # if episode ends first, break out of loop below

                obss.append(obs.copy()) # detach obss from obs with copy() 

                action = self.sess.run(self.actions_out, feed_dict={self.obs_ph: obs.reshape(1, -1)})
                # np.reshape(): "One shape dimension can be -1. In this case, the value is inferred from the length of the array and remaining dimensions."
                ## so I guess np.reshape() ensures that the batch dimension size is 1, e.g. takes the transpose
                ## it's weird that [0] was included in pg_cartpole.py, since actions_out already uses tf.squeeze()...

                actions.append(action)

                # if render==True:
                #    env.render()

                if not envs[e].unwrapped.spec.id=='Pendulum-v0': ## this is an ugly hack
                    obs, reward, done, _ = envs[e].step(action)
                else:
                    obs, reward, done, _ = envs[e].step(np.squeeze(action, axis=-1))                 
                ## a2c.total_timesteps += 1 ## if this is an arg of run(), updated and returned
                ## if envs[e]._elapsed_steps>=envs[e]._max_episode_steps-1: # check if time limit exceeded

                rewards.append(reward)
                dones.append(done)

                ##### *** TO PARALLELIZE, should collect separately for each worker and dump in buffer after all workers done
                # episodes are collected later and dumped in buffer, in prioritized_replay case
                if replay_buffer is not None: # note: I think that env.step() assumes 'action' has different shapes in discrete vs. continuous env's, so these should be treated differently to ensure same shape in replay_buffer in both cases
                    if isinstance(envs[0].action_space, Discrete):
                        a = np.expand_dims(action, axis=1) ## I'm not sure this is correct for multiple discrete dimensions
                    else:
                        a = np.squeeze(action, axis=0) ## this works for HalfCheetah, which has action_dims=6
                    replay_buffer.add(obss[-1], a, reward, obs, float(done), 1, 0, 0) # last 3 elements are not used since prioritized_replay=False in this case 

                if done: # episode ended
                    obs = envs[e].reset() # for A2C, we continue next iteration with new episode
                    n_eps += 1
                if i==nsteps-1:
                    obs_last[e] = obs.copy()

                if n_episodes is not None and n_eps==n_episodes: # optionally, stop after worker collects fixed num of episodes
                    obs_last[e] = obs.copy()
                    break

            obs_batch[e] = obss.copy()
            actions_batch[e] = actions.copy()  # np.squeeze(action,axis=1) ## axis=0 ?
            rewards_batch[e] = rewards.copy()
            dones_batch[e] = dones.copy()

        return obs_batch, actions_batch, rewards_batch, dones_batch, obs_last


##    def train(self, 

def train(env_name,
          logdir = os.environ["OPENAI_LOGDIR"],
          render_period=1e40, # defaults to never rendering the env
          use_model=True,
          plan_bool=True,
          buffer_size=10000, # divide by model-free batch size to get # iterations of data which the buffer includes
          store_priorities=False, # True if we want to use PER
          prioritized_replay=False, # choose between PrioritizedReplayBuffer() or ReplayBuffer()
          prioritized_replay_alpha=0.0,
          prioritized_replay_nsteps_horizon=5, # defines 'future' priorities
          load_env_model_path='data/cartpole_wind/pretrained_models/model0/env_model.ckpt'): # default None

    ## using spinning up logger to save hyperparams to file; not sure how to use logger for this
#    logger_kwargs = dict(output_dir=logdir, exp_name=None)
#    spinningup_logger = EpochLogger(**logger_kwargs)
#    spinningup_logger.save_config(locals())
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True)) # allows outputting of device that tensors/ops are placed on, when print(sess.run(...))

    a2c = A2C(env_name=env_name, sess=sess)
    ## cf. OpenAI Baselines PolicyWithValue() or a2c Model() classes
    ## like in a2c.learn(); **network_kwargs?

    if use_model:
        env_model = PredictiveModel(env=gym.make(env_name), sess=sess)
        ## currently, model uses separate copy of env to generate model-free trajectories to compare its rollouts to...
        kwargs_model = dict([('logdir', logdir), ('modelfree_alg', a2c)])
        if prioritized_replay:
            replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha=prioritized_replay_alpha, gamma=a2c.gamma) 
            print('Since model obs loss failed to drop very much when trained on PrioritizedReplayBuffer(alpha=0)-sampled data, there may be a problem with recovering uniform sampling in this case. Figure this out before using alpha any more...')
            exit()
        else:
            replay_buffer = ReplayBuffer(buffer_size)
    else:
        replay_buffer=None

    sess.run(tf.global_variables_initializer())

    train_writer = tf.summary.FileWriter(logdir, sess.graph)
    saver = tf.train.Saver()
    if use_model:
        saver_env_model = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'env_model'))
    if use_model and load_env_model_path is not None:
        saver_env_model.restore(sess, load_env_model_path)

    ## a2c.train(n_iters=A2C.n_iters)
    ## restructure this, as suggested at end of A2C().__init__ ?
    gamma = a2c.gamma
    n_envs = a2c.n_envs
    envs = a2c.envs
    n_iters = a2c.n_iters
    nsteps = a2c.nsteps

    total_episodes = 0

    episode_returns, episode_lengths = [0]*n_envs, [0]*n_envs # running returns and ep lengths for each worker
    episode_rewards, episode_obs_vals = [[] for _ in range(n_envs)], [[] for _ in range(n_envs)]  # for each worker, list of r's and V(s)'s for current batch of states
    episode_obs, episode_actions, episode_dones = [[] for _ in range(n_envs)], [[] for _ in range(n_envs)], [[] for _ in range(n_envs)]
    episode_td_errors = [[] for _ in range(n_envs)]

    obs_init = []
    for e in range(n_envs):
        obs_init.append(envs[e].reset())

    for i in range(1,int(n_iters)):

        returns_disc = [[] for _ in range(n_envs)] # will hold discounted+bootstrapped returns seen from each state s_t, for each worker, for the current batch

        if i % render_period==0: render = True

        if use_model and not prioritized_replay:
            buf = replay_buffer
        else: buf = None 
        #####
        obs, actions, rewards, dones, obs_last = a2c.run(obs_init, nsteps=a2c.nsteps, n_envs=a2c.n_envs, replay_buffer=buf, render=True)

        obs_init = obs_last # for next iteration

        a2c.total_timesteps += nsteps*n_envs

        # log information and get discounted returns 'R' for training
        for e in range(n_envs): # could be parallelized, but should be fast anyways
            t0 = 0
            for t in range(nsteps): # should be valid for nsteps >> episode lengths
                if dones[e][t] or t==nsteps-1: # an episode ended, or end of batch
                    episode_lengths[e] += t+1 - t0
                    episode_returns[e] += sum(rewards[e][t0:t+1]) # starts from 0 unless t0=0 (continuing episode)
                    episode_rewards[e] += list(rewards[e][t0:t+1])
                    episode_obs[e] += list(obs[e][t0:t+1])
                    episode_actions[e] += list(actions[e][t0:t+1])
                    episode_dones[e] += list(dones[e][t0:t+1])
                    # keep track of V(obs) for trajectories; note that V is updated every nsteps
                    obs_vals = sess.run(a2c.values, feed_dict={a2c.obs_ph: obs[e][t0:t+1]})
                    obs_vals = np.squeeze(sess.run(a2c.values, feed_dict={a2c.obs_ph: obs[e][t0:t+1]}), axis=1) # *** PARALLELIZE? ***
                    # may continue from previous batches
                    episode_obs_vals[e] += list(obs_vals) ## would make more sense for this to be np array throughout; also returns_disc
                    if dones[e][t]:
                        val_nstep = 0
                    else:
                        assert t+1==nsteps
                        val_nstep = np.squeeze(sess.run(a2c.values, feed_dict={a2c.obs_ph: obs_last[e].reshape(1,-1)})) # if target network: values_target, obs_value_target_ph
                        val_nstep = np.asscalar(val_nstep)
                    returns_disc[e] += list(returns_nstep(rewards[e][t0:t+1], val_nstep, gamma, terminal=dones[e][t]))
                    obs_next_vals = list(obs_vals[1:]) + [val_nstep]
                    td_errors = []
                    for i in range(len(obs_next_vals)): ## this is kind of a hack
                        td_errors.append(rewards[e][t0:t+1][i] + [v * gamma for v in obs_next_vals][i] - list(obs_vals)[i])
                    episode_td_errors[e] += td_errors
                    t0 = t+1 # new episode started at this timestep
                    if dones[e][t]: # episode ended
                        total_episodes += 1
                        assert episode_lengths[e]==len(episode_obs[e])
                        if use_model and not prioritized_replay:
                            a = 3 ## do nothing
                            #####replay_buffer.add_episode(episode_obs[e], episode_actions[e], episode_rewards[e], episode_dones[e], obs_last[e])
                        elif use_model and prioritized_replay:
                            if store_priorities:
                                priorities=episode_td_errors[e]
                            else:
                                priorities=None
                            # checked that this removes oldest episodes as needed, and adds episode to end of buffer
                            replay_buffer.add_episode(episode_obs[e], episode_actions[e], episode_rewards[e], episode_dones[e], obs_last[e],
                                                  priorities=priorities, horizons=[prioritized_replay_nsteps_horizon]*episode_lengths[e])
                        if a2c.env.unwrapped.spec.id=='CartPole-v0':
                            theta, x = [], []
                            for j in range(len(episode_obs[e])):
                                theta.append(episode_obs[e][j][2])
                                x.append(episode_obs[e][j][0])
                            log_list_to_file(theta, logdir + '/theta.txt')
                            log_list_to_file(x, logdir + '/x.txt')
                        #print('added to buffer')
                        #for i in range(len(replay_buffer._storage)-len(episode_obs[e]), len(replay_buffer._storage)):
                        #    print(replay_buffer._storage[i])
                        episode_returns_discounted = cum_sum_discounted(episode_rewards[e], gamma)
                        rewards_ev = explained_variance(np.array(episode_obs_vals[e]), np.array(episode_returns_discounted)) ## define as np arrays earlier?
                        logger.record_tabular('number_episodes', total_episodes)
                        logger.record_tabular('episode_lengths', episode_lengths[e])
                        logger.record_tabular('episode_returns', episode_returns[e])
                        logger.record_tabular('Explained variance in returns (discounted)', rewards_ev)
                        logger.record_tabular('Value ftn mean over episode states', sum(episode_obs_vals[e])/len(episode_obs_vals[e]))
                        # logger.record_tabular('Value ftn at final episode state', episode_obs_vals[e][-1])
                        episode_lengths[e], episode_returns[e] = 0, 0
                        episode_rewards[e], episode_obs_vals[e] = [], [] # restart this worker's lists of rewards r(s) and values V(s)
                        episode_obs[e], episode_actions[e], episode_dones[e] = [], [], []
                        episode_td_errors[e] = []
            assert episode_lengths[e]==envs[e]._elapsed_steps

        # learning rate schedule
        T = a2c.total_timesteps + a2c.total_timesteps_sim
        a2c.lr = a2c.lr0 * max(1 - T * a2c.lr_decay_rate, a2c.lr_min_to_max)

        # reshape to one big batch, for gradient update; checked that this is correct
        actions = [a for actions_e in actions for a in actions_e]
        rewards = [r for rewards_e in rewards for r in rewards_e]
        dones = [d for dones_e in dones for d in dones_e]
        returns_disc = [G for returns_disc_e in returns_disc for G in returns_disc_e]
        obs = np.array(obs)
        obs = np.squeeze(obs) #### added np.squeeze() when using InvertedPendulum
        actions = np.array(actions)
        if len(actions.shape)>2:
            actions = np.squeeze(actions, axis=1) ## NOT ONLY for discrete actions (?), need to squeeze axis=-1 or axis=1(=-2) ?  I think it's the LATTER
        rewards = np.array(rewards)
        dones = np.array(dones)
        returns_disc = np.array(returns_disc)

        feed_dict={a2c.obs_ph: obs, a2c.actions_ph: actions, a2c.returns_disc_ph: returns_disc, a2c.lr_ph: a2c.lr}

        #### for testing A2C in CONTINUOUS domains:
        #lp, ao, ameans, stdevs = sess.run([actions_logprobs, actions_out, mu, tf.exp(log_std)], feed_dict=feed_dict)
        #print(ameans)
        #print(stdevs)
        #print(actions)
        #print('END')

        if isinstance(a2c.env.action_space, Discrete):
        ## if discrete_actions:
            pi_probs1 = sess.run(a2c.actions_probs, feed_dict={a2c.obs_ph: obs})

        # take gradient step
        pi_loss_batch, v_loss_batch, grads_vars_pi_batch, _ = sess.run([a2c.pi_loss, a2c.v_loss, a2c.grads_vars_pi, a2c.train_op], feed_dict=feed_dict)
        ### grads_v_batch = sess.run(grads_v, feed_dict={obs_ph: obs, actions_ph: actions, returns_disc_ph: returns_disc_batch, lr_ph: lr_current}) ## this didn't work...

        if isinstance(a2c.env.action_space, Discrete):
        ## if discrete_actions:
            pi_probs2 = sess.run(a2c.actions_probs, feed_dict={a2c.obs_ph: obs})
            kl_divs_batch = kl(tf.convert_to_tensor(pi_probs1), tf.convert_to_tensor(pi_probs2)) ## not ideal: converting from tensor to numpy (via sess.run) to tensor
            kl_divs_batch = sess.run(kl_divs_batch)
            logger.record_tabular('KL(pi_new||pi_old), real data', np.mean(kl_divs_batch))

        # policy entropy
        policy_entropy = sess.run(a2c.pi_entropy, feed_dict={a2c.obs_ph: obs})

        if use_model:
            kwargs_model['total_timesteps'] = a2c.total_timesteps
            kwargs_model['current batch data'] = [obs, actions, rewards, dones]
            kwargs_model['plan_bool'] = plan_bool
            use_env_model(env_model, a2c, replay_buffer, **kwargs_model)

        ## print('SHARED WEIGHTS:')
        ## print('POLICY WEIGHTS:')
        ## vs = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='policy')] ## if v.name == "policy/dense_1/kernel:0"][0]
        ## for v in vs:  print(sess.run(v))

        print('pi_entropy CHANGE AFTER synthetic data update:')
        print(policy_entropy - sess.run(a2c.pi_entropy, feed_dict={a2c.obs_ph: obs}))

        ## TensorBoard ## these 3 lines by themselves give an error
        #merge = tf.summary.merge_all()
        #summary = sess.run(merge, feed_dict=feed_dict)
        #train_writer.add_summary(summary, global_step=a2c.total_timesteps)

        # log for TensorBoard
        logger.record_tabular('n_updates', i)
        logger.record_tabular('total_timesteps', a2c.total_timesteps)
        logger.record_tabular('total_timesteps (simulation included)', a2c.total_timesteps + a2c.total_timesteps_sim)
        logger.record_tabular('Policy Gradient L2 Norm (sort of)', gradients_norm(grads_vars_pi_batch))
        logger.record_tabular('Policy Gradient Linf Norm', gradients_norm(grads_vars_pi_batch, np.inf))
        # logger.record_tabular('Value Function Gradient Norm', np.linalg.norm(grads_v_batch))
        logger.record_tabular('policy_loss', pi_loss_batch)
        logger.record_tabular('value_loss', v_loss_batch)
        logger.record_tabular('policy_entropy', policy_entropy)
        logger.record_tabular('learning_rate', a2c.lr)
        logger.dump_tabular()
        save_path = saver.save(sess, logdir+'/model.ckpt')
        if use_model: save_path_env_model = saver_env_model.save(sess, logdir+'/env_model.ckpt')


if __name__ == '__main__':

    # parser = argparse.ArgumentParser()
    # parser.add_argument('logdir',type=str,help='log directory')
    # parser.add_argument('lr',type=float,help='learning rate')

    ## env_name = 'Breakout-v0'

    env_name = 'CartPole-v0'
    # env_name = 'MountainCar-v0'
    # env_name = 'MountainCarContinuous-v0'
    # env_name = 'LunarLander-v2'
    # env_name = 'Acrobot-v1'
    # env_name = 'Pendulum-v0'
    # env_name = 'Reacher-v2'
    # env_name = 'Hopper-v2'
    # env_name = 'InvertedPendulum-v2'
    # env_name = 'InvertedDoublePendulum-v2'
    # env_name = 'HalfCheetah-v2'
    # env_name = 'Walker2d-v2'
    # env_name = 'Swimmer-v2'

    ### Alternative:
    # env = gym.make(env_name)
    # n_envs = 3
    # env = DummyVecEnv([lambda: env]*n_envs)

    # env.tau = 0.002 # default 0.02

    # with tf.device('/cpu:0'):
    ## with tf.device('/device:GPU:0'): ## this led to errors
    train(env_name)
    exit()

    # render a saved model
    saver = tf.train.Saver()
    sess = tf.Session()
    render(saver, sess, saved_model_path=='data/trash/model.ckpt', env=gym.make('CartPole-v0'))


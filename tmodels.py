import math
import random
import numpy as np
import tensorflow as tf

import gym
from gym.spaces import Box, Discrete

from baselines.common import tf_util
from baselines import logger

from replay_buffer import ReplayBuffer, PrioritizedReplayBuffer 
from replay_buffer import normalize, obs_stdev, delta_obs_stdev

from think import plan, run_sim

from errors import obs_error

from log_data import log_list_to_file

from bias import bias_data


def use_env_model(env_model, model_free_alg, replay_buffer, **kwargs):

    if len(replay_buffer._storage)==0:
        return # wait until we have data to train with

    plan_bool = kwargs['plan_bool']
    total_timesteps = kwargs['total_timesteps']

    if env_model.updates_per_iter>0:
        env_model.lr = env_model.lr0 * max(1 - total_timesteps*env_model.lr_decay, env_model.lr_mintomax)
        logger.record_tabular('Env Model Learning Rate', env_model.lr)

    model_is_learning = True
    # loss_obs_list = [] # use for early stopping (see below)
    # n_ave = 10 # use for early stopping (see below)
    i = 0
    print('Training Env Model')
    while model_is_learning:
    ## (use this instead?) for _ in range(env_model.updates_per_iter):
        if i+1>env_model.updates_per_iter:
            break
        ###print('weights')
        ###print(env_model.sess.run(env_model.trainable_vars)) ####
        loss_model, loss_model_obs, loss_model_done, loss_model_r = env_model.train(replay_buffer)
        i += 1
        logger.record_tabular('Env Model Loss', loss_model)
        logger.record_tabular('Env Model Loss (obs)', loss_model_obs)
        logger.record_tabular('Env Model Loss (dones)', loss_model_done)
        logger.record_tabular('Env Model Loss (rewards)', loss_model_r)
# TEMPORARY HACK:
        logger.dump_tabular() # comment this out for same x-axis as use_model=False on Tensorboard
#        # early stopping of model training when loss plateaus:
#        loss_obs_list.append(loss_model_obs)
#        if i>=2*(n_ave+1):
#            loss_obs_ave1 = sum(loss_obs_list[-n_ave-1:-1])/n_ave
#            loss_obs_ave2 = sum(loss_obs_list[-2*n_ave-2:-n_ave-2])/n_ave
#            if loss_obs_ave1/loss_obs_ave2>0.9:
#                break # model learning has slowed down or stopped

    if env_model.updates_per_iter==0: # if we're not training the model, assume we can use it to plan
        env_model.plan = True
    elif loss_model_obs<env_model.loss_model_obs_plan and loss_model<env_model.loss_model_plan:
        env_model.plan = True
    elif not env_model.plan: # if model not good enough, don't plan yet
        return 

    if plan_bool:
        if len(replay_buffer._storage)>0: # wait till we have stored episodes/data
            print('PLAN():')
            plan(env_model, model_free_alg, replay_buffer, **kwargs)

    # env_model.test(replay_buffer) # compare s'(s,a) between env and env_model

    return


def fnn(s,a):

    x = tf.concat([s,a],1)

    n_hidden = 128
    n_layer = 2
    n_obs = s.get_shape().as_list()[1]
    n_out = n_obs + 3 # two extra done nodes (logits for True & False), and one reward node
    l2 = 1e-5

    activation_ftn = tf.nn.relu
    w_initializer = None # tf.random_normal_initializer(stddev=0.001)
    reg = tf.contrib.layers.l2_regularizer(scale=l2, scope='env_model')

    for _ in range(n_layer):
        x = tf.layers.dense(inputs=x, units=n_hidden, activation=activation_ftn, kernel_initializer=w_initializer, kernel_regularizer=reg)

    # default activation = linear
    output = tf.layers.dense(
        inputs=x,
        units=n_out,
        kernel_regularizer=reg,
        kernel_initializer=w_initializer,
        name='fnn_output')

    delta_obs_predicted = tf.slice(output, [0,0], [-1,n_obs])
    done_output = tf.slice(output, [0,n_obs], [-1,2])
    reward_predicted = tf.slice(output, [0,n_obs+2], [-1,1])

    return delta_obs_predicted, done_output, reward_predicted
    # for obs-only models, w/o rewards, just omit the reward output from the loss

# separate networks for each obs component
def fnn2(s,a):

    x = tf.concat([s,a],1)

    n_hidden = 128
    n_layer = 3
    n_hidden_done = 64
    n_layer_done = 4
    n_hidden_r = 32
    n_layer_r = 4 
    n_obs = s.get_shape().as_list()[1]
    n_out = n_obs
    l2 = 1e-5
    l2_done = 0e-5

    activation_ftn = tf.nn.relu
    w_initializer = None # tf.random_normal_initializer(stddev=0.001)
    reg = tf.contrib.layers.l2_regularizer(scale=l2, scope='env_model')

    for i in range(n_obs): # separate FNN for each obs dim
        x2 = x
        for _ in range(n_layer):
            x2 = tf.layers.dense(x2, units=n_hidden, activation=activation_ftn, kernel_initializer=w_initializer, kernel_regularizer=reg)
        output = tf.layers.dense(x2, units=1, activation=None, kernel_initializer=w_initializer, kernel_regularizer=reg)
        if i==0:
            d_obs_predicted = output
        if i>0 and i<n_obs:
            d_obs_predicted = tf.concat([d_obs_predicted, output], 1)

    done_output = x
    for _ in range(n_layer_done):
        done_output = tf.layers.dense(done_output, units=n_hidden_done, activation=activation_ftn, kernel_initializer=w_initializer, kernel_regularizer=reg)
    # 2 units for softmax logits
    done_output = tf.layers.dense(done_output, units=2, activation=None, kernel_initializer=w_initializer, kernel_regularizer=reg)

    r = x
    for _ in range(n_layer_r):
        r = tf.layers.dense(r, units=n_hidden_r, activation=activation_ftn, kernel_initializer=w_initializer, kernel_regularizer=reg)
    reward_predicted = tf.layers.dense(r, units=1, activation=None, kernel_initializer=w_initializer, kernel_regularizer=reg)

    return d_obs_predicted, done_output, reward_predicted


class PredictiveModel(object):

    def __init__(self,
                 env,
                 network=fnn2,
                 updates_per_iter=0,
                 lr=1e-3, lr_decay=5e-8, lr_mintomax=1e-3,
                 batch_size=500, # note that the gradient size is independent of this (model loss = mean, not sum)
                 r_loss_coef=0.1,
                 done_loss_coef=0.1,
                 # t_horizon_smooth=30, # scale for smoothing history of successful prediction timesteps
                 n_rollouts_per_iter=10, # (only used with uniform, state-independent, global horizon) number of real trajectories with which to evaluate model's errors / bias, per iteration
                 n_iters_errors=1, # (ideal to keep =1) number of (A2C) iterations over which we collect and average model errors / bias
                 loss_model_obs_plan=0.1, ## 1e20 ## 0.01, # don't plan until model is this good; infinity = plan from the start
                 loss_model_plan=0.25, ## 1e20 ## 0.05,
                 sess=None):

        self.plan = False # set to True when model loss is small enough
        self.loss_model_obs_plan = loss_model_obs_plan
        self.loss_model_plan = loss_model_plan

        self.env = env

        self.obs_dims = env.observation_space.shape[0] ## this returns the state space dimensionality for Cartpole-v0; does it work generally? (does this assume Cartpole?: github.com/jachiam/rl-intro/blob/master/pg_cartpole.py)
        if isinstance(env.action_space, Discrete):
            self.action_dims = 1 # a single discrete dimension indexes all possible (discrete) actions
        else:
            self.action_dims = env.action_space.shape[0]

        # model input placeholders
        self.obs = tf.placeholder(tf.float32, [None, self.obs_dims])
        self.actions = tf.placeholder(tf.float32, [None, self.action_dims]) ## here we assume action dtype can be cast to float32

        # define the model network
        with tf.variable_scope('env_model'):
            self.delta_obs_predicted, self.done_output, self.r_predicted = network(self.obs, self.actions)
            self.obs_predicted = self.obs + self.delta_obs_predicted
        self.trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='env_model')

        self.num_params=0
        for i in range(len(self.trainable_vars)):
            num = np.prod(np.asarray(self.trainable_vars[i].shape))
            self.num_params += int(num)

        self.sess = sess ## (bad?:) tf_util.get_session()

        self.done_probs = tf.nn.softmax(logits=self.done_output)
        self.done_probs = tf.slice(self.done_probs, begin=[0,1], size=[-1,1]) # keep just the 2nd element (begin[1]=1), since the 1st element is p(done=False)=1-p(done=True); this is because the one-hot encoding makes (0,1) correspond to index=1=True
        self.labels_done = tf.placeholder(tf.int32, [None])
        self.labels_done_onehot = tf.one_hot(indices=self.labels_done, depth=2)

        self.obs_next = tf.placeholder(tf.float32, [None, self.obs_dims])
        ### self.labels = tf.placeholder(tf.float32, [None, 1 + self.obs_dims]) # successor state s', concatenated with predicted dones
        self.delta_obs = self.obs_next - self.obs
        self.obs_stdev_ph = tf.placeholder(tf.float32, [self.obs_dims]) # feed current stdev of obs to this placeholder
        # self.delta_obs_stdev_ph = tf.placeholder(tf.float32, [self.obs_dims]) # feed current stdev of delta_obs to this placeholder
        self.delta_obs_error = (self.delta_obs - self.delta_obs_predicted) / self.obs_stdev_ph # CHECKED this normalization

        self.loss_obs = tf.reduce_mean(self.delta_obs_error**2) # mean of component-size normalized error; does not increase with obs_dim
        ## self.loss_obs = tf.losses.mean_squared_error(self.delta_obs, self.delta_obs_predicted)
        ## obs_fractional_errors = tf.div(self.labels_obs - self.obs_predicted, self.labels_obs + self.obs_predicted) 
        ## self.loss_obs = tf.reduce_mean(tf.square(obs_fractional_errors))

        self.done_weights = tf.placeholder(tf.float32, [None])
        self.loss_done = done_loss_coef * tf.losses.softmax_cross_entropy(onehot_labels=self.labels_done_onehot, logits=self.done_output, weights=self.done_weights)
        ## self.loss_done = tf.losses.mean_squared_error(self.labels_done, self.done_output)
        ## self.done_probs = tf.sigmoid(self.done_output)
        ## self.loss_done = tf.reduce_mean(self.labels_done * (-tf.log(self.done_probs)) + (1.-self.labels_done) * (-tf.log(1. - self.done_probs)))
        ### if label=True=1, loss_done = -log(done_prediction) 
        ### if label=False=0, loss_done = -log(1-done_prediction) 
        #### self.loss_done = self.labels_done * (-self.done_output) + (1.-self.labels_done) * (-tf.log(1 - tf.exp(self.done_output)))
        #### self.loss_done = tf.reduce_mean(self.loss_done)

        self.r = tf.placeholder(tf.float32, [None, 1])
        ## self.r_mean_ph = tf.placeholder(tf.float32, [1]) # feed current mean rewards to this placeholder
        self.r_error = self.r - self.r_predicted ## / self.r_mean_ph
        self.loss_r = r_loss_coef * tf.reduce_mean(self.r_error**2)

        self.loss = self.loss_obs + self.loss_done + self.loss_r
        self.loss += tf.losses.get_regularization_loss(scope='env_model')

        self.lr0 = lr # initial learning rate
        self.lr = lr # can be changed by learning rate schedule 
        self.lr_decay = lr_decay
        self.lr_mintomax = lr_mintomax
        self.lr_ph = tf.placeholder(tf.float32, []) 

        # self.optimizer = tf.train.GradientDescentOptimizer(self.lr_ph)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr_ph)
        # self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr_ph, decay=0.99, epsilon=1e-5)
        # self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.lr_ph, momentum=0.8)
        self.train_step = self.optimizer.minimize(self.loss, var_list=self.trainable_vars)
        self.gradients = self.optimizer.compute_gradients(self.loss, var_list=self.trainable_vars)

        self.batch_size = batch_size

        # initialize the model
        self.sess.run(tf.global_variables_initializer())
        ## (all vars are trainable, except optimizer params?)
        ## self.sess.run(tf.variables_initializer(self.trainable_vars))
        ## print(tf.GraphKeys.TRAINABLE_VARIABLES) # occasionally useful

        ##### this is ~ loss_ftn above, in STOCHASTIC case
        ##### WRITE THIS ANALOGOUSLY TO self.loss ABOVE
        # will be used by posterior over tmodel:
        # def loglikelihood(self, tmodel_params, obs, actions):
        #    self.tmodel(
        #    return tf.log(tmodel_output)     

        self.updates_per_iter = updates_per_iter

        # store real trajectories for testing rollout predictions
        self.n_rollouts_per_iter = n_rollouts_per_iter
        self.n_rollouts_average = n_rollouts_per_iter * n_iters_errors # number of rollouts used to compute mean model errors / bias
        self.trajectory_obs, self.trajectory_actions, self.trajectory_rewards, self.trajectory_dones = [], [], [], []
        self.partial_trajectory_obs, self.partial_trajectory_actions, self.partial_trajectory_rewards, self.partial_trajectory_dones = [], [], [], []
        self.list_rollout_errors = [] # a list of lists of errors e(t) for each of _ rollouts
        self.list_rollout_ratios_grad_length = []
        self.timestep_mean_errors = [1.]*1000 # initialize model errors to be very bad at each of t=1...1000 timesteps
        self.timestep_mean_errors_samplesizes = [0]*1000 # initialize the sample sizes with which the model's error at each timestep was estimated

        ### THESE ARE NOT CURRENTLY IN USE:
        self.planning_timesteps = 1 # initially, don't trust the model
        self.rollout_successful_timesteps = 0
        self.rollout_successful_timesteps_history = []
        self.obs_sim_init = env.reset()
        # self.t_horizon_smooth = t_horizon_smooth

        # value_errors[t] will be a list or array of errors in the value ftn gradient, due to differences between real & corresponding simulated trajectories


    def train(self, replay_buffer):

        sess = self.sess
        batch_size = min(self.batch_size, len(replay_buffer._storage))

        if isinstance(replay_buffer, PrioritizedReplayBuffer) and replay_buffer._alpha!=0.0:
            print('If I"m going to train the model and run PER-initialized rollouts at the same time, I need a way to input alpha_PER when sampling in either of those two cases. I could train the model with the SAME alpha_PER, and use beta below to re-weight, but PER is for training the policy not model...')
            exit()

        if isinstance(replay_buffer, PrioritizedReplayBuffer):
            data = replay_buffer.sample(batch_size, beta=0)
        else:
            data = replay_buffer.sample(batch_size)
        obs, actions, rewards, obs_next, dones = data[0], data[1], data[2], data[3], data[4]
        obs_std = obs_stdev(replay_buffer)

        # if any obs components just stay constant, they're irrelevant to the dynamics you're trying to learn -> ignore them (e.g. Reacher-v2) 
        #for ; WHAT IF i in range(len(obs_std)):
        #    print('IGNORING SOME OBSERVATION COMPONENTS IN MODEL TRAINING')
        #    if obs_std[i]<1e-15:
        #        obs_std[i]=1e50
        # obs_std = np.maximum(obs_std, 1e-3) # if some obs components have tiny stdev, replace that stdev with constant threshold, to prevent blowup of normalized data

        ## if using a2c baselines implementation
        ## obs = np.squeeze(obs, axis=1)
        ## obs_next = np.squeeze(obs_next, axis=1)

        rewards = np.expand_dims(rewards, axis=1)
        # rewards = np.squeeze(rewards, axis=1)
        # actions = np.expand_dims(actions, axis=1) #### added for Acrobot... as of June 12, may need this also with PrioritizedReplayBuffer()...
        done_weights = np.ones(shape=[batch_size]) # default, for all done labels = False; for large batch_size, we should have n_dones>0
        n_dones = 0
        for i in range(batch_size):
            if dones[i]:
                n_dones += 1
        if n_dones>0: # fix class imbalance by assigning equal weight to True and False done labels
            for i in range(batch_size):
                if dones[i]:
                    done_weights[i] = (batch_size/2.)/n_dones
                else:
                    done_weights[i] = (batch_size/2.)/(batch_size-n_dones)
        dones = dones.astype(int)
        ## dones = np.expand_dims(dones, axis=1)
        ## done_weights = np.expand_dims(done_weights, axis=1)

        obs, actions, rewards, obs_next, dones = bias_data(obs, actions, rewards, obs_next, dones)

        feed_dict={self.obs: obs, self.actions: actions, self.obs_next: obs_next, self.labels_done: dones, self.r: rewards, self.lr_ph: self.lr, self.obs_stdev_ph: obs_std, self.done_weights: done_weights}

        d_obs = obs_next - obs
        d_obs_pred = sess.run(self.delta_obs_predicted, feed_dict=feed_dict)
        #print(replay_buffer.__len__())
        #print('d_obs:')
        #print(d_obs)
        #print('d_obs_pred:')
        #print(d_obs_pred)
        #print('self.delta_obs_error:')
        #print(sess.run(self.delta_obs_error, feed_dict=feed_dict))
        ## print('done output + probs:')
        ## print('Env Model Gradients:')
        ### print(sess.run(self.lr_ph), feed_dict={self.lr_ph: [0.43]})

        loss, loss_obs, loss_done, loss_r,  _ = sess.run([self.loss, self.loss_obs, self.loss_done, self.loss_r, self.train_step], feed_dict=feed_dict)

        print('real dones & model done_probs:')
        print(dones[0:10])
        print(sess.run(self.done_probs[0:10], feed_dict=feed_dict))

        # components of obs
        #losses = sess.run(tf.reduce_mean(self.delta_obs_error, axis=0), feed_dict={self.obs: obs, self.actions: actions, self.obs_next: obs_next, self.obs_stdev_ph: obs_std})
        #logger.record_tabular('loss_obs_1', losses[0])
        #logger.record_tabular('loss_obs_2', losses[1])
        #logger.record_tabular('loss_obs_3', losses[2])
        #logger.record_tabular('loss_obs_4', losses[3])

        return loss, loss_obs, loss_done, loss_r

    def test(self, replay_buffer):

        env = self.env
        sess = self.sess

        obs = []

        # generate array of input states
        obs_std = obs_stdev(replay_buffer)
        for i in range(-100,100):
            o = obs_std.copy()
            o[0] = obs_std[0]*(3*i/100) # let input obs vary over 3sigma in one component
            obs.append(o)
        obs = np.array(obs)

        action = 0
        actions = [action]*len(obs)

        # compute ground-truth successor states
        obs_actual = []
        for i in range(len(obs)):
            env.reset()
            self.state = obs[i]
            o, _, _, _ = env.step(action)
            obs_actual.append(o)

        # compute predicted successor states
        obs_predicted = sess.run(self.obs_predicted, feed_dict={self.obs:obs.reshape(1,-1), self.actions:actions})

        #print(obs_actual)
        #print(obs_predicted)

        return

    # see older versions of this file for update_errors() using trajectories stored in self.trajectory_obs, etc.
    def update_errors(self, model_free_alg, replay_buffer, **kwargs):

        print('Modify this function to call rollout_errors().')
        exit()

        gamma = model_free_alg.gamma
        obs_ph = model_free_alg.obs_ph
        actions_ph = model_free_alg.actions_ph
        values = model_free_alg.values
        grads_vars = model_free_alg.grads_vars
        returns_disc_ph = model_free_alg.returns_disc_ph
        from A2C import returns_nstep

        sess = self.sess ## or could be model_free_alg.sess, or something better

        logdir = kwargs['logdir']

        if len(replay_buffer._storage)==0: return
        n = 1
        while n<=self.n_rollouts_per_iter: 

            # sample a transition from the buffer, and use that episode
            idx = random.randint(0,len(replay_buffer._storage)-1)

            obs, actions, rewards, dones = [], [], [], []
            while not replay_buffer._storage[idx][4]==1:
                if idx==len(replay_buffer._storage)-1: break
                idx += 1 # find the end of the episode
            if not replay_buffer._storage[idx][4]==1: continue # this episode is incomplete
            obs_last = replay_buffer._storage[idx][3] # checked that this is consistent with obs as defined below
            d = replay_buffer._storage[idx-1][4] # done (=False) for the 2nd-to-last transition
            while not d and idx>=0: # checked that this correctly adds observations from episode to obs, moving to preceding episode in buffer on next iteration
                data = replay_buffer._storage[idx]
                if not data[4] and idx<len(replay_buffer._storage)-1: # if non-terminal, assert that next transition in buffer is successor transition in same episode
                    assert data[3][0]==replay_buffer._storage[idx+1][0][0] 
                obs.insert(0, data[0])
                actions.insert(0, data[1])
                rewards.insert(0, data[2])
                dones.insert(0, data[4])
                idx -= 1 
                d = replay_buffer._storage[idx][4]
            n += 1

            obs_sim_init = obs[0]
            print('run_sim() now doesn"t use obs_sim_init input ... if I want to input this as arg, need to change run_sim(); otherwise change the code here in update_errors().')
            exit()
            obs_sim, _, rewards_sim, done_probs_cum, obs_sim_last, _ = run_sim(self, obs_sim_init, restrict_actions=True, actions_required=actions, stop_if_done=False, replay_buffer=replay_buffer, **kwargs) # don't need the last element 'real_trajectory' in run_sim(), since we already have the corresponding real trajectory and are comparing to it
            # stop_if_done=False allows for predicting past when model thinks episode ends

#hi

            # restrict_actions ensures len(obs_sim)<=len(obs); note that if real episode ends, run_sim() will end at same point
            assert len(obs)==len(obs_sim) # could be violated if done_prob_min allows model to terminate rollout with done=True
            # assert len(obs)>=len(obs_sim) # not equal only if done_prob_min allows model to terminate rollout with done=True

            # bootstrapped returns estimates, for simulated trajectory
            val_nstep = np.squeeze(sess.run(values, feed_dict={obs_ph: obs_last.reshape(1,-1)})) # if target network: values_target, obs_value_target_ph
            val_nstep_sim = np.squeeze(sess.run(values, feed_dict={obs_ph: obs_sim_last.reshape(1,-1)}))
            assert dones[-1]
            returns_disc = returns_nstep(rewards, val_nstep, gamma, terminal=dones[-1])
            returns_disc_sim = returns_nstep(rewards_sim, val_nstep_sim, gamma, done_probs_cum=done_probs_cum)

        # save (component of) trajectories to files
        # obs1 = obs[:,0]
        # log_list_to_file(obs1, 'obs.txt')
        # obs2 = np.array(obs_sim)[:,0]

            if isinstance(replay_buffer, PrioritizedReplayBuffer): ## not certain this is the right condition for this shape adjustment...
                actions = np.expand_dims(np.array(actions), axis=1)
            returns_disc = np.expand_dims(returns_disc, axis=1)
            returns_disc_sim = np.expand_dims(returns_disc_sim, axis=1)

            bit1, bit2 = False, False

            rollout_errors, rollout_ratios_grad_length = [], []

            # compare actual and predicted rollouts
            for i in range(len(obs_sim)):

                feed_dict_real = {obs_ph: obs[i].reshape(1, -1), actions_ph: actions[i], returns_disc_ph: returns_disc[i]}
                feed_dict_sim = {obs_ph: obs_sim[i].reshape(1, -1), actions_ph: actions[i], returns_disc_ph: returns_disc_sim[i]}
                g_real = sess.run(grads_vars, feed_dict=feed_dict_real)
                g_sim = sess.run(grads_vars, feed_dict=feed_dict_sim)
                g_real = list(zip(*g_real))[0]
                g_sim = list(zip(*g_sim))[0]
                for j in range(len(g_real)):
                    if j==0:
                        g1 = g_real[j].flatten()
                        g2 = g_sim[j].flatten()
                    else:
                        g1 = np.concatenate((g1, g_real[j].flatten()))
                        g2 = np.concatenate((g2, g_sim[j].flatten()))
                # float64 needed, because we're comparing direction of nearly-identical and very high-dimensional vectors
                ## some of these float64's might be redundant
                g12 = np.float64(np.dot(np.float64(g1),np.float64(g2)))
                g11 = np.float64(np.dot(np.float64(g1),np.float64(g1)))
                g22 = np.float64(np.dot(np.float64(g2),np.float64(g2)))
                error = 1.0 - math.sqrt(np.float64(g12*g12/g11/g22))
                ratio_grad_length = math.sqrt(np.float64(g22/g11))
                assert error>=0. # w/o float64, this failed
#hi
                rollout_errors.append(error) 
                rollout_ratios_grad_length.append(ratio_grad_length)

                error_obs = obs_error(replay_buffer, obs_sim[i], obs[i])
                if error_obs>0.5 and bit1==False:
                    logger.record_tabular('Model obs prediction time horizon (error<0.5)', i)
                    bit1=True
                if error>0.055 and bit2==False:
                    logger.record_tabular('Model obs prediction time horizon (error<0.055)', i)
                    bit2=True
#                if error>(DEFINE:)rollout_error_max:
##                    print('RESTART')
#                    break
#                else:

# define rollout_successful_timesteps

            self.list_rollout_errors.append(rollout_errors)
            self.list_rollout_ratios_grad_length.append(rollout_ratios_grad_length)
            log_list_to_file(rollout_errors, logdir + '/rollout_errors.txt')
            # log_list_to_file(rollout_ratios_grad_length, logdir + '/rollout_gradlength.txt')
#            logger.record_tabular('Model prediction time horizon', self.rollout_successful_timesteps)
#            self.rollout_successful_timesteps_history.append(self.rollout_successful_timesteps)
#            t_smooth = min(self.t_horizon_smooth, len(self.rollout_successful_timesteps_history)) # smoothing scale 
#            self.planning_timesteps = sum(self.rollout_successful_timesteps_history[-t_smooth-1:-1])/t_smooth # update the planning time horizon
#            logger.record_tabular('Model prediction time horizon (smoothed)', self.planning_timesteps)
#            self.planning_timesteps = int(self.planning_timesteps)

        # update the mean errors, given new rollout errors
        # first, drop the 1st n rollouts since we just added n new rollouts
        nn = len(self.list_rollout_errors)
        assert nn==len(self.list_rollout_ratios_grad_length)
        if nn>self.n_rollouts_average: # truncate; if n_iters_errors=1, this keeps only the entries appended within this function call 
            self.list_rollout_errors = self.list_rollout_errors[-self.n_rollouts_average:]
            self.list_rollout_ratios_grad_length = self.list_rollout_ratios_grad_length[-self.n_rollouts_average:]
            assert len(self.list_rollout_errors) == self.n_rollouts_average
        timestep_sum_errors = [0]*1000
        timestep_sum_errors_mag = [0]*1000
        self.timestep_mean_errors_samplesizes = [0]*1000
        for n in range(len(self.list_rollout_errors)):
            for t in range(len(self.list_rollout_errors[n])):
                if t<len(self.list_rollout_errors[n]):
                    timestep_sum_errors[t] += self.list_rollout_errors[n][t]
                    mag_error_coef = 0.100 # error tolerance for magnitude of simulation policy gradient
                    mag_error = mag_error_coef * (self.list_rollout_ratios_grad_length[n][t] - 1.0)
                    if mag_error<0.: mag_error *= 0. # no constraint on simulation gradients of shorter length than real gradients
                    timestep_sum_errors_mag[t] += mag_error
                    self.timestep_mean_errors_samplesizes[t] += 1
        for t in range(len(self.timestep_mean_errors)): # normalize mean errors by sample sizes
            if self.timestep_mean_errors_samplesizes[t]==0: # for large t, we won't have any samples
                break
            self.timestep_mean_errors[t] = timestep_sum_errors[t]/self.timestep_mean_errors_samplesizes[t]

        log_list_to_file(self.timestep_mean_errors[0:t], logdir + '/rollout_mean_errors.txt')
        logger.record_tabular('env model mean gradient direction error (5 timesteps)', self.timestep_mean_errors[5])
        logger.record_tabular('env model mean gradient direction error (10 timesteps)', self.timestep_mean_errors[10])
        logger.record_tabular('env model mean gradient direction error (20 timesteps)', self.timestep_mean_errors[20])

        return

    # not using this function; instead, just using the trajectories from A2C-generated real data, as stored in the replay buffer
    # update the model's list of current real trajectories, to compare predictions to
    def get_trajectories(self, model_free_alg, **kwargs):

        obs_init = np.expand_dims(self.env.reset(), axis=0)

        obs, actions, rewards, dones, obs_last = model_free_alg.run(obs_init, nsteps=int(1e50), n_envs=1, n_episodes=self.n_rollouts_per_iter, envs=[self.env])
        ## currently, this uses the model's copy of env instead of a2c's envs (otherwise _elapsed_steps for those envs get screwed up)...

        ## for A2C, we have to take the 0 component = 1st of 1 worker
        obs = obs[0]
        actions = actions[0]
        rewards = rewards[0]
        dones = dones[0]

        # flush old trajectories
        self.trajectory_obs, self.trajectory_actions, self.trajectory_rewards, self.trajectory_dones = [], [], [], []

        i0 = 0
        for i in range(len(obs)):
            if dones[i]:
                self.trajectory_obs.append(obs[i0:i+1])
                self.trajectory_actions.append(actions[i0:i+1])
                self.trajectory_rewards.append(rewards[i0:i+1])
                self.trajectory_dones.append(dones[i0:i+1])
                i0 = i+1

## ugh, I'd really need a list of these, one for each episode among those above
##**##        self.trajectory_obs_last = obs_last

        return

    ## be sure the [1:]'s and other indexing/slicing, are correct
    def update_horizon_a2cbaseline(self, obs, actions, rewards, dones, runner, error_max=1):

        nsteps = runner.nsteps
        assert len(obs)==nsteps

        actions = np.expand_dims(actions, axis=1)
        
        new_rollout = True # assume we'll start a new rollout

        # IF surviving rollout, CONTINUE; ELSE start new rollout (could generalize to keeping track of a set of rollouts, start one at each call to method
        if self.rollout_successful_timesteps>0:
            #print('CONTINUING ROLLOUT')
            #print('obs:')
            #print(obs)
            runner.obs_sim = self.obs_sim_init # make sure the rollout continues from the predicted state where it left off
            obs_sim, _, _, masks_sim, _, _ = runner.run_sim(self, nsteps, restrict_actions=True, actions_required=actions, init_obs=runner.obs_sim)
            ## <= last arg REDUNDANT
            error = prediction_error(obs_sim[:nsteps-1], obs[1:], masks_sim, dones)
            ## (I could include this but it's only a 20% difference:) obs_sim[nsteps-1] is prediction for obs[0] on NEXT iteration 
            ## if reward function part of tmodel, could have error args rewards_sim[:nsteps-1], rewards[:nsteps-1]; (when I had this before, I omitted rewards[0], which we could include by keeping track of predicted reward from last iteration, like we do with obs_sim_init)
            #print('obs_sim:')
            #print(obs_sim)
            #print('ERROR:')
            #print(error)
            if error<error_max: # rollout predictions haven't failed yet
                new_rollout = False  # continue with this rollout, don't start a new one
                self.obs_sim_init = np.expand_dims(obs_sim[nsteps-1], axis=0) # the last predicted state; start here when continuing rollout
                self.rollout_successful_timesteps += nsteps
                self.planning_timesteps = max(self.planning_timesteps, self.rollout_successful_timesteps) # update the planning horizon
            else: # start a new rollout
                self.rollout_successful_timesteps = 0

        if new_rollout==True:
            #print('NEW ROLLOUT')
            #print('obs:')
            #print(obs)
            runner.obs_sim = np.expand_dims(obs[0], axis=0)
            obs_sim, _, _, masks_sim, _, _ = runner.run_sim(self, nsteps, restrict_actions=True, actions_required=actions, init_obs=runner.obs_sim)
            ## <= last arg REDUNDANT
            error = prediction_error(obs_sim[1:], obs[1:], masks_sim, dones)
            # the first obs, obs[0]=obs_sim[0], is omitted; also, the last reward and last reward_sim follow the last obs, so omit them
            #print('obs_sim:')
            #print(obs_sim)
            #print('ERROR:')
            #print(error)
            if error<error_max:
                self.obs_sim_init = np.expand_dims(obs_sim[nsteps-1], axis=0)
                self.rollout_successful_timesteps = nsteps
                self.planning_timesteps = max(self.planning_timesteps, self.rollout_successful_timesteps) # update the planning horizon


##### PUT THESE METHODS IN DIFFERENT FILES? #####

# this weights all nsteps equally
def prediction_error_0(obs_sim, obs, dones_sim, dones, rewards_sim=None, rewards=None, y=0.5):

    print('obs')
    print(obs)
    print('obs_sim')
    print(obs_sim)

    error_obs = np.mean(np.square(obs_sim - obs)) / np.mean(np.square(obs_sim + obs))
    ## error_dones = np.mean(np.square(dones_sim - dones)) / np.mean(np.square(dones_sim + dones))
    # error_rewards = np.mean(np.square(rewards_sim - rewards)) / np.mean(np.square(rewards_sim + rewards))

    return error_obs
    ## return y*error_obs + (1.0-y)*error_dones


def a2c_transition_data(obs, actions, masks, rewards):
    
    actions = np.expand_dims(actions,axis=1)
    ## rewards = np.expand_dims(rewards,axis=1)
    obs2 = obs

    # modify (obs,actions,obs2) to be lists of s,a,s' values 
    # removing data from final (done) or initial episode state as needed
    ## checked that this is correct
    i=1 # no need to do anything if masks[0]=True
    while i<len(masks):
        if masks[i]==True:
            masks = np.delete(masks, i)
            masks[i-1] = True # marks terminal state
            obs = np.delete(obs, i-1, 0) # this and next two lines omit terminal states s from (s,a,s') training pairs
            actions = np.delete(actions, i-1, 0)
            obs2 = np.delete(obs2, i, 0)
            ## rewards = np.delete(rewards, i-1, 0)
        i+=1
    ### can reduce time by minimizing len() evaluations
    masks = np.delete(masks, 0, 0)
    obs = np.delete(obs, len(obs)-1, 0)
    ## rewards = np.delete(rewards, len(rewards)-1, 0)
    actions = np.delete(actions, len(actions)-1, 0)
    obs2 = np.delete(obs2, 0, 0)

    masks = np.expand_dims(masks,axis=1)

    return obs, actions, obs2, masks, rewards


import time
import functools
import sys
import tensorflow as tf
import numpy as np

import gym

from gym.spaces import Box, Discrete

from baselines.common import tf_util
from baselines import logger

## sys.path.append("/usr/local/anaconda3/lib/python3.6/site-packages")
## import gym
## sys.path.append("/home/e5nelson/baselines")

from baselines import a2c

from replay_buffer import ReplayBuffer, PrioritizedReplayBuffer 
from replay_buffer import normalize, obs_stdev, delta_obs_stdev

from errors import obs_error, rollout_errors, grad_error

from log_data import log_list_to_file, float_to_str


## this should probably be a method within env_model class
# currently, run_sim() just simulates 1 episode = not analogous to run() in A2C.py
def run_sim(env_model, obs_init=None, nsteps=None, compare_ground_truth=False, restrict_actions=False, actions_required=None, real_dones=False, done_prob_cum_min=0.99, stop_if_done=True, replay_buffer=None, obs_stdevs_max=6.0, perfect_model=False, **kwargs):

    env = env_model.env

    model_free_alg = kwargs['modelfree_alg']
    obs_ph = model_free_alg.obs_ph
    actions_out = model_free_alg.actions_out

    obs_batch, actions_batch, rewards_batch, done_probs, done_probs_cum = [], [], [], [], []
    # dones_batch = []
    obs_init = env.reset() ## this overrides any obs_init input ....
    obs = obs_init

    if compare_ground_truth:
        obs_real, rewards_real, dones_real = [], [], []

    if restrict_actions: # if we're comparing simulation to real trajectory of fixed length
        print('Restricting actions in run_sim() to compare to real trajectory.')
        nsteps = len(actions_required)
        print('NSTEPS (run_sim) = %d' % nsteps) ####
    elif compare_ground_truth:
        print('Start rollout in run_sim() and compare to rollout in the real env.')
        assert nsteps is None
        nsteps=1e6 # ~infinity; let rollouts continue until ground truth rollout terminates (or rollout diverges)
    elif nsteps is None: # default = adaptive horizon
        print('Start rollout in run_sim() with adaptive time horizon.')
        nsteps = max(1,env_model.planning_timesteps) # adaptive number of simulated steps
        print('NSTEPS (run_sim) = %d' % nsteps) ####

    sess = env_model.sess ## tf.get_default_session()

    if replay_buffer is not None:
        obs_std = obs_stdev(replay_buffer)
    else:
        obs_std = None

    i = 0 
    while i<nsteps:

        if compare_ground_truth and i==0: ## ideally, obs_real should be updates like obs_batch...
            obs_real.append(obs)

        obs_batch.append(obs.copy()) # detach obs_batch from obs with copy()

        if restrict_actions==False:
            action = sess.run(actions_out, feed_dict={obs_ph: obs.reshape(1,-1)})
            actions_batch.append(action.copy())
            action = np.expand_dims(action, axis=0)
        if restrict_actions==True: # override the policy sample above
            action = np.expand_dims(actions_required[i], axis=0)

        if perfect_model: # use the actual environment, rather than a learned model
            assert not compare_ground_truth
            if i==0:
                env.reset()
                assert env.unwrapped.spec.id=='CartPole-v0'
                env.env.state = obs_init # b/c of the env class wrapper, need 'env.env.state' NOT 'env.state'
                ## elif env.unwrapped.spec.id=='Acrobot-v1':
                # env.env.state = env.env._get_state(obs_init)
                print('starting sim with perfect model')
                print(obs_init)
                print(env.env.state)
                ##else:
                ##    print('Need to figure out how to initialize env.state from obs_init, for this env.')
                ##    exit()
            obs, reward, done, _ = env.step(np.squeeze(action))
            print(obs)
            print(env.env.state)
            rewards_batch.append(reward)
            # dones_batch.append(done)
            done_probs.append(int(done))
            done_probs_cum.append(int(done))
            obs = np.squeeze(obs)
            i += 1
            if done: break
            continue # go to the next iteration, skipping lines below
            print('Ooops, should not get here')
            exit()

        # simulate
        obs, reward, done_prob = sess.run([env_model.obs_predicted, env_model.r_predicted, env_model.done_probs], feed_dict={env_model.obs: obs.reshape(1,-1), env_model.actions: action.reshape(1,-1)})

        obs = np.squeeze(obs)

        if real_dones: # override the model's done prediction
            ### env._elapsed_steps = i ## might be incorrect if run_sim() resumes an existing rollout
            ### env.state = obs_batch[-1]
            ### obs_real, _, done, _ = env.step(np.squeeze(action)) ## this was returning obs_real values very different from obs or its rollout predecessor obs_batch[-1]
            if not env.unwrapped.spec.id=='CartPole-v0': 
                print('Helpppppp')
                exit()
            done_prob = obs[0] < -env.x_threshold \
                or obs[2] < -env.theta_threshold_radians \
                or obs[2] > env.theta_threshold_radians
            done_prob = int(done_prob)

        done_prob = np.squeeze(done_prob)
        done_probs.append(done_prob)
        if done_probs_cum==[]:
            d0 = 0.
        else:
            d0 = done_probs_cum[-1]
        done_prob_cum = 1 - (1-d0)*(1-done_prob)
        done_probs_cum.append(done_prob_cum)
        d = done_prob_cum>done_prob_cum_min
        # dones_batch.append(done)
        if env.unwrapped.spec.id=='CartPole-v0':
            print('obs, done_prob, done_prob_cum, ground truth done (CartPole):')
            print(obs)
            print(done_prob)
            print(done_prob_cum)
            dd = obs[0] < -env.x_threshold \
                or obs[0] > env.x_threshold \
                or obs[2] < -env.theta_threshold_radians \
                or obs[2] > env.theta_threshold_radians
            print(dd)

        # for now, assume prior knowledge of rewards; just try to learn and plan with (obs,done) predictions
        #if env.udone_probs_cumnwrapped.spec.id=='CartPole-v0' or env.unwrapped.spec.id=='InvertedPendulum-v2':
        #    reward = 1
        #elif env.unwrapped.spec.id=='Acrobot-v1':
        #    print('need to define terminal like: bool(-np.cos(s[0]) - np.cos(s[1] + s[0]) > 1.)') ### terminal = bool(obs[0] - np.cos(np.arc ... )
        #    print('but giving it the reward function is cheating? best to try to learn it from very sparse terminal reward data?')
        #    exit()
        #    ## reward = -1. if not terminal else 0.
        #else:
        #    print('need rewards in simulation')
        #    exit()
        rewards_batch.append(reward)

        if compare_ground_truth:
            o_real, r_real, d_real, _ = env.step(np.squeeze(action)) # take the actions the policy took in model simulation
            obs_real.append(o_real) ## this will put 'obs_last_real' as the final element of obs_real, unlike the simulated obs_batch list
            rewards_real.append(r_real)
            dones_real.append(d_real)
            assert env.unwrapped.spec.id=='CartPole-v0' or env.unwrapped.spec.id=='InvertedPendulum-v2' ## otherwise, env may (?) not allow continuing past done=True with a simple logger.warn() ?  Although I'm not quite sure about InvertedPendulum...
            if obs_error(replay_buffer, obs, o_real)>1.0:
                print('model rollout deteriorated')
                break
            
        if real_dones and done_prob==1:
            obs_init = env.reset()
            break
        elif not real_dones and d and stop_if_done: # used by plan(), to stop training in simulation when done=True predicted 
            print('model predicted Done in simulation after %d timesteps' % i)
            obs_init = env.reset()
            break

        # adaptively stop if the simulated data is outside the real data distribution
        if restrict_actions==False and obs_std is not None and np.amax(np.absolute(obs / obs_std))>obs_stdevs_max:
            print('Ending rollout: model is predicting obs at large # of stdevs of real data:')
            print('STDEVS:')
            print(obs / obs_std)
            break 

        i += 1

    print('end of simulation after %d timesteps' % i)

    # obs_init = env.reset() 

    if compare_ground_truth:
        real_trajectory = (obs_real, rewards_real, dones_real)
    else:
        real_trajectory = None
 
    return obs_batch, actions_batch, rewards_batch, done_probs_cum, obs, real_trajectory  # dones_batch


def plan(env_model,
         model_free_alg,
         replay_buffer,
         # parameters determining training procedure (given env_model and model rollouts)
         n_updates_sim=10, ## 30, 30
         nsteps_sim=200, # batch size; default should match batch size of model-free baseline
         ## nsteps_sim_update=200, # fixed; alternative to adaptive planning horizon
#### would prefer to use this instead of truncating model's buffer size...:     plan_buffer_size=300, # for sampling obs_init for rollouts
         lr_decay_rate_sim_to_real=1, ## 4 # anneal the learning rate to zero this much faster in simulation than in real environment
## <= could ALSO USE lr_sim = (const<1)*lr_real ...    
         entropy_coef_sim=None, # part of train_op; default = None = same as in model-free algorithm
         # parameters determining env_model
         done_prob_cum_min=0.99, # setting to 1.0 = model rollouts are never terminated
         real_dones=False, # 'cheat' by using done from env.step() instead of the model's prediction
         perfect_model=False, # override env_model with the actual real environment
         # parameters determining termination of model rollouts (given env_model)
         sim_stdevs_max=6.0, # num of stdevs of real obs data that model is allowed to simulate within
         nsteps_horizon=None, # default None = adaptive horizon, using prediction_error_max
         plan_horizon_smooth=0.8, # smoothing scale for uniform/global adaptive horizon
         prediction_error_max=0.03, # unused if nsteps_horizon is not None
         compare_ground_truth=True, # compare model rollouts to rollouts in the real env; set to False if nsteps_horizon!=None
         ## <= related parameter mag_error_coef is defined in update_errors()
         # parameters determining initialization of model rollouts (given env_model)
         obs_init_sim='reset', # how to initialize rollouts
         prioritized_replay_beta=0.5, # currently this is not used ... although we could use it to weight simulation gradients
         **kwargs):
         ## lr_sim, max_grad_norm_sim

    logdir = kwargs['logdir']

    if nsteps_horizon is not None:
        compare_ground_truth = False

    env = env_model.env
    sess = env_model.sess
    # sess = tf.get_default_session()

    gamma = model_free_alg.gamma
    actions_ph = model_free_alg.actions_ph
    obs_ph = model_free_alg.obs_ph
    returns_disc_ph = model_free_alg.returns_disc_ph
    values = model_free_alg.values
    pi_loss = model_free_alg.pi_loss
    v_loss = model_free_alg.v_loss
    # actions_logits = model_free_alg.actions_logits ## discrete policies
    train_op = model_free_alg.train_op
    grads_vars_pi = model_free_alg.grads_vars_pi
    lr_ph = model_free_alg.lr_ph
    lr = model_free_alg.lr

    from A2C import returns_nstep, gradients_norm, kl

    if nsteps_horizon is None and not compare_ground_truth: ## and model_free_alg.total_timesteps%2000==0: # = adaptive horizon
        print('updating model errors for planning horizon')
        # no longer in use: # env_model.get_trajectories(model_free_alg, **kwargs)
        env_model.update_errors(model_free_alg, replay_buffer, **kwargs)
        logger.dump_tabular()

    print('training in simulation for %d updates' % n_updates_sim)

    # update the (uniform) planning time horizon
    if not compare_ground_truth:
        for t in range(len(env_model.timestep_mean_errors)):
            if env_model.timestep_mean_errors[t]>prediction_error_max:
                env_model.planning_timesteps = plan_horizon_smooth*env_model.planning_timesteps + (1-plan_horizon_smooth)*t # max(t, env_model.planning_timesteps) # at t for which error exceeds threshold, increase planning horizon if it's smaller than t
                ## plan_horizon_smooth is a smoothing parameter, alternative to max() which introduces overestimation bias
                ## the variance of t_crit over iterations was O(10) when asymptoting to ~65; in this case the max() option fixed permanently to 75
                break
        logger.record_tabular('env_model.planning_timesteps', env_model.planning_timesteps) 
        if env_model.planning_timesteps==1 and nsteps_horizon is None: # if nsteps_horizon=constant, proceed with fixed length rollouts
            print('model cannot yet predict >1 timesteps')
            return # if we can't predict anything yet, don't try to plan

    # temporarily set entropy_coef to (higher) simulation value
    if entropy_coef_sim is not None:
        entropy_coef_real = model_free_alg.entropy_coef
        model_free_alg.entropy_coef = entropy_coef_sim

    for jj in range(n_updates_sim): 

        print('Starting iteration %d in plan()' % jj)

        # obs_init = env.reset() # start new rollout next time; use obs_last instead of env.reset() to continue existing rollout

        t = 0
        obs, actions, rewards, done_probs_cum = [], [], [], []
        returns_disc_batch = []
        while t<nsteps_sim:
            # sample initial rollout obs from replay buffer
            if obs_init_sim=='reset':
                pass ## obs_init = env.reset() is now done within run_sim()
            elif obs_init_sim=='buffer':
                if compare_ground_truth:
                    print('obs_init needs to be set within run_sim(), so that the real env we"re comparing to can get the right env.state by running obs_init=env.reset()')
                    exit()
                obs_init_info = replay_buffer.sample(1, prioritized_replay_beta)
                obs_init = np.squeeze(obs_init_info[0])
                print('obs_init')
                print(obs_init)
                idx = obs_init_info[-1][0]
                idxes = replay_buffer.idxes_episode(idx) # checked that correct transitions are updated below, using idxes
                # print(idx) # print(idxes)
                obs_ep, r_ep, td_errors = [], [], []
                for i in idxes: # update the priorities for the episode including obs_init; checked that this is correct, for correct idxes
                    data = replay_buffer._storage[i]
                    obs_ep.append(data[0])
                    r_ep.append(data[2])
                assert idxes[-1]==max(idxes)
                obs_ep.append(replay_buffer._storage[idxes[-1]][3]) # the episode's terminal obs
                obs_vals = np.squeeze(sess.run(model_free_alg.values, feed_dict={model_free_alg.obs_ph: np.array(obs_ep)}), axis=1)
                for j in range(len(r_ep)):
                    td_error = r_ep[j] + gamma * obs_vals[j+1] - obs_vals[j]
                    if j==len(r_ep)-1: td_error = r_ep[j] - obs_vals[j]
                    td_errors.append(td_error)
                replay_buffer.update_priorities(idxes, td_errors, idx_sampled=idx) # checked that priorities updated correctly from obs_vals, for the episode being sampled from
                #for j in range(len(replay_buffer._storage)):
                #    print(j)
                #    print(replay_buffer._storage[j])
            else:
                print('Specify how to initialize rollouts.')
                exit()
            # stop_if_done=True cuts off data when done=True, which returns_nstep() assumes, currently
            o, a, r, d_probs_cum, o_last, real_trajectory = run_sim(env_model, nsteps=nsteps_horizon, compare_ground_truth=compare_ground_truth, real_dones=real_dones, done_prob_cum_min=done_prob_cum_min, stop_if_done=True, replay_buffer=replay_buffer, obs_stdevs_max=sim_stdevs_max, perfect_model=perfect_model, **kwargs)
            if len(a)<=1: continue # skip length-1 trajectory, and try again
            oo = o + [o_last]
            if compare_ground_truth:
                o_real, r_real, d_real = real_trajectory[0], real_trajectory[1], real_trajectory[2]
                print('REAL')
                print(o_real)
                print('SIM')
                print(o)
                errors, obs_errors = rollout_errors(env_model, model_free_alg, a[1:], oo[1:], r[1:], d_probs_cum[1:], o_real[1:], r_real[1:], d_real[1:], replay_buffer=replay_buffer, logdir=logdir) # omit the initial state, since it's the same
                print('rollout errors:')
                print(','.join([float_to_str(xx) for xx in errors]))
                print('obs errors:')
                print(','.join([float_to_str(yy) for yy in obs_errors]))
                for tt in range(1,len(errors)+1): # start at 1 since we omitted s[0] above in errors
                    if errors[tt-1]>prediction_error_max: # then just keep the rollout up until this point
                        a = a[:tt]
                        o_last = o[tt]
                        o = o[:tt]
                        a = a[:tt]
                        r = r[:tt]
                        d_probs_cum = d_probs_cum[:tt]
                        break
            obs += o
            actions += a
            rewards += r
            done_probs_cum += d_probs_cum
            val_nstep = sess.run(values, feed_dict={obs_ph: o_last.reshape(1,-1)})
            returns_disc_batch += list(returns_nstep(r, val_nstep, gamma, done_probs_cum=d_probs_cum))
            t += len(o)
            print('DATA SO FAR IS = %d' % t)
        model_free_alg.total_timesteps_sim += t
        T = model_free_alg.total_timesteps + model_free_alg.total_timesteps_sim
        lr_decay_rate = model_free_alg.lr_decay_rate * lr_decay_rate_sim_to_real
        lr = model_free_alg.lr0 * max(1 - T * lr_decay_rate, model_free_alg.lr_min_to_max)

        obs = np.array(obs)
        actions = np.array(actions)
        if len(actions.shape)>2: actions = np.squeeze(actions, axis=-1)
        returns_disc_batch = np.array(returns_disc_batch)

        feed_dict={obs_ph: obs, actions_ph: actions, returns_disc_ph: returns_disc_batch, lr_ph: lr}

        if isinstance(env.action_space, Discrete):
            actions_probs = model_free_alg.actions_probs ## this is redundantly defined, iteratively ...
            pi_probs1 = sess.run(actions_probs, feed_dict={obs_ph: obs})

        _, pi_loss_sim, v_loss_sim, values_sim, grad_var_list = sess.run([train_op, pi_loss, v_loss, values, grads_vars_pi], feed_dict=feed_dict)
        # a_logits = sess.run(actions_logits, feed_dict=feed_dict)

        if isinstance(env.action_space, Discrete):
            pi_probs2 = sess.run(actions_probs, feed_dict={obs_ph: obs})
            kl_divs_batch = kl(tf.convert_to_tensor(pi_probs1), tf.convert_to_tensor(pi_probs2)) ## not ideal: converting from tensor to numpy (via sess.run) to tensor
            kl_divs_batch = sess.run(kl_divs_batch)
            logger.record_tabular('KL(pi_new||pi_old), sim data', np.mean(kl_divs_batch))

        # print obs, action_logits, actions, dones
        # print('grad_var_list from simulation:')
        #for gv in grad_var_list:
        #    print('parameter values, shape = ' + str(np.shape(gv[1])))
        #    print(gv[1])
        #    print('gradients, shape = ' + str(np.shape(gv[0])))
        #    print(gv[0])

        logger.record_tabular('learning_rate (simulation)', lr)
        logger.record_tabular('policy_loss (simulation)', pi_loss_sim) 
        logger.record_tabular('value_loss (simulation)', v_loss_sim)
        logger.record_tabular('Value ftn mean over episode (simulation)', np.mean(values_sim))
        logger.record_tabular('Policy Gradient L2 Norm (simulation)', gradients_norm(grad_var_list))
        logger.record_tabular('Policy Gradient Linf Norm (simulation)', gradients_norm(grad_var_list, np.inf))
        # logger.record_tabular('max action_logit for rollout states', np.amax(a_logits))

    if entropy_coef_sim is not None:
        model_free_alg.entropy_coef = entropy_coef_real # restore original entropy_coef

    return


def plan_a2c_baseline(policy_model, tmodel, runner, nsteps_update, nsteps_total):
    # the current obs is in runner.obs ## RIGHT?
    # the runner's step model should be from the same model class instance as the train model, policy_model
    ## remove one arg and infer the runner from policy_model? but I don't think you can do that, and I'd prefer to keep policy_model as explicit arg (alternatively, execute run_sim() outside of the plan() function)

#    tt = 0 #.#

    for i in range(int(nsteps_total/nsteps_update)):

#        t = time.time() #.#

        obs, states, rewards, masks, actions, values = runner.run_sim(tmodel, nsteps_update, init_real = not bool(i))
        # allow run_sim() to continue from the simulated observation runner.sim_obs where it left off
        ## obs, dones, rewards = tmodel.simulate(policy_model, nsteps_imagine, obs_current=obs_current)

#        tt += time.time() - t #.#

        # policy_loss, value_loss, policy_entropy = 
        _, _, _ = policy_model.train(obs, states, rewards, masks, actions, values)

#    print('run_sim() TIME = ' + str(tt)) #.#


def plan2(policy_model, posterior, tmodel, runners, nsteps_update, nsteps_total, error_min=0.25, n_sample_models=2):

    tmodel_samples = spn_sample(posterior.spn, n_sample_models)

    print('STARTING THE PLANNING LOOP.')

    new_episode = False
    for j in range(int(nsteps_total / n_sample_models)):

        print('j = ' + str(j)) ###

        # if no model has predicted end of episode yet, check if predictions diverged
        if new_episode==False and j>0:
            error = np.mean(np.square(obs_list[0]-obs_list[1])) / np.mean(np.square(obs_list[0]+obs_list[1]))
            ## call prediction error function here?
            print('ERROR = ' + str(error)) ###
            if error>error_min: # if model predictions diverged, start a new set of rollouts
                new_episode=True
                obs_last = obs[nsteps_update-1] ## (as noted below, may be unnecessary)

        obs_list = [] # list of the predicted (next nsteps_update steps) observations from all the model

        for i in range(n_sample_models):

            runner = runners[i]

            spn_sample_to_tmodel(tmodel_samples[i], posterior.tmodel)
            ## EXPENSIVE to swap out weights each iteration, just define tmodel1 and tmodel2, and DETACH from posterior ??

            if new_episode==True:
                runner.dones_sim = np.expand_dims(True, axis=0) ## (confirm that this resets runner.obs_sim at a new initial value)
                runner.obs_sim = np.expand_dims(obs_last, axis=0) ## (not sure obs_last is necessary...obs_sim may not affect the new initial value) restart all rollouts at the same state 
                ## (is states just always None?:) runner.states_sim = states_last

            if i==0: 
                obs, states, rewards, masks, actions, values = runner.run_sim(tmodel, nsteps_update, init_real = not bool(j))
                actions_required = np.expand_dims(actions, axis=1) # ** NOT SURE THIS IS RIGHT AXIS ** # store the actions chosen with first model, to reuse with all models
            if i!=0:
                obs, states, rewards, masks, actions, values = runner.run_sim(tmodel, nsteps_update, restrict_actions=True, actions_required=actions_required, init_real = not bool(j))
            # init_real arg ensures that all models start from the same initial state, the current real state

            if True in masks:  # then start a new simulated episode for all models
                new_episode = True
                obs_last = obs[nsteps_update-1]
                break # start a new rollout, starting with the first model

            if i==n_sample_models-1: # we've restarted all models now, and can proceed w/o resetting at the start of the loop
                new_episode = False

            obs_list.append(np.copy(obs))

            _, _, _ = policy_model.train(obs, states, rewards, masks, actions, values)


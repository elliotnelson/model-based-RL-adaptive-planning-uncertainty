import math
import random
import numpy as np
import tensorflow as tf

from replay_buffer import obs_stdev

from log_data import log_list_to_file


def obs_error(replay_buffer, obs_sim, obs):

    obs_std = obs_stdev(replay_buffer)

    obs_error = (obs_sim - obs) / obs_std # element-wise division
    return np.sqrt(np.mean(np.square(obs_error))) # rms error over obs components

def grad_error(sess, model_free_alg, action, obs, obs_real, return_disc, return_disc_real):

    obs_ph = model_free_alg.obs_ph
    actions_ph = model_free_alg.actions_ph
    returns_disc_ph = model_free_alg.returns_disc_ph

    # grads_vars = model_free_alg.grads_vars
    ## <= implicitly depends on entropy_coef, which may differ in simulation...
    grad_pi = model_free_alg.grad_pi
    grad_v = model_free_alg.grad_v

    feed_dict_real = {obs_ph: obs_real.reshape(1, -1), actions_ph: action, returns_disc_ph: return_disc_real}
    feed_dict_sim = {obs_ph: obs.reshape(1, -1), actions_ph: action, returns_disc_ph: return_disc}

    # g_real = sess.run(grads_vars, feed_dict=feed_dict_real)
    # g_sim = sess.run(grads_vars, feed_dict=feed_dict_sim)
    g_pi_real = sess.run(grad_pi, feed_dict=feed_dict_real)
    g_pi_sim = sess.run(grad_pi, feed_dict=feed_dict_sim)
    g_v_real = sess.run(grad_v, feed_dict=feed_dict_real)
    g_v_sim = sess.run(grad_v, feed_dict=feed_dict_sim)
    # g_real = list(zip(*g_real))[0]
    # g_sim = list(zip(*g_sim))[0]
    g_pi_real = list(zip(*g_pi_real))[0]
    g_pi_sim = list(zip(*g_pi_sim))[0]
    g_v_real = list(zip(*g_v_real))[0]
    g_v_sim = list(zip(*g_v_sim))[0]
    for j in range(len(g_pi_real)):
        if j==0:
            g1 = g_pi_real[j].flatten()
            g2 = g_pi_sim[j].flatten()
        else:
            g1 = np.concatenate((g1, g_pi_real[j].flatten()))
            g2 = np.concatenate((g2, g_pi_sim[j].flatten()))
    for k in range(len(g_v_real)):
        if k==0:
            gv1 = g_v_real[k].flatten()
            gv2 = g_v_sim[k].flatten()
        else:
            gv1 = np.concatenate((gv1, g_v_real[k].flatten()))
            gv2 = np.concatenate((gv2, g_v_sim[k].flatten()))
    # float64 needed, because we're comparing direction of nearly-identical and very high-dimensional vectors
    ## some of these np.float64's might be redundant
    g12 = np.float64(np.dot(np.float64(g1),np.float64(g2)))
    g11 = np.float64(np.dot(np.float64(g1),np.float64(g1)))
    g22 = np.float64(np.dot(np.float64(g2),np.float64(g2)))
    gv12 = np.float64(np.dot(np.float64(gv1),np.float64(gv2)))
    gv11 = np.float64(np.dot(np.float64(gv1),np.float64(gv1)))
    gv22 = np.float64(np.dot(np.float64(gv2),np.float64(gv2)))
    error_pi = 1.0 - math.sqrt(np.float64(g12*g12/g11/g22))
    error_v = 1.0 - math.sqrt(np.float64(gv12*gv12/gv11/gv22))
    #print('PI and V GRADIENT ERRORS')
    #print(error_pi)
    #print(error_v)
    assert error_pi>=0. and error_v>=0. # w/o float64, this failed
    # ratio_grad_length = math.sqrt(np.float64(g22/g11))

    return error_pi, error_v

def rollout_errors(env_model, model_free_alg, actions, obs, rewards, done_probs_cum, obs_real, rewards_real, dones_real, replay_buffer=None, logdir=None):
# actions are shared between the real and sim trajectories

    gamma = model_free_alg.gamma
    values = model_free_alg.values
    obs_ph = model_free_alg.obs_ph
    from A2C import returns_nstep
    sess = env_model.sess ## or could be model_free_alg.sess, or something better

    assert len(obs)==len(obs_real)

    obs_last = obs[-1]
    obs_last_real = obs_real[-1]
    obs_ph = model_free_alg.obs_ph

    # bootstrapped returns estimates
    val_nstep_real = np.squeeze(sess.run(values, feed_dict={obs_ph: obs_last_real.reshape(1,-1)})) # if target network: values_target, obs_value_target_ph
    val_nstep = np.squeeze(sess.run(values, feed_dict={obs_ph: obs_last.reshape(1,-1)}))
    returns_disc_real = returns_nstep(rewards_real, val_nstep_real, gamma, terminal=dones_real[-1])
    returns_disc = returns_nstep(rewards, val_nstep, gamma, done_probs_cum=done_probs_cum)

    actions = np.expand_dims(np.array(actions), axis=1)
### <= may need to omit this, depending on the env / settings ... OMIT for InvertedPendulum
    returns_disc_real = np.expand_dims(returns_disc_real, axis=1)
    returns_disc = np.expand_dims(returns_disc, axis=1)

    errors, errors_pi, errors_v = [], [], []
    obs_errors = []

    for i in range(len(obs)-1):

        error_pi, error_v = grad_error(sess, model_free_alg, actions[i], obs[i], obs_real[i], returns_disc[i], returns_disc_real[i])
        error = max(error_pi, error_v)
        errors.append(error)
        errors_pi.append(error_pi)
        errors_v.append(error_v)
        if replay_buffer is not None:
            obs_errors.append(obs_error(replay_buffer, obs[i], obs_real[i]))

    if logdir is not None:
        log_list_to_file(errors, logdir + '/rollout_errors_grad.txt')
        log_list_to_file(errors_pi, logdir + '/rollout_errors_grad_pi.txt')
        log_list_to_file(errors_v, logdir + '/rollout_errors_grad_v.txt')
        log_list_to_file(obs_errors, logdir + '/rollout_obs_errors.txt')

    return errors, obs_errors


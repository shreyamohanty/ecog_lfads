# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Optimization routines for LFADS"""


from __future__ import print_function, division, absolute_import

import datetime
import h5py

import jax.numpy as np
#EDIT: import device_put
#ORIGINAL: from jax import grad, jit, lax, random
from jax import grad, jit, lax, random, device_put
from jax.experimental import optimizers

import matplotlib.pyplot as plt
import numpy as onp  # original CPU-backed NumPy
import sklearn

#EDIT: import lfads_edited instead of lfads
#ORIGINAL: import lfads_tutorial.lfads as lfads
import lfads_tutorial.lfads_edited as lfads
import lfads_tutorial.utils as utils

import time

#EDIT: disable jit for debugging
########################################
# from jax.config import config
# config.update('jax_disable_jit', True)
########################################


def get_kl_warmup_fun(lfads_opt_hps):
  """Warmup KL cost to avoid a pathological condition early in training.

  Arguments:
    lfads_opt_hps : dictionary of optimization hyperparameters

  Returns:
    a function which yields the warmup value
  """

  kl_warmup_start = lfads_opt_hps['kl_warmup_start']
  kl_warmup_end = lfads_opt_hps['kl_warmup_end']
  kl_min = lfads_opt_hps['kl_min']
  kl_max = lfads_opt_hps['kl_max']
  def kl_warmup(batch_idx):
    progress_frac = ((batch_idx - kl_warmup_start) /
                     (kl_warmup_end - kl_warmup_start))
    kl_warmup = np.where(batch_idx < kl_warmup_start, kl_min,
                         (kl_max - kl_min) * progress_frac + kl_min)
    return np.where(batch_idx > kl_warmup_end, kl_max, kl_warmup)
  return kl_warmup


def optimize_lfads_core(key, batch_idx_start, num_batches,
                        update_fun, kl_warmup_fun,
                        opt_state, lfads_hps, lfads_opt_hps, train_data):
  """Make gradient updates to the LFADS model.

  Uses lax.fori_loop instead of a Python loop to reduce JAX overhead. This 
    loop will be jit'd and run on device.

  Arguments:
    init_params: a dict of parameters to be trained
    batch_idx_start: Where are we in the total number of batches
    num_batches: how many batches to run
    update_fun: the function that changes params based on grad of loss
    kl_warmup_fun: function to compute the kl warmup
    opt_state: the jax optimizer state, containing params and opt state
    lfads_hps: dict of lfads model HPs
    lfads_opt_hps: dict of optimization HPs
    train_data: nexamples x time x ndims np array of data for training

  Returns:
    opt_state: the jax optimizer state, containing params and optimizer state"""

  print("in optimize lfads core")

  key, dkeyg = utils.keygen(key, num_batches) # data
  key, fkeyg = utils.keygen(key, num_batches) # forward pass
  
  # Begin optimziation loop. Explicitly avoiding a python for-loop
  # so that jax will not trace it for the sake of a gradient we will not use.
  def run_update(batch_idx, opt_state):
    kl_warmup = kl_warmup_fun(batch_idx)
    didxs = random.randint(next(dkeyg), [lfads_hps['batch_size']], 0,
                           train_data.shape[0])
    #EDIT: jax raw np array vs tracer index error fix
    #ORIGINAL: x_bxt = train_data[didxs].astype(np.float32)
    x_bxt = device_put(train_data)[didxs].astype(np.float32)
    #EDIT: get values to check for param and grad nans (see optimize_lfads: update_w_gc)
    #ORIGINAL: 
    opt_state = update_fun(batch_idx, opt_state, lfads_hps, lfads_opt_hps,
                           next(fkeyg), x_bxt, kl_warmup)
    # opt_state, params_found, params_tracker, grads_found, grads_tracker = update_fun(batch_idx, opt_state, lfads_hps, lfads_opt_hps,
    #                                                                        next(fkeyg), x_bxt, kl_warmup)
    #EDIT: return values to check for param and grad nans (see optimize_lfads: update_w_gc)  
    #ORIGINAL: 
    return opt_state                                                              
    #return opt_state, params_found, params_tracker, grads_found, grads_tracker

  lower = batch_idx_start
  upper = batch_idx_start + num_batches
  #EDIT: replace the fori_loop while tuning so we can track params and grads
  #ORIGINAL: 
  return lax.fori_loop(lower, upper, run_update, opt_state)
  # for i in range(lower, upper):
  #   opt_state, params_found, params_tracker, grads_found, grads_tracker = run_update(i, opt_state)
  #   if params_found or grads_found:
  #     return opt_state, params_found, params_tracker, grads_found, grads_tracker
  # return opt_state, params_found, params_tracker, grads_found, grads_tracker

#EDIT: make all args static (except key) so checking for nans works hopefully :(
#ORIGINAL: 
optimize_lfads_core_jit = jit(optimize_lfads_core, static_argnums=(2,3,4,6,7))
#EDIT: optimize_lfads_core_jit = jit(optimize_lfads_core, static_argnums=(1,2,3,4,5,6,7,8))

def optimize_lfads(key, init_params, lfads_hps, lfads_opt_hps,
                   train_data, eval_data):
  """Optimize the LFADS model and print batch based optimization data.

  This loop is at the cpu nonjax-numpy level.

  Arguments:
    init_params: a dict of parameters to be trained
    lfads_hps: dict of lfads model HPs
    lfads_opt_hps: dict of optimization HPs
    train_data: nexamples x time x ndims np array of data for training

  Returns:
    a dictionary of trained parameters"""

  # Begin optimziation loop.
  all_tlosses = []
  all_elosses = []

  # Build some functions used in optimization.
  kl_warmup_fun = get_kl_warmup_fun(lfads_opt_hps)
  decay_fun = optimizers.exponential_decay(lfads_opt_hps['step_size'],
                                           lfads_opt_hps['decay_steps'],
                                           lfads_opt_hps['decay_factor'])

  opt_init, opt_update, get_params = optimizers.adam(step_size=decay_fun,
                                                     b1=lfads_opt_hps['adam_b1'],
                                                     b2=lfads_opt_hps['adam_b2'],
                                                     eps=lfads_opt_hps['adam_eps'])
  opt_state = opt_init(init_params)
  print("initial state initialized")

  #EDIT: create function to check dict for nans
  #######################################################
  # def check_dict(d):
  #   tracker = dict.fromkeys(d.keys(), 0)
  #   found = 0
  #   for k in d.keys():
  #     if type(d[k]) is dict:
  #       found, sub_tracker = check_dict(d[k])
  #       tracker[k] = sub_tracker
  #     elif np.isnan(d[k]).any():
  #       tracker[k] = 1
  #       found = 1
  #   return found, tracker
  #######################################################

  #EDIT: create function to append values in a dict to another tracker dict
  #NOTE: the two dicts must have the same keys
  #######################################################
  # def append_dict(tracker, d, func):
  #   for k in d.keys():
  #     if type(d[k]) is dict:
  #       sub_tracker = append_dict(tracker[k], d[k], func)
  #       #dicts are pass by reference so change will automatically be made
  #     else:
  #       #the values of each parameter is an array (some are arrays of arrays/lists)
  #       #np.max/min will go through nested lists
  #       if func == 'max':
  #         tracker[k].append(np.max(d[k]))
  #       elif func == 'min':
  #         tracker[k].append(np.min(d[k]))
  #######################################################


  def update_w_gc(i, opt_state, lfads_hps, lfads_opt_hps, key, x_bxt,
                  kl_warmup):
    #print(len(opt_state))
    """Update fun for gradients, includes gradient clipping."""
    params = get_params(opt_state)
    #print(params)
    #EDIT: check params for nans
    #######################################################
    # params_found, params_tracker = check_dict(params)
    #######################################################
    grads = grad(lfads.lfads_training_loss)(params, lfads_hps, key, x_bxt,
                                            kl_warmup,
                                            lfads_opt_hps['keep_rate'])
    #EDIT: check grads for nans
    #######################################################
    # grads_found, grads_tracker = check_dict(grads)
    #######################################################
    clipped_grads = optimizers.clip_grads(grads, lfads_opt_hps['max_grad_norm'])

    #EDIT: append grads and clipped grads to trackers
    #NOTE: update_w_gc always called from optimize lfads frame, so shouldn't need to change function signatures fro grad value trackers
    #######################################################
    # grads_value_tracker.append(grads)
    # clipped_grads_value_tracker.append(clipped_grads)
    #######################################################

    #EDIT: return params/grads_found and params/grads_tracker
    #ORIGINAL: 
    return opt_update(i, clipped_grads, opt_state)
    # return opt_update(i, clipped_grads, opt_state), params_found, params_tracker, grads_found, grads_tracker

 
  # Run the optimization, pausing every so often to collect data and
  # print status.
  batch_size = lfads_hps['batch_size']
  num_batches = lfads_opt_hps['num_batches']
  print_every = lfads_opt_hps['print_every']
  num_opt_loops = int(num_batches / print_every)
  params = get_params(opt_state)

  #EDIT: create function to copy a dict with custom list values and param value tracker dictionary
  #NOTE: hardcoding this function to use [] since if we input it as a variable, all values would reference same list
  #######################################################
  # def copy_and_replace_dict(d, func):
  #   new_dict = {}
  #   for k in d.keys():
  #     if type(d[k]) is dict:
  #       sub_dict = copy_and_replace_dict(d[k], func)
  #       new_dict[k] = sub_dict
  #     else:
  #       if func == 'max':
  #           new_dict[k] = [np.max(d[k])]
  #       elif func == 'min':
  #           new_dict[k] = [np.min(d[k])]
  #   return new_dict

  # max_param_value_tracker = copy_and_replace_dict(params, 'max')
  # min_param_value_tracker = copy_and_replace_dict(params, 'min')
  #######################################################

  #EDIT: for recording grads and clipped grads
  #######################################################
  # grads_value_tracker = []
  # clipped_grads_value_tracker = []
  #######################################################


  for oidx in range(num_opt_loops):
    print("oidx:", oidx)
    batch_idx_start = oidx * print_every
    start_time = time.time()
    key, tkey, dtkey, dekey = random.split(random.fold_in(key, oidx), 4)
    #EDIT: get values to check for param and grad nans (see optimize_lfads: update_w_gc)
    #ORIGINAL: 
    opt_state = optimize_lfads_core_jit(tkey, batch_idx_start,
                                       print_every, update_w_gc, kl_warmup_fun,
                                       opt_state, lfads_hps, lfads_opt_hps,
                                       train_data)
    # opt_state, params_found, params_tracker, grads_found, grads_tracker = optimize_lfads_core_jit(tkey, batch_idx_start,
    #                                                                                     print_every, update_w_gc, kl_warmup_fun,
    #                                                                                     opt_state, lfads_hps, lfads_opt_hps,
    #                                                                                     train_data)
                                                                              
    #EDIT: update param_value_tracker with new parameters
    #######################################################  
    # params = get_params(opt_state)
    # append_dict(max_param_value_tracker, params, "max")
    # append_dict(min_param_value_tracker, params, "min")  
    ####################################################### 

    #EDIT: print grad value tracker (just to check if it works)
    #######################################################
    #print(grads_value_tracker[-1])      
    ####################################################### 


    #EDIT: if nans found, return state
    #######################################################                                                                      
    # if params_found or grads_found:
    #   return params, optimizer_details, params_tracker, grads_tracker, batch_idx_start, max_param_value_tracker, min_param_value_tracker #, grads_value_tracker, clipped_grads_value_tracker
    #######################################################

    batch_time = time.time() - start_time

    # Losses
    params = get_params(opt_state)
    batch_pidx = batch_idx_start + print_every
    kl_warmup = kl_warmup_fun(batch_idx_start)
    # Training loss
    didxs = onp.random.randint(0, train_data.shape[0], batch_size)
    x_bxt = train_data[didxs].astype(onp.float32)
    tlosses = lfads.lfads_losses_jit(params, lfads_hps, dtkey, x_bxt,
                                     kl_warmup, 1.0)

    # Evaluation loss
    didxs = onp.random.randint(0, eval_data.shape[0], batch_size)
    ex_bxt = eval_data[didxs].astype(onp.float32)
    elosses = lfads.lfads_losses_jit(params, lfads_hps, dekey, ex_bxt,
                                     kl_warmup, 1.0)

    # #EDIT: check if any parameters go to nan
    # #######################################################
    # #print(params)
    # #######################################################

    # #EDIT: create dictionaries to record if any loss in nan
    # #######################################################
    # nan_found = 0
    # is_tnan = {'total' : 0, 'll_gaussian' : 0, 'kl_g0' : 0, 'kl_g0_prescale' : 0, 'kl_ii' : 0, 'kl_ii_prescale' : 0, 'l2' : 0}
    # is_enan = {'total' : 0, 'll_gaussian' : 0, 'kl_g0' : 0, 'kl_g0_prescale' : 0, 'kl_ii' : 0, 'kl_ii_prescale' : 0, 'l2' : 0}

    # loss_keys = is_tnan.keys()

    # for k in loss_keys:
    #   if np.isnan(tlosses[k]):
    #     is_tnan[k] = 1
    #     nan_found = 1
    #   if np.isnan(elosses[k]):
    #     is_enan[k] = 1
    #     nan_found = 1
    
    # if nan_found:
    #   return params, optimizer_details, is_tnan, is_enan, batch_idx_start
    # #######################################################


    # Saving, printing.
    all_tlosses.append(tlosses)
    all_elosses.append(elosses)
    s1 = "Batches {}-{} in {:0.2f} sec, Step size: {:0.5f}"
    s2 = "    Training losses {:0.0f} = NLL {:0.0f} + KL IC {:0.0f},{:0.0f} + KL II {:0.0f},{:0.0f} + L2 {:0.2f}"
    s3 = "        Eval losses {:0.0f} = NLL {:0.0f} + KL IC {:0.0f},{:0.0f} + KL II {:0.0f},{:0.0f} + L2 {:0.2f}"
    print(s1.format(batch_idx_start+1, batch_pidx, batch_time,
                   decay_fun(batch_pidx)))
    print(s2.format(tlosses['total'], 
                    #EDIT: changing nlog_p_xgz to ll_gaussian as done in lfads_edited
                    #ORIGINAL: tlosses['nlog_p_xgz'],
                    tlosses['ll_gaussian'],
                    tlosses['kl_g0_prescale'], tlosses['kl_g0'],
                    tlosses['kl_ii_prescale'], tlosses['kl_ii'],
                    tlosses['l2']))
    print(s3.format(elosses['total'], 
                    #EDIT: changing nlog_p_xgz to ll_gaussian as done in lfads_edited
                    #ORIGINAL: elosses['nlog_p_xgz'],
                    elosses['ll_gaussian'],
                    elosses['kl_g0_prescale'], elosses['kl_g0'],
                    elosses['kl_ii_prescale'], elosses['kl_ii'],
                    elosses['l2']))

    tlosses_thru_training = utils.merge_losses_dicts(all_tlosses)
    elosses_thru_training = utils.merge_losses_dicts(all_elosses)
    optimizer_details = {'tlosses' : tlosses_thru_training,
                         'elosses' : elosses_thru_training}

  #EDIT: return params/grads_tracker, batch_idx_start, and param_value_tracker
  #ORIGINAL: 
  return params, optimizer_details
  #######################################################
  # return params, optimizer_details, params_tracker, grads_tracker, batch_idx_start, max_param_value_tracker, min_param_value_tracker #, grads_value_tracker, clipped_grads_value_tracker
  #######################################################


  

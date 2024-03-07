# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import matplotlib.pyplot as plt
import torch as t
import time
import random as rd

# %%
device = t.device("cuda")

# %%
num_virus = 10**4
num_steps = 60*60*24*3 # 60*60*24*3=259200 seconds in 3 days
infection_prob = 0.2

# %% [markdown]
# x up from cells<br>
# y down nasal passage<br>
# z the other one

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### Starting attempt, trashed

# %%
# vir_sim = t.zeros(num_virus,3)

# start_time = time.time()
# for step in range(num_steps):
#     vir_sim += t.rand(num_virus,3)*2-1 # simulates next step
#     list_encounter = (vir_sim[:,0]<0).nonzero().reshape(-1,) # index of virions that encountered a cell
#     if len(list_encounter)>0:
#         dice_roll = np.random.rand(len(list_encounter)) # check for condition: probability to infect
#         vir_success = list_encounter[dice_roll < infection_prob] # index of virions that successfully infects
#         if len(vir_success)>0:
#             vir_fail = list_encounter[~(dice_roll < infection_prob)] # index of virions that encoutnered but did not infect
#             vir_sim[vir_fail,0] = -vir_sim[vir_fail,0] # if fail to infect, reflect back into ASL
#             print(step,vir_sim[vir_success])
# print(time.time() - start_time, "s")

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### Second attempt, very inefficient
#
# Looks elegant but way too slow, need to find way to improve (or give up?)

# %% [markdown]
# - $10^4$ virions, $60\cdot 60\cdot 24\cdot 3$ steps: ~43 seconds
# - $10^6$ virions, $60\cdot 60\cdot 24\cdot 3$ steps: ~62 seconds
# - $10^7$ virions, $60\cdot 60\cdot 24\cdot 3$ steps: kernel dies :(
# - $10^7$ virions, $60\cdot 60\cdot 24$ steps: kernel dies :(
# - $10^7$ virions, $60\cdot 60\cdot 12$ steps: kernel dies :(
# - $10^7$ virions, $60\cdot 60\cdot 6$ steps: ~41 seconds
# - $10^7$ virions, $60\cdot 60$ steps: ~8 seconds

# %%
vir_sim = t.zeros(num_virus,3,device=device) # starting all virions at (0,0,0)
infect_mask = t.zeros(num_virus, dtype=t.bool,device=device) # initialize, array of false
dice_roll = t.empty_like(infect_mask, dtype=t.float) # initialize, empty
vir_sim_every_min = []
vir_sim_every_min.append(np.copy(vir_sim.cpu())) # record starting position of virions

start_time = time.time()
for step in range(num_steps):
    num_active = num_virus - len(infect_mask.nonzero()) # number of active virions being simulated at this step
    if step%60==59: # save the location of virions every minute
        vir_sim_every_min.append(np.copy(vir_sim.cpu()))
    vir_sim[~infect_mask] += t.rand(num_active,3,device=device)*2-1 # simulates next step
    encounter_mask = t.zeros(num_virus, dtype=t.bool,device=device) # initialize encounter_mask as array of false
    encounter_mask[~infect_mask] = vir_sim[~infect_mask,0]<0 # mask of virions that encountered a cell (true if encountner)
    num_encounter = len(encounter_mask.nonzero()) # number of virions that encountered a cell
    if num_encounter>0:
        dice_roll[encounter_mask] = t.rand(size=(num_encounter,),device=device)
        infect_mask += encounter_mask & (dice_roll < infection_prob)
    # print(vir_sim)
    # print("encounter:", encounter_mask)
    # print("infected", infect_mask)    
print(time.time() - start_time, "s")

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### Vastly improved efficiency
#
# Record time spent on each line of code. Find ways to shorten it. (For reference: the original code takes less that a second to complete this step.)

# %%
# test speed of each step
t_initialize = []
t_active_vir = []
t_save_loc = []
t_save_loc_loop = []
t_new_step = []
t_encounter_initialize = []
t_encounter = []
t_num_encounter = []
t_dice_roll = []
t_infect_update = []
t_infect_loop = []
t_total_inner = []

start_time = time.time()
vir_sim = t.zeros(num_virus,3,device=device) # starting all virions at (0,0,0)
# infect_mask = t.zeros(num_virus, dtype=t.bool,device=device) # initialize, array of false
active_mask = t.ones(num_virus, dtype=t.bool,device=device) # initialize, array of false
dice_roll = t.empty_like(active_mask, dtype=t.float) # initialize, empty
vir_sim_every_min = []
vir_sim_every_min.append(np.copy(vir_sim.cpu())) # record starting position of virions
t_initialize.append(time.time() - start_time)

start_time_total_outer = time.time()
for step in range(num_steps):
    start_time_total_inner = time.time()
    
    start_time = time.time()
    num_active = active_mask.sum() # number of active virions being simulated at this step
    t_active_vir.append(time.time() - start_time)

    start_time_save_loc_loop = time.time()
    if step%60==59: # save the location of virions every minute
        start_time = time.time()
        vir_sim_every_min.append(np.copy(vir_sim.cpu()))
        t_save_loc.append(time.time() - start_time)
    t_save_loc_loop.append(time.time() - start_time_save_loc_loop)

    start_time = time.time()
    # vir_sim[~infect_mask] += t.rand(num_active,3,device=device)*2-1 # simulates next step
    # vir_sim += (t.rand(num_virus,3,device=device)*2-1) * (~infect_mask).reshape(-1,1) # simulates next step
    vir_sim += (t.rand(num_virus,3,device=device)*2-1) * active_mask.reshape(-1,1) # simulates next step
    t_new_step.append(time.time() - start_time)

    start_time = time.time()
    encounter_mask = t.zeros(num_virus, dtype=t.bool,device=device) # initialize encounter_mask as array of false
    t_encounter_initialize.append(time.time() - start_time)

    start_time = time.time()
    # encounter_mask[~infect_mask] = vir_sim[~infect_mask,0]<0 # mask of virions that encountered a cell (true if encountner)
    encounter_mask = (vir_sim[:,0]<0) * active_mask # mask of virions that encountered a cell (true if encountner)
    t_encounter.append(time.time() - start_time)

    start_time = time.time()
    num_encounter = encounter_mask.sum() # number of virions that encountered a cell
    t_num_encounter.append(time.time() - start_time)

    start_time_infect_loop = time.time()
    if num_encounter>0:
        start_time = time.time()
        dice_roll[encounter_mask] = t.rand(size=(num_encounter,),device=device)
        t_dice_roll.append(time.time() - start_time)

        start_time = time.time()
        active_mask ^= encounter_mask & (dice_roll < infection_prob) # flip true to false if condition holds
        t_infect_update.append(time.time() - start_time)
    t_infect_loop.append(time.time() - start_time_infect_loop)

    t_total_inner.append(time.time() - start_time_total_inner)

t_total_outer = time.time() - start_time_total_outer

# %%
t_total_outer

# %%
num_active.sum()

# %%
print("t_initialize:",np.sum(t_initialize))
print("t_active_vir:",np.sum(t_active_vir))
print("t_save_loc:",np.sum(t_save_loc))
print("t_save_loc_loop:",np.sum(t_save_loc_loop))
print("t_new_step:",np.sum(t_new_step))
print("t_encounter_initialize:",np.sum(t_encounter_initialize))
print("t_encounter:",np.sum(t_encounter))
print("t_num_encounter:",np.sum(t_num_encounter))
print("t_dice_roll:",np.sum(t_dice_roll))
print("t_infect_update:",np.sum(t_infect_update))
print("t_infect_loop:",np.sum(t_infect_loop))
print("t_total_inner:",np.sum(t_total_inner))

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### Efficient version, cleaned up

# %%
vir_sim = t.zeros(num_virus,3,device=device) # starting all virions at (0,0,0)
active_mask = t.ones(num_virus, dtype=t.bool,device=device) # initialize, array of false
dice_roll = t.empty_like(active_mask, dtype=t.float) # initialize, empty
vir_sim_every_min = []
vir_sim_every_min.append(np.copy(vir_sim.cpu())) # record starting position of virions

start_time = time.time()
for step in range(num_steps):  
    
    num_active = active_mask.sum() # number of active virions being simulated at this step
    
    if step%60==59: # save the location of virions every minute
        vir_sim_every_min.append(np.copy(vir_sim.cpu()))

    vir_sim += (t.rand(num_virus,3,device=device)*2-1) * active_mask.reshape(-1,1) # simulates next step

    encounter_mask = t.zeros(num_virus, dtype=t.bool,device=device) # initialize encounter_mask as array of false
    encounter_mask = (vir_sim[:,0]<0) * active_mask # mask of virions that encountered a cell (true if encountner)

    num_encounter = encounter_mask.sum() # number of virions that encountered a cell
    
    if num_encounter>0:
        dice_roll[encounter_mask] = t.rand(size=(num_encounter,),device=device) # for each encounter, roll random number from 0 to 1
        active_mask ^= encounter_mask & (dice_roll < infection_prob) # flip true to false if condition holds

print(time.time() - start_time)

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### Active virion count over time?
#
# Number of active virions becomes **very** low (~200 out of 10000) after just an hour, but the code still carries around the entire tensor. It's quite a waste compared to computing each virion individually... Maybe it will get better once more eliminating mechanisms are added, hopefully.

# %%
vir_sim = t.zeros(num_virus,3,device=device) # starting all virions at (0,0,0)
active_mask = t.ones(num_virus, dtype=t.bool,device=device) # initialize, array of false
dice_roll = t.empty_like(active_mask, dtype=t.float) # initialize, empty
vir_sim_every_min = []
vir_sim_every_min.append(np.copy(vir_sim.cpu())) # record starting position of virions

active_over_time = t.empty(num_steps, dtype=t.float) # number of active virions at any time

print(active_mask.sum())

start_time = time.time()
for step in range(num_steps):  
    
    active_over_time[step] = active_mask.sum() # number of active virions being simulated at this step
    
    if step%60==59: # save the location of virions every minute
        vir_sim_every_min.append(np.copy(vir_sim.cpu()))

    vir_sim += (t.rand(num_virus,3,device=device)*2-1) * active_mask.reshape(-1,1) # simulates next step

    encounter_mask = t.zeros(num_virus, dtype=t.bool,device=device) # initialize encounter_mask as array of false
    encounter_mask = (vir_sim[:,0]<0) * active_mask # mask of virions that encountered a cell (true if encountner)

    num_encounter = encounter_mask.sum() # number of virions that encountered a cell
    
    if num_encounter>0:
        dice_roll[encounter_mask] = t.rand(size=(num_encounter,),device=device) # for each encounter, roll random number from 0 to 1
        active_mask ^= encounter_mask & (dice_roll < infection_prob) # flip true to false if condition holds

print(time.time() - start_time)

print(active_mask.sum())

# %%
print("virion count after 1 hour: ", active_over_time[3600].item(), '\n')

plt.plot(active_over_time[[t*60 for t in range(60)]])
plt.xlabel("time (minute)")
plt.ylabel("free virions")
plt.show()

# %% [markdown]
# ### Reflective upper boundary!
#
# Now virions can't float away from cells infinitely. Will probably vastly improve speed.

# %%
reflective_boundary = 17 # microns

# %%
vir_sim = t.zeros(num_virus,3,device=device) # starting all virions at (0,0,0)
active_mask = t.ones(num_virus, dtype=t.bool,device=device) # initialize, array of false
dice_roll = t.empty_like(active_mask, dtype=t.float) # initialize, empty
vir_sim_every_min = [] # to record position of virions

start_time = time.time()
for step in range(num_steps):  
    
    vir_sim += (t.rand(num_virus,3,device=device)*2-1) * active_mask.reshape(-1,1) # simulates next step

    ################
    # TODO: reflective upper boundary, more efficient?
    ################
    reflect_mask = t.zeros(num_virus, dtype=t.bool,device=device)
    reflect_mask = (vir_sim[:,0]>reflective_boundary) * active_mask
    vir_sim[reflect_mask,0] = reflective_boundary*2 - vir_sim[reflect_mask,0]
    ################

    encounter_mask = t.zeros(num_virus, dtype=t.bool,device=device) # initialize encounter_mask as array of false
    encounter_mask = (vir_sim[:,0]<0) * active_mask # mask of virions that encountered a cell (true if encountner)

    num_encounter = encounter_mask.sum() # number of virions that encountered a cell
    
    if num_encounter>0:
        dice_roll[encounter_mask] = t.rand(size=(num_encounter,),device=device) # for each encounter, roll random number from 0 to 1
        active_mask ^= encounter_mask & (dice_roll < infection_prob) # flip true to false if condition holds

    if step%60==0: # save the location of virions every minute
        vir_sim_every_min.append(np.copy(vir_sim.cpu()))
    
    if active_mask.sum()==0: # number of active virions being simulated at this step
        print("no active virions remain after step:", step)
        break

print("total time:", time.time() - start_time)

# %%

# %%

# %% [markdown]
# ### 

# %% [markdown]
# ### ???

# %%

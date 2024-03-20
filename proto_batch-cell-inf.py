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
import pandas as pd
import torch as t
import time
import random as rd

# %%
device = t.device("cuda")

infection_prob = 0.2
reflective_boundary = 17 # microns
cell_diameter = 4

virion_prod_rate = 42 # per hour i.e. ~1000 per day
end_time = 72 # hours

latency_time = 6 # hours


# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### Code for virion path simulation
#
# copied from file `proto_pre-gen-paths.ipynb`, with minimal changes

# %%
def sim_vir_path(device, num_virus, num_steps, infection_prob, reflective_boundary, cell_diameter):
    
    vir_sim = t.zeros(num_virus,3,device=device) # starting all virions at (0,0,0)
    active_mask = t.ones(num_virus, dtype=t.bool,device=device) # initialize, array of true
    dice_roll = t.empty_like(active_mask, dtype=t.float) # initialize, empty
    vir_sim_every_min = [] # to record position of virions
    infected_cells = {}
    
    start_time = time.time()
    for step in range(num_steps):  
        
        vir_sim += (t.rand(num_virus,3,device=device)*2-1) * active_mask.reshape(-1,1) # simulates next step
    
        ################
        # reflective upper boundary 
        # TODO: more efficient?
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
            new_infection = encounter_mask & (dice_roll < infection_prob)
            active_mask ^= new_infection # remove infected cells from active mask
    
            ############
            # module for updating infected cells list
            # TODO: more efficient?
            ############
            new_cell_loc = (t.unique(vir_sim[new_infection,1:] // cell_diameter, dim=0)).int().tolist()
            new_cell_loc = set([str(loc) for loc in new_cell_loc]) - infected_cells.keys()
            infected_cells |= {loc: step+1 for loc in new_cell_loc}
            ############
    
        # if step%60==0: # save the location of virions every minute
        #     vir_sim_every_min.append(np.copy(vir_sim.cpu()))
        
        if active_mask.sum()==0: # number of active virions being simulated at this step
            print("no active virions remain after step:", step)
            break
    
    print("total time:", time.time() - start_time)

    return infected_cells


# %%
num_virus = 10**4
num_steps = 60*60*24*3 # 60*60*24*3=259200 seconds in 3 days

# virion_prod_rate = 42 # per hour i.e. ~1000 per day
# end_time = 72 # hours

# num_virus = (end_time - 0) * virion_prod_rate
# num_steps = end_time * 60 * 60

# print(num_virus, num_steps)

#sim_vir_path(device, num_virus, num_steps, infection_prob, reflective_boundary, cell_diameter)

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### Plan

# %% [markdown]
# specify starting cell, at `(0,0)` by default <br>
# calculate number of virions produced by the given cell: `n = (end_time - start_time) * virion_production_rate` <br>
# simulate the paths of `n` virions, create a list of `newly_infected_cells` and their `infection_times` <br>
# calculate the total number of virions produced by `newly_infected_cells` <br>
# repeat
#
# things to consider: <br>
# how to stop the loop? need to ignore cells infected after `end_time` <br>
# this might get a bit messy.....

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### First attempt & scratch work
#
# Significant changes needed to implement cell batch idea!"

# %%
num_virus = (end_time - 0) * virion_prod_rate
num_steps = end_time * 60 * 60

print(num_virus, num_steps)

infected_cells_wave0 = sim_vir_path(device, num_virus, num_steps, infection_prob, reflective_boundary, cell_diameter)

print("total infected cell:", len(infected_cells_wave0))

# %%
wave = 1

vir_prod_each_cell = [(num_steps - step - latency_time*3600) * virion_prod_rate // 3600 for step in infected_cells_wave0.values()]

remaining_time = end_time - latency_time*wave
num_virus = sum(vir_prod_each_cell)
num_steps = remaining_time * 60 * 60

print(num_virus, num_steps)

# %%
infected_cells_wave1 = sim_vir_path(device, num_virus, num_steps, infection_prob, reflective_boundary, cell_diameter)

# %%
new_infected_cells_wave1_keys = infected_cells_wave1.keys() - infected_cells_wave0

infected_cells = infected_cells_wave0.copy()
infected_cells |= {loc: infected_cells_wave1[loc] for loc in new_infected_cells_wave1_keys}

print(len(infected_cells) - len(new_infected_cells_wave1_keys))
print(sum(vir_prod_each_cell[:2]))

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### Try releasing the virion at the correct step
#
# Active mask starts all false. Every k steps, the entry corresponding to the virion being produced will be flipped to true. <br>
# Terrible performance.

# %%
# wave 0: starts from initial cell at [0,0]
virion_prod_rate = 42 # per hour i.e. ~1000 per day
end_time = 72 # hours

# %%
num_steps = end_time * 60 * 60

vir_prod_interval = (60 * 60 // virion_prod_rate) + 1 # seconds per virion
num_virus = (end_time*3600 // vir_prod_interval) + 1

print(num_virus, num_steps)


# %%
def sim_vir_path_wave0(device, num_virus, num_steps, infection_prob, reflective_boundary, cell_diameter, vir_prod_interval):
    
    vir_sim = t.zeros(num_virus,3,device=device) # starting all virions at (0,0,0)
    active_mask = t.zeros(num_virus, dtype=t.bool,device=device) # initialize, array of false
    dice_roll = t.empty_like(active_mask, dtype=t.float) # initialize, empty
    # vir_sim_every_min = [] # to record position of virions
    infected_cells = {} # initialize dictionary of all infected cells
    vir_counter = 0 # track the number of virions produced

    #### for debugging
    num_step_simulated = 0
    max_active = 0
    ####
    
    start_time = time.time()
    for step in range(num_steps):  

        # produce a virion at fixed interval
        if step % vir_prod_interval == 0:
            active_mask[vir_counter] = True
            vir_counter += 1

        if active_mask.sum() > 0 & vir_counter <= num_virus: # only do the simulation step if there is an active virion

            #### for debugging
            num_step_simulated += 1
            max_active = max(max_active, active_mask.sum())
            ####
        
            vir_sim += (t.rand(num_virus,3,device=device)*2-1) * active_mask.reshape(-1,1) # simulates next step
        
            ################
            # reflective upper boundary 
            # TODO: more efficient?
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
                new_infection = encounter_mask & (dice_roll < infection_prob)
                active_mask ^= new_infection # remove infected cells from active mask
        
                ############
                # module for updating infected cells list
                # TODO: more efficient?
                ############
                new_cell_loc = (t.unique(vir_sim[new_infection,1:] // cell_diameter, dim=0)).int().tolist()
                new_cell_loc = set([str(loc) for loc in new_cell_loc]) - infected_cells.keys()
                infected_cells |= {loc: step+1 for loc in new_cell_loc}
                ############
    
        # if step%60==0: # save the location of virions every minute
        #     vir_sim_every_min.append(np.copy(vir_sim.cpu()))
        
        if active_mask.sum() == 0 & vir_counter >= num_virus: # number of active virions being simulated at this step
            print("no active virions remain after step:", step)
            break
    
    print("total time:", time.time() - start_time)
    print("number of steps simulated:", num_step_simulated)
    print("maximal number of active virions:", max_active)

    return infected_cells


# %%
infected_cells_wave0 = sim_vir_path_wave0(device, num_virus, num_steps, infection_prob, reflective_boundary, cell_diameter, vir_prod_interval)

print("total infected cell:", len(infected_cells_wave0))

# %% [markdown]
# ### Try using `vir_sim` to store infection time
#
# Store relative infection times in `vir_sim`, along with virions' final destinations. Post process the output to extract list of infected cells.

# %% [markdown]
# #### **Wave 0:** Starts from initial cell at [0,0]

# %%
virion_prod_rate = 42 # per hour i.e. ~1000 per day
end_time = 72 # hours

num_steps = end_time * 60 * 60

# want each virion to be produced at an integer step
# this gives a close enough exact number of virions produced
vir_prod_interval = (60 * 60 // virion_prod_rate) + 1 # seconds per virion
num_virus = (num_steps // vir_prod_interval) + 1

print(num_virus, num_steps)


# %%
def sim_vir_path_wave0(device, num_virus, num_steps, infection_prob, reflective_boundary):
    
    vir_sim = t.zeros(num_virus,4,device=device) # starting all virions at (0,0,0), 4th column records infection time
    active_mask = t.ones(num_virus, dtype=t.bool,device=device) # initialize, array of true
    dice_roll = t.empty_like(active_mask, dtype=t.float) # initialize, empty
    
    start_time = time.time()
    for step in range(num_steps):  
    
        vir_sim[:,:3] += (t.rand(num_virus,3,device=device)*2-1) * active_mask.reshape(-1,1) # simulates next step
    
        ################
        # reflective upper boundary 
        # TODO: more efficient?
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
            new_infection = encounter_mask & (dice_roll < infection_prob)
            active_mask ^= new_infection # remove infected cells from active mask           
            ############
            vir_sim[new_infection,3] = step+1 # record infection time in the 4th column of vir_sim
            ############

        # if step%60==0: # save the location of virions every minute
        #     vir_sim_every_min.append(np.copy(vir_sim.cpu()))
        
        if active_mask.sum() == 0: # number of active virions being simulated at this step
            print("no active virions remain after step:", step)
            break
    
    print("total time:", time.time() - start_time)
    
    return vir_sim[:,1:]


# %%
# simulate
infected_cells_wave0 = sim_vir_path_wave0(device, num_virus, num_steps, infection_prob, reflective_boundary)

# %%
# the steps at which each virion is produced
vir_prod_modifier = t.tensor([x * vir_prod_interval for x in range(num_virus)], device=device)

infected_cells_wave0_adjusted = t.zeros_like(infected_cells_wave0)
# absolute y,z coordinates are converted to cell locations
infected_cells_wave0_adjusted[:,:2] = (infected_cells_wave0[:,:2] + cell_diameter/2) // cell_diameter
# add production times to relative infection times -> absolute infection time
infected_cells_wave0_adjusted[:,2] = infected_cells_wave0[:,2] + vir_prod_modifier

infected_cells_wave0_adjusted

# %%
# TODO: create a dictionary/dataframe/numpyarray of all infected cells

# %%
df = pd.DataFrame(infected_cells_wave0_adjusted.cpu())
for i in 0,1,2:
    df[i] = df[i].astype(int)
# eliminate repeated infections by selecting the minimum infection time
infected_cells_wave0_adjusted_df = df.groupby([0,1], as_index=False).agg('min')

# keep a numpy copy for computing the next wave
infected_cells_wave0_adjusted_np = infected_cells_wave0_adjusted_df.to_numpy(dtype=int)
# sort by infection time
infected_cells_wave0_adjusted_np = infected_cells_wave0_adjusted_np[infected_cells_wave0_adjusted_np[:,2].argsort()]

# remove the starting cell at (0,0)
new_infection_mask = [np.any(infected_cells_wave0_adjusted_np[i, :2] != 0) for i in range(len(infected_cells_wave0_adjusted_np))]
infected_cells_wave0_adjusted_np = infected_cells_wave0_adjusted_np[new_infection_mask]

# number of cells infected in this wave
num_new_infection = len(infected_cells_wave0_adjusted_np)
print("number of new infections:", num_new_infection)

# %% [markdown]
# #### **Wave 1:** Start from cells infected in wave 0

# %%
wave = 1

remaining_time = end_time - latency_time*wave
num_steps = remaining_time * 60 * 60

vir_prod_each_cell = [max(0, (num_steps - step) // vir_prod_interval + 1) for step in infected_cells_wave0_adjusted_np[:,2]]
num_virus = sum(vir_prod_each_cell)

print(num_virus, num_steps)


# %%
# not actually different from wave 0 function
def sim_vir_path_later_waves(device, num_virus, num_steps, infection_prob, reflective_boundary):
    
    vir_sim = t.zeros(num_virus,4,device=device) # starting all virions at (0,0,0), 4th column records infection time   
    active_mask = t.ones(num_virus, dtype=t.bool,device=device) # initialize, array of true
    dice_roll = t.empty_like(active_mask, dtype=t.float) # initialize, empty
    
    start_time = time.time()
    for step in range(num_steps):  
    
        vir_sim[:,:3] += (t.rand(num_virus,3,device=device)*2-1) * active_mask.reshape(-1,1) # simulates next step
    
        ################
        # reflective upper boundary 
        # TODO: more efficient?
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
            new_infection = encounter_mask & (dice_roll < infection_prob)
            active_mask ^= new_infection # remove infected cells from active mask           
            ############
            vir_sim[new_infection,3] = step+1 # record infection time in the 4th column of vir_sim
            ############

        # if step%60==0: # save the location of virions every minute
        #     vir_sim_every_min.append(np.copy(vir_sim.cpu()))
        
        if active_mask.sum() == 0: # number of active virions being simulated at this step
            print("no active virions remain after step:", step)
            break
    
    print("total time:", time.time() - start_time)
    
    return vir_sim[:,1:]


# %%
# simulate
infected_cells_wave1 = sim_vir_path_later_waves(device, num_virus, num_steps, infection_prob, reflective_boundary)

# %%
vir_subtotal = 0
vir_prod_modifier = t.zeros(num_virus, device=device)

for i,num in enumerate(vir_prod_each_cell):
    prod_start_time = infected_cells_wave0_adjusted_np[i, 2] + latency_time * 60 * 60
    vir_prod_modifier[vir_subtotal:vir_subtotal+num] += t.tensor([prod_start_time + x*vir_prod_interval for x in range(vir_prod_each_cell[i])], device=device)
    vir_subtotal += num
    # print(i, num)
    # print(infected_cells_wave0_adjusted_np[i, :2])

vir_prod_modifier

# %%
# convert infected_cells_wave1 to cell locations
# add vir_prod_modifier to infected_cells_wave1
infected_cells_wave1_adjusted = t.zeros_like(infected_cells_wave1)
infected_cells_wave1_adjusted[:,:2] = (infected_cells_wave1[:,:2] + cell_diameter/2) // cell_diameter
infected_cells_wave1_adjusted[:,2] = infected_cells_wave1[:,2] + vir_prod_modifier

infected_cells_wave1_adjusted

# %%
df = pd.DataFrame(infected_cells_wave1_adjusted.cpu())
for i in 0,1,2:
    df[i] = df[i].astype(int)

# eliminate repeated infections by selecting the minimum infection time
infected_cells_wave1_adjusted_df = df.groupby([0,1], as_index=False).agg('min')
print("number of unique infected cells in wave 1:", len(infected_cells_wave1_adjusted_df))

# remove previously infected cells, only keep new ones
infected_cells_wave1_adjusted_df = pd.concat([infected_cells_wave0_adjusted_df, infected_cells_wave1_adjusted_df]).drop_duplicates(subset=[0,1],keep=False)
print("number of new infected cells among those:", len(infected_cells_wave1_adjusted_df))

# keep a numpy copy for computing the next wave
infected_cells_wave1_adjusted_np = infected_cells_wave1_adjusted_df.to_numpy(dtype=int)
# sort by infection time
infected_cells_wave1_adjusted_np = infected_cells_wave1_adjusted_np[infected_cells_wave1_adjusted_np[:,2].argsort()]

# %%

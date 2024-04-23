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

infectable_fraction = 0.5 # fraction of infectable cells
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
    infect_roll = t.empty_like(active_mask, dtype=t.float) # initialize, empty
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
            infect_roll[encounter_mask] = t.rand(size=(num_encounter,),device=device) # for each encounter, roll random number from 0 to 1
            new_infection = encounter_mask & (infect_roll < infection_prob)
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
    infect_roll = t.empty_like(active_mask, dtype=t.float) # initialize, empty
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
                infect_roll[encounter_mask] = t.rand(size=(num_encounter,),device=device) # for each encounter, roll random number from 0 to 1
                new_infection = encounter_mask & (infect_roll < infection_prob)
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


# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### Try using `vir_sim` to store infection time
#
# Store relative infection times in `vir_sim`, along with virions' final destinations. Post process the output to extract list of infected cells.

# %% [markdown]
# #### **Wave 0:** Starts from initial cell at [0,0]

# %%
def sim_vir_path_wave0(device, num_virus, num_steps, infection_prob, reflective_boundary, cell_diameter):
    
    vir_sim = t.zeros(num_virus,4,device=device) # starting all virions at (0,0,0), 4th column records infection time
    active_mask = t.ones(num_virus, dtype=t.bool,device=device) # initialize, array of true
    infect_roll = t.empty_like(active_mask, dtype=t.float) # initialize, empty
    infectable_dict = {(0,0): True}

    infectability_time_total = 0
    
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

        ################
        # check and update dictionary of cell infectability
        # TODO: more efficient?
        ################
        infectability_time_start = time.time()
        if num_encounter>0:
            encounter_index = t.where(encounter_mask)[0] # indecies of encounters
            for encounter in encounter_index:
                cell_loc = tuple(vir_sim[encounter,1:4] // cell_diameter) 
            if cell_loc not in infectable_dict.keys():
                infectable_roll = rd.random() > 0.5 # roll against infectability
                infectable_dict |= {cell_loc: infectable_roll} # update dictionary
            else:
                infectable_roll = infectable_dict[cell_loc]
            encounter_mask[encounter] = infectable_roll # remove from encounter if cell not infectable
            num_encounter += infectable_roll - 1 # update number of encounters
            if infectable_roll == False:
                vir_sim[encounter,0] = -vir_sim[encounter,0] # reflect back into asl
        infectability_time_total += time.time() - infectability_time_start
        ################
        
        if num_encounter>0:
            infect_roll[encounter_mask] = t.rand(size=(num_encounter,),device=device) # for each encounter, roll random number from 0 to 1
            new_infection = encounter_mask & (infect_roll < infection_prob)
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
    print("time spent checking infecability:", infectability_time_total)
    
    return vir_sim[:,1:], infectable_dict


# %%
virion_prod_rate = 42 # 42 per hour i.e. ~1000 per day
end_time = 72 # 72 hours

num_steps = end_time * 60 * 60

# want each virion to be produced at an integer step
# this gives a close enough exact number of virions produced
vir_prod_interval = (60 * 60 // virion_prod_rate) + 1 # seconds per virion
num_virus = (num_steps // vir_prod_interval) + 1

print(num_virus, num_steps)

# simulate wave 0
infected_cells_wave0, infectable_dict_wave0 = sim_vir_path_wave0(device, num_virus, num_steps, infection_prob, reflective_boundary, cell_diameter)

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
all_infected_cells = pd.DataFrame({0: [0], 1: [0], 2:[0]}) # start with 1 infected cell at (0,0)

df = pd.DataFrame(infected_cells_wave0_adjusted.cpu())
for i in 0,1,2:
    df[i] = df[i].astype(int)

# eliminate repeated infections by selecting the minimum infection time
infected_cells_wave0_adjusted_df = df.groupby([0,1], as_index=False).agg('min')
print("number of unique infected cells in wave 0:", len(infected_cells_wave0_adjusted_df))

# # remove previously infected cells, only keep new ones
infected_cells_wave0_adjusted_coordinate_series = pd.Series(zip(infected_cells_wave0_adjusted_df[0],infected_cells_wave0_adjusted_df[1]))
all_infected_cells_coordinate_series = pd.Series(zip(all_infected_cells[0],all_infected_cells[1]))
shared_cells = infected_cells_wave0_adjusted_coordinate_series.isin(all_infected_cells_coordinate_series)
infected_cells_wave0_adjusted_df = infected_cells_wave0_adjusted_df[~shared_cells]
print("number of new infected cells among those:", len(infected_cells_wave0_adjusted_df))

# add new infected cells to all infected cells table
all_infected_cells = pd.concat([all_infected_cells, infected_cells_wave0_adjusted_df])
print("total number of infected cells:", len(all_infected_cells))

# keep a numpy copy for computing the next wave
infected_cells_wave0_adjusted_np = infected_cells_wave0_adjusted_df.to_numpy(dtype=int)
# sort by infection time
infected_cells_wave0_adjusted_np = infected_cells_wave0_adjusted_np[infected_cells_wave0_adjusted_np[:,2].argsort()]


# %% [markdown]
# #### **Wave 1:** Start from cells infected in wave 0

# %%
# includes starting location of each virion
def sim_vir_path_later_waves(device, num_virus, num_steps, infection_prob, reflective_boundary, cell_diameter, inf_cell_last_wave, infectable_dict):
    
    # vir_sim = t.zeros(num_virus,4,device=device) # starting all virions at (0,0,0), 4th column records infection time  
    # initialize virion path with starting locations
    vir_subtotal = 0
    vir_sim = t.zeros(num_virus,4,device=device)
    for i,num in enumerate(vir_prod_each_cell):
        vir_sim[vir_subtotal:vir_subtotal+num, 1:3] = t.from_numpy((inf_cell_last_wave[i, :2] + 1/2) * cell_diameter)
        vir_subtotal += num
        
    active_mask = t.ones(num_virus, dtype=t.bool,device=device) # initialize, array of true
    infect_roll = t.empty_like(active_mask, dtype=t.float) # initialize, empty

    infectability_time_total = 0
    
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

        ################
        # check and update dictionary of cell infectability
        # TODO: more efficient?
        ################
        infectability_time_start = time.time()
        if num_encounter>0:
            encounter_index = t.where(encounter_mask)[0] # indecies of encounters
            for encounter in encounter_index:
                cell_loc = tuple(vir_sim[encounter,1:4] // cell_diameter) 
            if cell_loc not in infectable_dict.keys():
                infectable_roll = rd.random() > 0.5 # roll against infectability
                infectable_dict |= {cell_loc: infectable_roll} # update dictionary
            else:
                infectable_roll = infectable_dict[cell_loc]
            encounter_mask[encounter] = infectable_roll # remove from encounter if cell not infectable
            num_encounter += infectable_roll - 1 # update number of encounters
            if infectable_roll == False:
                vir_sim[encounter,0] = -vir_sim[encounter,0] # reflect back into asl
        infectability_time_total += time.time() - infectability_time_start
        ################
        
        if num_encounter>0:
            infect_roll[encounter_mask] = t.rand(size=(num_encounter,),device=device) # for each encounter, roll random number from 0 to 1
            new_infection = encounter_mask & (infect_roll < infection_prob)
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
    print("time spent checking infecability:", infectability_time_total)

    return vir_sim[:,1:], infectable_dict


# %%
wave = 1

remaining_time = end_time - latency_time*wave
num_steps = remaining_time * 60 * 60

vir_prod_each_cell = [max(0, (num_steps - step) // vir_prod_interval + 1) for step in infected_cells_wave0_adjusted_np[:,2]]
num_virus = sum(vir_prod_each_cell)

print(num_virus, num_steps)

# simulate wave 1
infected_cells_wave1, infectable_dict_wave1 = sim_vir_path_later_waves(device, num_virus, num_steps, infection_prob, reflective_boundary, 
                                                cell_diameter, infected_cells_wave0_adjusted_np, infectable_dict_wave0)

# %%
# find out the step at which each virion is produced
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
infected_cells_wave1_adjusted[:,:2] = infected_cells_wave1[:,:2] // cell_diameter
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
infected_cells_wave1_adjusted_coordinate_series = pd.Series(zip(infected_cells_wave1_adjusted_df[0],infected_cells_wave1_adjusted_df[1]))
all_infected_cells_coordinate_series = pd.Series(zip(all_infected_cells[0],all_infected_cells[1]))
shared_cells = infected_cells_wave1_adjusted_coordinate_series.isin(all_infected_cells_coordinate_series)
infected_cells_wave1_adjusted_df = infected_cells_wave1_adjusted_df[~shared_cells]
print("number of new infected cells among those:", len(infected_cells_wave1_adjusted_df))

# keep a numpy copy for computing the next wave
infected_cells_wave1_adjusted_np = infected_cells_wave1_adjusted_df.to_numpy(dtype=int)
# sort by infection time
infected_cells_wave1_adjusted_np = infected_cells_wave1_adjusted_np[infected_cells_wave1_adjusted_np[:,2].argsort()]

# add new infected cells to all infected cells table
all_infected_cells = pd.concat([all_infected_cells, infected_cells_wave1_adjusted_df])
print("total number of infected cells:", len(all_infected_cells))

# %% [markdown]
# #### **Wave 2:** Start from cells infected in wave 1

# %%
wave = 2

remaining_time = end_time - latency_time*wave
num_steps = remaining_time * 60 * 60

vir_prod_each_cell = [max(0, (num_steps - step) // vir_prod_interval + 1) for step in infected_cells_wave1_adjusted_np[:,2]]
num_virus = sum(vir_prod_each_cell)

print(num_virus, num_steps)

# simulate wave 2
infected_cells_wave2, infectable_dict_wave2 = sim_vir_path_later_waves(device, num_virus, num_steps, infection_prob, reflective_boundary, 
                                                cell_diameter, infected_cells_wave1_adjusted_np, infectable_dict_wave1)

# %%
# find out the step at which each virion is produced
vir_subtotal = 0
vir_prod_modifier = t.zeros(num_virus, device=device)

for i,num in enumerate(vir_prod_each_cell):
    prod_start_time = infected_cells_wave1_adjusted_np[i, 2] + latency_time * wave * 60 * 60
    vir_prod_modifier[vir_subtotal:vir_subtotal+num] += t.tensor([prod_start_time + x*vir_prod_interval for x in range(vir_prod_each_cell[i])], device=device)
    vir_subtotal += num
    # print(i, num)
    # print(infected_cells_wave0_adjusted_np[i, :2])

vir_prod_modifier

# %%
# convert infected_cells_wave1 to cell locations
# add vir_prod_modifier to infected_cells_wave1
infected_cells_wave2_adjusted = t.zeros_like(infected_cells_wave2)
infected_cells_wave2_adjusted[:,:2] = infected_cells_wave2[:,:2] // cell_diameter
infected_cells_wave2_adjusted[:,2] = infected_cells_wave2[:,2] + vir_prod_modifier

infected_cells_wave2_adjusted

# %%
df = pd.DataFrame(infected_cells_wave2_adjusted.cpu())
for i in 0,1,2:
    df[i] = df[i].astype(int)

all_infected_cells = pd.concat([infected_cells_wave0_adjusted_df, infected_cells_wave1_adjusted_df])

# eliminate repeated infections by selecting the minimum infection time
infected_cells_wave2_adjusted_df = df.groupby([0,1], as_index=False).agg('min')
print("number of unique infected cells in wave 2:", len(infected_cells_wave2_adjusted_df))

# remove previously infected cells, only keep new ones
infected_cells_wave2_adjusted_coordinate_series = pd.Series(zip(infected_cells_wave2_adjusted_df[0],infected_cells_wave2_adjusted_df[1]))
all_infected_cells_coordinate_series = pd.Series(zip(all_infected_cells[0],all_infected_cells[1]))
shared_cells = infected_cells_wave2_adjusted_coordinate_series.isin(all_infected_cells_coordinate_series)
infected_cells_wave2_adjusted_df = infected_cells_wave2_adjusted_df[~shared_cells]
print("number of new infected cells among those:", len(infected_cells_wave2_adjusted_df))

# keep a numpy copy for computing the next wave
infected_cells_wave2_adjusted_np = infected_cells_wave2_adjusted_df.to_numpy(dtype=int)
# sort by infection time
infected_cells_wave2_adjusted_np = infected_cells_wave2_adjusted_np[infected_cells_wave2_adjusted_np[:,2].argsort()]

# add new infected cells to all infected cells table
all_infected_cells = pd.concat([all_infected_cells, infected_cells_wave2_adjusted_df])
print("total number of infected cells:", len(all_infected_cells))

# %% [markdown]
# #### **Wave 3:** Start from cells infected in wave 2

# %%
wave = 3

remaining_time = end_time - latency_time*wave
num_steps = remaining_time * 60 * 60

vir_prod_each_cell = [max(0, (num_steps - step) // vir_prod_interval + 1) for step in infected_cells_wave2_adjusted_np[:,2]]
num_virus = sum(vir_prod_each_cell)

print(num_virus, num_steps)

# simulate wave 3
infected_cells_wave3, infectable_dict_wave3 = sim_vir_path_later_waves(device, num_virus, num_steps, infection_prob, reflective_boundary, 
                                                cell_diameter, infected_cells_wave2_adjusted_np, infectable_dict_wave2)

# %%
# find out the step at which each virion is produced
vir_subtotal = 0
vir_prod_modifier = t.zeros(num_virus, device=device)

for i,num in enumerate(vir_prod_each_cell):
    prod_start_time = infected_cells_wave2_adjusted_np[i, 2] + latency_time * wave * 60 * 60
    vir_prod_modifier[vir_subtotal:vir_subtotal+num] += t.tensor([prod_start_time + x*vir_prod_interval for x in range(vir_prod_each_cell[i])], device=device)
    vir_subtotal += num
    # print(i, num)
    # print(infected_cells_wave0_adjusted_np[i, :2])

vir_prod_modifier

# %%
# convert infected_cells_wave2 to cell locations
# add vir_prod_modifier to infected_cells_wave2
infected_cells_wave3_adjusted = t.zeros_like(infected_cells_wave3)
infected_cells_wave3_adjusted[:,:2] = infected_cells_wave3[:,:2] // cell_diameter
infected_cells_wave3_adjusted[:,2] = infected_cells_wave3[:,2] + vir_prod_modifier

infected_cells_wave3_adjusted

# %%
df = pd.DataFrame(infected_cells_wave3_adjusted.cpu())
for i in 0,1,2:
    df[i] = df[i].astype(int)

all_infected_cells = pd.concat([infected_cells_wave0_adjusted_df, infected_cells_wave1_adjusted_df, infected_cells_wave2_adjusted_df])

# eliminate repeated infections by selecting the minimum infection time
infected_cells_wave3_adjusted_df = df.groupby([0,1], as_index=False).agg('min')
print("number of unique infected cells in wave 3:", len(infected_cells_wave3_adjusted_df))

# remove previously infected cells, only keep new ones
infected_cells_wave3_adjusted_coordinate_series = pd.Series(zip(infected_cells_wave3_adjusted_df[0],infected_cells_wave3_adjusted_df[1]))
all_infected_cells_coordinate_series = pd.Series(zip(all_infected_cells[0],all_infected_cells[1]))
shared_cells = infected_cells_wave3_adjusted_coordinate_series.isin(all_infected_cells_coordinate_series)
infected_cells_wave3_adjusted_df = infected_cells_wave3_adjusted_df[~shared_cells]
print("number of new infected cells among those:", len(infected_cells_wave3_adjusted_df))

# keep a numpy copy for computing the next wave
infected_cells_wave3_adjusted_np = infected_cells_wave3_adjusted_df.to_numpy(dtype=int)
# sort by infection time
infected_cells_wave3_adjusted_np = infected_cells_wave3_adjusted_np[infected_cells_wave3_adjusted_np[:,2].argsort()]

# add new infected cells to all infected cells table
all_infected_cells = pd.concat([all_infected_cells, infected_cells_wave3_adjusted_df])
print("total number of infected cells:", len(all_infected_cells))

# %% [markdown]
# #### **Wave 4:** Start from cells infected in wave 3

# %%
wave = 4

remaining_time = end_time - latency_time*wave
num_steps = remaining_time * 60 * 60

vir_prod_each_cell = [max(0, (num_steps - step) // vir_prod_interval + 1) for step in infected_cells_wave3_adjusted_np[:,2]]
num_virus = sum(vir_prod_each_cell)

print(num_virus, num_steps)

# simulate wave 4
infected_cells_wave4, infectable_dict_wave4 = sim_vir_path_later_waves(device, num_virus, num_steps, infection_prob, reflective_boundary, 
                                                cell_diameter, infected_cells_wave3_adjusted_np, infectable_dict_wave3)

# %%
# find out the step at which each virion is produced
vir_subtotal = 0
vir_prod_modifier = t.zeros(num_virus, device=device)

for i,num in enumerate(vir_prod_each_cell):
    prod_start_time = infected_cells_wave3_adjusted_np[i, 2] + latency_time * wave * 60 * 60
    vir_prod_modifier[vir_subtotal:vir_subtotal+num] += t.tensor([prod_start_time + x*vir_prod_interval for x in range(vir_prod_each_cell[i])], device=device)
    vir_subtotal += num
    # print(i, num)
    # print(infected_cells_wave0_adjusted_np[i, :2])

vir_prod_modifier

# %%
# convert infected_cells_wave1 to cell locations
# add vir_prod_modifier to infected_cells_wave1
infected_cells_wave4_adjusted = t.zeros_like(infected_cells_wave4)
infected_cells_wave4_adjusted[:,:2] = infected_cells_wave4[:,:2] // cell_diameter
infected_cells_wave4_adjusted[:,2] = infected_cells_wave4[:,2] + vir_prod_modifier

infected_cells_wave4_adjusted

# %%
df = pd.DataFrame(infected_cells_wave4_adjusted.cpu())
for i in 0,1,2:
    df[i] = df[i].astype(int)

all_infected_cells = pd.concat([infected_cells_wave0_adjusted_df, infected_cells_wave1_adjusted_df, infected_cells_wave2_adjusted_df, infected_cells_wave3_adjusted_df])

# eliminate repeated infections by selecting the minimum infection time
infected_cells_wave4_adjusted_df = df.groupby([0,1], as_index=False).agg('min')
print("number of unique infected cells in wave 4:", len(infected_cells_wave4_adjusted_df))

# remove previously infected cells, only keep new ones
infected_cells_wave4_adjusted_coordinate_series = pd.Series(zip(infected_cells_wave4_adjusted_df[0],infected_cells_wave4_adjusted_df[1]))
all_infected_cells_coordinate_series = pd.Series(zip(all_infected_cells[0],all_infected_cells[1]))
shared_cells = infected_cells_wave4_adjusted_coordinate_series.isin(all_infected_cells_coordinate_series)
infected_cells_wave4_adjusted_df = infected_cells_wave4_adjusted_df[~shared_cells]
print("number of new infected cells among those:", len(infected_cells_wave4_adjusted_df))

# keep a numpy copy for computing the next wave
infected_cells_wave4_adjusted_np = infected_cells_wave4_adjusted_df.to_numpy(dtype=int)
# sort by infection time
infected_cells_wave4_adjusted_np = infected_cells_wave4_adjusted_np[infected_cells_wave4_adjusted_np[:,2].argsort()]

# add new infected cells to all infected cells table
all_infected_cells = pd.concat([all_infected_cells, infected_cells_wave4_adjusted_df])
print("total number of infected cells:", len(all_infected_cells))


# %%

# %% [markdown]
# ### Block cell infectability

# %%
def sim_vir_path_wave0(device, num_virus, num_steps, infection_prob, reflective_boundary, cell_diameter):
    
    vir_sim = t.zeros(num_virus,4,device=device) # starting all virions at (0,0,0), 4th column records infection time
    active_mask = t.ones(num_virus, dtype=t.bool,device=device) # initialize, array of true
    infectable_roll = t.empty_like(active_mask, dtype=t.bool) # initialize, empty
    infect_roll = t.empty_like(active_mask, dtype=t.float) # initialize, empty

    infectability_time_total = 0
    
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

        ################
        # # check cell infectability
        ################
        infectability_time_start = time.time()
        if num_encounter>0:
            encounters_cell_loc = vir_sim[encounter_mask,1:3]//cell_diameter
            infectable_roll_index = ((encounters_cell_loc[:,0])%infectable_block_size)*infectable_block_size + ((encounters_cell_loc[:,1]//cell_diameter)%infectable_block_size)
            infectable_roll_index = infectable_roll_index.to(t.int)
            infectable_roll[encounter_mask] = infectable_sim_block[infectable_roll_index]
            uninfectable = encounter_mask & (~infectable_roll)
            vir_sim[uninfectable,0] = - vir_sim[uninfectable,0]
            encounter_mask ^= uninfectable
        infectability_time_total += time.time() - infectability_time_start
        ################

        num_encounter = encounter_mask.sum() # number of virions that encountered a cell

        if num_encounter>0:
            infect_roll[encounter_mask] = t.rand(size=(num_encounter,),device=device) # for each encounter, roll random number from 0 to 1
            new_infection = encounter_mask & (infect_roll < infection_prob)
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
    print("time spent checking infecability:", infectability_time_total)
    
    return vir_sim[:,1:]


# %%
infectable_block_size = 1000
infectable_sim_block = t.rand(infectable_block_size*infectable_block_size, device=device) < infectable_fraction
virion_prod_rate = 42 # 42 per hour i.e. ~1000 per day
end_time = 72 # 72 hours

num_steps = end_time * 60 * 60

# want each virion to be produced at an integer step
# this gives a close enough exact number of virions produced
vir_prod_interval = (60 * 60 // virion_prod_rate) + 1 # seconds per virion
num_virus = (num_steps // vir_prod_interval) + 1

print(num_virus, num_steps)

# simulate wave 0
infected_cells_wave0 = sim_vir_path_wave0(device, num_virus, num_steps, infection_prob, reflective_boundary, cell_diameter)

# %%

# %%
infectable_fraction = 0.5
infectable_sim_block_size = 1000
infectable_sim_block = t.rand(infectable_sim_block_size*infectable_sim_block_size) < infectable_fraction

# %%
infectable_sim_block

# %%
bleh = t.tensor([[1,2,4], [3,7,5],[4,9,3]])
(bleh[:,1] % 3) * 3 + (bleh[:,2] % 3)

# %%
bleh = t.tensor([1,2,4, 3,7,5, 4,9,3])
beh = t.tensor([0,2,4])
bleh[beh]

# %%
vir_sim = t.zeros(num_virus,4,device=device)
((vir_sim[encounter_mask,1]//cell_diameter)%infectable_block_size)*infectable_block_size + ((vir_sim[encounter_mask,2]//cell_diameter)%infectable_block_size)

# %%

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
import itertools


# %%
def sim_vir_path_wave0(device, num_virus, num_steps, diffusion_cell, infection_prob, ref_bound_cell, adv_bound_cell, adv_vel_cell, exit_bound_cell, 
                       prd_bound_cell, infectable_block_size, infectable_sim_block):

    print("device:", device)
    print("num virus:", num_virus)
    print("num_steps:", num_steps)
    print("infection_prob:", infection_prob)
    print("ref_bound_cell:", ref_bound_cell)
    print("adv_bound_cell:", adv_bound_cell)
    print("adv_vel_cell:", adv_vel_cell)
    print("exit_bound_cell:", exit_bound_cell)
    print("prd_bound_cell:", prd_bound_cell)
    print("infectable_block_size:", infectable_block_size)
    print("infectable_sim_block:", infectable_sim_block)

    reflection_time_total = 0
    advection_time_total = 0
    flush_time_total = 0
    encounter_mask_time_total = 0
    infectability_time_total = 0
    infect_time_total = 0

    start_time = time.time()
    vir_sim = t.zeros(num_virus, 4, device=device) # starting all virions at (0,0,0), 4th column records infection time
    active_mask = t.ones(num_virus, dtype=t.bool,device=device) # initialize, array of true
    infectable_roll = t.empty_like(active_mask, dtype=t.bool) # initialize, empty
    infect_roll = t.empty_like(active_mask, dtype=t.bool) # initialize, empty
    # vir_sim_every_min = []
    print("time to initialize:", time.time()-start_time, "seconds")

    vir_sim[:,1:3] = vir_sim[:,1:3] + 0.5
    
    start_time = time.time()
    for step in range(num_steps):  
    
        vir_sim[:,:3] += (t.rand(num_virus,3,device=device)*2-1) * diffusion_cell * active_mask.reshape(-1,1) # simulates next step
    
        ################
        # reflective upper boundary 
        # TODO: more efficient?
        ################
        reflection_time_start = time.time()
        reflect_mask = t.zeros(num_virus, dtype=t.bool,device=device)
        reflect_mask = (vir_sim[:,0]>ref_bound_cell) & active_mask
        vir_sim[reflect_mask,0] = ref_bound_cell*2 - vir_sim[reflect_mask,0]
        reflection_time_total += time.time() - reflection_time_start
        ################

        ################
        # apply advection to y if x coordinate \geq 7
        ################
        advection_time_start = time.time()
        advection_mask = t.zeros(num_virus, dtype=t.bool,device=device)
        advection_mask = (vir_sim[:,0]>adv_bound_cell) & active_mask
        vir_sim[advection_mask,1] += adv_vel_cell
        advection_time_total += time.time() - advection_time_start
        ################

        ################
        # turn off virions that are flushed out
        ################
        flush_time_start = time.time()
        flux_mask = t.zeros(num_virus, dtype=t.bool,device=device)
        flux_mask = (vir_sim[:,1]>exit_bound_cell) & active_mask
        vir_sim[flux_mask,3] = -(step+1)
        active_mask ^= flux_mask
        flush_time_total += time.time() - flush_time_start
        ################

        ################
        # periodic boundary condition
        ################
        # vir_sim[:,2] = vir_sim[:,2] % prd_bound_cell
        ################

        encounter_mask_time_start = time.time()
        encounter_mask = t.zeros(num_virus, dtype=t.bool,device=device) # initialize encounter_mask as array of false
        encounter_mask = (vir_sim[:,0]<0) * active_mask # mask of virions that encountered a cell (true if encountner)
        encounter_mask_time_total += time.time() - encounter_mask_time_start
    
        num_encounter = encounter_mask.sum() # number of virions that encountered a cell

        ################
        # # check cell infectability
        ################
        infectability_time_start = time.time()
        if num_encounter>0:
            infectable_roll_index = (t.floor(vir_sim[encounter_mask,1])%infectable_block_size)*infectable_block_size + (t.floor(vir_sim[encounter_mask,2])%infectable_block_size)
            infectable_roll_index = infectable_roll_index.to(t.int)
            infectable_roll[encounter_mask] = infectable_sim_block[infectable_roll_index]
            uninfectable = encounter_mask & (~infectable_roll)
            vir_sim[uninfectable,0] = - vir_sim[uninfectable,0]
            encounter_mask ^= uninfectable
        num_encounter = encounter_mask.sum() # number of virions that encountered a cell
        infectability_time_total += time.time() - infectability_time_start
        ################

        infect_time_start = time.time()
        if num_encounter>0:
            infect_roll[encounter_mask] = t.rand(size=(num_encounter,),device=device) < infection_prob # for each encounter, roll random number from 0 to 1
            new_infection = encounter_mask & infect_roll
            active_mask ^= new_infection # remove infected cells from active mask           
            ############
            vir_sim[new_infection,3] = step+1 # record infection time in the 4th column of vir_sim
            ############
        infect_time_total += time.time() - infect_time_start

        # if step%600==0: # save the location of virions every 10 minutes
        #     vir_sim_every_min.append(np.copy(vir_sim[:,1:].cpu()))
        
        if active_mask.sum() == 0: # number of active virions being simulated at this step
            print("no active virions remain after step:", step)
            break
    
    print("time spent on reflective boundary:", reflection_time_total)
    print("time spent on advection:", advection_time_total)
    print("time spent removing flushed out virions:", flush_time_total)
    print("time spent checking encounters:", encounter_mask_time_total)
    print("time spent checking infecability:", infectability_time_total)
    print("time spent on infection:", infect_time_total)
    print("total time:", time.time() - start_time)
    
    return vir_sim[:,1:]


# %%
# includes starting location of each virion
def sim_vir_path_later_waves(num_virus, device, vir_prod_each_cell, infected_cells_old_adjusted_np, num_steps, diffusion_cell, 
                             ref_bound_cell, adv_bound_cell, adv_vel_cell, exit_bound_cell, prd_bound_cell,
                             infectable_block_size, infectable_sim_block, infection_prob):

    print("starting simulation.")

    print("device:", device)
    print("num virus:", num_virus)
    print("num_steps:", num_steps)
    print("infection_prob:", infection_prob)
    print("ref_bound_cell:", ref_bound_cell)
    print("adv_bound_cell:", adv_bound_cell)
    print("adv_vel_cell:", adv_vel_cell)
    print("exit_bound_cell:", exit_bound_cell)
    print("prd_bound_cell:", prd_bound_cell)
    print("infectable_block_size:", infectable_block_size)
    print("infectable_sim_block:", infectable_sim_block)

    sim_step_time_total = 0
    reflection_time_total = 0
    advection_time_total = 0
    flush_time_total = 0
    encounter_mask_time_total = 0
    infectability_time_total = 0
    infect_time_total = 0
    
    start_time = time.time()
    ################
    # initialize virion path with starting locations
    ################
    vir_prod = np.trim_zeros(vir_prod_each_cell, 'b') # ignore cells that didn't produce any virions
    vir_initial_coords = infected_cells_old_adjusted_np[:len(vir_prod),:2] + 0.5 # start each virion at center of cells
    vir_sim = t.zeros(num_virus, 4, device=device) # starting all virions at (0,0,0), 4th column records infection time  
    vir_sim[:,1:3] = t.from_numpy(np.repeat(vir_initial_coords, vir_prod, axis=0)[:num_virus]) # populate with initial locations
    ################
    print("time to initialize virion locations:", time.time()-start_time, "seconds")  

    start_time = time.time()
    ################
    # other initialization
    ################
    active_mask = t.ones(num_virus, dtype=t.bool,device=device) # initialize, array of true
    infectable_roll = t.empty_like(active_mask, dtype=t.bool) # initialize, empty
    infect_roll = t.empty_like(active_mask, dtype=t.bool) # initialize, empty
    ################
    print("time to initialize other things:", time.time()-start_time, "seconds")

    infectability_time_total = 0
    
    start_time = time.time()
    for step in range(num_steps):  

        sim_step_time_start = time.time()
        vir_sim[:,:3] += (t.rand(num_virus,3,device=device)*2-1) * diffusion_cell * active_mask.reshape(-1,1) # simulates next step
        sim_step_time_total += time.time() - sim_step_time_start
    
        ################
        # reflective upper boundary 
        # TODO: more efficient?
        ################
        reflection_time_start = time.time()
        reflect_mask = t.zeros(num_virus, dtype=t.bool,device=device)
        reflect_mask = (vir_sim[:,0]>ref_bound_cell) & active_mask
        vir_sim[reflect_mask,0] = ref_bound_cell*2 - vir_sim[reflect_mask,0]
        reflection_time_total += time.time() - reflection_time_start
        ################

        ################
        # apply advection if y coordinate \geq 7
        # TODO: more efficient?
        ################
        advection_time_start = time.time()
        advection_mask = t.zeros(num_virus, dtype=t.bool,device=device)
        advection_mask = (vir_sim[:,0]>adv_bound_cell) & active_mask
        vir_sim[advection_mask,1] += adv_vel_cell
        advection_time_total += time.time() - advection_time_start
        ################

        ################
        # turn off virions that are flushed out
        ################
        flush_time_start = time.time()
        flux_mask = t.zeros(num_virus, dtype=t.bool,device=device)
        flux_mask = (vir_sim[:,1]>exit_bound_cell) & active_mask
        vir_sim[flux_mask,3] = -(step+1)
        active_mask ^= flux_mask
        flush_time_total += time.time() - flush_time_start
        ################

        ################
        # periodic boundary condition
        ################
        # vir_sim[:,2] = vir_sim[:,2] % prd_bound_cell
        ################

        encounter_mask_time_start = time.time()
        encounter_mask = t.zeros(num_virus, dtype=t.bool,device=device) # initialize encounter_mask as array of false
        encounter_mask = (vir_sim[:,0]<0) * active_mask # mask of virions that encountered a cell (true if encountner)
        encounter_mask_time_total += time.time() - encounter_mask_time_start
    
        num_encounter = encounter_mask.sum() # number of virions that encountered a cell

        ################
        # # check cell infectability
        ################
        infectability_time_start = time.time()
        if num_encounter>0:
            infectable_roll_index = (t.floor(vir_sim[encounter_mask,1])%infectable_block_size)*infectable_block_size + (t.floor(vir_sim[encounter_mask,2])%infectable_block_size)
            infectable_roll_index = infectable_roll_index.to(t.int)
            infectable_roll[encounter_mask] = infectable_sim_block[infectable_roll_index]
            uninfectable = encounter_mask & (~infectable_roll)
            vir_sim[uninfectable,0] = - vir_sim[uninfectable,0]
            encounter_mask ^= uninfectable
        num_encounter = encounter_mask.sum() # number of virions that encountered a cell
        infectability_time_total += time.time() - infectability_time_start
        ################

        infect_time_start = time.time()
        if num_encounter>0:
            infect_roll[encounter_mask] = t.rand(size=(num_encounter,),device=device) < infection_prob # for each encounter, roll random number from 0 to 1
            new_infection = encounter_mask & infect_roll
            active_mask ^= new_infection # remove infected cells from active mask           
            ############
            vir_sim[new_infection,3] = step+1 # record infection time in the 4th column of vir_sim
            ############
        infect_time_total += time.time() - infect_time_start

        # if step%600==0: # save the location of virions every minute
        #     vir_sim_every_min.append(np.copy(vir_sim.cpu()))
        
        if active_mask.sum() == 0: # number of active virions being simulated at this step
            print("no active virions after:", step, "steps")
            break

        if step%1000==0:
            print("now at step", step, "after", time.time() - start_time, "seconds")

    print("time spent simulating next step:", sim_step_time_total)
    print("time spent on reflective boundary:", reflection_time_total)
    print("time spent on advection:", advection_time_total)
    print("time spent removing flushed out virions:", flush_time_total)
    print("time spent checking encounters:", encounter_mask_time_total)
    print("time spent checking infecability:", infectability_time_total)
    print("time spent on infection:", infect_time_total)
    print("total time:", time.time() - start_time)

    return vir_sim[:,1:]


# %%
def secondary_para_later_waves(end_time, latency_time, vir_prod_interval, infected_cells_old_adjusted_np, device):
    
    start_time = time.time()    
    num_steps = (end_time - latency_time) * 60 * 60
    
    vir_prod_each_cell = [max(0, (num_steps - step) // vir_prod_interval + 1) for step in infected_cells_old_adjusted_np[:,2]]
    num_virus = int(sum(vir_prod_each_cell))
    
    print("number of virions:", num_virus)
    print("number of steps:", num_steps)
    print("time to count virions and steps:", time.time()-start_time, "seconds")

    start_time = time.time()    
    # find out the step at which each virion is produced
    vir_subtotal = 0
    vir_prod_modifier = t.zeros(num_virus, device=device)
    
    for i,num in enumerate(vir_prod_each_cell):
        prod_start_time = infected_cells_old_adjusted_np[i, 2] + latency_time * 60 * 60
        vir_prod_modifier[int(vir_subtotal):int(vir_subtotal+num)] += t.tensor([prod_start_time + x*vir_prod_interval for x in range(int(vir_prod_each_cell[i]))], device=device)
        vir_subtotal += num
        # print(i, num)
        # print(infected_cells_wave0_adjusted_np[i, :2])
    print("time to compute when is each virion produced:", time.time()-start_time, "seconds")

    return num_steps, num_virus, vir_prod_each_cell, vir_prod_modifier


# %%
def inf_cell_wave0(vir_prod_modifier, num_virus, device, num_steps, infected_cells_wave0):

    start_time = time.time()

    print(infected_cells_wave0.shape)
    
    # the steps at which each virion is produced
    # vir_prod_modifier = t.tensor([x * vir_prod_interval for x in range(num_virus)], device=device)
    vir_prod_modifier = t.unsqueeze(vir_prod_modifier, 1)
    print(vir_prod_modifier.shape)

    # concatenate so later corresponding virions also get deleted from modifier
    infected_cells_wave0 = t.cat((infected_cells_wave0, vir_prod_modifier), 1)
    print(infected_cells_wave0.shape)
    # remove virions that were flushed out, i.e. ones with infection time 0
    infected_cells_wave0 = infected_cells_wave0[infected_cells_wave0[:,2] > 0]    
    # add production times to relative infection times -> absolute infection time
    infected_cells_wave0[:,2] = infected_cells_wave0[:,2] + infected_cells_wave0[:,3]
    # remove cells that were infected past the simulation time
    infected_cells_wave0 = infected_cells_wave0[infected_cells_wave0[:,2] < num_steps]
    # absolute y,z coordinates are converted to cell locations
    infected_cells_wave0[:,:2] = t.floor(infected_cells_wave0[:,:2])
    
    all_infected_cells = pd.DataFrame({0: [0], 1: [0], 2:[0]}) # start with 1 infected cell at (0,0)
    
    df = pd.DataFrame(infected_cells_wave0[:,:3].cpu())
    for i in 0,1,2:
        df[i] = df[i].astype(int)
    
    # eliminate repeated infections by selecting the minimum infection time
    infected_cells_wave0_adjusted_df = df.groupby([0,1], as_index=False).agg('min')
    print("number of unique infected cells in wave 0:", len(infected_cells_wave0_adjusted_df))
    
    # add new infected cells to all infected cells table
    all_infected_cells = pd.concat([all_infected_cells, infected_cells_wave0_adjusted_df])
    print("total number of infected cells:", len(all_infected_cells))
    
    # keep a numpy copy for computing the next wave
    infected_cells_wave0_adjusted_np = infected_cells_wave0_adjusted_df.to_numpy(dtype=int)
    # sort by infection time
    infected_cells_wave0_adjusted_np = infected_cells_wave0_adjusted_np[infected_cells_wave0_adjusted_np[:,2].argsort()]

    print("time post processing:", time.time() - start_time, "seconds")
    
    return infected_cells_wave0_adjusted_np, all_infected_cells


# %%
def inf_cell_later_waves(infected_cells_new, vir_prod_modifier, end_time, all_infected_cells):

    start_time = time.time()

    vir_prod_modifier = t.unsqueeze(vir_prod_modifier, 1)
    # concatenate so later corresponding virions also get deleted from modifier
    infected_cells_new = t.cat((infected_cells_new, vir_prod_modifier), 1)
    # remove virions that were flushed out, i.e. ones with infection time <= 0
    infected_cells_new = infected_cells_new[infected_cells_new[:,2] > 0]    
    # add vir_prod_modifier to infected_cells_new
    infected_cells_new[:,2] = infected_cells_new[:,2] + infected_cells_new[:,3]
    print("max inf time:", infected_cells_new[:,2].max() / 3600)
    # remove cells that were infected past the simulated time
    infected_cells_new = infected_cells_new[infected_cells_new[:,2] <= end_time*60*60]    
    print("max inf time within simulation time:", infected_cells_new[:,2].max() / 3600)
    # convert infected_cells_new to cell locations
    infected_cells_new[:,:2] = t.floor(infected_cells_new[:,:2])

    if len(infected_cells_new) > 0:
        df = pd.DataFrame(infected_cells_new[:,:3].cpu())
        for i in 0,1,2:
            df[i] = df[i].astype(int)
        print("max inf time in df:", df[2].max() / 3600)
    
        # eliminate repeated infections by selecting the minimum infection time
        infected_cells_new_adjusted_df = df.groupby([0,1], as_index=False).agg('min')
        print("unique infected cells:", len(infected_cells_new_adjusted_df))
        
        # remove previously infected cells, only keep new ones
        infected_cells_new_adjusted_coordinate_series = pd.Series(zip(infected_cells_new_adjusted_df[0],infected_cells_new_adjusted_df[1]))
        all_infected_cells_coordinate_series = pd.Series(zip(all_infected_cells[0],all_infected_cells[1]))
        shared_cells = infected_cells_new_adjusted_coordinate_series.isin(all_infected_cells_coordinate_series)
        infected_cells_new_adjusted_df = infected_cells_new_adjusted_df[~shared_cells]
        print("new infected cells among those:", len(infected_cells_new_adjusted_df))
        
        # keep a numpy copy for computing the next wave
        infected_cells_new_adjusted_np = infected_cells_new_adjusted_df.to_numpy(dtype=int)
        # sort by infection time
        infected_cells_new_adjusted_np = infected_cells_new_adjusted_np[infected_cells_new_adjusted_np[:,2].argsort()]
        print("max inf time in np:", infected_cells_new_adjusted_np[:,2].max() / 3600)
        
        # add new infected cells to all infected cells table
        all_infected_cells = pd.concat([all_infected_cells, infected_cells_new_adjusted_df])
        print("max inf time in all:", all_infected_cells[2].max() / 3600)
        print("total infected cells:", len(all_infected_cells))

    else: 
        print("no new cells infected")
        infected_cells_new_adjusted_np = pd.DataFrame(infected_cells_new[:,:3].cpu()).to_numpy(dtype=int)

    print("time post processing:", time.time() - start_time, "seconds")

    return infected_cells_new_adjusted_np, all_infected_cells


# %%
def count_viral_load_over_time(record_increment, vir_prod_modifier, infected_cells_new, viral_load_over_time):

    start_time = time.time()

    vir_prod_time = vir_prod_modifier.cpu().numpy()
    vir_exit_time = np.abs(infected_cells_new[:,2].cpu().numpy()) + vir_prod_time

    for time_step in range(len(viral_load_over_time)):
        step_in_sec = time_step * record_increment
        viral_load_over_time[time_step] += (np.all([(vir_prod_time < step_in_sec), (step_in_sec < vir_exit_time)], axis=0)).sum()

    print("time counting viral load:", time.time() - start_time, "seconds")

    return viral_load_over_time


# %%
device = t.device("cuda")

diffusion_coeff = 1.27 # microns per second

infectable_fraction = 0.5 # fraction of infectable cells
infection_prob = 0.2
reflective_boundary = 17 # microns in nasal passage
exit_boundary = 130000 # microns
advection_boundary = 7 # microns in nasal passage
advection_velocity = 146.67  # 146.67 microns per second in nasal passage
periodic_boundary = 50000 # microns
cell_diameter = 4

virion_prod_rate = 42 # 42 per hour i.e. ~1000 per day
end_time = 48 # 72 hours

latency_time = 6 # hours

infectable_block_size = 2**10
infectable_sim_block = t.rand(infectable_block_size*infectable_block_size, device=device) < infectable_fraction
record_increment = 60*10 # record data every this many seconds

ref_bound_cell = reflective_boundary / cell_diameter
exit_bound_cell = exit_boundary / cell_diameter
adv_bound_cell = advection_boundary / cell_diameter
adv_vel_cell = advection_velocity / cell_diameter
prd_bound_cell = periodic_boundary / cell_diameter
diffusion_cell = diffusion_coeff / cell_diameter

# %%
num_steps = end_time * 60 * 60
viral_load_over_time = np.zeros(num_steps//record_increment)

# want each virion to be produced at an integer step
# this gives a close enough exact number of virions produced
vir_prod_interval = (60 * 60 // virion_prod_rate) + 1 # seconds per virion
num_virus = (num_steps // vir_prod_interval) + 1

print(num_virus, num_steps)

# simulate wave 0
infected_cells_wave0 = sim_vir_path_wave0(device, num_virus, num_steps, diffusion_cell, infection_prob, ref_bound_cell, adv_bound_cell, adv_vel_cell, exit_bound_cell, 
                                         prd_bound_cell, infectable_block_size, infectable_sim_block)

vir_prod_modifier = t.tensor([x * vir_prod_interval for x in range(num_virus)], device=device)

viral_load_over_time = count_viral_load_over_time(record_increment, vir_prod_modifier, infected_cells_wave0, viral_load_over_time)

infected_cells_wave0_adjusted_np, all_infected_cells_after_wave0 = inf_cell_wave0(vir_prod_modifier, num_virus, device, num_steps, infected_cells_wave0)

plt.plot(viral_load_over_time)
x_ticks = np.arange(0, len(viral_load_over_time)+1, 6*3600/record_increment)
plt.xticks(x_ticks, x_ticks/(3600/record_increment))
plt.grid(True)
plt.show()

# %%
num_waves = end_time // latency_time
print("number of waves:", num_waves)
memory_cutoff = int(10**8)

infected_cells_old_adjusted_np = infected_cells_wave0_adjusted_np.copy()
all_infected_cells = all_infected_cells_after_wave0.copy()

start_time_total = time.time()
num_virus_total = 0

for wave in range(1,num_waves+1):
# for wave in range(1,3):
    print("wave number:", wave)
    
    start_time_wave = time.time()
    start_time = time.time()
    num_steps, num_virus, vir_prod_each_cell, vir_prod_modifier = secondary_para_later_waves(end_time, latency_time, 
                                                                                             vir_prod_interval, infected_cells_old_adjusted_np, device)
    print("actual time to run function to generate num_virus:", time.time()-start_time, "seconds")
    
    if num_virus==0:
        print("no virions produced. simulation stops.", "\n")
        break
    
    num_virus_total += num_virus

    
    if num_virus > memory_cutoff: 
        infected_cells_new_adjusted_np = np.empty((0,3))
        vir_prod_subtotal = list(itertools.accumulate(vir_prod_each_cell))
        cell_cutoff_old = 0

        start_time_all_batch = time.time()
        for batch in range(num_virus // memory_cutoff):
            start_time_batch = time.time()
            print("batch:", batch+1)
            cell_cutoff_new = list((x > ((batch+1) * memory_cutoff)) for x in vir_prod_subtotal).index(True)
            
            start_time = time.time()
            print("cell cutoff:", cell_cutoff_new)
            infected_cells_new = sim_vir_path_later_waves(memory_cutoff, device, vir_prod_each_cell[cell_cutoff_old:cell_cutoff_new+1], 
                                                          infected_cells_old_adjusted_np[cell_cutoff_old:cell_cutoff_new+1], num_steps, diffusion_cell,
                                                          ref_bound_cell, adv_bound_cell, adv_vel_cell, exit_bound_cell, prd_bound_cell,
                                                          infectable_block_size, infectable_sim_block, infection_prob)
            print("actual time to run function to simulate:", time.time()-start_time, "seconds")
            start_time = time.time()
            viral_load_over_time = count_viral_load_over_time(record_increment, vir_prod_modifier[batch*memory_cutoff:(batch+1)*memory_cutoff], 
                                                              infected_cells_new, viral_load_over_time)
            print("actual time to run function to count viral load:", time.time()-start_time, "seconds")
            start_time = time.time()
            infected_cells_new_adjusted_np_batch, all_infected_cells = inf_cell_later_waves(infected_cells_new,
                                                                                            vir_prod_modifier[batch*memory_cutoff:(batch+1)*memory_cutoff], 
                                                                                            end_time, all_infected_cells) 
            print("actual time to run function to postprocess:", time.time()-start_time, "seconds")
            start_time = time.time()
            infected_cells_new_adjusted_np = np.concatenate((infected_cells_new_adjusted_np, infected_cells_new_adjusted_np_batch), axis=0)
            cell_cutoff_old = cell_cutoff_new
            print("this batch took:", time.time()-start_time_batch, "seconds")
            print("\n")
            
        batch += 1
        print("last batch:", batch+1)
        start_time_batch = time.time()
        start_time = time.time()
        infected_cells_new = sim_vir_path_later_waves(int(num_virus % memory_cutoff), device, vir_prod_each_cell[cell_cutoff_old:], 
                                                      infected_cells_old_adjusted_np, num_steps, diffusion_cell,
                                                      ref_bound_cell, adv_bound_cell, adv_vel_cell, exit_bound_cell, prd_bound_cell,
                                                      infectable_block_size, infectable_sim_block, infection_prob)
        print("actual time to run function to simulate:", time.time()-start_time, "seconds")
        start_time = time.time()
        viral_load_over_time = count_viral_load_over_time(record_increment, vir_prod_modifier[batch*memory_cutoff:], 
                                                          infected_cells_new, viral_load_over_time)
        print("actual time to run function to count viral load:", time.time()-start_time, "seconds")
        start_time = time.time()
        infected_cells_new_adjusted_np_batch, all_infected_cells = inf_cell_later_waves(infected_cells_new, vir_prod_modifier[batch*memory_cutoff:], 
                                                                                        end_time, all_infected_cells) 
        print("actual time to run function to postprocess:", time.time()-start_time, "seconds")
        infected_cells_new_adjusted_np = np.concatenate((infected_cells_new_adjusted_np, infected_cells_new_adjusted_np_batch), axis=0)
        print("this batch took:", time.time()-start_time_batch)
        print("\n")
        print("infected cells for the wave:", len(infected_cells_new_adjusted_np), "\n")
        print("all batches took:", time.time()-start_time_all_batch, "seconds in total")
            
    else:
        start_time_batch = time.time()
        start_time = time.time()
        infected_cells_new = sim_vir_path_later_waves(num_virus, device, vir_prod_each_cell, infected_cells_old_adjusted_np, num_steps, diffusion_cell,
                                                      ref_bound_cell, adv_bound_cell, adv_vel_cell, exit_bound_cell, prd_bound_cell,
                                                      infectable_block_size, infectable_sim_block, infection_prob)
        print("actual time to run function to simulate:", time.time()-start_time, "seconds")
        start_time = time.time()
        viral_load_over_time = count_viral_load_over_time(record_increment, vir_prod_modifier, infected_cells_new, viral_load_over_time)
        print("actual time to run function to count viral load:", time.time()-start_time, "seconds")
        start_time = time.time()
        infected_cells_new_adjusted_np, all_infected_cells = inf_cell_later_waves(infected_cells_new, vir_prod_modifier, end_time, all_infected_cells)

        print("actual time to run function to postprocess:", time.time()-start_time, "seconds")
        print("this only batch took:", time.time()-start_time_batch, "seconds")
        #print("\n")

    if len(infected_cells_new_adjusted_np)==0:
        print("no new cells infected. simulation stops.", "\n")
        break

    infected_cells_old_adjusted_np = infected_cells_new_adjusted_np.copy()
    print("this wave took:", time.time()-start_time_wave, "seconds in total")
    print("\n")

    plt.plot(viral_load_over_time)
    x_ticks = np.arange(0, len(viral_load_over_time)+1, 6*3600/record_increment)
    plt.xticks(x_ticks, x_ticks/(3600/record_increment))
    plt.grid(True)
    plt.show()

print("total time:", time.time() - start_time_total)
print("total infected cells:", len(all_infected_cells))
print("total virions simulated:", num_virus_total)

# %%
all_infected_cells.to_csv("infected_cells.csv", index=False) 
(pd.DataFrame(infected_cells_new.cpu())).to_csv("infected_cells_last_wave.csv", index=False) 
(pd.DataFrame(infected_cells_new_adjusted_np)).to_csv("infected_cells_last_wave_raw.csv", index=False)
(pd.DataFrame(viral_load_over_time)).to_csv("viral_load_over_time.csv", index=False) 

# %%
plt.plot(viral_load_over_time)
x_ticks = np.arange(0, len(viral_load_over_time)+1, 6*3600/record_increment)
plt.xticks(x_ticks, x_ticks/(3600/record_increment))
plt.title("total viral load over 48 hours post infection")
plt.xlabel("hours")
plt.ylabel("total viral load")
plt.grid(True)
plt.show()

# %%
x_ticks = np.arange(0, len(viral_load_over_time)+1, 6*3600/record_increment)

plt.plot(viral_load_over_time)
plt.yscale("log")
plt.xticks(x_ticks, x_ticks/(3600/record_increment))
plt.title("total viral load (log10) over 48 hours post infection")
plt.xlabel("hours")
plt.ylabel("total viral load")
plt.grid(True)
plt.show()

# %%
viral_load_over_time.max()

# %%

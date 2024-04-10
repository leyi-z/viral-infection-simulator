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
def sim_vir_path_wave0(device, num_virus, num_steps, infection_prob, reflective_boundary, advection_boundary, advection_velocity, exit_boundary, 
                       periodic_boundary, cell_diameter, infectable_block_size, infectable_sim_block):

    print("device:", device)
    print("num virus:", num_virus)
    print("num_steps:", num_steps)
    print("infection_prob:", infection_prob)
    print("reflective_boundary:", reflective_boundary)
    print("advection_boundary:", advection_boundary)
    print("advection_velocity:", advection_velocity)
    print("exit_boundary:", exit_boundary)
    print("periodic_boundary:", periodic_boundary)
    print("cell_diameter:", cell_diameter)
    print("infectable_block_size:", infectable_block_size)
    print("infectable_sim_block:", infectable_sim_block)

    start_time = time.time()
    vir_sim = t.zeros(num_virus, 4, device=device) # starting all virions at (0,0,0), 4th column records infection time
    active_mask = t.ones(num_virus, dtype=t.bool,device=device) # initialize, array of true
    infectable_roll = t.empty_like(active_mask, dtype=t.bool) # initialize, empty
    infect_roll = t.empty_like(active_mask, dtype=t.bool) # initialize, empty
    print("time to initialize:", time.time()-start_time, "seconds")

    infectability_time_total = 0
    
    start_time = time.time()
    for step in range(num_steps):  
    
        vir_sim[:,:3] += (t.rand(num_virus,3,device=device)*2-1) * 1.27 * active_mask.reshape(-1,1) # simulates next step
    
        ################
        # reflective upper boundary 
        # TODO: more efficient?
        ################
        reflect_mask = t.zeros(num_virus, dtype=t.bool,device=device)
        reflect_mask = (vir_sim[:,0]>reflective_boundary) & active_mask
        vir_sim[reflect_mask,0] = reflective_boundary*2 - vir_sim[reflect_mask,0]
        ################

        ################
        # apply advection to y if x coordinate \geq 7
        ################
        advection_mask = t.zeros(num_virus, dtype=t.bool,device=device)
        advection_mask = (vir_sim[:,0]>advection_boundary) & active_mask
        vir_sim[advection_mask,1] += advection_velocity
        ################

        ################
        # turn off virions that are flushed out
        ################
        flux_mask = t.zeros(num_virus, dtype=t.bool,device=device)
        flux_mask = (vir_sim[:,1]>exit_boundary) & active_mask
        active_mask ^= flux_mask
        ################

        ################
        # periodic boundary condition
        ################
        # vir_sim[:,2] = vir_sim[:,2] % periodic_boundary
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
        num_encounter = encounter_mask.sum() # number of virions that encountered a cell
        infectability_time_total += time.time() - infectability_time_start
        ################

        if num_encounter>0:
            infect_roll[encounter_mask] = t.rand(size=(num_encounter,),device=device) < infection_prob # for each encounter, roll random number from 0 to 1
            new_infection = encounter_mask & infect_roll
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
# includes starting location of each virion
def sim_vir_path_later_waves(num_virus, device, vir_prod_each_cell, infected_cells_old_adjusted_np, cell_diameter, num_steps, 
                             reflective_boundary, advection_boundary, advection_velocity, exit_boundary, periodic_boundary,
                             infectable_block_size, infectable_sim_block, infection_prob):

    print("starting simulation.")

    print("device:", device)
    print("num virus:", num_virus)
    print("num_steps:", num_steps)
    print("infection_prob:", infection_prob)
    print("reflective_boundary:", reflective_boundary)
    print("advection_boundary:", advection_boundary)
    print("advection_velocity:", advection_velocity)
    print("exit_boundary:", exit_boundary)
    print("periodic_boundary:", periodic_boundary)
    print("cell_diameter:", cell_diameter)
    print("infectable_block_size:", infectable_block_size)
    print("infectable_sim_block:", infectable_sim_block)
    
    start_time = time.time()
    ################
    # initialize virion path with starting locations
    ################
    vir_prod = np.trim_zeros(vir_prod_each_cell, 'b') # ignore cells that didn't produce any virions
    cell_coords_to_vir = (infected_cells_old_adjusted_np[:len(vir_prod),:2] + 0.5) * cell_diameter # convert from cell coord to microns
    vir_sim = t.zeros(num_virus, 4, device=device) # starting all virions at (0,0,0), 4th column records infection time  
    vir_sim[:,1:3] = t.from_numpy(np.repeat(cell_coords_to_vir, vir_prod, axis=0)[:num_virus]) # populate with initial locations
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
    
        vir_sim[:,:3] += (t.rand(num_virus,3,device=device)*2-1) * 1.27 * active_mask.reshape(-1,1) # simulates next step
    
        ################
        # reflective upper boundary 
        # TODO: more efficient?
        ################
        reflect_mask = t.zeros(num_virus, dtype=t.bool,device=device)
        reflect_mask = (vir_sim[:,0]>reflective_boundary) & active_mask
        vir_sim[reflect_mask,0] = reflective_boundary*2 - vir_sim[reflect_mask,0]
        ################

        ################
        # apply advection if y coordinate \geq 7
        # TODO: more efficient?
        ################
        advection_mask = t.zeros(num_virus, dtype=t.bool,device=device)
        advection_mask = (vir_sim[:,0]>advection_boundary) & active_mask
        vir_sim[advection_mask,1] += advection_velocity
        ################

        ################
        # turn off virions that are flushed out
        ################
        flux_mask = t.zeros(num_virus, dtype=t.bool,device=device)
        flux_mask = (vir_sim[:,1]>exit_boundary) & active_mask
        active_mask ^= flux_mask
        ################

        ################
        # periodic boundary condition
        ################
        # vir_sim[:,2] = vir_sim[:,2] % periodic_boundary
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
        num_encounter = encounter_mask.sum() # number of virions that encountered a cell
        infectability_time_total += time.time() - infectability_time_start
        ################
        
        if num_encounter>0:
            infect_roll[encounter_mask] = t.rand(size=(num_encounter,),device=device) < infection_prob # for each encounter, roll random number from 0 to 1
            new_infection = encounter_mask & infect_roll
            active_mask ^= new_infection # remove infected cells from active mask           
            ############
            vir_sim[new_infection,3] = step+1 # record infection time in the 4th column of vir_sim
            ############

        # if step%60==0: # save the location of virions every minute
        #     vir_sim_every_min.append(np.copy(vir_sim.cpu()))
        
        if active_mask.sum() == 0: # number of active virions being simulated at this step
            print("no active virions after:", step, "steps")
            break

        if step%1000==0:
            print("now at step", step, "after", time.time() - start_time, "seconds")
    
    print("simulation time:", time.time() - start_time, "seconds")
    print("checking infecability:", infectability_time_total, "seconds")

    return vir_sim[:,1:]


# %%
def secondary_para_later_waves(end_time, latency_time, wave, vir_prod_interval, infected_cells_old_adjusted_np, device):

    #secondary_para_later_waves(end_time, latency_time, wave, vir_prod_interval, infected_cells_old_adjusted_np, device)

    start_time = time.time()    
    remaining_time = end_time - latency_time * wave
    num_steps = remaining_time * 60 * 60
    
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
def inf_cell_wave0(vir_prod_interval, num_virus, device, infected_cells_wave0, cell_diameter):

    start_time = time.time()

    print(infected_cells_wave0.shape)
    
    # the steps at which each virion is produced
    vir_prod_modifier = t.tensor([x * vir_prod_interval for x in range(num_virus)], device=device)
    vir_prod_modifier = t.unsqueeze(vir_prod_modifier, 1)
    print(vir_prod_modifier.shape)

    # concatenate so later corresponding virions also get deleted from modifier
    infected_cells_wave0 = t.cat((infected_cells_wave0, vir_prod_modifier), 1)
    print(infected_cells_wave0.shape)
    # remove virions that were flushed out, i.e. ones with infection time 0
    infected_cells_wave0 = infected_cells_wave0[infected_cells_wave0[:,2] > 0]    
    # absolute y,z coordinates are converted to cell locations
    infected_cells_wave0[:,:2] = (infected_cells_wave0[:,:2] + cell_diameter/2) // cell_diameter
    # add production times to relative infection times -> absolute infection time
    infected_cells_wave0[:,2] = infected_cells_wave0[:,2] + infected_cells_wave0[:,3]
    
    all_infected_cells = pd.DataFrame({0: [0], 1: [0], 2:[0]}) # start with 1 infected cell at (0,0)
    
    df = pd.DataFrame(infected_cells_wave0[:,:3].cpu())
    for i in 0,1,2:
        df[i] = df[i].astype(int)
    
    # eliminate repeated infections by selecting the minimum infection time
    infected_cells_wave0_adjusted_df = df.groupby([0,1], as_index=False).agg('min')
    print("number of unique infected cells in wave 0:", len(infected_cells_wave0_adjusted_df))
    
    # # remove previously infected cells, only keep new ones
    # infected_cells_wave0_adjusted_coordinate_series = pd.Series(zip(infected_cells_wave0_adjusted_df[0],infected_cells_wave0_adjusted_df[1]))
    # all_infected_cells_coordinate_series = pd.Series(zip(all_infected_cells[0],all_infected_cells[1]))
    # shared_cells = infected_cells_wave0_adjusted_coordinate_series.isin(all_infected_cells_coordinate_series)
    # infected_cells_wave0_adjusted_df = infected_cells_wave0_adjusted_df[~shared_cells]
    # print("number of new infected cells among those:", len(infected_cells_wave0_adjusted_df))
    
    # add new infected cells to all infected cells table
    all_infected_cells_after_wave0 = pd.concat([all_infected_cells, infected_cells_wave0_adjusted_df])
    print("total number of infected cells:", len(all_infected_cells_after_wave0))
    
    # keep a numpy copy for computing the next wave
    infected_cells_wave0_adjusted_np = infected_cells_wave0_adjusted_df.to_numpy(dtype=int)
    # sort by infection time
    infected_cells_wave0_adjusted_np = infected_cells_wave0_adjusted_np[infected_cells_wave0_adjusted_np[:,2].argsort()]

    print("time post processing:", time.time() - start_time, "seconds")
    
    return infected_cells_wave0_adjusted_np, all_infected_cells_after_wave0


# %%
def inf_cell_later_waves(infected_cells_new, cell_diameter, vir_prod_modifier, all_infected_cells):

    start_time = time.time()

    vir_prod_modifier = t.unsqueeze(vir_prod_modifier, 1)
    # concatenate so later corresponding virions also get deleted from modifier
    infected_cells_new = t.cat((infected_cells_new, vir_prod_modifier), 1)
    # remove virions that were flushed out, i.e. ones with infection time 0
    infected_cells_new = infected_cells_new[infected_cells_new[:,2] > 0]    
    # convert infected_cells_new to cell locations
    infected_cells_new[:,:2] = infected_cells_new[:,:2] // cell_diameter
    # add vir_prod_modifier to infected_cells_new
    infected_cells_new[:,2] = infected_cells_new[:,2] + infected_cells_new[:,3]
    
    df = pd.DataFrame(infected_cells_new[:,:3].cpu())
    for i in 0,1,2:
        df[i] = df[i].astype(int)
    
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
    
    # add new infected cells to all infected cells table
    all_infected_cells = pd.concat([all_infected_cells, infected_cells_new_adjusted_df])
    print("total infected cells:", len(all_infected_cells))

    print("time post processing:", time.time() - start_time, "seconds")

    return infected_cells_new_adjusted_np, all_infected_cells


# %%
device = t.device("cuda")

infectable_fraction = 0.5 # fraction of infectable cells
infection_prob = 0.002 # 0.2
reflective_boundary = 17 # microns
exit_boundary = 130000 # microns
advection_boundary = 7 # microns
advection_velocity = 0.01  # 146.67 microns
periodic_boundary = 50000 # microns
cell_diameter = 4

virion_prod_rate = 42 # 42 per hour i.e. ~1000 per day
end_time = 48 # 72 hours

latency_time = 6 # hours

infectable_block_size = 2**10
infectable_sim_block = t.rand(infectable_block_size*infectable_block_size, device=device) < infectable_fraction

# %%
num_steps = end_time * 60 * 60

# want each virion to be produced at an integer step
# this gives a close enough exact number of virions produced
vir_prod_interval = (60 * 60 // virion_prod_rate) + 1 # seconds per virion
num_virus = (num_steps // vir_prod_interval) + 1

print(num_virus, num_steps)

# simulate wave 0
infected_cells_wave0 = sim_vir_path_wave0(device, num_virus, num_steps, infection_prob, reflective_boundary, advection_boundary, advection_velocity, exit_boundary, 
                                         periodic_boundary, cell_diameter, infectable_block_size, infectable_sim_block)

infected_cells_wave0_adjusted_np, all_infected_cells_after_wave0 = inf_cell_wave0(vir_prod_interval, num_virus, device, infected_cells_wave0, cell_diameter)

# %%
num_waves = end_time // latency_time
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
    num_steps, num_virus, vir_prod_each_cell, vir_prod_modifier = secondary_para_later_waves(end_time, latency_time, wave, 
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
                                                          infected_cells_old_adjusted_np[cell_cutoff_old:cell_cutoff_new+1], cell_diameter, num_steps,
                                                          reflective_boundary, advection_boundary, advection_velocity, exit_boundary, periodic_boundary,
                                                          infectable_block_size, infectable_sim_block, infection_prob)
            print("actual time to run function to simulate:", time.time()-start_time, "seconds")
            start_time = time.time()
            infected_cells_new_adjusted_np_batch, all_infected_cells = inf_cell_later_waves(infected_cells_new, cell_diameter, 
                                                                                            vir_prod_modifier[batch*memory_cutoff:(batch+1)*memory_cutoff], 
                                                                                            all_infected_cells) 
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
                                                      infected_cells_old_adjusted_np, cell_diameter, num_steps,
                                                      reflective_boundary, advection_boundary, advection_velocity, exit_boundary, periodic_boundary,
                                                      infectable_block_size, infectable_sim_block, infection_prob)
        print("actual time to run function to simulate:", time.time()-start_time, "seconds")
        start_time = time.time()
        infected_cells_new_adjusted_np_batch, all_infected_cells = inf_cell_later_waves(infected_cells_new, cell_diameter, 
                                                                                        vir_prod_modifier[batch*memory_cutoff:], 
                                                                                        all_infected_cells) 
        print("actual time to run function to postprocess:", time.time()-start_time, "seconds")
        infected_cells_new_adjusted_np = np.concatenate((infected_cells_new_adjusted_np, infected_cells_new_adjusted_np_batch), axis=0)
        print("this batch took:", time.time()-start_time_batch)
        print("\n")
        print("infected cells for the wave:", len(infected_cells_new_adjusted_np), "\n")
        print("all batches took:", time.time()-start_time_all_batch, "seconds in total")
            
    else:
        start_time_batch = time.time()
        start_time = time.time()
        infected_cells_new = sim_vir_path_later_waves(num_virus, device, vir_prod_each_cell, infected_cells_old_adjusted_np, cell_diameter, num_steps, 
                                                      reflective_boundary, advection_boundary, advection_velocity, exit_boundary, periodic_boundary,
                                                      infectable_block_size, infectable_sim_block, infection_prob)
        print("actual time to run function to simulate:", time.time()-start_time, "seconds")
        start_time = time.time()
        infected_cells_new_adjusted_np, all_infected_cells = inf_cell_later_waves(infected_cells_new, cell_diameter, vir_prod_modifier, all_infected_cells)
        print("actual time to run function to postprocess:", time.time()-start_time, "seconds")
        print("this only batch took:", time.time()-start_time_batch, "seconds")
        #print("\n")

    infected_cells_old_adjusted_np = infected_cells_new_adjusted_np.copy()
    print("this wave took:", time.time()-start_time_wave, "seconds in total")
    print("\n")

print("total time:", time.time() - start_time_total)
print("total infected cells:", len(all_infected_cells))
print("total virions simulated:", num_virus_total)

# %%
# 240403-6
all_infected_cells.to_csv("infected_cells.csv", index=False) 
(pd.DataFrame(infected_cells_new.cpu())).to_csv("infected_cells_last_wave.csv", index=False) 
(pd.DataFrame(infected_cells_new_adjusted_np)).to_csv("infected_cells_last_wave_raw.csv", index=False)

# %%

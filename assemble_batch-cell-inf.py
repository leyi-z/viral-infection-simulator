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

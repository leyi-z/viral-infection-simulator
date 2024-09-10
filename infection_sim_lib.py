"""contains all functions needed for simulations"""

import numpy as np
import pandas as pd
import torch as t
import time
import itertools
from typing import Tuple,List



def generate_secondary_parameters(
    end_time:int, 
    latency_time:int, 
    vir_prod_interval:float, 
    infected_cells_old_adjusted_np:np.ndarray, 
    device:t.device,
) -> Tuple[int,int,List[float],t.Tensor]:
    """Computes some values and arrays needed for the upcoming wave of simulations

    Args:
        end_time: total number of hours being simulated
        latency_time: number of hours a cell stays dormant post-infection before releasing virions
        vir_prod_interval: number of seconds between two virions being released by a cell
        infected_cells_old_adjusted_np: position of cells infected in the previous wave
        device: the device to be used by pytorch

    Returns:
         num_steps: total number of seconds simulated in this wave
         num_virus: total number virions produced in this wave
         vir_prod_each_cell: total number of virions produced by each cell infected in the previous wave
         vir_prod_modifier: time each virion is produced, used to offset infection times
    """
    
    start_time = time.time()    
    num_steps = (end_time - latency_time) * 60 * 60
    
    vir_prod_each_cell = [max(0, (num_steps - step) // vir_prod_interval + 1) for step in infected_cells_old_adjusted_np[:,2]]
    num_virus = int(sum(vir_prod_each_cell))
    num_steps = num_steps - np.min(infected_cells_old_adjusted_np[:,2])
    
    print("number of virions:", num_virus)
    print("number of steps:", num_steps)
    print("time to count virions and steps:", time.time()-start_time, "seconds")

    start_time = time.time()    
    # find out the step at which each virion is produced
    vir_subtotal = 0
    print(device)
    vir_prod_modifier = t.zeros(num_virus, device=device)
    
    for i,num in enumerate(vir_prod_each_cell):
        prod_start_time = infected_cells_old_adjusted_np[i, 2] + latency_time * 60 * 60
        vir_prod_modifier[int(vir_subtotal):int(vir_subtotal+num)] += t.tensor([prod_start_time + x*vir_prod_interval for x in range(int(vir_prod_each_cell[i]))], device=device)
        vir_subtotal += num
    print("time to compute when is each virion produced:", time.time()-start_time, "seconds")

    return int(num_steps), int(num_virus), vir_prod_each_cell, vir_prod_modifier



def create_batches_by_memory_cutoff(
    num_virus_wave:int, 
    memory_cutoff:int, 
    vir_prod_each_cell:List[float],
) -> List[list]:
    """Divide a wave of simulation into multiple smaller batches handleable by machine

    Args:
        num_virus_wave: total number of virions to be simulated in this wave
        memory_cutoff: divide this wave of virions into batches smaller than this selected limit
        vir_prod_each_cell: number of virions produced by each cell infected in the previous wave

    Returns:
        batch_config: the configuration (number of cells and number of virions) of each batch to be simulated in this wave
    """

    cell_cutoff_old = 0
    num_virus_subtotal = 0
    batch_config = []
    vir_prod_subtotal = list(itertools.accumulate(vir_prod_each_cell))
    
    while num_virus_subtotal < num_virus_wave:
        cell_cutoff_new = list((x > (num_virus_subtotal + memory_cutoff)) for x in vir_prod_subtotal).index(True)
        num_virus = int(sum(vir_prod_each_cell[cell_cutoff_old:cell_cutoff_new]))
        batch_config.append([cell_cutoff_new, num_virus])
        cell_cutoff_old = cell_cutoff_new
        num_virus_subtotal += num_virus
        if num_virus_wave - num_virus_subtotal < memory_cutoff:
            num_virus = int(sum(vir_prod_each_cell[cell_cutoff_old:]))
            batch_config.append([len(vir_prod_each_cell), num_virus])
            num_virus_subtotal += num_virus
            break
    print("batch configuration:", batch_config)
    return batch_config



# includes starting location of each virion
def simulate_virion_paths(
    num_virus:int, 
    device:t.device, 
    vir_prod_each_cell:List[float], 
    infected_cells_old_adjusted_np:np.ndarray, 
    num_steps:int, 
    diffusion_cell:float, 
    ref_bound_cell:float, 
    adv_bound_cell:float, 
    adv_vel_cell:float, 
    exit_bound_cell:float, 
    prd_bound_cell:float,
    infectable_block_size:float, 
    infectable_sim_block:t.Tensor, 
    infection_prob:float,
) -> t.Tensor:
    """Simulate trajectory of each virion produced by cells infected in the previous wave
    
    Args:
        num_virus: total number of virions to be simulated 
        device: the device to be used by pytorch
        vir_prod_each_cell: number of virions produced by each cell infected in the previous wave
        infected_cells_old_adjusted_np: position of cells infected in the previous wave
        num_steps: number of seconds to be simulated
        diffusion_cell: diffusion coefficient in cell diameter unit
        ref_bound_cell: reflective boundary in cell diameter unit
        adv_bound_cell: advection boundary in cell diameter unit, advection starts at this height 
        adv_vel_cell: advection velocity in cell diameter unit
        exit_bound_cell: exit boundary in cell diameter unit 
        prd_bound_cell: periodic boundary in cell diameter unit
        infectable_block_size: side length of the randomized cell infectability block
        infectable_sim_block: a randomized block simulating which cells are infectable and which are not 
        infection_prob: probablity to infect during virion-cell encounter each second
    
    Returns:
        vir_sim[:,1:]: final locations of all simulated virions, and their exit or infection times
    """

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
    print("number of cells that produced nonzero virions", len(vir_prod))
    print("num virus according to vir_prod", sum(vir_prod))
    vir_initial_coords = infected_cells_old_adjusted_np[:len(vir_prod),:2] + 0.5 # start each virion at center of cells
    vir_sim = t.zeros(num_virus, 4, device=device) # starting all virions at (0,0,0), 4th column records infection time  
    vir_sim[:,1:3] = t.from_numpy(np.repeat(vir_initial_coords, vir_prod, axis=0)) # populate with initial locations
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



def infected_cells_location_and_time(
    infected_cells_new:t.Tensor, 
    vir_prod_modifier:t.Tensor, 
    end_time:int, 
    all_infected_cells:pd.DataFrame,
) -> Tuple[np.ndarray,pd.DataFrame]:
    """Tally new cells infected by virions simulated in the current wave
    
    Args:
        infected_cells_new: final locations of all virions simulated in this wave, and their exit or infection times
        vir_prod_modifier: time each virion is produced, used to offset infection times
        end_time: total number of hours simulated
        all_infected_cells: all cells that have been infected from the start of the entire simulation
    
    Returns:
        infected_cells_new_adjusted_np: cells that are actually infected by virions simulated in this wave
        all_infected_cells: updated record of all infected cells
    """

    start_time = time.time()

    vir_prod_modifier = t.unsqueeze(vir_prod_modifier, 1)
    # concatenate so later corresponding virions also get deleted from modifier
    infected_cells_new = t.cat((infected_cells_new, vir_prod_modifier), 1)
    # remove virions that were flushed out, i.e. ones with infection time <= 0
    infected_cells_new = infected_cells_new[infected_cells_new[:,2] > 0]    
    # add vir_prod_modifier to infected_cells_new
    infected_cells_new[:,2] = infected_cells_new[:,2] + infected_cells_new[:,3]
    # print("max inf time:", infected_cells_new[:,2].max() / 3600)
    # remove cells that were infected past the simulated time
    infected_cells_new = infected_cells_new[infected_cells_new[:,2] <= end_time*60*60]    
    # print("max inf time within simulation time:", infected_cells_new[:,2].max() / 3600)
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
        print("max inf time in np:", infected_cells_new_adjusted_np[:,2].max() / 3600, "hours")
        
        # add new infected cells to all infected cells table
        all_infected_cells = pd.concat([all_infected_cells, infected_cells_new_adjusted_df])
        print("max inf time in all:", all_infected_cells[2].max() / 3600, "hours")
        print("total infected cells:", len(all_infected_cells))

    else: 
        print("no new cells infected")
        infected_cells_new_adjusted_np = pd.DataFrame(infected_cells_new[:,:3].cpu()).to_numpy(dtype=int)

    print("time post processing:", time.time() - start_time, "seconds")

    return infected_cells_new_adjusted_np, all_infected_cells



def count_viral_load_over_time(
    record_increment:int, 
    vir_prod_modifier:t.Tensor,
    infected_cells_new:t.Tensor, 
    viral_load_over_time:np.ndarray,
) -> np.ndarray:
    """Count total number of active virions at each time point

    Args:
        record_increment: record data every this many seconds
        vir_prod_modifier: time each virion is produced, used to offset infection times
        infected_cells_new: final locations of all virions simulated in this wave, and their exit or infection times
        viral_load_over_time: number of active virions at each time point

    Returns:
        viral_load_over_time: updated record of viral load over time
    """

    start_time = time.time()

    vir_prod_time = vir_prod_modifier.cpu().numpy()
    vir_exit_time = np.abs(infected_cells_new[:,2].cpu().numpy()) + vir_prod_time

    for time_step in range(len(viral_load_over_time)):
        step_in_sec = time_step * record_increment
        viral_load_over_time[time_step] += (np.all([(vir_prod_time < step_in_sec), (step_in_sec < vir_exit_time)], axis=0)).sum()

    print("time counting viral load:", time.time() - start_time, "seconds")

    return viral_load_over_time



# TODO: consider moving this function to the class file
def count_cell_inf_over_time(
    record_increment:int, 
    end_time:int, 
    all_infected_cells:pd.DataFrame,
) -> np.ndarray:
    """Count total number of infected cells at each time point

    Args:
        record_increment: record data every this many seconds
        end_time: total number of hours simulated
        all_infected_cells: all cells that have been infected from the start of the entire simulation

    Returns:
        cell_inf_over_time: total number of infected cells at each time point
    """

    start_time = time.time()
    
    num_steps = end_time * 60 * 60
    cell_inf_over_time =  np.zeros(num_steps//record_increment)
    
    for time_step in range(len(cell_inf_over_time)):
            step_in_sec = time_step * record_increment
            cell_inf_over_time[time_step] += (all_infected_cells[2] < step_in_sec).sum()

    print("time counting infected cells:", time.time() - start_time, "seconds")

    return cell_inf_over_time
    

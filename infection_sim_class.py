"""organize parameters and the code for running simulations into a class"""

import numpy as np
import pandas as pd
import torch as t
import time

from infection_sim_lib import *



class InfectionSim:

    def __init__(
        self,
        device = t.device("cuda"),
        memory_cutoff = int(10**8), # max number of virions simulated simultaneously
        seed = 2024, # default seed for simulation
        reflective_boundary = 7+17, # microns in nasal passage
        exit_boundary = 130000, # microns, length of nasal passage
        advection_boundary = 7, # microns in nasal passage, advection starts at this height
        advection_velocity = 146.67,  # 146.67 microns per second in nasal passage
        periodic_boundary = 50000, # microns, circumference of nasal passage
        diffusion_coeff = 1.27, # microns per second
        cell_diameter = 4, # microns
        infectable_fraction = 0.5, # fraction of infectable cells
        infection_prob = 0.2,
        virion_prod_rate = 42, # 42 per hour i.e. ~1000 per day
        end_time = 48, # hours, total number of simulated hours
        latency_time = 6, # hours
        infectable_block_size = 2**10, # side length of the randomized cell infectability block
        record_increment = 60*10, # record data every this many seconds
    ):
        # system parameters
        self.device = device
        self.memory_cutoff = memory_cutoff
        self.record_increment = record_increment
        # convert all lengths into cell diameter unit
        self.cell_diameter = cell_diameter
        self.ref_bound_cell = reflective_boundary / cell_diameter
        self.exit_bound_cell = exit_boundary / cell_diameter
        self.adv_bound_cell = advection_boundary / cell_diameter
        self.adv_vel_cell = advection_velocity / cell_diameter
        self.prd_bound_cell = periodic_boundary / cell_diameter
        self.diffusion_cell = diffusion_coeff / cell_diameter
        # other parameters
        self.infectable_fraction = infectable_fraction
        self.infection_prob = infection_prob
        self.virion_prod_rate = virion_prod_rate
        self.end_time = end_time
        self.latency_time = latency_time

        # set seed for reproducibility
        t.manual_seed(seed)

        # create block pattern for cell infectability to draw from
        self.infectable_block_size = infectable_block_size
        self.infectable_sim_block = t.rand(infectable_block_size * infectable_block_size, device=device) < infectable_fraction
        
        # want each virion to be produced at an integer step
        # this gives a close enough exact number of virions produced
        self.vir_prod_interval = (60 * 60 // virion_prod_rate) + 1 # seconds per virion

    
    def run(self):
        """This is the main function that carries out the simulation
        """

        # compute number of waves
        num_waves = int(self.end_time // self.latency_time)
        print("number of waves:", num_waves, "\n")
        
        # start with 1 infected cell at (y,z)=(0,0) position
        infected_cells_old_adjusted_np = np.array([[0,0,0]]) # initialize "cells infected in previous wave"
        all_infected_cells = pd.DataFrame({0: [0], 1: [0], 2:[0]}) # initialize "all infected cells"

        # initialize array to record and plot viral load later
        viral_load_over_time = np.zeros(self.end_time * 60 * 60//self.record_increment)
        # x_ticks = np.arange(0, len(viral_load_over_time)+1, 6*3600/self.record_increment)

        # start simulation
        start_time_total = time.time() # start recording run time
        num_virus_total = 0 # for sanity check: initialize total number of virions simulated
        
        for wave in range(num_waves):
            print("wave number:", wave)
            
            start_time_wave = time.time()
            start_time = time.time()
            num_steps, num_virus_wave, vir_prod_each_cell, vir_prod_modifier = generate_secondary_parameters(
                self.end_time, self.latency_time, self.vir_prod_interval, 
                infected_cells_old_adjusted_np, self.device
            )
            print("actual time to run function to generate num_virus:", time.time()-start_time, "seconds")
            
            if num_virus_wave==0:
                print("no virions produced. simulation stops.", "\n")
                break
        
            num_virus_total += num_virus_wave # sanity check: total number of virions simulated
        
            if num_virus_wave > self.memory_cutoff: 
        
                batch_config = create_batches_by_memory_cutoff(num_virus_wave, self.memory_cutoff, vir_prod_each_cell)
                
                infected_cells_new_adjusted_np = np.empty((0,3))
                cell_cutoff_old = 0
                num_virus_subtotal = 0
                
                start_time_all_batch = time.time()
                for batch in range(len(batch_config)):
                    start_time_batch = time.time()
                    print("batch:", batch)
                    cell_cutoff_new, num_virus = batch_config[batch]
                    
                    print("cell cutoff new:", cell_cutoff_new)
                    print("num_virus:", num_virus)
                    
                    start_time = time.time()
                    
                    infected_cells_new = simulate_virion_paths(num_virus, self.device, vir_prod_each_cell[cell_cutoff_old:cell_cutoff_new], 
                                                               infected_cells_old_adjusted_np[cell_cutoff_old:cell_cutoff_new], num_steps, self.diffusion_cell,
                                                               self.ref_bound_cell, self.adv_bound_cell, self.adv_vel_cell, self.exit_bound_cell, self.prd_bound_cell,
                                                               self.infectable_block_size, self.infectable_sim_block, self.infection_prob)
                    print("actual time to run function to simulate:", time.time()-start_time, "seconds")
                    start_time = time.time()
                    viral_load_over_time = count_viral_load_over_time(self.record_increment, vir_prod_modifier[num_virus_subtotal:num_virus_subtotal+num_virus], 
                                                                      infected_cells_new, viral_load_over_time)
                    print("actual time to run function to count viral load:", time.time()-start_time, "seconds")
                    start_time = time.time()
                    infected_cells_new_adjusted_np_batch, all_infected_cells = infected_cells_location_and_time(infected_cells_new,
                                                                                                                vir_prod_modifier[num_virus_subtotal:num_virus_subtotal+num_virus], 
                                                                                                                self.end_time, all_infected_cells) 
                    print("actual time to run function to postprocess:", time.time()-start_time, "seconds")
                    start_time = time.time()
                    infected_cells_new_adjusted_np = np.concatenate((infected_cells_new_adjusted_np, infected_cells_new_adjusted_np_batch), axis=0)
                    cell_cutoff_old = cell_cutoff_new
                    num_virus_subtotal += num_virus
                    print("this batch took:", time.time()-start_time_batch, "seconds")
                    print("\n")
                    
                print("infected cells for the wave:", len(infected_cells_new_adjusted_np), "\n")
                print("all batches took:", time.time()-start_time_all_batch, "seconds in total")
                    
            else:
                start_time_batch = time.time()
                start_time = time.time()
                infected_cells_new = simulate_virion_paths(num_virus_wave, self.device, vir_prod_each_cell, infected_cells_old_adjusted_np, num_steps, self.diffusion_cell,
                                                           self.ref_bound_cell, self.adv_bound_cell, self.adv_vel_cell, self.exit_bound_cell, self.prd_bound_cell,
                                                           self.infectable_block_size, self.infectable_sim_block, self.infection_prob)
                print("actual time to run function to simulate:", time.time()-start_time, "seconds")
                start_time = time.time()
                viral_load_over_time = count_viral_load_over_time(self.record_increment, vir_prod_modifier, infected_cells_new, viral_load_over_time)
                print("actual time to run function to count viral load:", time.time()-start_time, "seconds")
                start_time = time.time()
                infected_cells_new_adjusted_np, all_infected_cells = infected_cells_location_and_time(infected_cells_new, vir_prod_modifier, self.end_time, all_infected_cells)
        
                print("actual time to run function to postprocess:", time.time()-start_time, "seconds")
                print("this only batch took:", time.time()-start_time_batch, "seconds")
        
            if len(infected_cells_new_adjusted_np)==0:
                print("no new cells infected. simulation stops.", "\n")
                break
        
            infected_cells_old_adjusted_np = infected_cells_new_adjusted_np.copy()
            print("this wave took:", time.time()-start_time_wave, "seconds in total")
            print("\n")
        
        print("total time:", time.time() - start_time_total)
        print("total infected cells:", len(all_infected_cells))
        print("total virions simulated:", num_virus_total)

        return viral_load_over_time, all_infected_cells

    

    def cell_count(self, all_infected_cells):
        """Count total number of infected cells at each time point
        """

        cell_inf_over_time = count_cell_inf_over_time(self.record_increment, self.end_time, all_infected_cells)

        return cell_inf_over_time





        
        

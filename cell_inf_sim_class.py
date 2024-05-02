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
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from infection_sim_class import InfectionSim
#from infection_sim_lib import count_cell_count_over_time

# %%
sim = InfectionSim(
    end_time=18, 
    virion_prod_rate=5
)
viral_load_over_time, all_infected_cells = sim.run()

# %%
sim.cell_count(all_infected_cells)

# %%
##########
# just for testing, will merge into class later
##########
all_infected_cells.to_csv("results/infected_cells.csv", index=False) 
(pd.DataFrame(viral_load_over_time)).to_csv("results/viral_load_over_time.csv", index=False) 

# %%

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
import torch as t

from infection_sim_class import InfectionSim

# %%
# to run one single realization
sim = InfectionSim(
    end_time=18, 
    virion_prod_rate=5,
    device=t.device("mps")
)
viral_load_over_time, all_infected_cells = sim.run()
cell_inf_over_time = sim.cell_count(all_infected_cells)

# # %%
# # to read parameters from table
# parameter_table = pd.read_csv("parameter_combinations.csv")
# for parameter_id, parameter_combo in parameter_table.iterrows():
#     print(parameter_id)
#     print(parameter_combo)
#     sim = InfectionSim(**parameter_combo)
#     viral_load_over_time, all_infected_cells = sim.run()
#     cell_inf_over_time = sim.cell_count(all_infected_cells)

# %%
all_infected_cells.to_csv("results/infected_cells.csv", index=False)
(pd.DataFrame(viral_load_over_time)).to_csv("results/viral_load_over_time.csv", index=False)
(pd.DataFrame(cell_inf_over_time)).to_csv("results/cell_inf_over_time.csv", index=False)

# %%

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
import pandas as pd
import matplotlib.pyplot as plt

from infection_sim_class import InfectionSim

# %%
sim = InfectionSim(end_time=30, virion_prod_rate=8)

# %%
viral_load_over_time, all_infected_cells = sim.run()

# %%
##########
# just for testing, will merge into class later
##########
all_infected_cells.to_csv("results/infected_cells.csv", index=False) 
(pd.DataFrame(viral_load_over_time)).to_csv("results/viral_load_over_time.csv", index=False) 

# %%
##########
# just for testing, will merge into class later
##########
record_increment = 60*10
x_ticks = np.arange(0, len(viral_load_over_time)+1, 6*3600/record_increment)

plt.plot(viral_load_over_time)
plt.xticks(x_ticks, x_ticks/(3600/record_increment))
plt.title("total viral load over 48 hours post infection")
plt.xlabel("hours")
plt.ylabel("total viral load")
plt.grid(True)
plt.show()

plt.plot(viral_load_over_time)
plt.yscale("log")
plt.xticks(x_ticks, x_ticks/(3600/record_increment))
plt.title("total viral load (log10) over 48 hours post infection")
plt.xlabel("hours")
plt.ylabel("total viral load")
plt.grid(True)
plt.show()

# %%

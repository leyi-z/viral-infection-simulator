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
from infection_sim import InfectionSim

# %%
latency_times = [5,6,7]
viral_loads_by_time = []
for latency_time in latency_times:
    sim = InfectionSim(end_time=16, latency_time=latency_time)
    viral_loads_by_time.append(sim.run())

# %%
[vl.max() for vl in viral_loads_by_time]

# %%
plot_viral_load()

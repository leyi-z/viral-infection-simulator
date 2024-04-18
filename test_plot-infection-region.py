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
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# %%
# corr = np.corrcoef(np.random.randn(10, 200))
# mask = np.zeros_like(corr)
# mask[np.triu_indices_from(mask)] = True
# g = sns.heatmap(corr, mask=mask, vmax=.3, square=True)
# g.set_facecolor('xkcd:salmon')

# %%
df = pd.read_csv("../sim_results/infected_cells_240417.csv")
df = df.sort_values('2')#[:10]
df

# %%
(df["2"] < 3600).sum()

# %%
y = df["0"]
z = df["1"]
t = df["2"]/3600

# %%
end_time = 48 # 72 hours
record_increment = 60*10

num_steps = end_time * 60 * 60
cell_inf_over_time =  np.zeros(num_steps//record_increment)

for time_step in range(len(cell_inf_over_time)):
        step_in_sec = time_step * record_increment
        cell_inf_over_time[time_step] += (df["2"] < step_in_sec).sum()

print("peak infected cells count:", cell_inf_over_time.max())

# %%
x_ticks = np.arange(0, len(cell_inf_over_time)+1, 6*3600/record_increment)

plt.plot(cell_inf_over_time)
plt.xticks(x_ticks, x_ticks/(3600/record_increment))
plt.title("infected cells count over 48 hours post infection")
plt.xlabel("hours")
plt.ylabel("infected cells count")
plt.grid(True)
plt.show()

# %%
x_ticks = np.arange(0, len(cell_inf_over_time)+1, 6*3600/record_increment)

plt.plot(cell_inf_over_time+1)
plt.yscale('log')
plt.xticks(x_ticks, x_ticks/(3600/record_increment))
plt.title("infected cells count (log10) over 48 hours post infection")
plt.xlabel("hours")
plt.ylabel("infected cells count")
plt.grid(True)
plt.show()

# %%
# np.quantile([1,2,3,4,5],0.05)
# lambda a : np.quantile(a,0.05)

# %%
# Create heatmap plot
plt.figure(figsize=(12,9))
plt.hexbin(y,z, C=t, cmap='viridis', reduce_C_function=np.min)
plt.colorbar(label='Hours')  # Add colorbar with label
plt.xlabel('Y')
plt.ylabel('Z')
plt.title('Minimum Infection Times Over 48 Hours')
plt.show()

# %%

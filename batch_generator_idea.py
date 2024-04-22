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
# in:

memory_cutoff
num_virus
vir_prod_each_cell

# out:
# batch indices / cell cutoffs / w/e

# %%
import itertools
import numpy as np

# %%
np.array(list(itertools.accumulate(vir_prod_each_cell[7:]))) >=memory_cutoff

# %%
memory_cutoff=100
vir_prod_each_cell = np.array([30,10,25,66,12,13,30,10,25,66])
num_virus = sum(vir_prod_each_cell)
vir_prod_subtotal = np.array(list(itertools.accumulate(vir_prod_each_cell)))
np.where(vir_prod_subtotal>=memory_cutoff)[0]


# %%
def batch_generator(
    memory_cutoff,
    vir_prod_each_cell,
):
    num_virus = sum(vir_prod_each_cell)
    cell_index_start = 0
    while True:
        num_cells = np.where( np.array(list(itertools.accumulate(vir_prod_each_cell[cell_index_start:]))) >=memory_cutoff)[0][0]
        cell_index_end = cell_index_start + num_cells
        yield cell_index_start, cell_index_end
        cell_index_start = cell_index_end


# %%
for cell_index_start, cell_index_end in batch_generator(100,vir_prod_each_cell = np.array([30,10,25,66,12,13,30,10,25,66])):
    print(cell_index_start, cell_index_end)

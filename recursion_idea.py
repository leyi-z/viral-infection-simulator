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
import itertools
import numpy as np

memory_cutoff=100
vir_prod_each_cell = np.array([30,10,25,66,12,13,30,10,25,66])
num_virus = sum(vir_prod_each_cell)
vir_prod_subtotal = np.array(list(itertools.accumulate(vir_prod_each_cell)))


# %%
def run_batchwise(memory_cutoff, vir_prod_each_cell, results=None):

    if results=None:
        # initialize results ...
    
    num_virus = sum(vir_prod_each_cell)
    vir_prod_subtotal = np.array(list(itertools.accumulate(vir_prod_each_cell)))
    num_virus_for_this_batch = vir_prod_each_cell[vir_prod_subtotal<memory_cutoff].sum()

    # simulate the first num_virus_for_this_batch

    vir_prod_each_cell_remaining = vir_prod_each_cell[vir_prod_subtotal>=memory_cutoff]

    if len(vir_prod_each_cell_remaining) > 0:
        results = run_batchwise(memory_cutoff, vir_prod_each_cell_remaining, results)
    return results


# %%
vir_prod_subtotal

# %%
vir_prod_each_cell[vir_prod_subtotal<memory_cutoff].sum()

# %%
vir_prod_each_cell[vir_prod_subtotal>=memory_cutoff]

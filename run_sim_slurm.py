import os
import click
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from infection_sim_class import InfectionSim

results_folder = 'results/'
output_folder = results_folder + 'output/'

parameter_table = pd.read_csv("parameter_combinations.csv")

@click.command()
@click.option('--parameter_id', prompt='parameter id', help='which parameter combination to use')
@click.option('--realization', prompt='realization', help='which realization of the given parameter combination')
@click.option('--seed', prompt='seed', help='for reproducing simulation results')
def run_sim(parameter_id, realization, seed):
    """run one realization of the simulation given the parameter combination"""

    parameters = parameter_table.loc[int(parameter_id)].to_dict()
    output_suffix = "_para_id_{0}_rlzt_{1}".format(parameter_id, realization)

    # to run one single realization
    sim = InfectionSim(**parameters, seed=seed)
    viral_load_over_time, all_infected_cells = sim.run()
    cell_inf_over_time = sim.cell_count(all_infected_cells)

    # save simulation outputs
    (pd.DataFrame(viral_load_over_time)).to_csv(output_folder + "viral_load" + output_suffix, index=False)
    (pd.DataFrame(cell_inf_over_time)).to_csv(output_folder + "cell_inf" + output_suffix, index=False)

if __name__ == '__main__':
    run_sim()
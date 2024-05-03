import click
import pandas as pd

parameter_table = pd.read_csv("parameter_combinations.csv")

def bleh(latency_time=2, virion_prod_rate=2, advection_velocity=1):
    print("bleh")
    print("latency_time:", latency_time)
    print("virion_prod_rate:", virion_prod_rate)
    print("advection_velocity:", advection_velocity)

@click.command()
@click.option('--parameter_id', prompt='parameter id', help='which parameter combination to use')
@click.option('--realization', prompt='realization', help='which realization of the given parameter combination')
def run_sim(parameter_id, realization):
    """run one realization of the simulation given the parameter combination"""
    parameters = parameter_table.loc[int(parameter_id)].to_dict()
    print("parameter_id:", parameter_id)
    print("parameters", parameters)
    print("realization:", realization)
    bleh(**parameters)

if __name__ == '__main__':
    run_sim()
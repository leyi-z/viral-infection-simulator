import os
import subprocess
import pandas as pd

##################
# the total number of realizations for each parameter combo
##################
num_realization = 1
##################

# folder where where all results are stored
results_folder = 'results/'
os.makedirs(results_folder, exist_ok=True)

# create a folder inside results folder to store output files
output_folder = results_folder + 'output/'
os.makedirs(output_folder, exist_ok=True)

# find total number of parameter combinations
parameter_table = pd.read_csv("parameter_combinations.csv")
num_parameter_id = len(parameter_table.index)

for parameter_id in range(num_parameter_id):
    for realization in range(num_realization):
        job_name = "p{0}-r{1}".format(parameter_id, realization)
        
        subprocess.run(['sbatch', '-J', job_name, './job_submit_template.sh', str(parameter_id), str(realization)])

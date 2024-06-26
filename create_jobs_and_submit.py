import os
import csv
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

# create a csv file for recording job ids and seeds
submit_record_file = results_folder+'submission_record.csv'
fieldnames = ['parameter_id', 'realization', 'job_id', 'seed']
with open(submit_record_file, 'w') as csvfile:
    csv.DictWriter(csvfile, fieldnames=fieldnames).writeheader()

# record git commit hash for better reproducibility
git_hash = subprocess.run(['git', 'log', '-1', '--format="%H"'], capture_output=True, text=True)
with open("git_commit_hash.txt", "w") as txtfile:
    txtfile.write(git_hash.stdout)

# find total number of parameter combinations
parameter_table = pd.read_csv("parameter_combinations.csv")
num_parameter_id = len(parameter_table.index)

for parameter_id in range(num_parameter_id):
    for realization in range(num_realization):
        # define random seed for reproducibility
        seed = int.from_bytes(os.urandom(3), "big")

        job_name = "p{0}-r{1}".format(parameter_id, realization)
        job_submit = ['sbatch', '-J', job_name, './job_submit_template.sh', str(parameter_id), str(realization), str(seed)]
        
        job_submit_output = subprocess.run(job_submit, capture_output=True, text=True)

        # get the job id
        # TODO: maybe there's a better way to do this?
        job_id = job_submit_output.stdout.split()[-1]
        # record info about this job
        with open(submit_record_file, 'a') as csvfile:
            csv.writer(csvfile).writerow([parameter_id, realization, job_id, seed])

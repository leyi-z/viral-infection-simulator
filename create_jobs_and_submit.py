import os
import subprocess

parameter_id = 0
num_realization = 25
print("num_realization:", num_realization)


subprocess.run(['bash','./job_submit_template.sh',str(parameter_id),str(num_realization)])

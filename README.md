# Read Me (or a better title)

This is an agent-based viral infection simulator, set up to model and simulate the progression of SARS-CoV-2 infection in the human nasal passage.


<br>

## Files

The file `infection_sim_lib.py` contains functions used in the simulations. They are used in the file `infection_sim_class.py` which defines the structure of the model and simulations.

The file `run_sim_local.py` is used to run simulations locally on your own computer. 

The following three files are used to run simulations on the UNC computer cluster Longleaf: \
`create_jobs_and_submit.py`is the file the user will run to setup and submit the simulations jobs on Longleaf to run. It uses `job_submit_template.sh` as template to submit the simulation job to SLURM which is the task manager used by Longleaf. In the submitted job, the file `run_sim_slurm.py` is run to carry out the simulations using user specified paramter values from the table `parameter_combinations.csv`.

The file `plot_data.py` is used to create some of the plots. Feel free to modify for the plots you need.

The file `check_jobs_and_resubmit.py` is used to check the complete status of simulations jobs on Longleaf and resubmit if the initial job fails.


<br>

## Running simulations locally on your own computer

#### Set up environment

Start by opening a new Terminal (or Powershell/GitBash for Windows) window and navigating to the directory containing this code base. **For Windows users, you might need to install Python first.**

If you prefer to use this code base in a contained environment, you could create a Python virtual environment by running
```sh
python -m venv <venv_folder_name>
```
Activate the venv with the following command
```sh
# for Linux and MacOS users
. <venv_folder_name>/bin/activate

# for Windows users
.\<venv_folder_name>\Scripts\activate
```
The text file `requirements.txt` contains all python packages required to run simulations locally. Run the following command to install them
```sh
pip install -r requirements.txt
```

#### GPU and PyTorch

The simulations are intended to be run mostly on GPU using PyTorch. 

In the `run_sim_local.py` file, the currently selected GPU is set to MPS for my Macbook, which overrides the default CUDA in the `infection_sim_class.py`. 

Check the hardware of your computer to determine. If it has MPS, then no need to change anything. If it has CUDA configured, then you can either delete the line `device=t.device("mps")` or simply replace `mps` by `cuda` in `run_sim_local.py`. 

See [this page](https://pytorch.org/docs/stable/tensor_attributes.html#torch-device) for more information on device selection.

If you would like to verify that everything works, open Python in your Terminal and try the following commands
```python
import torch
device = torch.device("<your_device_name>")
torch.rand(3,4,device=device)
```

**Alternatively**, if your computer doesn't have a GPU or if you are having trouble setting it up, you could totally run the code on CPU only by changing `device=t.device("mps")` to `device=t.device("cpu")` in `run_sim_local.py`. Although the simulations might take a bit longer depending on the chosen paramter values.

#### Run simulations
Now you are ready to run simulations! To run one single simulation locally, simply use the command below. You might need to replace `python3` by `python`.
```sh
python3 run_sim_local.py 
```
The default paramter values are defined in `infection_sim_class.py` and you can override them by specifying the new parameter values in `run_sim_local.py`.

However, you might want to test a number of paramter values. In that case, comment out the single-simulation block and uncomment the read-from-csv block below. Then it will read paramter values from the file `parameter_combinations.csv`. 


<br>

## Running simulations on Longleaf

[under construction]

import os
import numpy as np
import argparse
import mdevaluate as md
import re

from multiprocessing import Pool
from scipy.optimize import curve_fit
from scipy.special import gamma
from functools import partial


#----------------------------functions-------------------------------
def main():
    """
    Main is the only function directly run by this script.
    Other functions are called by this main function.
    All other used functions are alphabethically ordered below.
    This script can be run through bash command with --system_dir path to run in given path.
    If no --system_dir is given, manual_path is used instead.
    """

    manual_path = "/data/bfuesser/FHEC_local/DoubleFHEC/25_07_07/no_bond/temperature_variation"
    manual_axis = "xy"
    manual_atm_names = ["Li", "OW", "HW"]

    # Initializing Parsing
    parser = argparse.ArgumentParser()

    # First argument to be parsed
    parser.add_argument(
        "--system_dir",
        type=str,
        default=manual_path,
        help="Path to the System"
    )

    # Second argument to be parsed
    parser.add_argument(
        "--axis",
        type=str,
        choices=["x", "y", "z", "xy", "xz", "yz", "all"],
        default=manual_axis,
        help="Axis to consider regarding msd calculations"
    )

    # Third argument to be parsed
    parser.add_argument(
        "--atom_names",
        nargs='+',
        default=manual_atm_names,
        help="List of atoms names to calculate self van hove for"
    )

    # Collect parsed arguments
    args = parser.parse_args()
    grandparent_dir = args.system_dir
    axis = args.axis
    atm_names = args.atom_names

    collect_data_dir = os.path.join(grandparent_dir, 'collected_data_plots/data')
    os.makedirs(collect_data_dir, exist_ok=True)

    # Search for directories to work in
    simulations_dir_list = sorted(get_dir_list(grandparent_dir))
    # simulations_dir_list = simulations_dir_list[15:]
    print(f"Simulations on which analysis will be performed: {simulations_dir_list}")


    for work_dir in simulations_dir_list:

        import_path = os.path.join(work_dir, '03_prod')


        meta_data = {}
        meta_data["axis"] = axis
        meta_data["temp"] = get_temp(import_path)

        # Extract slitsize from directory name
        work_dir_basename = os.path.basename(work_dir)
        if work_dir_basename.startswith("slit"):
            match = re.search(r'slitsize_(\d+p\d+)', work_dir)
            meta_data["slitsize"] = match.group(1) if match else "unknown"
        else:
            meta_data["slitsize"] = work_dir_basename

        trajectory = md.open(import_path, nojump=True, topology='FHEC_prod.tpr')

        sub_trjs = []

        for atm_name in atm_names:
            if atm_name == "OW" or atm_name == "HW" or atm_name == "HW1" or atm_name == "HW2":
                residue_name = "SOL"
            else:
                residue_name = "FHEC"
            trj = trajectory.subset(atom_name=atm_name, residue_name=residue_name).nojump
            sub_trjs.append(trj)

        # sub_trjs = [
        #     trajectory.subset(atom_name="OW", residue_name="SOL").nojump,
        #     trajectory.subset(atom_name="Li", residue_name="FHEC").nojump,
        #     trajectory.subset(atom_name="HW", residue_name="FHEC").nojump
        # ]

        # Here everything is happening...
        process_subtrajectory(sub_trjs, atm_names, collect_data_dir, meta_data)
    return None


def get_dir_list(dir):
    """
    Search algorithm for folders which contain 03_prod/FHEC_prod.gro.
    This indicates the simulation has finished and therefore be able to be analysed.
    It returns a list with all parent folder directories for 03_prod/FHEC_prod.gro.

    Parameters:
        dir (str): A given directory on wich the search will be performed

    Returns:
        list: A list of paths to directories which contain simulation data.
    """

    simulations_dirs_list = []
    for root, dirs, files in os.walk(dir):
        for dir in dirs:
            if dir == "03_prod" and os.path.exists(os.path.join(root, dir, "FHEC_prod.gro")):
                simulation_dirs = os.path.dirname(os.path.join(root, dir))
                simulations_dirs_list.append(os.path.dirname(os.path.join(root, dir)))
    return simulations_dirs_list


def get_mdp(directory):
    """
    Search for .mdp file which is not mdout.mdp and return it.
    Code can be used on 02_eq and 03_prod with this approach.
    Returns the full path of the .mdp file in directory

    Parameters:
        directory (str): Any directory which contains a .mdp file.

    Returns:
        str: Full path of .mdp file.
    """

    files = os.listdir(directory)
    for file in files:
        if file.endswith('.mdp') and not file.startswith('mdout'):
            mdb_file = os.path.join(directory, file)
            break
    return mdb_file


def get_temp(directory):
    """
    Uses get_mdp to get .mdp file.
    Performs a line search for temperature and returns it as an int.

    Parameters:
        directory (str): Any directory which contains a .mdp file.

    Returns:
        int: Temperature as an integer.
    """

    mdp_file = get_mdp(directory)
    with open(mdp_file, 'r') as file:
        for line in file:
            if line.strip().startswith('ref-t'):
                parts = line.split('=')
                if len(parts) > 1:
                    temperature = float(parts[1].strip())
                break
    return int(temperature)


def get_van_hove_self(args):
    """
    function to calculate self part of van hove.

    Parameter:
        args (list): contain all of the following objects:
            sub_trjs (list): contains md evaluate trajectory object
            collect_data_dir (str): directory were data will be saved
            meta_data_local (dict): dictionary containing keys: 'temp' and 'atm_name'
    """

    sub_trj, collected_data_dir, meta_data_local = args
    axis = meta_data_local["axis"]
    bins = np.linspace(0, 5, 501)
    time, van_hove_self = md.correlation.shifted_correlation(partial(md.correlation.van_hove_self, bins=bins, axis=axis), sub_trj, segments=1000, skip=0.1, average=True)

    meta_data_local['export_data'] = f"{axis}_vhs"

    # Construct file name and path
    data_dir = os.path.join(collected_data_dir, meta_data_local['export_data'], meta_data_local['slitsize'], str(meta_data_local['temp']), meta_data_local['atm_name'])
    os.makedirs(data_dir, exist_ok=True)

    # Export data into seperated folder to be loaded later
    for index in range(1, len(time)):
        data = np.array([van_hove_self[index]])
        file_name = f"vhs_{index:03}.dat"
        full_path = os.path.join(data_dir, file_name)
        np.savetxt(full_path, data)

    # Export time steps into same folder
    file_name = "time.dat"
    full_path = os.path.join(data_dir, file_name)
    np.savetxt(full_path, time[1:])
    file_name = "bins.dat"
    full_path = os.path.join(data_dir, file_name)
    np.savetxt(full_path, np.asarray(bins))
    print(f"vhs saved in {data_dir}!")

    return None


def process_subtrajectory(sub_trjs, atm_names, collect_data_dir, meta_data):
    """
    Allocated tasks with their own arguments for multiple agents to perform multiprocessed calculation.
    Tasks are defined in process_task().

    Parameters:
        sub_trjs (list of trajectory object from md evaluate): Trajectories to perform calculations on. Must have the same lengh as atm_names and matching order.
        atm_names (list of str): Contains atom names which are alocated to trajectories in sub_trjs. Must have the same lengh as sub_trjs and matching order.
        collect_data_dir (str): Directory where data will be saved.
        meta_data (dict): Dictionary containing keys 'temp' and 'atm_name'
    """
    num_tasks = 1
    num_sub_trjs = len(sub_trjs)
    num_processes = num_tasks * num_sub_trjs
    args_list = []
    for idx in range(0, len(sub_trjs)):
        task_meta_data = meta_data.copy()
        task_meta_data["atm_name"] = atm_names[idx]
        args_list.append((sub_trjs[idx], collect_data_dir, task_meta_data))

    with Pool(processes=num_processes) as pool:
        pool.map(get_van_hove_self, args_list)

    return None


#----------------------------main-function---------------------------

main()
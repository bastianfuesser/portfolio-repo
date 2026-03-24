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
    manual_axis = "all"
    manual_atm_names = ["ob"]

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
        help="List of atoms names to calculate msd and isf for"
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
    # simulations_dir_list = simulations_dir_list[1]
    print(f"Simulations on which analysis will be performed: {simulations_dir_list}")


    for work_dir in simulations_dir_list:

        import_path = os.path.join(work_dir, '03_prod')


        meta_data = {}
        meta_data["temp"] = get_temp(import_path)
        meta_data["axis"] = axis

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

        # Here everything is happening...
        process_subtrajectory(sub_trjs, atm_names, collect_data_dir, meta_data)
        # break
    return None


def export_data(data_array, path, meta_data):
    """
    Exports data to a .dat file in the specific path given.
    Meta_data will generate the file name.

    Parameters:
        data_array (list or np.ndarray): Data to be exported.
        path (str): Directory path where data will be saved.
        meta_data (dict): Dictionary containing keys 'export_data', 'temp', 'slitsize' and 'atm_name'
    """

    # Convert data into np.ndarray if not already correctly formatted
    if not isinstance(data_array, np.ndarray):
        if isinstance(data_array, (float, int)):
            data_array = np.array([data_array])
        else:
            data_array = np.asarray(data_array)


    # Construct file name and path
    data_dir = os.path.join(path, meta_data['export_data'], meta_data['slitsize'])
    os.makedirs(data_dir, exist_ok=True)

    base_name = f"{meta_data['export_data']}_{meta_data['temp']}_{meta_data['atm_name']}"
    dat_path = os.path.join(data_dir, base_name + ".dat")

    # Export .dat file
    np.savetxt(dat_path, data_array)

    return None


def get_diffusion_2d(trajectory, collect_data_dir, meta_data):
    """
    Here MSD gets calculated along the xy axes.
    Afterwards 2d-diffusion is fitted and all data is saved in the given folder.

    Parameters:
        trajectory (trajectory object from md evaluate): Data to be processed
        collect_data_dir (str): Directory where data will be saved
        idx (int): Index for atm_name
        meta_data (dict): Dictionary containing keys 'temp', 'atm_name' and 'slitsize'
    """

    axis = meta_data["axis"]
    time, msd = md.correlation.shifted_correlation(partial(md.correlation.msd, axis=axis), trajectory, segments=1000, skip=0.1, average=True)

    def diffusion_2d(t, D):
        return np.log(4 * D * t)

    if axis == 'xy':
        try:
            # Get indices of the three highest MSD values
            top_indices = np.argsort(msd)[-3:]  # Indices of the three highest values
            top_indices = np.sort(top_indices)  # Sort indices to maintain order in time

            # Select corresponding time and msd values
            time_top = time[top_indices]
            msd_top = msd[top_indices]

            # Fit the function to the selected points
            popt, pcov = curve_fit(diffusion_2d, time_top, np.log(msd_top), maxfev=10000, bounds=(0, np.inf))
            diffusion_value = popt[0]
            diffusion_std = np.sqrt(np.diag(pcov))[0]
            diffusion = (diffusion_value, diffusion_std)

            list_data = np.array([time[1:], msd[1:], np.exp(diffusion_2d(time[1:], *popt))])

            # Data export, one for the diffusion coefficient and one for the msd data
            meta_data["export_data"] = "diff"
            export_data(diffusion, collect_data_dir, meta_data)
            meta_data["export_data"] = f"{axis}_msd"
            export_data(list_data.T, collect_data_dir, meta_data)
            print(f"Temp {meta_data['temp']}, {meta_data['slitsize']}, {meta_data['atm_name']} successful finished diffusion export!")
            return None
        except Exception as e:
            print(f"An error occurred: {e}")
            meta_data["export_data"] = f"{axis}_msd"
            list_data = np.array([time[1:], msd[1:]])
            print("Data saved without fit data as fit not working!")
            export_data(list_data.T, collect_data_dir, meta_data)
    else:
            meta_data["export_data"] = f"{axis}_msd"
            list_data = np.array([time[1:], msd[1:]])
            export_data(list_data.T, collect_data_dir, meta_data)
            print("Data saved without diffusion as choosen axis is not 2D!")


    return None


def get_isf(trajectory, collect_data_dir, meta_data):
    """
    Here ISF gets calculated along all axes.
    Afterwards mean tau and KWW=1/e (time) is calculated and saved in the given data_path.

    Parameters:
        trajectory (trajectory object from md evaluate): Data to be processed
        collect_data_dir (str): Directory where data will be saved
        idx (int): Index for atm_name
        meta_data (dict): Dictionary containing keys 'temp', 'atm_name' and 'slitsize'
    """

    # mdevaluate is used to calculate ISF values with bin=segments.
    time, isf = md.correlation.shifted_correlation(partial(md.correlation.isf, q=22.7), trajectory, segments=1000, average=True)
    mask = time > 3e-1
    try:
        fit, cov= curve_fit(md.functions.kww, time[mask], isf[mask], maxfev=10000)
        # bounds = ([0, 0, 0], [np.inf, np.inf, np.inf])
        # fit, cov= curve_fit(md.functions.kww, time[mask], isf[mask], maxfev=10000, bounds=bounds)

        tau = fit[1]
        beta = fit[2]

        print(fit[0])
        print(fit[1])
        print(fit[2])

        # Calculate uncertainty in tau and beta
        tau_err1 = np.sqrt(cov[1, 1])
        beta_err1 = np.sqrt(cov[2, 2])

        # Calculate mean_tau
        mean_relaxation_time = np.array([(tau / beta) * gamma(1 / beta)])
        t_1e = md.functions.kww_1e(fit[0], fit[1], fit[2])
        t_1e = np.array([t_1e])

        print(t_1e)
        print(mean_relaxation_time)

        # Prepare data to be saved in dat file
        list_data = np.array([time[1:], isf[1:], md.functions.kww(time[1:], *fit)])

        # Data export
        meta_data["export_data"] = "isf"
        export_data(list_data.T, collect_data_dir, meta_data)
        meta_data["export_data"] = "mean_tau"
        export_data(mean_relaxation_time, collect_data_dir, meta_data)
        meta_data["export_data"] = "kww1e"
        export_data(t_1e, collect_data_dir, meta_data)
        print(f"Temp {meta_data['temp']}, {meta_data['slitsize']}, {meta_data['atm_name']} successful finished ISF export!")

        return None

    except Exception as e:
        print(f"An error occurred: {e}")
        meta_data["export_data"] = "isf"
        list_data = np.array([time[1:], isf[1:]])
        print("Data saved without fit data as fit not working!")
        export_data(list_data.T, collect_data_dir, meta_data)

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


def process_subtrajectory(sub_trjs, atm_names, collect_data_dir, meta_data):
    """
    Allocated tasks with their own arguments for multiple agents to perform multiprocessed calculation.
    Tasks are defined in process_task().

    Parameters:
        sub_trjs (list of trajectory object from md evaluate): Trajectories to perform calculations on. Must have the same lengh as atm_names and matching order.
        atm_names (list of str): Contains atom names which are alocated to trajectories in sub_trjs. Must have the same lengh as sub_trjs and matching order.
        collect_data_dir (str): Directory where data will be saved.
        meta_data (dict): Dictionary containing keys 'temp' and 'slitsize'
    """

    task_types = ["get_isf", "get_diffusion_2d"]

    runner_num = len(task_types) * len(atm_names)

    args_list = []
    for idx in range(0, len(sub_trjs)):
        task_meta_data = meta_data.copy()
        task_meta_data["atm_name"] = atm_names[idx]
        for task in task_types:
            args_list.append((sub_trjs[idx], collect_data_dir, task, task_meta_data))

    with Pool(processes=runner_num) as pool:
        pool.map(process_task, args_list)

    return None


def process_task(args):
    """
    Define tasks for calculating isf and msd including mean tau, kww=1/e and diffusion. Diffusion is only calculated if choosen axis is xy!

    Parameters:
        args (list): Includes all necessaray parameters for tasks to run. The args list must contain the following objects in excaly this order:
            sub_trj (trajectory object from md evaluate): Data to be processed
            collect_data_dir (str): Directory where data will be saved.
            task_type (str): Given task to be performed on the data set.
            idx (int): Index for atm_name
            meta_data (dict): Dictionary containing keys 'temp', 'atm_name' and 'slitsize'
    """

    sub_trj, collect_data_dir, task_type, meta_data = args

    if task_type == "get_isf":
        get_isf(sub_trj, collect_data_dir, meta_data)
    elif task_type == "get_diffusion_2d":
        get_diffusion_2d(sub_trj, collect_data_dir, meta_data)

    return None

#----------------------------main-function---------------------------

main()
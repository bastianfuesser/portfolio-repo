import os
import numpy as np
import argparse
import mdevaluate as md
import re
import sys

from multiprocessing import Pool


def main():
    """
    Main is the only function directly run by this script.
    Other functions are called by this main function.
    All other used functions are alphabethically ordered below.
    This script can be run through bash command with --system_dir path to run in given path.
    If no --system_dir is given, manual_path is used instead.
    """

    manual_path = "/data/bfuesser/FHEC_local/DoubleFHEC/25_07_07/no_bond/temperature_variation"
    manual_atm_names = ["HW", "OW", "Li", "whole"] # "whole" is possible
    number_bins_x = 100

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
        "--atom_names",
        nargs='+',
        default=manual_atm_names,
        help="List of atoms names to calculate msd and isf for"
    )

    # Collect parsed arguments
    args = parser.parse_args()
    grandparent_dir = args.system_dir
    atm_names = args.atom_names

    collect_data_dir = os.path.join(grandparent_dir, 'collected_data_plots/data')
    os.makedirs(collect_data_dir, exist_ok=True)

    # Search for directories to work in
    simulations_dir_list = sorted(get_dir_list(grandparent_dir))
    # simulations_dir_list = simulations_dir_list[:2]   #debug
    print(f"Simulations on which analysis will be performed: {simulations_dir_list}")


    for work_dir in simulations_dir_list:

        import_path = os.path.join(work_dir, '03_prod')


        meta_data = {}
        meta_data["temp"] = get_temp(import_path)

        # Extract slitsize from directory name
        work_dir_basename = os.path.basename(work_dir)
        if work_dir_basename.startswith("slit"):
            match = re.search(r'slitsize_(\d+p\d+)', work_dir)
            meta_data["slitsize"] = match.group(1) if match else "unknown"
        else:
            meta_data["slitsize"] = work_dir_basename

        trajectory = md.open(import_path, nojump=False, topology='FHEC_prod.tpr')

        sub_trjs = []

        for atm_name in atm_names:
            if atm_name == "OW" or atm_name == "HW" or atm_name == "HW1" or atm_name == "HW2":
                residue_name = "SOL"
                trj = trajectory.subset(atom_name=atm_name, residue_name=residue_name).pbc
                sub_trjs.append(trj)
            elif atm_name == "whole":
                sub_trjs.append(trajectory.pbc)
            else:
                residue_name = "FHEC"
                trj = trajectory.subset(atom_name=atm_name, residue_name=residue_name).pbc
                sub_trjs.append(trj)

            # trj = trajectory.subset(atom_name=atm_name, residue_name=residue_name).nojump
            # sub_trjs.append(trj)

        lio_trj = trajectory.subset(atom_name="lio", residue_name="FHEC").pbc

        center_pos = np.zeros((len(lio_trj), 2))
        for frame in range(len(lio_trj)):
            center_pos[frame] = get_layers_centerpos(lio_trj[frame])


        # Here everything is happening...
        process_subtrajectory(sub_trjs, atm_names, collect_data_dir, meta_data, number_bins_x, center_pos)


    return None


#----------------------------------------------------------------------------------------
#--- used functions (alphabetically ordered)---
#----------------------------------------------------------------------------------------

def calculate_xy_density(sub_trj, collect_data_dir, meta_data, number_bins_x, center_pos):
    """
    Function to calculate the time-averaged spatial density distribution of atoms
    in the xy-plane, resolved in z-slabs, and export it as a normalized 3D histogram.

    Parameters:
        sub_trj (trajectory object of mdevaluate): Trajectory of atoms to calcuate on
        collect_data_dir (str): Directory where the output .npy file will be saved
        meta_data (dict): Metadata containing simulation conditions (keys: "temp", "atm_name", "slitsize")
        number_bins_x (int): Number of bins along the x-axis used for spatial binning
        center_pos (np.ndarray): Array of shape (n_frames, 2) containing the center positions of both clay layers
                                 at each frame, used for re-centering and wrapping

    Returns:
        None: Saves a 3D numpy array of shape (number_slabs, number_bins_x, number_bins_y) to disk,
              containing the normalized atom density [atoms / Å³] in the xy-plane for the selected atom type.
    """

    box = np.diagonal(sub_trj[0].box)

    # Length of supercell to be plotted
    xy_length_plot_supercell = np.array([box[0] / 8, box[1] / 4])

    # number of bins for xy binning
    number_bins_y = round(number_bins_x * ((box[1] / 8) / (box[0] / 16)))

    # calculate bin edges for binning, len(number_x_bins) == len(x_bin_edges + 1) !
    x_bin_edges = np.linspace(0, xy_length_plot_supercell[0], number_bins_x + 1)
    y_bin_edges = np.linspace(0, xy_length_plot_supercell[1], number_bins_y + 1)

    # Get number of slabs if all are 0.2 Angstrom wide for a distance of 20 Angstrom
    slab_width = 0.02
    binned_height = 2
    if (box[2] / 2) < binned_height:
        binned_height = round_down_to_step(box[2] / 2, 0.02)
        number_slabs = int(binned_height / slab_width)
        print("As half box size is smaller as 2nm, new number of slabs are choosed.")
        print(f"Number slabs: {number_slabs}. Determined distance: {binned_height} nm.")
    else:
        number_slabs = 100
        print(f"Number slabs: {number_slabs}, Determined distance: {binned_height} nm.")
    z_bin_edges = np.linspace(0, binned_height, number_slabs + 1)

    bins = z_bin_edges, x_bin_edges, y_bin_edges

    # get length of trajectory for iterating and normalizing
    sim_step_num = len(sub_trj)

    count = np.zeros((number_slabs, number_bins_x, number_bins_y), dtype=int)
    for idx_frame in range(sim_step_num):

        frame = sub_trj[idx_frame]

        # Detect lower layer
        layer_pos = np.min(center_pos[idx_frame])

        # Wrap data to create single slab system and preserve statistics, combining the counts -important note for nomalization, this means /2 is necessary
        frame[:, 2] = np.mod(frame[:, 2], (box[2] / 2))

        # Basic idea is to collect all atoms below the center of the layer and put them on top to get one combined water layer with slab on top and bottom
        diff = frame[:, 2] - layer_pos

        mask_combine_water = diff <= 0
        frame[:, 2][mask_combine_water] += box[2] / 2
        frame[:, 2] = frame[:, 2] - layer_pos # Adjust to new origin

        if np.min(frame[:, 2]) < 0:
            print("Somethings wrong with adjusting the origin. Exiting...")
            print(f"{meta_data["atm_name"]}, {meta_data["temp"]}, {meta_data["slitsize"]}")
            sys.exit()

        mask = frame[:, 2] <= binned_height
        data = frame[mask][:,[2, 0, 1]]

        binned = np.histogramdd(data, bins=bins)
        count += binned[0].astype(int)

        if idx_frame % 5000 == 0:
            print(f"Step {idx_frame} finished!")

    # normalization of data
    bin_volume = (
        (bins[0][1] - bins[0][0]) *
        (bins[1][1] - bins[1][0]) *
        (bins[2][1] - bins[2][0])
    )
    wrapping_multiplier = 8 * 4 * 2
    norm = wrapping_multiplier * bin_volume * sim_step_num
    normalized_data = count.astype(float) / norm

    # Export of normalized data
    save_dir = os.path.join(collect_data_dir, "xy_density", meta_data["slitsize"])
    os.makedirs(save_dir, exist_ok=True)
    file_name = f"xy_density_{meta_data['temp']}_{meta_data['atm_name']}.npy"
    file_path = os.path.join(save_dir, file_name)
    np.save(file_path, normalized_data)

    # Export bins
    export_bins = np.array([
        [bins[0][1] - bins[0][0], len(bins[0])],
        [bins[1][1] - bins[1][0], len(bins[1])],
        [bins[2][1] - bins[2][0], len(bins[2])]
    ])
    file_name_bins = f"xy_density_{meta_data['temp']}_bins.npy"
    file_path_bins = os.path.join(save_dir, file_name_bins)
    np.save(file_path_bins, export_bins)

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


def get_layers_centerpos(lio_frame):
    """
    Function to get centerpositions of both layers and output them, using lithium frame.

    Parameters:
        lio_frame (frame object from md evaluate): Sliced frame of octahedral lithium trajectory

    Returns:
        np.array(1,2): Both mean positions of hectorite sheet
    """

    halve_length = len(lio_frame) // 2
    layer_pos = (
        np.mean(lio_frame[:halve_length,2]),
        np.mean(lio_frame[halve_length:,2])
    )
    return np.array(layer_pos)


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


def process_subtrajectory(sub_trjs, atm_names, collect_data_dir, meta_data, number_bins_x, center_pos):
    """
    Allocated tasks with their own arguments for multiple agents to perform multiprocessed calculation.
    Tasks are defined in process_task().

    Parameters:
        sub_trjs (list of trajectory object from md evaluate): Trajectories to perform calculations on. Must have the same lengh as atm_names and matching order.
        atm_names (list of str): Contains atom names which are alocated to trajectories in sub_trjs. Must have the same lengh as sub_trjs and matching order.
        collect_data_dir (str): Directory where data will be saved.
        meta_data (dict): Dictionary containing keys 'temp' and 'slitsize'
    """

    task_types = ["xy_density"]

    runner_num = len(task_types) * len(atm_names)

    args_list = []
    print(len(sub_trjs))
    print(len(atm_names))
    for idx in range(0, len(sub_trjs)):
        task_meta_data = meta_data.copy()
        task_meta_data["atm_name"] = atm_names[idx]
        for task in task_types:
            args_list.append((sub_trjs[idx], collect_data_dir, task_meta_data, number_bins_x, center_pos))

    with Pool(processes=runner_num) as pool:
        pool.map(process_task, args_list)

    return None


def process_task(args):
    """
    Define tasks for calculating isf and msd including mean tau, kww=1/e and diffusion.
    Function used to make multiprocessing easier

    Parameters:
        args (list): Includes all necessaray parameters for tasks to run. The args list must contain the following objects in excaly this order:
            sub_trj (trajectory object from md evaluate): Data to be processed
            collect_data_dir (str): Directory where data will be saved.
            task_type (str): Given task to be performed on the data set.
            idx (int): Index for atm_name
            meta_data (dict): Dictionary containing keys 'temp', 'atm_name' and 'slitsize'
    """

    sub_trj, collect_data_dir, meta_data, number_bins_x, center_pos = args
    calculate_xy_density(sub_trj, collect_data_dir, meta_data, number_bins_x, center_pos)

    return None


def round_down_to_step(value, step=0.02):
    return round(np.floor(value / step) * step, 2)

#----------------------------main-function---------------------------
main()
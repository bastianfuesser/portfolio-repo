import os
import numpy as np
import argparse
import mdevaluate as md
import re

from multiprocessing import Pool
from functools import partial

from scipy.spatial import KDTree
from mdevaluate.pbc import pbc_points


# ----------------------------functions-------------------------------
def main():
    """
    Main is the only function directly run by this script.
    Other functions are called by this main function.
    All other used functions are alphabethically ordered below.
    This script can be run through bash command with --system_dir path to run in given path.
    If no --system_dir is given, manual_path is used instead.
    """

    manual_path = (
        "/data/bfuesser/FHEC_local/DoubleFHEC/25_07_07/no_bond/temperature_variation"
    )
    manual_atm_name_a = "Li"
    manual_atm_name_b = ["OW", "HW", "ob","Li"]
    manual_distance_to_layer = 0.8

    # Initializing Parsing
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--system_dir", type=str, default=manual_path, help="Path to the System"
    )
    parser.add_argument(
        "--atom_name_a",
        type=str,
        default=manual_atm_name_a,
        help="Axis to consider regarding msd calculations",
    )
    parser.add_argument(
        "--atom_name_b",
        nargs="+",
        default=manual_atm_name_b,
        help="List of atoms names to calculate msd and isf for",
    )
    parser.add_argument(
        "--distance_to_layer",
        default=manual_distance_to_layer,
        help="If provided only calculates for atoms with distance given to center of layer",
    )

    # Collect parsed arguments
    args = parser.parse_args()
    grandparent_dir = args.system_dir
    atm_res_name_a = add_res_name(args.atom_name_a)
    atm_res_name_b = add_res_name(args.atom_name_b)

    distance = None
    if args.distance_to_layer is not None:
        try:
            distance = float(args.distance_to_layer)
        except:
            print(
                "Float conversion of given --distance_to_layer failed. None is used instead..."
            )

    collect_data_dir = os.path.join(grandparent_dir, "collected_data_plots/data")
    os.makedirs(collect_data_dir, exist_ok=True)

    # Search for directories to work in
    simulations_dir_list = sorted(get_dir_list(grandparent_dir), reverse=False)
    # simulations_dir_list = [simulations_dir_list[0]]  # Debug
    print(f"Simulations on which analysis will be performed: {simulations_dir_list}")

    for work_dir in simulations_dir_list:

        import_path = os.path.join(work_dir, "03_prod")

        meta_data = {}
        meta_data["temp"] = get_temp(import_path)

        # Extract slitsize from directory name
        work_dir_basename = os.path.basename(work_dir)
        if work_dir_basename.startswith("slit"):
            match = re.search(r"slitsize_(\d+p\d+)", work_dir)
            meta_data["slitsize"] = match.group(1) if match else "unknown"
        else:
            meta_data["slitsize"] = work_dir_basename

        trajectory = md.open(import_path, nojump=False, topology="FHEC_prod.tpr")

        process_subtrajectory(
            trajectory,
            atm_res_name_a,
            atm_res_name_b,
            distance,
            collect_data_dir,
            meta_data,
        )

    return None


def add_res_name(list_atm_name):
    """
    Assigns residue names to atom names.

    Parameters:
        list_atm_name (list of str or str): A list of atom names or just a single string.

    Returns:
        list of [str, str]: A list where each element is a pair [atom_name, residue_name].
                            The residue name is "FHEC" for specific atom types,
                            and "SOL" for all others.
    """
    atm_res_name = []
    if isinstance(list_atm_name, str):
        if list_atm_name in ["lio", "mgo", "ob", "obss", "F", "Li", "st"]:
            atm_res_name = [list_atm_name, "FHEC"]
        else:
            atm_res_name = [list_atm_name, "SOL"]
    else:
        for atm_name in list_atm_name:
            if atm_name in ["lio", "mgo", "ob", "obss", "F", "Li", "st"]:
                atm_res_name.append([atm_name, "FHEC"])
            else:
                atm_res_name.append([atm_name, "SOL"])
    return atm_res_name


def coordination_number(
    frame_a,
    frame_b,
    lio_frame=None,
    bins=np.linspace(0, 0.5, 21),
    cumulative=False,
    layer_distance=None,
    distinct=True,
):
    """
    This function calculates how many atoms from `frame_b` are found within spherical shells around atoms from
    `frame_a`, optionally restricted to atoms in specific layers defined by `lio_frame`. The number of neighbors
    is computed using increasing radii (bin edges), and the result can be returned either cumulatively or
    shell-by-shell.

    Args:
        frame_a (np.ndarray): Reference atomic coordinates of shape (N, 3), typically the central species (e.g., Li).
        frame_b (np.ndarray): Neighbor atomic coordinates of shape (M, 3), typically the surrounding species (e.g., OW).
        lio_frame (np.ndarray, optional): Atomic coordinates used to determine the positions of layers.
                                          If provided, only atoms in `frame_a` within `layer_distance` of one of the layer centers
                                          along the z-axis will be considered.
        bins (np.ndarray, optional): Array of bin edges defining the spherical shells (in nm). Defaults to 20 bins up to 0.5 nm.
        cumulative (bool, optional): If True, returns cumulative coordination numbers; if False, returns per-shell values.
        layer_distance (float, optional): Distance threshold (in nm) for selecting atoms in `frame_a` from the center of the layer.
                                          defined by `lio_frame`. Default is 0.7 nm. If no 'lio_frame' is provided, no spacial
                                          selection is performed.

    Returns:
        np.ndarray: Coordination number values for each bin (length: len(bins) - 1). If no atoms remain after filtering,
                    an array of zeros is returned.
    """
    if layer_distance is None:
        layer_distance = 0.7
    if lio_frame is not None:
        center_pos1, center_pos2 = get_layers_centerpos(lio_frame)
        z_coords_frame_a = frame_a[:, 2]
        mask = (np.abs(z_coords_frame_a - center_pos1) < layer_distance) | (
            np.abs(z_coords_frame_a - center_pos2) < layer_distance
        )
        frame_a = frame_a[mask].copy()

    bins = bins[1:]
    coordination_numbers = np.zeros((len(bins)))
    latest_number = 0.0
    len_frame_a = len(frame_a)
    if len_frame_a == 0:
        print(
            f"No atoms found in layer distance {layer_distance} nm around lio center."
        )
        return coordination_numbers
    for bin_idx in range(len(bins)):
        new_number = (
            np.sum(
                md.coordinates.number_of_neighbors(
                    frame_b, query_atoms=frame_a, r_max=bins[bin_idx], distinct=distinct
                )
            )
            / len_frame_a
        )
        if cumulative == False:
            coordination_numbers[bin_idx] = new_number - latest_number
        elif cumulative == True:
            coordination_numbers[bin_idx] = new_number
        latest_number = new_number

    return coordination_numbers


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
    data_dir = os.path.join(path, meta_data["export_data"], meta_data["slitsize"])
    os.makedirs(data_dir, exist_ok=True)

    base_name = (
        f"{meta_data['export_data']}_{meta_data['temp']}_{meta_data['atm_name']}"
    )
    dat_path = os.path.join(data_dir, base_name + ".dat")

    # Export .dat file
    np.savetxt(dat_path, data_array)
    print(f"Data saved to {dat_path}")

    return None


def get_coordination_number_per_distance(
    trajectory,
    atm_res_name_a,
    atm_res_name_b,
    distance,
    collect_data_dir,
    task_meta_data,
    cumulative=False,
):
    bins = np.linspace(0, 0.5, 101)

    trj_a = trajectory.subset(
        atom_name=atm_res_name_a[0], residue_name=atm_res_name_a[1]
    )

    if distance is None:

        for atm_name, res_name in atm_res_name_b:
            distinct = True
            if atm_name == atm_res_name_a[0]:
                distinct = False
            trj_b = trajectory.subset(atom_name=atm_name, residue_name=res_name)

            results = time_average(
                partial(
                    coordination_number,
                    bins=bins,
                    cumulative=cumulative,
                    distinct=distinct,
                ),
                coordinates=trj_a,
                coordinates_b=trj_b,
                skip=0.1,
                segments=1000,
            )
            list_data = np.array([bins[1:], results])
            task_meta_data["atm_name"] = f"{atm_res_name_a[0]}_{atm_name}"
            if cumulative == False:
                task_meta_data["export_data"] = "CN"
            else:
                task_meta_data["export_data"] = "CN_sum"
            export_data(list_data.T, collect_data_dir, task_meta_data)
    else:

        lio_trj = trajectory.subset(atom_name="lio", residue_name="FHEC")
        for atm_name, res_name in atm_res_name_b:
            distinct = True
            if atm_name == atm_res_name_a[0]:
                distinct = False

            trj_b = trajectory.subset(atom_name=atm_name, residue_name=res_name)

            results = time_average_three_inputs(
                partial(
                    coordination_number,
                    bins=bins,
                    cumulative=cumulative,
                    layer_distance=distance,
                    distinct=distinct,
                ),
                coordinates=trj_a,
                coordinates_b=trj_b,
                coordinates_c=lio_trj,
                skip=0.1,
                segments=1000,
            )
            list_data = np.array([bins[1:], results])
            str_dist = f"{distance:.1f}".replace(".", "p")
            task_meta_data["atm_name"] = f"{atm_res_name_a[0]}_{atm_name}"
            if cumulative == False:
                task_meta_data["export_data"] = f"CN_dist_{str_dist}"
            else:
                task_meta_data["export_data"] = f"CN_sum_dist_{str_dist}"
            export_data(list_data.T, collect_data_dir, task_meta_data)

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
            if dir == "03_prod" and os.path.exists(
                os.path.join(root, dir, "FHEC_prod.gro")
            ):
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
        if file.endswith(".mdp") and not file.startswith("mdout"):
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
    with open(mdp_file, "r") as file:
        for line in file:
            if line.strip().startswith("ref-t"):
                parts = line.split("=")
                if len(parts) > 1:
                    temperature = float(parts[1].strip())
                break
    return int(temperature)


def get_layers_centerpos(lio_frame):
    halve_length = len(lio_frame) // 2
    layer_pos = (
        np.mean(lio_frame[:halve_length, 2]),
        np.mean(lio_frame[halve_length:, 2]),
    )
    return np.array(layer_pos)


def get_rdf_per_distance(
    trajectory,
    atm_res_name_a,
    atm_res_name_b,
    distance,
    collect_data_dir,
    task_meta_data,
):
    # Define bins
    bins = np.linspace(0, 0.5, 101)
    trj_a = trajectory.subset(
        atom_name=atm_res_name_a[0], residue_name=atm_res_name_a[1]
    )
    if distance is None:
        for atm_name, res_name in atm_res_name_b:
            # Check wheter both sets represent the same atoms
            distinct = True
            if atm_name == atm_res_name_a[0]:
                distinct = False
            trj_b = trajectory.subset(atom_name=atm_name, residue_name=res_name)
            # Calculation of time averaged radial density function, both are modified and included in this script
            results = time_average(
                partial(rdf, bins=bins, remove_intra=False, distinct=distinct),
                coordinates=trj_a,
                coordinates_b=trj_b,
                skip=0.1,
                segments=1000,
            )
            list_data = np.array([bins[1:], results])
            task_meta_data["atm_name"] = f"{atm_res_name_a[0]}_{atm_name}"
            task_meta_data["export_data"] = "rdf"
            export_data(list_data.T, collect_data_dir, task_meta_data)
    else:

        lio_trj = trajectory.subset(atom_name="lio", residue_name="FHEC")
        for atm_name, res_name in atm_res_name_b:
            trj_b = trajectory.subset(atom_name=atm_name, residue_name=res_name)
            distinct = True
            if atm_res_name_a[0] == atm_res_name_b[0]:
                distinct = False

            results = time_average_three_inputs(
                partial(
                    spatially_resolved_rdf,
                    bins=bins,
                    layer_distance=distance,
                    distinct=distinct,
                ),
                coordinates=trj_a,
                coordinates_b=trj_b,
                coordinates_c=lio_trj,
                skip=0.1,
                segments=1000,
            )
            list_data = np.array([bins[1:], results])
            str_dist = f"{distance:.1f}".replace(".", "p")
            task_meta_data["atm_name"] = f"{atm_res_name_a[0]}_{atm_name}"
            task_meta_data["export_data"] = f"rdf_dist_{str_dist}"
            export_data(list_data.T, collect_data_dir, task_meta_data)

    return None


def process_subtrajectory(
    trajectory, atm_res_name_a, atm_res_name_b, distance, collect_data_dir, meta_data
):
    task_types = ["rdf", "CN", "CN_cumulative"]
    runner_num = len(task_types)

    args_list = []
    for task in task_types:
        task_meta_data = meta_data.copy()
        args_list.append(
            (
                task,
                trajectory,
                atm_res_name_a,
                atm_res_name_b,
                distance,
                collect_data_dir,
                task_meta_data,
            )
        )

    with Pool(processes=runner_num) as pool:
        pool.map(process_task, args_list)

    return None


def process_task(args):
    (
        task,
        trajectory,
        atm_res_name_a,
        atm_res_name_b,
        distance,
        collect_data_dir,
        task_meta_data,
    ) = args
    if task == "rdf":
        get_rdf_per_distance(
            trajectory,
            atm_res_name_a,
            atm_res_name_b,
            distance,
            collect_data_dir,
            task_meta_data,
        )
    elif task == "CN":
        get_coordination_number_per_distance(
            trajectory,
            atm_res_name_a,
            atm_res_name_b,
            distance,
            collect_data_dir,
            task_meta_data,
            cumulative=False,
        )
    elif task == "CN_cumulative":
        get_coordination_number_per_distance(
            trajectory,
            atm_res_name_a,
            atm_res_name_b,
            distance,
            collect_data_dir,
            task_meta_data,
            cumulative=True,
        )


def spatially_resolved_rdf(
    frame_a, frame_b, lio_frame, layer_distance=None, bins=None, distinct=False
):

    if layer_distance is None:
        layer_distance = 0.7
    if bins is None:
        bins = np.linspace(0, 0.5, 101)

    # Get center of layers and create mask for absolute z-pos difference smaller than given layer_distance
    center_pos1, center_pos2 = get_layers_centerpos(lio_frame)
    z_coords_frame_a = frame_a[:, 2]
    mask = (np.abs(z_coords_frame_a - center_pos1) < layer_distance) | (
        np.abs(z_coords_frame_a - center_pos2) < layer_distance
    )
    frame_a = frame_a[mask].copy()

    result = np.zeros(len(bins) - 1)
    if len(frame_a) > 0:
        result = rdf(frame_a, frame_b, bins=bins, remove_intra=False, distinct=distinct)
    return result


# --------------------------------------------------------------------
# ---modified-functions-from-mdevaluate---
# --------------------------------------------------------------------

def next_neighbors(
    atoms,
    query_atoms=None,
    number_of_neighbors=1,
    distance_upper_bound=np.inf,
    distinct=False,
    **kwargs,
):

    dnn = 0
    if query_atoms is None:
        query_atoms = atoms
        dnn = 1
    elif not distinct:
        dnn = 1

    box = atoms.box
    if np.all(np.diag(np.diag(box)) == box):
        atoms = atoms % np.diag(box)
        tree = KDTree(atoms, boxsize=np.diag(box))
        distances, indices = tree.query(
            query_atoms,
            number_of_neighbors + dnn,
            distance_upper_bound=distance_upper_bound,
        )
        distances = np.atleast_2d(distances)
        indices = np.atleast_2d(indices)
        distances = distances[:, dnn:]
        indices = indices[:, dnn:]
        distances_new = []
        indices_new = []
        for dist, ind in zip(distances, indices):
            distances_new.append(dist[dist <= distance_upper_bound])
            indices_new.append(ind[dist <= distance_upper_bound])
        return distances_new, indices_new
    else:
        atoms_pbc, atoms_pbc_index = pbc_points(
            atoms, box, thickness=distance_upper_bound + 0.1, index=True, **kwargs
        )
        tree = KDTree(atoms_pbc)
        distances, indices = tree.query(
            query_atoms,
            number_of_neighbors + dnn,
            distance_upper_bound=distance_upper_bound,
        )
        distances = np.atleast_2d(distances)
        indices = np.atleast_2d(indices)
        distances = distances[:, dnn:]
        indices = indices[:, dnn:]
        distances_new = []
        indices_new = []
        for dist, ind in zip(distances, indices):
            distances_new.append(dist[dist <= distance_upper_bound])
            indices_new.append(atoms_pbc_index[ind[dist <= distance_upper_bound]])
        return distances_new, indices_new


def rdf(atoms_a, atoms_b=None, bins=None, remove_intra=False, distinct=False, **kwargs):

    particles_in_volume = int(
        np.max(
            md.coordinates.number_of_neighbors(
                atoms_b, query_atoms=atoms_a, r_max=bins[-1] * 2, distinct=True
            )
        )
        * 1.1
    )
    distances, indices = next_neighbors(
        atoms_a,
        atoms_b,
        number_of_neighbors=particles_in_volume,
        distance_upper_bound=bins[-1],
        distinct=distinct,
        **kwargs,
    )
    if remove_intra:
        new_distances = []
        for entry in list(zip(atoms_a.residue_ids, distances, indices)):
            mask = entry[1] < np.inf
            new_distances.append(
                entry[1][mask][atoms_b.residue_ids[entry[2][mask]] != entry[0]]
            )
        distances = np.concatenate(new_distances)
    else:
        distances = [d for dist in distances for d in dist]

    hist, bins = np.histogram(distances, bins=bins, range=(0, bins[-1]), density=False)
    hist = hist / len(atoms_b)
    hist = hist / (4 / 3 * np.pi * bins[1:] ** 3 - 4 / 3 * np.pi * bins[:-1] ** 3)
    n = len(atoms_a) / np.prod(np.diag(atoms_a.box))
    hist = hist / n

    return hist


def time_average(
    function,
    coordinates,
    coordinates_b=None,
    skip=0.1,
    segments=100,
):

    frame_indices = np.unique(
        np.int_(
            np.linspace(len(coordinates) * skip, len(coordinates) - 1, num=segments)
        )
    )
    if coordinates_b is None:
        result = [function(coordinates[frame_index]) for frame_index in frame_indices]
    else:
        result = [
            function(coordinates[frame_index], coordinates_b[frame_index])
            for frame_index in frame_indices
        ]
    # Clean up results if no neighbors are found.
    result = np.array(result)
    zero_elem = np.zeros_like(result[0])
    mask = np.all(result == zero_elem, axis=tuple(range(1, result.ndim)))

    filtered_results = zero_elem
    if len(result[~mask]) > 0:
        filtered_results = np.mean(result[~mask], axis=0)
    return filtered_results


def time_average_three_inputs(
    function,
    coordinates,
    coordinates_b,
    coordinates_c,
    skip=0.1,
    segments=100,
):

    frame_indices = np.unique(
        np.int_(
            np.linspace(len(coordinates) * skip, len(coordinates) - 1, num=segments)
        )
    )
    result = [
        function(
            coordinates[frame_index],
            coordinates_b[frame_index],
            coordinates_c[frame_index],
        )
        for frame_index in frame_indices
    ]
    # Clean up results if no neighbors are found.
    result = np.array(result)
    zero_elem = np.zeros_like(result[0])
    mask = np.all(result == zero_elem, axis=tuple(range(1, result.ndim)))

    filtered_results = zero_elem
    if len(result[~mask]) > 0:
        filtered_results = np.mean(result[~mask], axis=0)
    return filtered_results


# ----------------------------main-function---------------------------

main()
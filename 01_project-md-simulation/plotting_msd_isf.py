import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import FuncFormatter
import tudplot

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
    manual_atm_names = ["HW", "OW", "Li"]
    # ["ob"]
    # Initializing Parsing
    parser = argparse.ArgumentParser()

    # Work directory to be parsed
    parser.add_argument(
        "--system_dir",
        type=str,
        default=manual_path,
        help="Path to the System"
    )

    # Atom names to be parsed
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
    collect_plots_dir = os.path.join(grandparent_dir, 'collected_data_plots/plots')
    os.makedirs(collect_data_dir, exist_ok=True)
    os.makedirs(collect_plots_dir, exist_ok=True)

    property_name = ["msd", "isf"]
    data_dirs = get_dir_list_data(collect_data_dir, property_name)


    # Collect all data in a nested dictionary. Keys represent meta data.
    data_dict = {}
    for data_dir in data_dirs:
        for file in os.listdir(data_dir):
            if file.endswith(".dat"):

                # Collect all meta data from directory and file names
                import_file_path = os.path.join(data_dir, file)
                data = import_data(import_file_path)
                data_property, temp, atm_name = file.rsplit('.', 1)[0].rsplit('_', 2)
                slitsize = os.path.basename(data_dir)

                if atm_name in atm_names:
                    slitsize = os.path.basename(data_dir)
                    if data_property not in data_dict:
                        data_dict[data_property] = {}
                    if slitsize not in data_dict[data_property]:
                        data_dict[data_property][slitsize] = {}
                    if atm_name not in data_dict[data_property][slitsize]:
                        data_dict[data_property][slitsize][atm_name] = {}

                    data_dict[data_property][slitsize][atm_name][temp] = data

    plot_data(data_dict, collect_plots_dir)


def get_dir_list_data(dir, property_name):
    """
    Search for directories in the base directory that contain files related to the specified property.

    Parameters:
        base_dir (str): The base directory to search in.
        property_name (str or list): The property/properties to look for in the directory names or files.

    Returns:
        list: A list of paths to directories containing the specified property.
    """

    matching_dirs = set()
    if isinstance(property_name, str):
        # Single property case
        for root, dirs, files in os.walk(dir):
            if any(property_name in file for file in files):
                matching_dirs.add(root)
    elif isinstance(property_name, (list, tuple)):
        # Multiple property case
        for single_property_name in property_name:
            for root, dirs, files in os.walk(dir):
                if any(single_property_name in file for file in files):
                    matching_dirs.add(root)
    else:
        raise ValueError("property_name must be a string or list of strings!")
    return sorted(list(matching_dirs))


def import_data(file_path):
    """
    Import data from a given file path. Handles scalar values and lists of data.

    Parameters:
        file_path (str): Path to the data file.

    Returns:
        float or numpy.ndarray: Returns a float for scalar data or a numpy array for list data.
    """

    try:
        with open(file_path, 'r') as file:
            # Read the first line to check the data format
            first_line = file.readline().strip()

            # Determine if it's scalar or list data
            if " " not in first_line:  # Scalar data (single number)
                return float(first_line)
            else:  # List data
                # Rewind and load the data as a numpy array
                file.seek(0)
                data = np.loadtxt(file).T
                return data
    except Exception as e:
        print(f"Error importing data from {file_path}: {e}")
        return None


def plot_msd(msd_data_dict, collect_plot_dir, msd_axis):
    # data_dict[data_property][slitsize][atm_name][temp]
    for str_slitsize in msd_data_dict.keys():

        fig_combined, ax_combined = plt.subplots(figsize=(4,3))

        plot_dir = os.path.join(collect_plot_dir, msd_axis, str_slitsize)
        os.makedirs(plot_dir, exist_ok=True)

        boolean_single_temp = True

        for str_atm_name in sorted(msd_data_dict[str_slitsize].keys()):

            if len(msd_data_dict[str_slitsize][str_atm_name]) == 1:

                # Plots for single temperatures

                int_temp = next(iter(msd_data_dict[str_slitsize][str_atm_name].keys()))

                file_name = f"{str_slitsize}_msd_{str_atm_name}.pdf"
                plot_full_path = os.path.join(plot_dir, file_name)

                if str_atm_name == "HW":
                    color='black'
                    zorder = 3
                    combined_marker = "x"
                    marker_size = 4
                elif str_atm_name == "OW":
                    color='dodgerblue'
                    zorder = 2
                    combined_marker = "o"
                    marker_size = 5
                elif str_atm_name == "Li":
                    color="red"
                    zorder = 2
                    combined_marker = "^"
                    marker_size = 5
                else:
                    color="brown"

                fig_single, ax_single = plt.subplots(figsize=(4,3))

                try:
                    time, data, fit = msd_data_dict[str_slitsize][str_atm_name][int_temp]

                    ax_combined.plot(
                        time*1e-12,
                        data*1e-18,
                        color=color,
                        linewidth=1,
                        label=f"{str_atm_name}",
                        zorder=zorder,
                        marker=combined_marker,
                        markersize=marker_size,
                    )

                    ax_single.plot(
                        time,
                        data,
                        color=color,
                        linewidth=2,
                        label=f"{str_atm_name}"
                    )

                    mask = fit > np.min(data)

                    ax_combined.plot(
                        time[mask]*1e-12,
                        fit[mask]*1e-18,
                        color=color,
                        linestyle="--",
                        linewidth=0.5,
                        label=f"{str_atm_name} fit"
                    )

                    ax_single.plot(
                        time[mask],
                        fit[mask],
                        color=color,
                        linestyle="--",
                        linewidth=0.5,
                        label=f"fit"
                    )

                except:
                    print(f"No fit data found for {str_slitsize}_{msd_axis}_{str_atm_name}")

                    time, data = msd_data_dict[str_slitsize][str_atm_name][int_temp]

                    ax_combined.plot(
                        time*1e-12,
                        data*1e-18,
                        color=color,
                        linewidth=2,
                        label=f"{str_atm_name}"
                    )

                    ax_single.plot(
                        time,
                        data,
                        color=color,
                        linewidth=2,
                        label=f"{str_atm_name}"
                    )

                axis = msd_axis.split("_", 1)[0]

                ax_single.set_xscale("log")
                ax_single.set_yscale("log")
                ax_single.set_xlabel(r"$t~/~(\mathrm{ps})$")
                ax_single.set_ylabel(rf"$<r_\mathrm{{{axis}}}^2(t)>~/~(\mathrm{{m}}^2)$")
                ax_single.grid(True)
                ax_single.legend()
                fig_single.tight_layout()

                file_name = f"{str_slitsize}_{msd_axis}_{str_atm_name}.pdf"
                single_full_path = os.path.join(plot_dir, file_name)

                fig_single.savefig(single_full_path)

                plt.figure(fig_single.number)
                tudplot.saveagr(single_full_path.replace(".pdf", ".agr"))
                print(f"Saved plot as {single_full_path} and .agr")

                plt.close(fig_single)

            else:
                plt.close(fig_combined)
                boolean_single_temp = False

                tuple_temps = [int(item) for item in msd_data_dict[str_slitsize][str_atm_name].keys()]
                int_temp_max = max(tuple_temps)
                int_temp_min = min(tuple_temps)
                fig, ax = plt.subplots(figsize=(4, 3))

                if int_temp_max - int_temp_min < 110:
                    step = 10
                else:
                    step = 20

                range_cb_ticks = range(int_temp_min, int_temp_max + 1, step)

                for idx_temp, temp in enumerate(sorted(msd_data_dict[str_slitsize][str_atm_name].keys())):

                    norm = mcolors.LogNorm(vmin=int_temp_min, vmax=int_temp_max)

                    cmap = plt.get_cmap('inferno', 256)
                    cmap = mcolors.LinearSegmentedColormap.from_list(
                        'truncated_inferno', cmap(np.linspace(0, 0.85, 256))
                    )
                    color = cmap(norm(int(temp)))



                    try:
                        time, data, fit = msd_data_dict[str_slitsize][str_atm_name][temp]

                        ax.plot(
                            time*1e-12,
                            data*1e-18,
                            color=color,
                            linewidth=2,
                            label=f"{temp}K"
                        )


                        fit = msd_data_dict[str_slitsize][str_atm_name][temp][2]
                        mask = fit > np.min(1e-15)

                        ax.plot(
                            time[mask]*1e-12,
                            fit[mask]*1e-18,
                            color=color,
                            linewidth=0.5,
                            linestyle="--",
                            label=f"{temp}k fit"
                        )

                    except:
                        print(f"No fit data found for {str_slitsize}_{msd_axis}_{str_atm_name}")

                        time, data = msd_data_dict[str_slitsize][str_atm_name][temp]
                        data = data * 1e-18

                        ax.plot(
                            time,
                            data,
                            color=color,
                            linewidth=2,
                            label=f"{temp}K"
                        )


                sm = cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])

                # Create colorbar
                cbar = plt.colorbar(sm, ax=ax, label=r"Temp / (K)", shrink=0.7)

                ticks = np.arange(int_temp_min, int_temp_max + 1, step)

                cbar.set_ticks(ticks)
                cbar.set_ticklabels([f"{int(t)}" for t in ticks])

                # plt.title(f"{msd_axis}, layer distance: {str_slitsize}, {str_atm_name}")

                ax.grid(True)

                plt.figure(fig.number)

                plt.xscale("log")
                plt.xlabel(r"$t~/~(\mathrm{s})$")

                plt.yscale("log")
                plt.ylim(1e-21, 1e-15)
                axis = msd_axis.split("_", 1)[0]
                plt.ylabel(rf"$<r_\mathrm{{{axis}}}^2(t)>~/~(\mathrm{{m}}^2)$")

                file_name = f"{str_slitsize}_{msd_axis}_{str_atm_name}.pdf"
                plot_full_path = os.path.join(plot_dir, file_name)

                plt.tight_layout()

                plt.savefig(plot_full_path)

                # Export as .agr
                tudplot.saveagr(plot_full_path.replace(".pdf", ".agr"))

                print(f"Plot successfull saved as {plot_full_path} and .agr")



        if boolean_single_temp == True:
            ax_combined.set_xscale("log")
            ax_combined.set_yscale("log")
            ax_combined.set_xlabel(r"$t~/~(\mathrm{s})$")
            ax_combined.set_ylabel(rf"$<r_\mathrm{{{axis}}}^2(t)>~/~(\mathrm{{m}}^2)$")
            ax_combined.set_ylim(1e-21, 1e-15)
            ax_combined.grid(True)
            ax_combined.legend()
            fig_combined.tight_layout()

            file_name = f"{str_slitsize}_{msd_axis}_combined.pdf"
            combined_full_path = os.path.join(plot_dir, file_name)

            fig_combined.savefig(combined_full_path)

            plt.figure(fig_combined.number)
            tudplot.saveagr(combined_full_path.replace(".pdf", ".agr"))
            print(f"Saved plot as {combined_full_path} and .agr")

            plt.close(fig_combined)


            # Plots for multiple temperatures
            # for int_temp in msd_data_dict[str_slitsize][str_atm_name].keys():
            #     break



    return None

def plot_isf(isf_data_dict, collect_plot_dir):
    # data_dict[data_property][slitsize][atm_name][temp]
    for str_slitsize in isf_data_dict.keys():

        fig_combined, ax_combined = plt.subplots(figsize=(4,3))

        plot_dir = os.path.join(collect_plot_dir, "isf", str_slitsize)
        os.makedirs(plot_dir, exist_ok=True)

        boolean_single_temp = True

        for str_atm_name in isf_data_dict[str_slitsize].keys():

            if len(isf_data_dict[str_slitsize][str_atm_name]) == 1:

                # Plots for single temperatures

                int_temp = next(iter(isf_data_dict[str_slitsize][str_atm_name].keys()))

                file_name = f"{str_slitsize}_isf_{str_atm_name}.pdf"
                plot_full_path = os.path.join(plot_dir, file_name)

                if str_atm_name == "HW":
                    color='black'
                    zorder = 3
                    combined_marker = "x"
                elif str_atm_name == "OW":
                    color='dodgerblue'
                    zorder = 2
                    combined_marker = "o"
                else:
                    color="red"
                    zorder = 1
                    combined_marker = "^"

                fig_single, ax_single = plt.subplots(figsize=(4,3))

                try:
                    time, data, fit = isf_data_dict[str_slitsize][str_atm_name][int_temp]

                    ax_combined.plot(
                        time,
                        data,
                        color=color,
                        linewidth=2,
                        label=f"{str_atm_name}"
                    )

                    ax_single.plot(
                        time,
                        data,
                        color=color,
                        linewidth=2,
                        label=f"{str_atm_name}"
                    )

                    mask = fit > np.min(data)

                    ax_combined.plot(
                        time[mask],
                        fit[mask],
                        color=color,
                        linestyle="--",
                        linewidth=0.5,
                        label=f"{str_atm_name} fit"
                    )

                    ax_single.plot(
                        time[mask],
                        fit[mask],
                        color=color,
                        linestyle="--",
                        linewidth=0.5,
                        label=f"fit"
                    )

                except:
                    print(f"No fit data found for {str_slitsize}_{str_atm_name}")

                    time, data = isf_data_dict[str_slitsize][str_atm_name][int_temp]

                    ax_combined.plot(
                        time,
                        data,
                        color=color,
                        linewidth=2,
                        label=f"{str_atm_name}"
                    )

                    ax_single.plot(
                        time,
                        data,
                        color=color,
                        linewidth=2,
                        label=f"{str_atm_name}"
                    )

                ax_single.set_xscale("log")
                ax_single.set_yscale("linear")
                ax_single.set_xlabel(r"$t~/~(\mathrm{ps})$")
                ax_single.set_ylabel(r"$F(\mathrm{q}, t)$")
                ax_single.grid(True)
                ax_single.legend()
                fig_single.tight_layout()

                file_name = f"{str_slitsize}_{str_atm_name}.pdf"
                single_full_path = os.path.join(plot_dir, file_name)

                fig_single.savefig(single_full_path)

                plt.figure(fig_single.number)
                tudplot.saveagr(single_full_path.replace(".pdf", ".agr"))
                print(f"Saved plot as {single_full_path} and .agr")

                plt.close(fig_single)

            else:
                plt.close(fig_combined)
                boolean_single_temp = False

                tuple_temps = [int(item) for item in isf_data_dict[str_slitsize][str_atm_name].keys()]
                int_temp_max = max(tuple_temps)
                int_temp_min = min(tuple_temps)
                fig, ax = plt.subplots(figsize=(6, 4))

                if int_temp_max - int_temp_min < 100:
                    step = 10
                else:
                    step = 20

                range_cb_ticks = range(int_temp_min, int_temp_max + 1, step)

                for idx_temp, temp in enumerate(sorted(isf_data_dict[str_slitsize][str_atm_name].keys())):

                    norm = mcolors.LogNorm(vmin=int_temp_min, vmax=int_temp_max)
                    cmap = plt.get_cmap('inferno', 256)
                    cmap = mcolors.LinearSegmentedColormap.from_list(
                        'truncated_inferno', cmap(np.linspace(0, 0.85, 256))
                    )
                    color = cmap(norm(int(temp)))



                    try:
                        time, data, fit = isf_data_dict[str_slitsize][str_atm_name][temp]

                        ax.plot(
                            time,
                            data,
                            color=color,
                            linewidth=2,
                            label=f"{temp}K"
                        )


                        fit = isf_data_dict[str_slitsize][str_atm_name][temp][2]
                        mask = fit > np.min(data)

                        ax.plot(
                            time[mask],
                            fit[mask],
                            color=color,
                            linewidth=0.5,
                            linestyle="--",
                            label=f"{temp}k fit"
                        )

                    except:
                        print(f"No fit data found for {str_slitsize}_{str_atm_name}")

                        time, data = isf_data_dict[str_slitsize][str_atm_name][temp]

                        ax.plot(
                            time,
                            data,
                            color=color,
                            linewidth=2,
                            label=f"{temp}K"
                        )


                sm = cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])

                # Create colorbar
                cbar = plt.colorbar(sm, ax=ax, label=r"$T~/~(\mathrm{K})$", shrink=0.7)

                # Force integer formatting
                cbar.ax.set_yticks(range_cb_ticks)
                cbar.ax.set_yticklabels([str(item) for item in range_cb_ticks])
                # plt.title(f"ISF, layer distance: {str_slitsize}, {str_atm_name}")

                ax.grid(True)

                plt.figure(fig.number)

                plt.xscale("log")
                plt.xlabel(r"$t~/~(\mathrm{ps})$")

                plt.yscale("linear")
                plt.ylabel(rf"$\mathrm{{ISF}} \, / \, (\mathrm{{nm}}^2) $")

                file_name = f"{str_slitsize}_{str_atm_name}.pdf"
                plot_full_path = os.path.join(plot_dir, file_name)

                plt.tight_layout()

                plt.savefig(plot_full_path)

                # Export as .agr
                tudplot.saveagr(plot_full_path.replace(".pdf", ".agr"))

                print(f"Plot successfull saved as {plot_full_path} and .agr")

        if boolean_single_temp == True:
            ax_combined.set_xscale("log")
            ax_combined.set_yscale("linear")
            ax_combined.set_ylim(0, 1)
            ax_combined.set_xlabel(r"$ t~/~(\mathrm{ps}) $")
            ax_combined.set_ylabel(r"$ F_\mathrm{s}(\mathrm{q}, t)~/~(-)$")
            ax_combined.grid(True)
            ax_combined.legend()
            ax_combined.set_title("")
            fig_combined.tight_layout()

            file_name = f"{str_slitsize}_combined.pdf"
            combined_full_path = os.path.join(plot_dir, file_name)

            fig_combined.savefig(combined_full_path)

            plt.figure(fig_combined.number)
            tudplot.saveagr(combined_full_path.replace(".pdf", ".agr"))
            print(f"Saved plot as {combined_full_path} and .agr")

            plt.close(fig_combined)

def plot_data(data_dict, collect_plot_dir):
    """
    Plot data from data dictionary.
    Keys represent meta data.

    Parameters:
        data_dict(dictionary): Dictionary containing nested dictionaries, keys represent meta data.
            key1: Property to plot (isf etc.)
                key2: Slitsize
                    key3: Atom name (lio etc)
                        key4: Temperature
        collect_plot_dir(str): main directory where plots will be saved to
    """


    for data_property in data_dict.keys():
        if "msd" in data_property:
            plot_msd(data_dict[data_property], collect_plot_dir, data_property)
        elif "isf" in data_property:
            plot_isf(data_dict[data_property], collect_plot_dir)
        else:
            print(f"Error, {data_property} as data property not recognized!")


main()
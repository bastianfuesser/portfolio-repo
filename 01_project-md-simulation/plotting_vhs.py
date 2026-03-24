import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
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

    manual_dir = "/data/bfuesser/FHEC_local/DoubleFHEC/25_07_07/no_bond/temperature_variation"
    parser = argparse.ArgumentParser()
    parser.add_argument("--system_dir", type=str, default=manual_dir, help="Path to the System")
    args = parser.parse_args()

    grandparent_dir = args.system_dir
    collect_data_dir = os.path.join(grandparent_dir, 'collected_data_plots/data')
    collect_plots_dir = os.path.join(grandparent_dir, 'collected_data_plots/plots')
    os.makedirs(collect_data_dir, exist_ok=True)
    os.makedirs(collect_plots_dir, exist_ok=True)

    data_dirs = get_dir_list_data(collect_data_dir, 'vhs')

    data_dict = {}
    for data_dir in data_dirs:
        axis, slitsize, temp, atm_name = data_dir.rsplit("/", 4)[-4:]

        # extend dictionary by new keys
        if slitsize not in data_dict:
            data_dict[slitsize] = {}
        if temp not in data_dict[slitsize]:
            data_dict[slitsize][temp] = {}
        if atm_name not in data_dict[slitsize][temp]:
            data_dict[slitsize][temp][atm_name] = {}
        if axis not in data_dict[slitsize][temp][atm_name]:
            data_dict[slitsize][temp][atm_name][axis] = {}

        # import data into data_dict
        for file in sorted(os.listdir(data_dir)):
            import_full_path = os.path.join(data_dir, file)
            if file == "bins.dat":
                bins = np.loadtxt(import_full_path)
                data_dict[slitsize][temp][atm_name][axis]["bins"] = bins
            elif file == "time.dat":
                time = np.loadtxt(import_full_path)
                data_dict[slitsize][temp][atm_name][axis]["time"] = time
            else:
                data_raw = np.loadtxt(import_full_path)
                number = int(file.split("_")[1].split(".")[0])
                data_dict[slitsize][temp][atm_name][axis][number] = data_raw

    plot_vhs(data_dict, collect_plots_dir)


def plot_vhs(data_dict, collect_plots_dir):
    """
    Plot data from data dictionary.
    Keys represent meta data.

    Parameters:
        data_dict(dictionary): Dictionary containing nested dictionaries, keys represent meta data.
            key1: Slitsize
                key2: Temperature
                    key3: Atom name (lio etc)
                        key4: Time interval number (int) or 'bins' or 'time'
        collect_plot_dir(str): main directory where plots will be saved to
    """

    for str_slitsize in data_dict.keys():

        float_slitsize = float(str_slitsize.replace("p", "."))

        for int_temp in data_dict[str_slitsize].keys():

            for str_atm_name in data_dict[str_slitsize][int_temp].keys():

                for str_axis in data_dict[str_slitsize][int_temp][str_atm_name].keys():

                    axis = str_axis

                    str_local_plots_path = os.path.join(collect_plots_dir, f"{axis}", str_slitsize, str(int_temp))
                    os.makedirs(str_local_plots_path, exist_ok=True)
                    plot_full_path = os.path.join(str_local_plots_path, f"{str_axis}_{str_atm_name}_{int_temp}_{str_slitsize}.pdf")

                    fig_only_2d, ax_only_2d = plt.subplots(figsize=(5,5))

                    fig = plt.figure(figsize=(8,4))
                    ax2d = fig.add_subplot(1, 2, 1)
                    ax3d = fig.add_subplot(1, 2, 2, projection='3d')

                    array_x_data = data_dict[str_slitsize][int_temp][str_atm_name][str_axis]['bins']

                    if str_axis == "normal_vhs" or str_axis == "OH_vhs":
                        array_x_data = array_x_data[1:]
                    else:
                        array_x_data = array_x_data[1:] * 10
                    array_time_intervals = data_dict[str_slitsize][int_temp][str_atm_name][str_axis]['time']

                    norm = mcolors.LogNorm(vmin=np.min(array_time_intervals), vmax=np.max(array_time_intervals))
                    cmap = plt.get_cmap('inferno', 256)
                    cmap = mcolors.LinearSegmentedColormap.from_list(
                        'truncated_inferno', cmap(np.linspace(0, 0.85, 256))
                    )

                    for int_number_time_bin in data_dict[str_slitsize][int_temp][str_atm_name][str_axis].keys():

                        if isinstance(int_number_time_bin, int):
                            float_time_interval = array_time_intervals[int_number_time_bin - 1]
                            array_y_data = data_dict[str_slitsize][int_temp][str_atm_name][str_axis][int_number_time_bin]

                            # color map adjustments
                            float_color_value = (int_number_time_bin / (len(data_dict[str_slitsize][int_temp][str_atm_name][str_axis].keys()) - 2) * 0.9)
                            color = cmap(norm(float_time_interval))
                            if str_axis == "z_vhs":
                                mask = array_x_data <= float_slitsize + 1
                            elif str_axis == "normal_vhs" or str_axis == "OH_vhs":
                                mask = array_x_data > 0
                            else:
                                mask = array_x_data <= 10
                            if int_number_time_bin % 3 == 0:
                                zorder = 100 - int_number_time_bin
                                ax2d.plot(array_x_data[mask], array_y_data[mask], label=f"Time {float_time_interval:5.2f}", color=color, linestyle="-")
                                ax3d.plot(array_x_data[mask], np.log(float_time_interval), array_y_data[mask], color=color, linestyle="-", zorder=zorder)
                                ax_only_2d.plot(array_x_data[mask], array_y_data[mask], label=f"Time {float_time_interval:5.2f}", color=color, linestyle="-")

                    # Set Correct dependencies in ylabel
                    possibilities = ["x", "y", "z", "xy", "xz", "yz", "all", "normal", "OH"]
                    returns = [
                        r"$\mathrm{G}(x, t)$",
                        r"$\mathrm{G}(y, t)$",
                        r"$\mathrm{G}(z, t)$",
                        r"$\mathrm{G}((x, y), t)$",
                        r"$\mathrm{G}((x, z), t)$",
                        r"$\mathrm{G}((y, z), t)$",
                        rf"$ \mathrm{{G}}_\mathrm{{{str_atm_name}}}(r, t) $",
                        r"$ \mathrm{G}_\mathrm{normal-H2O}(\varphi, t) $",
                        r"$ \mathrm{G}_\mathrm{OH}(\varphi, t) $"
                        ]


                    for item_index, item in enumerate(possibilities):
                        if axis.startswith(item):
                            axis_label = returns[item_index]

                    value_label = axis_label
                    if str_axis == "normal_vhs" or str_axis == "OH_vhs":
                        variable_label = r'$\varphi$'
                    else:
                        variable_label = "$ r~/~(\AA) $"

                    # Plot visuals
                    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
                    sm.set_array([])
                    plt.colorbar(sm, ax=ax2d, label=r"$ t~/~(ps) $", shrink=0.7)

                    ax_only_2d.set_xlabel(variable_label)
                    ax_only_2d.set_ylabel(value_label)
                    ax2d.set_xlabel(variable_label)
                    ax2d.set_ylabel(value_label)
                    ax2d.set_ylim(0, 0.012)
                    ax2d.grid(True)
                    ax3d.set_xlabel(variable_label)
                    ax3d.set_ylabel(r"", labelpad=-5)
                    ax3d.set_zlabel(value_label, labelpad=10)
                    ax3d.yaxis.set_ticks([])
                    ax3d.yaxis.set_ticklabels([])

                    # ax3d.set_yscale("log")
                    # fig.suptitle(f"Layer Distance: {str_slitsize} Å, {str_atm_name}, {str(int_temp)} K")
                    plt.savefig(plot_full_path)

                    # Export .agr
                    plt.figure(fig_only_2d.number)
                    tudplot.saveagr(plot_full_path.replace(".pdf", ".agr"))
                    plt.close(fig_only_2d)

                    print(f"Plot saved to: {plot_full_path} and .agr")
                    plt.close(fig)



    return None


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


main()
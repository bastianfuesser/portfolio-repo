#!/bin/bash

PYTHON_INTERPRETER='/nfsopt/mdevaluate/mdevaluate-24.02/bin/python'

PYTHON_CREATE_NOJUMP='/data/bfuesser/scripts/python/dynamics/create_nojump.py'
PYTHON_MSD_ISF='/data/bfuesser/scripts/python/dynamics/calculate_msd_isf.py'
PYTHON_MSD_ISF_PLOTTING='/data/bfuesser/scripts/python/dynamics/plotting_msd_isf.py'
PYTHON_DIFF_KWW1E_PLOTTING='/data/bfuesser/scripts/python/dynamics/diff_kww1e_plotting.py'
PYTHON_VHS='/data/bfuesser/scripts/python/dynamics/calculate_van_hove_self.py'
PYTHON_VHS_PLOTTING='/data/bfuesser/scripts/python/dynamics/plotting_vhs.py'
PYTHON_ZDENSITY_PROFILE='/data/bfuesser/scripts/python/layered_density/calculate_z_density_profile.py'
PYTHON_ZDENSITY_PROFILE_PLOTTING='/data/bfuesser/scripts/python/layered_density/plot_z_density_profiles.py'

BASH_RDF_CN='/autohome/bfuesser/Documents/main/scripts/bash_scripts/collect_rdf_CN.sh'
WORK_DIR=$(pwd)

#--------------------------------------------------------------------------------------------------
#--- DYNAMICS ---
#--------------------------------------------------------------------------------------------------

# --- Makes sure, nojump trajectories are calculated only once.
if ! $PYTHON_INTERPRETER $PYTHON_CREATE_NOJUMP --system_dir $WORK_DIR; then
    echo "Creating nojump failed! Exiting..."
    exit 1
fi

# 2D MSD, ISF, diffusion, kww1e, mean tau, for HW, OW and Li
(
    $PYTHON_INTERPRETER $PYTHON_MSD_ISF --system_dir $WORK_DIR --axis xy --atom_names Li OW HW
    $PYTHON_INTERPRETER $PYTHON_MSD_ISF_PLOTTING --system_dir $WORK_DIR --axis xy --atom_names Li OW HW
    $PYTHON_INTERPRETER $PYTHON_DIFF_KWW1E_PLOTTING --system_dir $WORK_DIR --atom_names Li OW HW
) &

# 3D MSD and ISF for basal oxygen ob
(
    $PYTHON_INTERPRETER $PYTHON_MSD_ISF --system_dir $WORK_DIR --axis all --atom_names ob
    $PYTHON_INTERPRETER $PYTHON_MSD_ISF_PLOTTING --system_dir $WORK_DIR --axis all --atom_names ob
) &

# VHS for Li, OW, HW in x, y, z and all directions respectivly
(
    $PYTHON_INTERPRETER $PYTHON_VHS --system_dir $WORK_DIR --axis all --atom_names Li OW HW
    $PYTHON_INTERPRETER $PYTHON_VHS_PLOTTING --system_dir $WORK_DIR
) &

(
    $PYTHON_INTERPRETER $PYTHON_VHS --system_dir $WORK_DIR --axis x --atom_names Li OW HW
    $PYTHON_INTERPRETER $PYTHON_VHS_PLOTTING --system_dir $WORK_DIR
) &

(
    $PYTHON_INTERPRETER $PYTHON_VHS --system_dir $WORK_DIR --axis y --atom_names Li OW HW
    $PYTHON_INTERPRETER $PYTHON_VHS_PLOTTING --system_dir $WORK_DIR
) &

(
    $PYTHON_INTERPRETER $PYTHON_VHS --system_dir $WORK_DIR --axis z --atom_names Li OW HW
    $PYTHON_INTERPRETER $PYTHON_VHS_PLOTTING --system_dir $WORK_DIR
) &

#--------------------------------------------------------------------------------------------------
#--- STATICS ---
#--------------------------------------------------------------------------------------------------

# Z density profiles for OW, HW and Li
(
    $PYTHON_INTERPRETER $PYTHON_ZDENSITY_PROFILE --system_dir $WORK_DIR --atom_names Li OW HW
    $PYTHON_INTERPRETER $PYTHON_ZDENSITY_PROFILE_PLOTTING --system_dir $WORK_DIR
)

#--------------------------------------------------------------------------------------------------
#--- Wait condition ---
#--------------------------------------------------------------------------------------------------

wait
bash $BASH_RDF_CN
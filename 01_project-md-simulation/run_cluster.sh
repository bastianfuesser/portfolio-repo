#!/bin/bash -l

set -e

WORK_DIR=$(realpath "$(dirname "$0")")
TARGET_SCRIPT="$WORK_DIR/cluster.sh"
TEMP=300

sed -i "/^#SBATCH --chdir=/c\\#SBATCH --chdir=$WORK_DIR" "$TARGET_SCRIPT"
sed -i "/^WORK_DIR=/c\\WORK_DIR=$WORK_DIR" "$TARGET_SCRIPT"

while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
        -temp|temperature)
        TEMP="$2"
        shift
        shift
        ;;
    *)
        echo "Unknown option: $1"
        exit 1
        ;;
    esac
done

#-----------------------define-paths-----------------------
SETUP_DIR="$WORK_DIR/00_setup"
EQ_DIR="$WORK_DIR/02_eq"
PROD_DIR="$WORK_DIR/03_prod"
TOPOL_FILE="$SETUP_DIR/topol.top"

#-----------------------set-temperature--------------------
sed -i "/^ref-t                    = /c\\ref-t                    = $TEMP" "$EQ_DIR/eq.mdp"
sed -i "/^gen-temp                 = /c\\gen-temp                 = $TEMP" "$EQ_DIR/eq.mdp"
sed -i "/^ref-t                    = /c\\ref-t                    = $TEMP" "$PROD_DIR/prod.mdp"
sed -i "/^gen-temp                 = /c\\gen-temp                 = $TEMP" "$PROD_DIR/prod.mdp"

echo "Temperature set to $TEMP"

echo "Run sbatch..."
sbatch $WORK_DIR/cluster.sh
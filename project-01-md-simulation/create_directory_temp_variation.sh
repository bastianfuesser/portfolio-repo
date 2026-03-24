#!/bin/bash

TMIN=250
TMAX=350
TSTEP=10

TEMPLATE_DIRECTORY='/data/bfuesser/FHEC_local/DoubleFHEC/templates/16_08_01_template'           # Enter template-path here!
WORK_DIR=$(realpath "$(dirname "$0")")

find_directories_with_script() {
  find "$1" -type f -name "run_cluster.sh" -exec dirname {} \;
}

for ((i=$TMIN; i<=$TMAX; i+=TSTEP)); do
    NEW_DIRECTORY="$WORK_DIR/temp_$i"
    mkdir -p "$NEW_DIRECTORY"
    echo "Folder ${NEW_DIRECTORY} created!"

    cp -a "$TEMPLATE_DIRECTORY/"* "$NEW_DIRECTORY"
    echo "Template copied to ${NEW_DIRECTORY}"

    mapfile -t DIRECTORIES < <(find_directories_with_script "$NEW_DIRECTORY")

    for DIR in "${DIRECTORIES[@]}"; do
        sed -i "/^TEMP=/c \\TEMP=$i" "$DIR/run_cluster.sh"
        echo "Temperature changed to ${i}K!"
    done
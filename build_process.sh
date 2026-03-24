#!/bin/bash
mkdir -p build
cp ./start_sim.sh ./build/start_sim.sh
cd ./build/
mkdir -p results
cmake ..
make
bash start_sim.sh

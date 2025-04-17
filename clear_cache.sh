#!/bin/bash

find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +

find . -type d -name "__pycache__" -exec rm -rf {} +

if [ -d "dock_comp" ]; then
    find dock_comp -type d -name "run_[0-9]*" -exec rm -rf {} +
fi

if [ -d "dock_comp" ]; then
    find dock_comp -type f -name "*.json" -delete
fi

rm -r dock_comp/unidock_run/etkdg
rm -r dock_comp/unidock_run/savedir
rm -r dock_comp/unidock_run/workdir
mkdir dock_comp/unidock_run/savedir
mkdir dock_comp/unidock_run/workdir


echo "Done"

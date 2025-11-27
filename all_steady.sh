#!/bin/bash
set -e

if [ $# -ne 1 ]; then
  echo "Usage: $0 <input_directory>"
  exit 1
fi

INPUT_DIR="$1"
EXEC="./heat_cond_steady_state"

# Loop over all .msh files in the input directory
for mesh_file in "$INPUT_DIR"/*.msh; do
    [ -e "$mesh_file" ] || { echo "No .msh files in $INPUT_DIR"; exit 1; }
    mesh_name=$(basename "$mesh_file")

    echo "-------------------------------------------------"
    echo "Running simulation for: $mesh_file"
    echo "-------------------------------------------------"

    "$EXEC" -m "$mesh_file"
done

echo "All steady-state simulations completed."

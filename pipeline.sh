#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status
#set -x  # Uncomment for verbose debug output

echo "===== Starting full thermal pipeline ====="

# # 1️⃣ Generate STEP files
# (
#     cd ../occt_prac/build
#     echo ">>> Generating STEP files"
#     ./more_holes
# )

# 2️⃣ Generate surface meshes
(
    cd ../occt_prac/build
    echo ">>> Generating surface meshes"
    python3 ../mesh_all.py test_step_files surface_meshes 1
)

# 3️⃣ Generate volume meshes
(
    cd ../occt_prac/
    echo ">>> Generating volume meshes"
    ./volume_mesh.sh
)

# # 4️⃣ Run steady state heat conduction simulations
# (
#     echo ">>> Running steady state heat conduction simulations"
#     ./all_steady.sh ../occt_prac/build/vol_meshes
# )

# # 5️⃣ Extract thermal statistics
# (
#     echo ">>> Extracting thermal statistics"
#     python3 heat_stats.py heat_cond_steady_state_solns
# )

echo "===== All executions complete ====="

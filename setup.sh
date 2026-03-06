#!/bin/bash

# Prepare conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda create -n maniskill_env python=3.10 -y
conda activate maniskill_env

# Clone ManiSkill
git clone https://github.com/haosulab/ManiSkill.git
cd ManiSkill

# Install ManiSkill
pip install -e .

# Clean up and (re-)download the PickCube-v1 dataset
rm -rf ~/.maniskill/demos/PickCube-v1
python -m mani_skill.utils.download_demo "PickCube-v1"

# Apply patches to unwrap various objects
git apply fix_wrappers.patch
git add .
git commit -m 'Patched code for accessing priviledge states'

# Then run the replays
python -m mani_skill.trajectory.replay_trajectory \
  --traj-path ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.h5 \
  --use-first-env-state -c pd_ee_delta_pos -o state \
  --save-traj --num-envs 1 -b physx_cpu

#!/bin/bash
cd /home/berrayan/planner/planner_executor_DiscreteDiffusion

# Activate the virtual environment
source llada_env/bin/activate

# Set CUDA devices (adjust as needed)
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Run the training script
python pts/train/train.py



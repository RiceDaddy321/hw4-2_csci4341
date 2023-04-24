#!/bin/bash

### Sets the job's name.
#SBATCH --job-name=hw4_1

### Sets the job's output file and path.
#SBATCH --output=result.out.%j

### Sets the job's error output file and path.
#SBTACH --error=result.err.%j

### Requested number of nodes for this job. Can be a single number or a range.
#SBATCH -N 1

### Requested partition (group of nodes, i.e. compute, bigmem, gpu, etc.) for the resource allocation.
#SBATCH -p kimq

### Requested number of gpus
#SBATCH --gres=gpu:1

### Limit on the total run time of the job allocation.
#SBATCH --time=12:00:00

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

echo "Activating Atari Environment"
conda activate atariEnv

echo "Running testTF.py"
python ~/hw4-1_csci4341/pinball.py

echo "Deactivating Atari environment"
conda deactivate

echo "Done."


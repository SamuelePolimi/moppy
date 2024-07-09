#!/bin/bash
#SBATCH --time=0-20:00
#SBATCH --mem=24000M
#SBATCH --cpus-per-task=8
## #SBATCH --gres=gpu:1
## #SBATCH --output=%x-%j.out

module purge
module load Anaconda3/2023.10/miniconda-base-2023.10
eval "$($UIBK_CONDA_DIR/bin/conda shell.bash hook)"
conda activate moppy

echo "Configurations for job:"
echo $1

python main.py $1

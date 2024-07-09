#!/bin/bash
#SBATCH --time=0-10:00
#SBATCH --mem=24000M
#SBATCH --cpus-per-task=4
## #SBATCH --gres=gpu:1


module purge
module load Anaconda3/2023.10/miniconda-base-2023.10
eval "$($UIBK_CONDA_DIR/bin/conda shell.bash hook)"
conda activate moppy
echo "Configurations for job:"
echo $1

echo "Start:"
python main_sinus.py $1
echo "Finish"
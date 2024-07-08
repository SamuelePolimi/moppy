#!/bin/bash
#SBATCH --time=0-20:00
#SBATCH --mem=24000M
#SBATCH --cpus-per-task=8
## #SBATCH --gres=gpu:1
## #SBATCH --output=%x-%j.out

echo "Configurations for job:"
echo $1

python main.py $1

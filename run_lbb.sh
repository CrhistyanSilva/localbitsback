#!/bin/bash
#SBATCH --job-name=lbb
#SBATCH --ntasks=1
#SBATCH --mem=32768
#SBATCH --time=6:00:00
#SBATCH --tmp=9G
#SBATCH --partition=normal
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cr.silper@gmail.com

export CUDA_VISIBLE_DEVICES=0,1

source /etc/profile.d/modules.sh
source ~/anaconda/bin/activate
conda activate lbb
cd ~/proyecto_grado/lbb/localbitsback
PYTHONPATH=./:compression/ans/build/ python scripts/run_compression.py


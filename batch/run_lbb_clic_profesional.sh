#!/bin/bash
#SBATCH --job-name=lbb_clic_professional
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --partition=normal
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=user@mail
#SBATCH --output=%x_%j.out

source /etc/profile.d/modules.sh
source $CONDA_ACTIVATE
conda activate lbb
cd ~/proyecto_grado/lbb/localbitsback

PYTHONPATH=.:compression/ans/build/ python scripts/run_compression_custom.py --input /clusteruy/home03/compresion_imgRN/professional_valid_cropped/ --dataset imagenet64

#!/bin/bash
#SBATCH --job-name=lbb_clic_mobile
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
source $CONDA_PYTHON_EXE
conda activate lbb
cd ~/proyecto_grado/lbb/localbitsback

PYTHONPATH=.:compression/ans/build/ python scripts/run_compression_custom.py --input /clusteruy/home03/compresion_imgRN/mobile_valid_cropped/ --dataset imagenet64

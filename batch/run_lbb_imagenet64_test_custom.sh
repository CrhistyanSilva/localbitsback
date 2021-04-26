#!/bin/bash
#SBATCH --job-name=lbb_imagenet64
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

PYTHONPATH=./:compression/ans/build/ python scripts/run_compression_custom.py --mode test --input /clusteruy/home03/compresion_imgRN/mobile_valid_cropped/0067.png --dataset imagenet64 --single_image --test_output_filename ~/lbb_output.json


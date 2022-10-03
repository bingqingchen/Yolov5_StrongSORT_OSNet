#!/bin/bash
#SBATCH --job-name=YOLOEvalTest
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8GB
#SBATCH --time=05:00:00
#SBATCH --account=madan
#SBATCH --cpus-per-task=2
#SBATCH --nodelist=node002
#SBATCH --exclude=node005,node006,node007

echo $SLURM_JOB_GPUS
source ~/.bashrc
export CUDA_VISIBLE_DEVICES="${SLURM_JOB_GPUS}"
conda activate yolo_env
cd /scratch/madan/Yolov5_StrongSORT_OSNet
setproxy 
bash scripts/eval_mot16.sh osnet_x0_25_distilled_e99 crowdhuman_yolov5m

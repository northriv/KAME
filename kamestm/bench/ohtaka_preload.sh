#!/bin/bash
#SBATCH --partition=F1cpu
#SBATCH --job-name=kame_preload
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=24:00:00
#SBATCH --mem=100G
#SBATCH --output=ohtaka_%x_%j.log
#SBATCH --error=ohtaka_%x_%j.err

#cd $HOME/KAME/bench
VARIANT=${VARIANT:-full} LABEL=ohtaka KAME_ROOT=$HOME/KAME \
  bash run_v0_full_preload.sh


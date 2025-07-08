#!/bin/bash
#SBATCH --nodes=1                 
#SBATCH --ntasks-per-node=1      
#SBATCH --cpus-per-task=1         
#SBATCH --mem=2G                  
#SBATCH --partition=GPU           
#SBATCH --job-name=GeneratingData
#SBATCH --error=error.log
#SBATCH --output=output.log

source /home/est_posgrado_alfredo.carreras/myenv/bin/activate
srun python generate_data.py --gpu 0
#!/bin/bash
#SBATCH -A ai-for-trap-processing-now
#SBATCH --job-name=my_python_job
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=job_output.log

module load python/3.10
python demo.py

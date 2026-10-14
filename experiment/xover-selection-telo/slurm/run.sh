#!/bin/bash
# Run the whole workflow inside one Slurm allocation, using every core it was given.
#
#     cd experiment/xover-selection-telo
#     SBATCH_ACCOUNT=project_XXXXXXX sbatch slurm/run.sh                    # the full study
#     SBATCH_ACCOUNT=project_XXXXXXX sbatch slurm/run.sh budgets            # stop after phase 1
#
# Submit from this directory: Slurm runs a copy of this script, so the working directory is the only
# reliable pointer back to the workflow. Results go under results/ here, so this directory must be on
# storage the compute node can write to and that outlives the job (scratch, not $TMPDIR).
#
# The workflow resumes where it stopped, so a job that runs out of time is continued by submitting it
# again. Before the first submission, on a login node:
#
#     uv sync --inexact --group experiment                 # DESDEO and Snakemake
#     uv sync --project envs/ela --frozen                  # pflacco's environment (needs the network)
#
# If the compute nodes cannot reach the network, also run the smoke configuration once on the login
# node, so that every package and dataset the full run needs is already in place.
#
#SBATCH --job-name=xover-telo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem-per-cpu=1500M
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit with sbatch from the study directory}"

uv run --group experiment snakemake \
    --cores "${SLURM_CPUS_PER_TASK}" \
    --rerun-incomplete \
    --keep-going \
    --printshellcmds \
    "$@"

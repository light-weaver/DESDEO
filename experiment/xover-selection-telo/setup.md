# Running the study on Roihu

These steps move the study to CSC's Roihu cluster, build its environment and start the run. The
commands use project `project_2020514` and user `bssainix`; change both if yours differ.

The setup and launch steps were rehearsed on a workstation from a shallow clone with an empty uv
cache and uv in offline mode, so compute nodes need no network. They have not been run on Roihu
itself.

## 1. Push the code

On your own machine, from the repository root:

```bash
git push personal experiment/xover-selection-telo
```

Everything the study needs is in git, including the MetallApplication data under `datasets/`.

## 2. Log in

Roihu requires a signed SSH certificate as well as your key, and the certificate lasts 24 hours.

1. Log in to [MyCSC](https://my.csc.fi) and open Profile.
2. Under SSH PUBLIC KEYS, click the three dots next to your key and choose "Sign and download SSH
   certificate".
3. Save the file beside your private key as `<key>-cert.pub`, for example
   `~/.ssh/id_ed25519-cert.pub`.

Then connect to the CPU side:

```bash
ssh bssainix@roihu-cpu.csc.fi
```

The GPU login nodes are ARM. Anything built there will not run on the CPU nodes.

## 3. Set up the environment

Do this once, on the login node. Run `csc-workspaces` first and check the project path.

The caches go on scratch because `$HOME` allows only 15 GiB and 150 000 files. `sbatch` passes these
variables on to the job.

```bash
cat >> ~/.bashrc <<'EOF'
export SCR=/scratch/project_2020514/$USER
export SBATCH_ACCOUNT=project_2020514
export UV_CACHE_DIR=$SCR/.cache/uv
export XDG_CACHE_HOME=$SCR/.cache
export PATH=$HOME/.local/bin:$PATH
EOF
source ~/.bashrc

curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.13 3.11

mkdir -p $SCR && cd $SCR
git clone --depth 1 --branch experiment/xover-selection-telo https://github.com/light-weaver/DESDEO.git
cd DESDEO/experiment/xover-selection-telo

uv sync --frozen --inexact --python 3.13 --group experiment
uv sync --project envs/ela --frozen
```

If the repository is private, the clone needs a GitHub token. Python 3.13 is for DESDEO and Python
3.11 is for pflacco, which runs in its own environment under `envs/ela`.

The old study's `roihu/` scripts and Tykky container belong to the previous pipeline and are not used
here. uv installs exactly what `uv.lock` pins, so Roihu runs the same versions as the workstation.

## 4. Run the smoke configuration

Run this from the study directory:

```bash
sbatch --partition=test --time=00:15:00 --cpus-per-task=8 slurm/run.sh --configfile config/smoke.yaml
```

`slurm-<jobid>.out` should end with `86 of 86 steps (100%) done`. It took about 2.5 minutes on 4
cores of a workstation.

If it takes far longer here, suspect Python start-up on Lustre. The study starts Python more than
20 000 times, and this cost has not been measured on Roihu. Move the environment into a Tykky
container if it turns out to be the bottleneck.

Then delete the smoke results:

```bash
rm -rf results_smoke
```

## 5. Start the full run

`slurm/run.sh` sets no partition, so give it on the command line:

```bash
sbatch --partition=small slurm/run.sh
```

To watch it:

```bash
squeue --me
tail -f slurm-<jobid>.out
```

The job plans 22 504 steps. Expect at least 7.5 hours on 128 cores. That estimate comes from the EMO
paper's rates and leaves out calibration and the landscape features, and AGE-MOEA-II may push it
higher. The script asks for 24 hours.

At the 0.75 billing units per core-hour the old study recorded, 7.5 hours costs about 720 units.
Check that the rate still applies and that the project has enough.

The run should use about 133 000 files on scratch, against a default limit of 500 000: about 34 000
results, as many Snakemake metadata files, and 65 000 for the environments and the clone. The results
take 12 GiB, well under the 250 GiB limit.

## Adding arms to a finished run

Add the arm to `crossovers` and to `run.crossovers` in `config/config.yaml`, push, then `git pull` on
Roihu and submit the same command as for the full run. Snakemake keeps the existing run tars, because
each block depends only on its own crossover's options and the instance budget. It runs the new
blocks of the main grid and the new arms' front runs. Then it re-freezes every reference front and
rescores every instance, since `reference_front.crossovers: all` puts the new arms into the fronts.

Check the plan first with `uv run --group experiment snakemake -n` from the study directory. For
the three SBX arms added on 2026-11-07 it should list 2 250 `run_block`, 2 250 `front_runs`, 150
`reference_front` and 750 `score` jobs, plus the targets that collect them. Many more `run_block`
jobs means Snakemake has decided to rerun the existing grid; stop and find out why before
submitting.

## If the job stops

Submit the same command again. The workflow continues from the finished steps.

If Snakemake says the directory is locked, clear the lock first:

```bash
uv run --group experiment snakemake --unlock
```

## Updating the code

Push from your machine, then on Roihu:

```bash
git pull
uv sync --frozen --inexact --python 3.13 --group experiment
```

Do not edit code while a job is running. Snakemake reads each script when its step starts.

## Afterwards

Scratch deletes files that have not been accessed for 180 days. Copy the results off Roihu once the
run finishes. From your own machine:

```bash
rsync -av bssainix@roihu-cpu.csc.fi:/scratch/project_2020514/bssainix/DESDEO/experiment/xover-selection-telo/results/ results/
```

That copies 12 GiB, almost all of it raw runs:

| Folder | Size |
|---|---|
| `runs/` | 6.2 GiB |
| `front_runs/` | 5.2 GiB |
| `indicators/` | 242 MiB |
| `landscape/` | 117 MiB |
| `mechanism/` | 13 MiB |
| `fronts/` | 5.9 MiB |
| `budgets/` | 1.2 MiB |

Then check the copy from the study directory:

```bash
uv run --group experiment python -m analysis.check
```

It fails if any front or table is missing, was built from fewer runs than the config asks for, or
was scored against an earlier front. A copy taken while the job is still running fails it.

If disk space is short, the analysis tables alone take about 400 MiB:

```bash
rsync -av --exclude='/runs/' --exclude='/front_runs/' \
  bssainix@roihu-cpu.csc.fi:/scratch/project_2020514/bssainix/DESDEO/experiment/xover-selection-telo/results/ results/
```

`front_runs/` only feeds the reference fronts, which are frozen in `fronts/`. `runs/` holds the raw
runs, one tar per block, and some of their contents reach no table:

- run timings, generation and message counts, and per-run provenance;
- the populations at each snapshot, needed for a new indicator or a plot of the populations;
- the per-generation mechanism series, since `mechanism/` keeps only the mean and final value.

If you skip these folders, copy them somewhere else before scratch deletes them. Getting them back
means repeating the run. See [README.md](README.md) for the output layout.

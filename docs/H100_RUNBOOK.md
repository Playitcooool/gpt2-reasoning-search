# One-H100 runbook

## Start the pipeline

Use the repository launcher. It installs the locked environment, prepares the pinned data, and
submits three separate eight-hour jobs. Edit only the scheduler settings in `config/ssh.env`.

```bash
git clone git@github.com:Playitcooool/gpt2-reasoning-search.git
cd gpt2-reasoning-search
./train-ssh setup
# Set SLURM_ACCOUNT, SLURM_TIME, and any required SLURM_PARTITION/SLURM_GRES.
scripts/slurm/submit_8h_pipeline.sh
squeue -u "$USER"
```

The wrapper runs `doctor` separately when a batch job starts and runs data preparation before
submitting the GPU jobs. It selects the fixed repository paths; no manual data-file setup is
required.

## Eight-hour stage profile

The default profile deliberately omits proxy ablations. Each stage leaves approximately 30 minutes
for scheduler startup and checkpoint finalization:

- 350M 70% main pretraining: 7.5 hours, up to a 2.5B-token cap;
- tool SFT: 7.5 hours;
- search RL: 7.5 hours, with the revision-pinned Qwen/Qwen3.5-2B judge enabled by default.

The trainer calibrates throughput and reduces the effective token cap if needed. A cap is a maximum,
not a promise that every reservation consumes it.

The optional equal-budget 0%/30%/70% proxy study requires separate reservations. Preserve those
results separately from the default pipeline.

## Resume and monitoring

Metrics are appended to `metrics.jsonl`. Watch loss, gradient norm, learning rate, tokens/second,
MFU estimate, peak memory, reasoning/general token counters, and the calibrated final token budget.
Direct single-stage retries use the stage-specific batch script and resume from the newest complete checkpoint:

```bash
scripts/slurm/submit_stage.sh pretrain
# Equivalent direct Slurm invocation with the checked-in eight-hour defaults:
# sbatch scripts/slurm/pretrain.sbatch
```

If a stage reaches its walltime, cancel pending dependent jobs and rerun
`scripts/slurm/submit_8h_pipeline.sh`. Completed stages are skipped, and the incomplete stage
resumes automatically. Do not combine metrics from runs with different tokenizer hashes, data
hashes, model configuration, or token budgets.

## Serving after training

After the main checkpoint completes, the dependent SFT and RL jobs run automatically. For a direct
server, use:

```bash
./train-ssh sft
./train-ssh rl
```

The default auxiliary judge is revision-pinned `Qwen/Qwen3.5-2B`, run greedily with a 4,096-token
input cap. Its roughly 2B parameters are practical on an 80 GB H100 alongside the 350M policy and
frozen reference when generation is sequential. Do not use the model's 262K maximum context here.

Compare tool-SFT and RL checkpoints on the frozen held-out set before deploying the RL checkpoint.
Then start the service. Keep the API bound to localhost unless it
is behind authenticated TLS termination. Configure secrets through the host secret manager, not a
repository file. The default single-request concurrency avoids unsafe simultaneous generation on
one model instance; raise it only after memory and latency measurement.

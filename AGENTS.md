# Markdown Guidelines

- Use **absolute paths** instead of relative paths.

# Python Guidelines

- At the top of each file, include a **docstring** with simple instructions on how to run the code.
- When writing new code: preview existing code first, reuse existing modules where possible, and keep the new code’s style consistent with the codebase.
- Before running scripts: source `~/.zshrc` and activate conda env `dllm` (e.g. `conda activate ~/miniconda3/envs/dllm`).
- For tasks requiring a GPU, use the following command: `srun -p $PARTITION -q=$QUOTATYPE --gres=gpu:1 --cpus-per-task=24 --time=03:00:00 python ...`.

# Agent Guidelines
- Never run a salloc, sbatch or srun command on your own. Give the commands to run to the user so they can run them for you.
- When checking things in the GPU or the server machine, NEVER use CUDA_VISIBLE_DEVICES anywhere.

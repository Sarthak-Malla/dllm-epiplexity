# Ablation 5: Token-generation temperature

Evaluate token-generation temperatures **0.5 and 0.8** using entropy budget 2,
uncapped within the 64-token block (`dependency_max_action_size=64`). These
temperatures enable Gumbel noise when predicting token values. Candidate seed
position temperature remains 1.0.

Both runs retain the reference configuration: full GSM8K, five-shot prompting,
four soft-full dependency candidates, entropy-drop lookahead, per-token scoring,
256 generated tokens, and the same random seeds. Each independent job requests two
GPUs, 32 CPUs, 64 GB RAM, and 4 hours 30 minutes. Running both jobs concurrently
uses four GPUs. Submit with the following two commands; the exclusion avoids
`gpu-51`, where the previous attempts failed during CUDA initialization:

```bash
sbatch --exclude=gpu-51 /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/ablation/5_token_temperature/run_gsm8k_token_temperature0.5.slurm.sh
sbatch --exclude=gpu-51 /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/ablation/5_token_temperature/run_gsm8k_token_temperature0.8.slurm.sh
```

Each launcher fixes its temperature and receives its own job ID. Submit only the
corresponding command if you want one temperature. Logs use the job name and job
ID under `/home/sarthak.malla/dllm-selection-ensemble/.logs`.

Results, response caches, and completion markers are separated by temperature at
`/home/sarthak.malla/dllm-selection-ensemble/eval_results/path_selection/ablation/5_token_temperature/token_temperature_v1/gsm8k_cot/entropy_budget2.0/temperature<T>/seed42`.
W&B names and configuration include token temperature. `ABLATION5_RUN_TAG`
overrides the run tag; completed runs use the shared runner's skip behavior.

The temperature-zero reference is already available at
`/home/sarthak.malla/dllm-selection-ensemble/eval_results/path_selection/ablation/2_entropy_budget_without_k_limit/vectorized_soft_full_v1/gsm8k_cot/entropy_budget2.0/seed42`.
The `seed42` directory label refers to dependency proposal generation; the
evaluation seed tuple remains `0,1234,1234,1234`, including PyTorch seed 1234
for token-sampling randomness.

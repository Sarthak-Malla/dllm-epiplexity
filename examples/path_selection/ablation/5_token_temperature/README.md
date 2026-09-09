# Ablation 5: Token-generation temperature

Evaluate token-generation temperatures **0.5 and 1.0** using entropy budget 2,
uncapped within the 64-token block (`dependency_max_action_size=64`). These
temperatures enable Gumbel noise when predicting token values. Candidate seed
position temperature remains 1.0.

Both runs retain the reference configuration: full GSM8K, five-shot prompting,
four soft-full dependency candidates, entropy-drop lookahead, per-token scoring,
256 generated tokens, and the same random seeds. Each array task requests two
GPUs, 32 CPUs, 64 GB RAM, and 16 hours. Running both tasks concurrently uses four
GPUs. Submit both with:

```bash
sbatch /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/ablation/5_token_temperature/run_gsm8k_token_temperature.slurm.sh
```

Array task 0 uses temperature 0.5; task 1 uses 1.0. To run the tasks sequentially,
pass `--array=0-1%1`. To submit only one temperature:

```bash
sbatch --array=0 --export=ALL,ABLATION5_TEMPERATURE=1.0 /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/ablation/5_token_temperature/run_gsm8k_token_temperature.slurm.sh
```

Dry-run both configurations without loading a model or creating result directories
after sourcing the shell setup and activating the `dllm` environment:

```bash
for temperature in 0.5 1.0; do
    ABLATION2_DRY_RUN=1 ABLATION5_TEMPERATURE=${temperature} bash /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/ablation/5_token_temperature/run_gsm8k_token_temperature.slurm.sh
done
```

Results, response caches, and completion markers are separated by temperature at
`/home/sarthak.malla/dllm-selection-ensemble/eval_results/path_selection/ablation/5_token_temperature/token_temperature_v1/gsm8k_cot/entropy_budget2.0/temperature<T>/seed42`.
W&B names and configuration include token temperature. `ABLATION5_RUN_TAG`
overrides the run tag; completed runs use the shared runner's skip behavior.

The temperature-zero reference is already available at
`/home/sarthak.malla/dllm-selection-ensemble/eval_results/path_selection/ablation/2_entropy_budget_without_k_limit/vectorized_soft_full_v1/gsm8k_cot/entropy_budget2.0/seed42`.
The `seed42` directory label refers to dependency proposal generation; the
evaluation seed tuple remains `0,1234,1234,1234`, including PyTorch seed 1234
for token-sampling randomness.

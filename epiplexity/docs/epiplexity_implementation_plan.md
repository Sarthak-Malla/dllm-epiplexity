## Plan: Implement Epiplexity in LLaDA SFT

This plan details integrating Epiplexity into the Masked Diffusion Language Modeling (MDLM) SFT pipeline leveraging PEFT to avoid Out-Of-Memory errors.

**Steps**
1. **Prepare File Structure**
   - Create `epiplexity/llada_sft_epiplexity.py` as the main entry point (modeled after `examples/llada/sft.py`).
   - Create `epiplexity/utils.py` for epiplexity-specific helper functions (e.g., model cloning via LoRA adapters).

2. **Integrate PEFT (LoRA) for Memory-Safe "Cloning"**
   - Wrapping the 7B/8B parameter LLM in a `peft` LoRA configuration.
   - Instead of cloning the full model, we maintain a base frozen model and instantiate two separate active LoRA adapters (`adapter_A` and `adapter_B`).
   - We will need to maintain two separate optimizer instances (one for `adapter_A` parameters and one for `adapter_B` parameters) to quickly test the duel without having to manually rewind optimizer states.

3. **Implement The Epiplexity Duel Logic**
   - **Duel Mechanism**: The masking ratio will drive the duel. `adapter_A` will see inputs mapped with a low mask schedule bias, `adapter_B` with a high mask bias.
   - **Training Loop Setup**: Implement a training step that processes:
     1. Switch active adapter to `adapter_A`, compute `loss_before_A`, `backward()`, `step_A()`.
     2. Re-evaluate on same batch to compute `loss_after_A`. Calculate $\Delta_A$.
     3. Switch active adapter to `adapter_B`, compute `loss_before_B`, `backward()`, `step_B()`.
     4. Re-evaluate on same batch to compute `loss_after_B`. Calculate $\Delta_B$.
     5. Winner selection: The adapter with the higher $\Delta$ overwrites the weights of the loser adapter. We sync their states and proceed to the next step.

4. **Address Training Iteration (HF Trainer vs Custom Loop)**
   - *Consideration*: Because Epiplexity requires manually stepping the optimizer *during* a single forward/backward conceptual step, Hugging Face `Trainer`'s rigid state logic gets in the way.
   - *Action*: We will use a custom `Accelerate` PyTorch training loop inside `epiplexity/llada_sft_epiplexity.py`. This mirrors the control shown in the Alchemy notebook while naturally scaling across GPUs via the repository's `accelerate_configs/`.

**Relevant files**
- `epiplexity/llada_sft_epiplexity.py` — Main custom training script.
- `dllm/core/trainers/mdlm.py` — We will import the `compute_loss` logic from `MDLMTrainer` as a standalone utility function within our custom loop so we don't have to duplicate the diffusion noise math.
- `examples/llada/sft.py` — Reference for loading the `tulu` dataset and `default_sft_map_fn`.

**Verification**
1. Test PEFT adapter switching visually by asserting the weights of A and B are identical after a sync step but diverge after a duel step.
2. Run single-batch overfitting: Ensure that over consecutive iterations, the masking schedule chosen by the Duel successfully drives `loss_after` to 0.
3. Compare the peak GPU VRAM use on one local step against standard SFT to ensure our 2-adapter strategy fits.

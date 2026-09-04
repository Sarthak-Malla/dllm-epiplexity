"""Decoding-risk path-selection sampler.

Run its focused CPU integration tests with:
    source /apps/local/conda_init.sh
    conda activate /home/sarthak.malla/.conda/envs/dllm
    pytest /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_dependency_guided_decoder.py -v
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F

from dllm.core.samplers.base import BaseSamplerOutput
from dllm.core.samplers.batched_lookahead import (
    candidate_batch_from_mask_mapping,
    evaluate_batched_lookahead,
)
from dllm.core.samplers.counterfactual import decoding_risk_per_token
from dllm.core.samplers.dependency_guided import (
    DependencyGuidedSamplerConfig,
    build_step_diagnostics,
    is_fixed_k_strategy,
    resolve_dependency_guided_config,
    run_base_forward_with_cfg_inputs,
    select_fixed_k_candidate,
    validate_fixed_k_schedule,
)
from dllm.core.samplers.mdlm import MDLMSampler
from dllm.core.samplers.utils import add_gumbel_noise, get_num_transfer_tokens
from dllm.core.samplers.parallel_candidates import (
    initialize_committed_anchor_state,
    record_committed_anchors,
)


@dataclass
class RiskReductionSamplerConfig(DependencyGuidedSamplerConfig):
    # The heuristic for generating candidate masks.
    risk_candidate_strategy: str = "mixed"
class RiskReductionSampler(MDLMSampler):
    """
    Epiplexity sampler using held-out decoding-risk reduction as the verifier.

    For each candidate reveal set U, the sampler reveals the current model
    predictions at U, runs one lookahead pass, and selects the candidate that
    most reduces 1 - max probability on the positions still masked after U.
    """

    def get_decoding_risk_per_token(self, logits: torch.Tensor) -> torch.Tensor:
        """Calculate decoding risk, 1 - max probability, for each position [B, T]."""
        return decoding_risk_per_token(logits)

    def generate_candidate_sets(
        self,
        confidence: torch.Tensor,
        mask_idx: torch.Tensor,
        num_transfer: Union[int, List[int], torch.Tensor],
        strategy: str = "mixed",
    ) -> Dict[str, torch.Tensor]:
        """
        Generate different sets of token indices to reveal (U).
        Returns a dictionary of boolean tensors matching mask_idx shape.
        """
        B, _ = mask_idx.shape
        num_transfer = self._normalize_num_transfer(num_transfer, B, mask_idx.device)

        if strategy == "mixed":
            # candidate_names = ["greedy", "spaced_0", "spaced_1"]
            candidate_names = ["spaced_0", "spaced_1"]
        elif strategy == "greedy":
            candidate_names = ["greedy"]
        elif strategy == "high_entropy":
            candidate_names = ["high_entropy"]
        elif strategy == "random":
            candidate_names = ["random"]
        else:
            raise ValueError(f"Unknown risk_candidate_strategy: {strategy}")

        candidates = {
            name: torch.zeros_like(mask_idx, dtype=torch.bool)
            for name in candidate_names
        }

        for b in range(B):
            valid_indices = torch.where(mask_idx[b])[0]
            num_valid = valid_indices.numel()
            k = int(num_transfer[b].item())
            k = max(0, min(k, num_valid))
            if k == 0:
                continue

            batch_conf = torch.where(mask_idx[b], confidence[b], -torch.inf)

            if strategy == "greedy":
                _, greedy_idx = torch.topk(batch_conf, k=k)
                candidates["greedy"][b, greedy_idx] = True

            if strategy == "high_entropy":
                inv_conf = torch.where(mask_idx[b], -confidence[b], -torch.inf)
                _, high_ent_idx = torch.topk(inv_conf, k=k)
                candidates["high_entropy"][b, high_ent_idx] = True

            if strategy == "mixed":
                for offset in range(2):
                    step_float = num_valid / k
                    start_offset = (offset * step_float) / 2
                    float_indices = (
                        torch.arange(k, device=valid_indices.device) * step_float
                        + start_offset
                    )
                    int_indices = float_indices.long().clamp(max=num_valid - 1)
                    selected_idx = valid_indices[int_indices]
                    candidates[f"spaced_{offset}"][b, selected_idx] = True

            if strategy == "random":
                rand_idx = valid_indices[
                    torch.randperm(num_valid, device=valid_indices.device)[:k]
                ]
                candidates["random"][b, rand_idx] = True

        return candidates

    def _normalize_num_transfer(
        self,
        num_transfer: Union[int, List[int], torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Normalize scalar or per-row transfer counts to a [B] tensor."""
        if isinstance(num_transfer, torch.Tensor):
            transfer = num_transfer.to(device=device, dtype=torch.long).flatten()
        elif isinstance(num_transfer, int):
            transfer = torch.full(
                (batch_size,), num_transfer, device=device, dtype=torch.long
            )
        else:
            transfer = torch.as_tensor(
                num_transfer, device=device, dtype=torch.long
            ).flatten()

        if transfer.numel() == 1 and batch_size > 1:
            transfer = transfer.expand(batch_size)
        if transfer.numel() != batch_size:
            raise ValueError(
                f"num_transfer must be scalar or length {batch_size}, got {transfer.numel()}"
            )
        return transfer

    def _select_best_candidate(
        self,
        x: torch.Tensor,
        x0: torch.Tensor,
        mask_index: torch.Tensor,
        candidates: Dict[str, torch.Tensor],
        attention_mask: torch.Tensor,
        base_risk_map: torch.Tensor,
        candidate_chunk_size: Optional[int] = None,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor], List[str]]:
        """Evaluate lookahead candidates and select the best mask per batch row."""
        candidate_batch = candidate_batch_from_mask_mapping(
            candidates,
            eligible_mask=mask_index,
        )
        result = evaluate_batched_lookahead(
            self.model,
            x,
            x0,
            candidate_batch,
            base_metric_map=base_risk_map,
            metric="risk_reduction",
            attention_mask=attention_mask,
            masked_active_mask=mask_index,
            candidate_chunk_size=candidate_chunk_size,
        )
        risk_reductions = {
            name: result.scores[index]
            for index, name in enumerate(candidate_batch.names)
        }
        best_candidate_names = [name or "" for name in result.best_names]
        return result.best_mask, risk_reductions, best_candidate_names

    @torch.no_grad()
    def sample(
        self,
        inputs: List[Union[torch.Tensor, List[int]]],
        config: Optional[RiskReductionSamplerConfig] = None,
        **kwargs,
    ) -> Union[BaseSamplerOutput, torch.Tensor]:
        if config is None:
            config = RiskReductionSamplerConfig()
        config = resolve_dependency_guided_config(config, kwargs)

        steps = kwargs.get("steps", config.steps)
        max_new_tokens = kwargs.get("max_new_tokens", config.max_new_tokens)
        max_length = kwargs.get("max_length", config.max_length)
        block_size = kwargs.get("block_size", config.block_size)
        temperature = kwargs.get("temperature", config.temperature)
        cfg_scale = kwargs.get("cfg_scale", config.cfg_scale)
        cfg_keep_tokens = kwargs.get("cfg_keep_tokens", config.cfg_keep_tokens)
        suppress_tokens = kwargs.get("suppress_tokens", config.suppress_tokens)
        stochastic_transfer = kwargs.get(
            "stochastic_transfer", config.stochastic_transfer
        )
        return_dict = kwargs.get("return_dict", config.return_dict)
        right_shift_logits = kwargs.get("right_shift_logits", config.right_shift_logits)
        begin_suppress_tokens = kwargs.get(
            "begin_suppress_tokens", config.begin_suppress_tokens
        )
        risk_candidate_strategy = kwargs.get(
            "risk_candidate_strategy", config.risk_candidate_strategy
        )
        candidate_chunk_size = kwargs.get(
            "candidate_chunk_size",
            config.candidate_chunk_size,
        )
        proposal_strategy = config.proposal_strategy
        cardinality_strategy = config.dependency_cardinality_strategy
        diagnostic_metadata = config.diagnostic_metadata

        assert 1 <= block_size
        assert 1 <= steps
        mask_id = self.tokenizer.mask_token_id
        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id

        if right_shift_logits:
            inputs = [
                [bos_id] if isinstance(p, list) and len(p) == 0 else p
                for p in inputs
            ]

        if isinstance(inputs[0], list):
            inputs = [
                torch.as_tensor(p, dtype=torch.long, device=self.model.device)
                for p in inputs
            ]
        prompt_lens = [p.shape[0] for p in inputs]

        if max_new_tokens:
            max_length = max_new_tokens + max(prompt_lens)
        else:
            max_new_tokens = max_length - max(prompt_lens)

        B = len(inputs)
        T = max_length

        x = torch.full((B, T), eos_id, dtype=torch.long, device=self.model.device)
        for i, p in enumerate(inputs):
            x[i, : prompt_lens[i]] = p
            x[i, prompt_lens[i] : prompt_lens[i] + max_new_tokens] = mask_id

        attention_mask = torch.zeros((B, T), dtype=torch.long, device=self.model.device)
        response_mask = torch.zeros((B, T), dtype=torch.bool, device=self.model.device)
        for i, pl in enumerate(prompt_lens):
            valid_end = min(pl + max_new_tokens, T)
            attention_mask[i, :valid_end] = 1
            response_mask[i, pl:valid_end] = True

        unmasked_index = (x != mask_id) & attention_mask.bool()
        if cfg_keep_tokens and len(cfg_keep_tokens) > 0:
            keep_mask = torch.isin(
                x, torch.as_tensor(cfg_keep_tokens, device=self.model.device)
            )
            unmasked_index = unmasked_index & ~keep_mask

        num_blocks = math.ceil(max_new_tokens / block_size)
        steps = math.ceil(steps / num_blocks)
        histories = [x.clone()] if return_dict else None
        selected_candidates = [[] for _ in range(B)]
        diagnostics = [[] for _ in range(B)] if diagnostic_metadata else None
        global_step_index = 0

        for b in range(num_blocks):
            anchor_state = (
                initialize_committed_anchor_state(
                    x,
                    confidence_threshold=(
                        config.dependency_anchor_confidence_threshold
                    ),
                )
                if proposal_strategy == "dependency"
                else None
            )
            block_mask_index = torch.zeros(
                (B, block_size), dtype=torch.bool, device=x.device
            )
            block_span_mask = torch.zeros_like(x, dtype=torch.bool)

            for j in range(B):
                start = prompt_lens[j] + b * block_size
                end = min(start + block_size, prompt_lens[j] + max_new_tokens, T)
                if start < end:
                    block_mask_index[j, : end - start] = x[j, start:end] == mask_id
                    block_span_mask[j, start:end] = True

            if cardinality_strategy in {
                "marginal_utility",
                "joint_k",
                "entropy_budget",
            }:
                num_transfer_tokens = None
                effective_steps = int(block_mask_index.sum(dim=-1).max().item())
            else:
                num_transfer_tokens = get_num_transfer_tokens(
                    mask_index=block_mask_index,
                    steps=steps,
                    scheduler=self.scheduler,
                    stochastic=stochastic_transfer,
                )
                effective_steps = num_transfer_tokens.size(1)
            if (
                is_fixed_k_strategy(proposal_strategy)
                and cardinality_strategy == "fixed"
            ):
                validate_fixed_k_schedule(
                    num_transfer_tokens,
                    (
                        config.dependency_commit_k
                        if proposal_strategy == "dependency"
                        else 1
                    ),
                )

            for i in range(effective_steps):
                if num_transfer_tokens is None:
                    remaining_by_row = (
                        (x == mask_id) & block_span_mask
                    ).sum(dim=-1, dtype=torch.long)
                    num_transfer = torch.minimum(
                        remaining_by_row,
                        torch.full_like(
                            remaining_by_row,
                            config.dependency_max_action_size,
                        ),
                    )
                else:
                    num_transfer = num_transfer_tokens[:, i]
                if torch.all(num_transfer == 0):
                    if num_transfer_tokens is None:
                        break
                    continue

                mask_index = x == mask_id
                current_block_mask = mask_index & block_span_mask

                unconditional_ids = None
                if cfg_scale > 0.0:
                    unconditional_ids = x.clone()
                    unconditional_ids[unmasked_index] = mask_id
                base_forward = run_base_forward_with_cfg_inputs(
                    self.model,
                    x,
                    attention_mask,
                    cfg_scale=cfg_scale,
                    unconditional_input_ids=unconditional_ids,
                    capture_dependency=proposal_strategy == "dependency",
                    dependency_last_n_layers=config.dependency_last_n_layers,
                    measure_timing=diagnostic_metadata,
                )
                logits = base_forward.logits

                if suppress_tokens and len(suppress_tokens) > 0:
                    for token_id in suppress_tokens:
                        logits[:, :, token_id] = -torch.inf

                if right_shift_logits:
                    logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)

                if begin_suppress_tokens and len(begin_suppress_tokens) > 0:
                    for token_id in begin_suppress_tokens:
                        logits[:, :, token_id] = -torch.inf

                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                )
                confidence = torch.where(current_block_mask, x0_p, -float("inf"))

                base_risk_map = self.get_decoding_risk_per_token(logits)
                if proposal_strategy == "legacy":
                    candidates = self.generate_candidate_sets(
                        confidence,
                        current_block_mask,
                        num_transfer,
                        strategy=risk_candidate_strategy,
                    )
                    best_c_idx, _, best_candidate_names = self._select_best_candidate(
                        x=x,
                        x0=x0,
                        mask_index=mask_index,
                        candidates=candidates,
                        attention_mask=attention_mask,
                        base_risk_map=base_risk_map,
                        candidate_chunk_size=candidate_chunk_size,
                    )
                else:
                    active_rows = num_transfer > 0
                    proposal_mask = current_block_mask & active_rows[:, None]
                    step_seed = config.dependency_generation_seed + global_step_index
                    entropy_map = -torch.sum(
                        p.float() * torch.log(p.float().clamp_min(1e-12)),
                        dim=-1,
                    )
                    selection = select_fixed_k_candidate(
                        self.model,
                        x,
                        x0,
                        base_forward=base_forward,
                        base_metric_map=base_risk_map,
                        entropy_map=entropy_map,
                        confidence=x0_p,
                        metric="risk_reduction",
                        active_mask=proposal_mask,
                        requested_k=num_transfer,
                        anchor_state=anchor_state,
                        masked_active_mask=mask_index,
                        response_mask=response_mask,
                        attention_mask=attention_mask,
                        config=config,
                        generation_seed=step_seed,
                    )
                    best_c_idx = selection.lookahead.best_mask
                    best_candidate_names = [
                        name or "" for name in selection.lookahead.best_names
                    ]
                    anchor_state_before = anchor_state
                    if anchor_state is not None:
                        anchor_state_after = record_committed_anchors(
                            anchor_state,
                            best_c_idx,
                            x0_p,
                            x0,
                            commit_step=global_step_index,
                        )
                    else:
                        anchor_state_after = None
                    if diagnostics is not None:
                        step_diagnostics = build_step_diagnostics(
                            selection,
                            base_forward,
                            config=config,
                            metric="risk_reduction",
                            masked_active_mask=mask_index,
                            response_mask=response_mask,
                            block_index=b,
                            step_index=i,
                            global_step_index=global_step_index,
                            generation_seed=step_seed,
                            base_metric_map=base_risk_map,
                            predicted_token_ids=x0,
                            anchor_state_before=anchor_state_before,
                            anchor_state_after=anchor_state_after,
                        )
                        for batch_index, record in enumerate(step_diagnostics):
                            diagnostics[batch_index].append(record)
                    anchor_state = anchor_state_after

                for b_idx in range(B):
                    if best_candidate_names[b_idx]:
                        if best_candidate_names[b_idx] not in selected_candidates[b_idx]:
                            selected_candidates[b_idx].append(best_candidate_names[b_idx])

                x[best_c_idx] = x0[best_c_idx]

                if return_dict:
                    histories.append(x.clone())
                global_step_index += 1

        if return_dict:
            return BaseSamplerOutput(
                sequences=x,
                histories=histories,
                selected_candidates=selected_candidates,
                diagnostics=diagnostics,
            )
        return x

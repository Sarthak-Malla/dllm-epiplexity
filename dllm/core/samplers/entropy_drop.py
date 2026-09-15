"""Entropy-drop path-selection sampler.

Run its focused integration tests on a compute node with:
    source ~/.zshrc
    conda activate ~/miniconda3/envs/dllm
    pytest /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_dependency_guided_decoder.py -v
"""

import math
import time
import torch
import torch.nn.functional as F
from dataclasses import dataclass, fields, replace
from typing import Dict, Optional, List, Union

from dllm.core.samplers.base import BaseSamplerOutput
from dllm.core.samplers.adaptive_cardinality import apply_size_aware_scoring
from dllm.core.samplers.decoding_state import (
    CommitValues, DecodeState, PreparedStep, RNGState, StepSelection, isolated_rng,
)
from dllm.core.samplers.batched_lookahead import (
    candidate_batch_from_mask_mapping,
    evaluate_batched_lookahead,
)
from dllm.core.samplers.counterfactual import entropy_per_token
from dllm.core.samplers.dependency_guided import (
    DependencyGuidedSamplerConfig,
    FixedKSelectionOutput,
    _build_baseline_candidates,
    _synchronize_for_timing,
    build_confidence_threshold_candidates,
    build_dependency_candidates,
    build_step_diagnostics,
    dependency_capture_required,
    is_fixed_k_strategy,
    release_dependency_capture_tensors,
    resolve_dependency_guided_config,
    run_base_forward_with_cfg_inputs,
    validate_fixed_k_schedule,
)
from dllm.core.samplers.mdlm import MDLMSampler
from dllm.core.samplers.non_lookahead import (
    NON_LOOKAHEAD_SELECTORS,
    select_candidates_without_lookahead,
    top2_probability_margin,
)
from dllm.core.samplers.utils import add_gumbel_noise, get_num_transfer_tokens
from dllm.core.samplers.parallel_candidates import (
    initialize_committed_anchor_state,
    record_committed_anchors,
)

@dataclass
class EntropyDropSamplerConfig(DependencyGuidedSamplerConfig):
    """
    Configuration for the EntropyDropSampler.
    """
    # The heuristic for generating candidate masks
    # "mixed" generates top-entropy and spaced anchors
    oracle_candidate_strategy: str = "mixed"
    dependency_candidate_selector: str = "entropy_drop"
class EntropyDropSampler(MDLMSampler):
    """
    Epiplexity Oracle Sampler with Entropy Drop for path selection.

    Uses a 1-step lookahead to evaluate candidate sets of unmasked tokens based on
    their "Structure Gain" quantified by the entropy drop in the model's predictions.
    Structure Gain proxy = Expected baseline future entropy - Lookahead future entropy

    This uses Entropy Drop as the epiplexity proxy, which is not quite the proxy we want.
    """

    def get_entropy_per_token(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute the entropy per token given the logits.

        Args:
            logits (torch.Tensor): The logits from the model of shape (batch_size, seq_len, vocab_size).

        Returns:
            torch.Tensor: The entropy per token of shape (batch_size, seq_len) => [B, T].
        """
        return entropy_per_token(logits)
    

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
        B, T = mask_idx.shape
        num_transfer = self._normalize_num_transfer(num_transfer, B, mask_idx.device)

        if strategy == "mixed":
            candidate_names = ["spaced_0", "spaced_1", "random", "high_entropy"]
        elif strategy == "greedy":
            candidate_names = ["greedy"]
        elif strategy == "high_entropy":
            candidate_names = ["high_entropy"]
        elif strategy == "random":
            candidate_names = ["random"]
        else:
            raise ValueError(f"Unknown oracle_candidate_strategy: {strategy}")

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
            
            # if strategy == "mixed" or strategy == "greedy":
            if strategy == "greedy":
                # Candidate: Greedy (Highest confidence)
                _, greedy_idx = torch.topk(batch_conf, k=k)
                candidates["greedy"][b, greedy_idx] = True
            
            if strategy == "mixed" or strategy == "high_entropy":
            # if strategy == "high_entropy":
                # Candidate: High Entropy (Lowest confidence)
                inv_conf = torch.where(mask_idx[b], -confidence[b], -torch.inf)
                _, high_ent_idx = torch.topk(inv_conf, k=k)
                candidates["high_entropy"][b, high_ent_idx] = True

            if strategy == "mixed":
                # Candidates: Deterministic Spaced Anchors
                for offset in range(2):
                    # Create evenly spaced floating point indices across the span
                    # Shift the second pattern by half a step size
                    step_float = num_valid / k
                    start_offset = (offset * step_float) / 2
                    
                    # Compute integer indices
                    float_indices = torch.arange(k, device=valid_indices.device) * step_float + start_offset
                    int_indices = float_indices.long().clamp(max=num_valid - 1)
                    
                    selected_idx = valid_indices[int_indices]
                    candidates[f"spaced_{offset}"][b, selected_idx] = True

            if strategy == "mixed" or strategy == "random":
                # Baseline for structure gain definition
                rand_idx = valid_indices[torch.randperm(num_valid, device=valid_indices.device)[:k]]
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
        base_entropy_map: torch.Tensor,
        candidate_chunk_size: Optional[int] = None,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor], List[str]]:
        """
        Select the best candidate set based on entropy drop.

        This is where the "epiplexity" heuristic is applied. The candidate set that results in the largest drop in entropy (i.e., the most confident predictions) is selected.
        
        Returns:
            - best_c_idx: boolean mask of best candidate positions
            - structure_gains: dict of entropy drops per candidate
            - best_candidate_names: list of best candidate name per batch element
        """
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
            base_metric_map=base_entropy_map,
            metric="entropy_drop",
            attention_mask=attention_mask,
            masked_active_mask=mask_index,
            candidate_chunk_size=candidate_chunk_size,
        )
        structure_gains = {
            name: result.scores[index]
            for index, name in enumerate(candidate_batch.names)
        }
        best_candidate_names = [name or "" for name in result.best_names]
        return result.best_mask, structure_gains, best_candidate_names

    
    def _resolve_config(self, config=None, **kwargs):
        """Resolve every sampler dataclass field, including selector overrides."""
        config = config or EntropyDropSamplerConfig()
        names = {item.name for item in fields(config)}
        config = replace(config, **{key: value for key, value in kwargs.items() if key in names})
        config = resolve_dependency_guided_config(config, {})
        selectors = ("entropy_drop", *NON_LOOKAHEAD_SELECTORS)
        if config.dependency_candidate_selector not in selectors:
            raise ValueError(f"dependency_candidate_selector must be one of {selectors}.")
        if config.dependency_candidate_selector != "entropy_drop":
            if config.proposal_strategy not in {"dependency", "confidence_threshold"}:
                raise ValueError("Non-lookahead candidate selection requires dependency proposals.")
        if config.commit_mode == "seed_first" and config.temperature != 0.0:
            raise ValueError("seed_first currently requires deterministic token temperature=0.")
        return config

    @torch.no_grad()
    def initialize_state(self, inputs, config=None, **kwargs) -> DecodeState:
        """Initialize the ordinary decoder without evaluating the model."""
        config = self._resolve_config(config, **kwargs)
        if not inputs:
            raise ValueError("At least one prompt is required.")
        if config.block_size < 1 or config.steps < 1:
            raise ValueError("block_size and steps must be positive.")
        if config.right_shift_logits:
            inputs = [
                [self.tokenizer.bos_token_id] if isinstance(p, list) and not p else p
                for p in inputs
            ]
        inputs = [
            torch.as_tensor(p, dtype=torch.long, device=self.model.device) for p in inputs
        ]
        prompt_lens = [len(prompt) for prompt in inputs]
        max_new_tokens = config.max_new_tokens
        if max_new_tokens:
            max_length = max_new_tokens + max(prompt_lens)
        else:
            max_length = config.max_length
            max_new_tokens = max_length - max(prompt_lens)
        if max_new_tokens <= 0:
            raise ValueError("The response must contain at least one position.")
        x = torch.full(
            (len(inputs), max_length), self.tokenizer.eos_token_id,
            dtype=torch.long, device=self.model.device,
        )
        attention_mask = torch.zeros_like(x)
        response_mask = torch.zeros_like(x, dtype=torch.bool)
        for row, prompt in enumerate(inputs):
            start, end = prompt_lens[row], prompt_lens[row] + max_new_tokens
            x[row, :start] = prompt
            x[row, start:end] = self.tokenizer.mask_token_id
            attention_mask[row, :end] = 1
            response_mask[row, start:end] = True
        unmasked_index = (x != self.tokenizer.mask_token_id) & attention_mask.bool()
        if config.cfg_keep_tokens:
            unmasked_index &= ~torch.isin(
                x, torch.as_tensor(config.cfg_keep_tokens, device=x.device)
            )
        num_blocks = math.ceil(max_new_tokens / config.block_size)
        return DecodeState(
            input_ids=x, attention_mask=attention_mask, response_mask=response_mask,
            unmasked_index=unmasked_index, prompt_lens=prompt_lens,
            max_new_tokens=max_new_tokens, num_blocks=num_blocks,
            steps_per_block=math.ceil(config.steps / num_blocks), config=config,
            rng=RNGState.capture(),
            histories=[x.clone()] if config.return_dict else None,
            selected_candidates=[[] for _ in inputs],
            diagnostics=[[] for _ in inputs] if config.diagnostic_metadata else None,
        )

    def _ensure_ready(self, state: DecodeState) -> torch.Tensor | None:
        """Advance empty schedule entries and initialize block-local anchors."""
        config = state.config
        while not state.done:
            if state.block_span_mask is None:
                state.step_index = 0
                state.anchor_state = (
                    initialize_committed_anchor_state(
                        state.input_ids,
                        confidence_threshold=config.dependency_anchor_confidence_threshold,
                    )
                    if config.proposal_strategy == "dependency" else None
                )
                block_mask = torch.zeros(
                    (len(state.prompt_lens), config.block_size),
                    device=state.input_ids.device, dtype=torch.bool,
                )
                state.block_span_mask = torch.zeros_like(state.input_ids, dtype=torch.bool)
                for row, prompt_length in enumerate(state.prompt_lens):
                    start = prompt_length + state.block_index * config.block_size
                    end = min(start + config.block_size, prompt_length + state.max_new_tokens)
                    if start < end:
                        block_mask[row, :end - start] = (
                            state.input_ids[row, start:end] == self.tokenizer.mask_token_id
                        )
                        state.block_span_mask[row, start:end] = True
                adaptive = (
                    config.proposal_strategy == "confidence_threshold"
                    or config.dependency_cardinality_strategy in {
                        "marginal_utility", "joint_k", "entropy_budget",
                    }
                )
                if adaptive:
                    state.num_transfer_tokens = None
                    state.effective_steps = int(block_mask.sum(-1).max().item())
                else:
                    with isolated_rng(state.rng):
                        state.num_transfer_tokens = get_num_transfer_tokens(
                            mask_index=block_mask, steps=state.steps_per_block,
                            scheduler=self.scheduler, stochastic=config.stochastic_transfer,
                        )
                        state.rng = RNGState.capture()
                    state.effective_steps = state.num_transfer_tokens.size(1)
                    if (
                        is_fixed_k_strategy(config.proposal_strategy)
                        and config.dependency_cardinality_strategy == "fixed"
                    ):
                        validate_fixed_k_schedule(
                            state.num_transfer_tokens,
                            config.dependency_commit_k
                            if config.proposal_strategy == "dependency" else 1,
                        )
            if state.step_index >= state.effective_steps:
                state.block_index += 1
                state.block_span_mask = None
                continue
            if state.num_transfer_tokens is None:
                remaining = (
                    (state.input_ids == self.tokenizer.mask_token_id) & state.block_span_mask
                ).sum(-1, dtype=torch.long)
                requested = torch.minimum(
                    remaining, torch.full_like(remaining, config.dependency_max_action_size)
                )
            else:
                requested = state.num_transfer_tokens[:, state.step_index]
            if bool(torch.any(requested > 0)):
                return requested
            if state.num_transfer_tokens is None:
                state.step_index = state.effective_steps
            else:
                state.step_index += 1
        return None

    def _prediction_maps(self, logits, config):
        """Apply the existing token transforms in their original order."""
        if config.suppress_tokens:
            for token_id in config.suppress_tokens:
                logits[:, :, token_id] = -torch.inf
        if config.right_shift_logits:
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
        predicted = torch.argmax(
            add_gumbel_noise(logits, temperature=config.temperature), dim=-1
        )
        if config.begin_suppress_tokens:
            for token_id in config.begin_suppress_tokens:
                logits[:, :, token_id] = -torch.inf
        probabilities = F.softmax(logits, dim=-1)
        confidence = probabilities.gather(-1, predicted.unsqueeze(-1)).squeeze(-1)
        return predicted, probabilities, confidence, self.get_entropy_per_token(logits)

    @torch.no_grad()
    def prepare_step(
        self, state: DecodeState, config=None, *, retain_capture: bool = False
    ) -> PreparedStep | None:
        """Prepare one immutable decision; repeated calls replay its RNG."""
        if config is not None:
            config = self._resolve_config(config)
            if (
                config.block_size != state.config.block_size
                or config.steps != state.config.steps
            ):
                raise ValueError("A resumed state cannot change its block geometry or schedule.")
            state.config = config
        config = state.config
        requested = self._ensure_ready(state)
        if requested is None:
            return None
        x = state.input_ids
        masked = x == self.tokenizer.mask_token_id
        active = masked & state.block_span_mask & (requested > 0)[:, None]
        step_seed = config.dependency_generation_seed + state.global_step_index
        with isolated_rng(state.rng):
            unconditional = None
            if config.cfg_scale > 0:
                unconditional = x.clone()
                unconditional[state.unmasked_index] = self.tokenizer.mask_token_id
            base = run_base_forward_with_cfg_inputs(
                self.model, x, state.attention_mask, cfg_scale=config.cfg_scale,
                unconditional_input_ids=unconditional,
                capture_dependency=dependency_capture_required(config),
                dependency_last_n_layers=config.dependency_last_n_layers,
                measure_timing=config.diagnostic_metadata,
            )
            predicted, probabilities, confidence, entropy = self._prediction_maps(
                base.logits, config
            )
            kwargs = dict(
                active_mask=active, requested_k=requested, anchor_state=state.anchor_state,
                response_mask=state.response_mask, attention_mask=state.attention_mask,
                entropy_map=entropy, confidence=confidence, config=config,
                generation_seed=step_seed,
            )
            if config.proposal_strategy == "dependency":
                candidates, dependency, reconstruction_seconds, proposal_seconds = (
                    build_dependency_candidates(base, **kwargs)
                )
            elif config.proposal_strategy == "confidence_threshold":
                candidates, dependency, reconstruction_seconds, proposal_seconds = (
                    build_confidence_threshold_candidates(base, **kwargs)
                )
            else:
                dependency, reconstruction_seconds = None, 0.0
                started = time.perf_counter()
                if config.proposal_strategy == "legacy":
                    mappings = self.generate_candidate_sets(
                        confidence=torch.where(active, confidence, -torch.inf),
                        mask_idx=active, num_transfer=requested,
                        strategy=config.oracle_candidate_strategy,
                    )
                    candidates = candidate_batch_from_mask_mapping(mappings, eligible_mask=masked)
                else:
                    candidates = _build_baseline_candidates(
                        config.proposal_strategy, active_mask=active, confidence=confidence,
                        config=config, generation_seed=step_seed,
                    )
                proposal_seconds = time.perf_counter() - started
            rng_after = RNGState.capture()
        if not retain_capture:
            release_dependency_capture_tensors(base)
        return PreparedStep(
            state=state.clone(include_history=False), config=config, base_forward=base,
            x0=predicted, confidence=confidence, entropy=entropy,
            top2_margin=(
                top2_probability_margin(probabilities)
                if retain_capture or config.dependency_candidate_selector == "min_top2_margin"
                else None
            ), probabilities=probabilities,
            candidates=candidates, dependency=dependency, active_mask=active,
            masked_active_mask=masked, requested_k=requested, step_seed=step_seed,
            reconstruction_seconds=reconstruction_seconds, proposal_seconds=proposal_seconds,
            rng_after=rng_after,
        )

    @torch.no_grad()
    def select_step(self, prepared: PreparedStep, selector=None) -> StepSelection:
        """Select from the frozen pool, caching each selector's result."""
        config = prepared.config
        selector = selector or config.dependency_candidate_selector
        if config.proposal_strategy == "confidence_threshold":
            selector = "direct"
        if selector in prepared.selections:
            return prepared.selections[selector]
        release_dependency_capture_tensors(prepared.base_forward)
        if selector == "entropy_drop":
            _synchronize_for_timing(prepared.x0, config.diagnostic_metadata)
            started = time.perf_counter()
            with isolated_rng(prepared.rng_after):
                raw = evaluate_batched_lookahead(
                    self.model, prepared.state.input_ids, prepared.x0, prepared.candidates,
                    base_metric_map=prepared.entropy, metric="entropy_drop",
                    attention_mask=prepared.state.attention_mask,
                    masked_active_mask=prepared.masked_active_mask,
                    candidate_chunk_size=config.candidate_chunk_size,
                )
            size = apply_size_aware_scoring(
                raw, prepared.candidates, prepared.entropy,
                rule="raw" if config.proposal_strategy == "legacy" else config.dependency_size_scoring,
                immediate_cost_weight=config.dependency_immediate_cost_weight,
                size_penalty=config.dependency_size_penalty,
            )
            _synchronize_for_timing(prepared.x0, config.diagnostic_metadata)
            detail = FixedKSelectionOutput(
                candidates=prepared.candidates, lookahead=size.lookahead,
                dependency=prepared.dependency,
                attention_reconstruction_seconds=prepared.reconstruction_seconds,
                proposal_generation_seconds=prepared.proposal_seconds,
                candidate_lookahead_seconds=time.perf_counter() - started,
                raw_lookahead_scores=size.raw_scores,
                immediate_action_costs=size.immediate_costs,
                size_scoring_rule=size.rule,
            )
            winner = size.lookahead
            result = StepSelection(
                selector, winner.best_mask, winner.best_index, winner.best_names,
                winner.scores, detail,
            )
        else:
            if selector == "min_top2_margin" and prepared.top2_margin is None:
                prepared.top2_margin = top2_probability_margin(prepared.probabilities)
            detail = select_candidates_without_lookahead(
                prepared.candidates, confidence=prepared.confidence,
                entropy=prepared.entropy, top2_margin=prepared.top2_margin,
                selector="max_confidence" if selector == "direct" else selector,
            )
            result = StepSelection(
                selector, detail.best_mask, detail.best_index, detail.best_names,
                detail.selection_scores, detail,
            )
        prepared.selections[selector] = result
        return result

    @torch.no_grad()
    def probe_seed_first(
        self, prepared: PreparedStep, candidate_index, *, reverse: bool = False
    ) -> CommitValues:
        """Reveal only the recorded seed, refresh masked companions, and restore RNG."""
        if prepared.config.temperature != 0.0:
            raise ValueError("Seed-precedence probes require temperature=0.")
        release_dependency_capture_tensors(prepared.base_forward)
        candidates = prepared.candidates
        batch_size = prepared.x0.shape[0]
        indexes = torch.as_tensor(
            candidate_index, device=prepared.x0.device, dtype=torch.long
        ).flatten()
        if indexes.numel() == 1:
            indexes = indexes.expand(batch_size)
        if indexes.numel() != batch_size:
            raise ValueError("candidate_index must be scalar or have one entry per row.")
        first = torch.full((batch_size,), -1, device=prepared.x0.device, dtype=torch.long)
        action_mask = torch.zeros_like(prepared.x0, dtype=torch.bool)
        for row, index in enumerate(indexes.tolist()):
            if index < 0:
                continue
            if not bool(candidates.candidate_valid[index, row]):
                raise ValueError("Cannot probe an invalid candidate.")
            action_mask[row] = candidates.candidate_masks[index, row]
            seed = int(candidates.seed_anchors[index, row].item())
            if seed < 0 or not bool(action_mask[row, seed]):
                raise ValueError("Seed precedence requires a recorded seed inside the action.")
            if reverse:
                positions = torch.where(action_mask[row])[0]
                if positions.numel() != 2:
                    raise ValueError("Reverse-order probes require two-token actions.")
                seed = int(positions[positions != seed][0].item())
            first[row] = seed
        companions = action_mask.clone()
        revealed = prepared.state.input_ids.clone()
        for row, seed in enumerate(first.tolist()):
            if seed >= 0:
                revealed[row, seed] = prepared.x0[row, seed]
                companions[row, seed] = False
        if not bool(torch.any(companions)):
            return CommitValues(
                prepared.x0.clone(), prepared.confidence.clone(), first, companions,
                torch.zeros_like(companions), prepared.confidence.clone(),
                prepared.probabilities, prepared.x0.clone(),
            )
        with isolated_rng(prepared.rng_after):
            unconditional = None
            if prepared.config.cfg_scale > 0:
                unconditional = revealed.clone()
                unconditional[prepared.state.unmasked_index] = self.tokenizer.mask_token_id
            refreshed = run_base_forward_with_cfg_inputs(
                self.model, revealed, prepared.state.attention_mask,
                cfg_scale=prepared.config.cfg_scale,
                unconditional_input_ids=unconditional, capture_dependency=False,
                dependency_last_n_layers=prepared.config.dependency_last_n_layers,
                measure_timing=prepared.config.diagnostic_metadata,
            )
            ids, probabilities, confidence, _ = self._prediction_maps(
                refreshed.logits, prepared.config
            )
        token_ids = torch.where(companions, ids, prepared.x0)
        committed_confidence = torch.where(companions, confidence, prepared.confidence)
        original_probability = probabilities.gather(-1, prepared.x0.unsqueeze(-1)).squeeze(-1)
        return CommitValues(
            token_ids, committed_confidence, first, companions,
            companions & (token_ids != prepared.x0), original_probability, probabilities,
            ids,
        )

    def _generic_diagnostics(self, state, prepared, selection, action_mask, token_ids):
        """Emit selector-independent records without inventing lookahead evidence."""
        records = []
        for row in range(state.input_ids.shape[0]):
            candidates = []
            for index, name in enumerate(prepared.candidates.names):
                valid = bool(prepared.candidates.candidate_valid[index, row])
                positions = torch.where(prepared.candidates.candidate_masks[index, row])[0]
                score = (
                    float(selection.scores[index, row].item())
                    if selection is not None and valid else None
                )
                candidates.append({
                    "name": name, "valid": valid, "positions": positions.tolist(),
                    "action_size": int(positions.numel()),
                    "seed_position": int(prepared.candidates.seed_anchors[index, row]),
                    "token_ids": prepared.x0[row, positions].tolist(),
                    "selection_score": score,
                    "mean_within_set_conflict": float(
                        prepared.candidates.mean_within_set_dependency[index, row]
                    ) if valid else None,
                })
            positions = torch.where(action_mask[row])[0]
            selected = None
            if selection is not None and int(selection.best_index[row]) >= 0:
                selected = dict(candidates[int(selection.best_index[row])])
                selected["token_ids"] = token_ids[row, positions].tolist()
            records.append({
                "block_index": state.block_index, "step_index": state.step_index,
                "global_step_index": state.global_step_index,
                "generation_seed": prepared.step_seed,
                "candidate_selector": selection.selector if selection else "forced",
                "verifier_metric": None, "lookahead_model_calls": 0,
                "captured_base_forward_count": prepared.base_forward.captured_base_forward_count,
                "candidate_count_realized": sum(item["valid"] for item in candidates),
                "commit_k": int(positions.numel()), "candidates": candidates,
                "selected_candidate": selected,
                "dependency_seed_strategy": prepared.config.dependency_seed_strategy,
                "base_forward_seconds": prepared.base_forward.base_forward_seconds,
                "attention_reconstruction_seconds": prepared.reconstruction_seconds,
                "proposal_generation_seconds": prepared.proposal_seconds,
                "candidate_lookahead_seconds": 0.0,
            })
        return records

    @torch.no_grad()
    def commit_step(
        self, state: DecodeState, prepared: PreparedStep, action_mask, *,
        token_ids=None, confidence=None, selection: StepSelection | None = None,
    ) -> DecodeState:
        """Commit one macro-action in place; defaults freeze base anchor confidence."""
        if (
            state.step_index != prepared.state.step_index
            or state.global_step_index != prepared.state.global_step_index
            or state.block_index != prepared.state.block_index
            or not torch.equal(state.input_ids, prepared.state.input_ids)
        ):
            raise ValueError("A prepared action can only commit to its original frozen state.")
        if action_mask.shape != state.input_ids.shape or action_mask.dtype != torch.bool:
            raise ValueError("action_mask must be a boolean tensor matching the state.")
        if bool(torch.any(action_mask & ~prepared.active_mask)):
            raise ValueError("An action may only reveal currently eligible positions.")
        if bool(torch.any((prepared.requested_k > 0) & ~action_mask.any(-1))):
            raise ValueError("Every active row must make progress.")
        token_ids = prepared.x0 if token_ids is None else token_ids
        confidence = prepared.confidence if confidence is None else confidence
        if token_ids.shape != state.input_ids.shape or confidence.shape != state.input_ids.shape:
            raise ValueError("Commit values must match the state shape.")
        before = state.anchor_state
        after = (
            record_committed_anchors(
                before, action_mask, confidence, token_ids,
                commit_step=state.global_step_index,
            )
            if before is not None else None
        )
        if state.diagnostics is not None:
            if selection is not None and isinstance(selection.detail, FixedKSelectionOutput):
                records = build_step_diagnostics(
                    selection.detail, prepared.base_forward, config=prepared.config,
                    metric="entropy_drop", masked_active_mask=prepared.masked_active_mask,
                    response_mask=state.response_mask, block_index=state.block_index,
                    step_index=state.step_index, global_step_index=state.global_step_index,
                    generation_seed=prepared.step_seed, base_metric_map=prepared.entropy,
                    predicted_token_ids=prepared.x0, anchor_state_before=before,
                    anchor_state_after=after,
                )
            else:
                records = self._generic_diagnostics(
                    state, prepared, selection, action_mask, token_ids
                )
            for row, record in enumerate(records):
                positions = torch.where(action_mask[row])[0]
                record["committed_token_ids"] = token_ids[row, positions].tolist()
                selected_record = record.get("selected_candidate")
                if selected_record is not None:
                    selected_record["pre_reveal_token_ids"] = prepared.x0[row, positions].tolist()
                    selected_record["token_ids"] = record["committed_token_ids"]
                record["commit_mode"] = state.config.commit_mode
                record["committed_value_changes"] = int(
                    (token_ids[row, positions] != prepared.x0[row, positions]).sum().item()
                )
                state.diagnostics[row].append(record)
        state.anchor_state = after
        state.input_ids[action_mask] = token_ids[action_mask]
        if selection is not None:
            for row, name in enumerate(selection.best_names):
                if name and name not in state.selected_candidates[row]:
                    state.selected_candidates[row].append(name)
        if state.histories is not None:
            state.histories.append(state.input_ids.clone())
        state.rng = prepared.rng_after
        state.step_index += 1
        state.global_step_index += 1
        return state

    @torch.no_grad()
    def continue_from_state(self, state: DecodeState, config=None, *, observer=None):
        """Finish a saved state through the same prepare/select/commit implementation."""
        if config is not None:
            config = self._resolve_config(config)
            if config.block_size != state.config.block_size or config.steps != state.config.steps:
                raise ValueError("Continuation cannot change block geometry or schedule.")
            state.config = config
        while True:
            prepared = self.prepare_step(state, retain_capture=observer is not None)
            if prepared is None:
                break
            if observer is not None:
                with isolated_rng():
                    observer(prepared)
            release_dependency_capture_tensors(prepared.base_forward)
            selection = self.select_step(prepared)
            token_ids, confidence = None, None
            if state.config.commit_mode == "seed_first":
                values = self.probe_seed_first(prepared, selection.best_index)
                token_ids, confidence = values.token_ids, values.confidence
            self.commit_step(
                state, prepared, selection.best_mask, token_ids=token_ids,
                confidence=confidence, selection=selection,
            )
        return BaseSamplerOutput(
            sequences=state.input_ids, histories=state.histories,
            selected_candidates=state.selected_candidates, diagnostics=state.diagnostics,
        )

    @torch.no_grad()
    def sample(
        self, inputs: List[Union[torch.Tensor, List[int]]],
        config: Optional[EntropyDropSamplerConfig] = None, **kwargs,
    ) -> Union[BaseSamplerOutput, torch.Tensor]:
        """Run the ordinary decoder using replayable state transitions."""
        observer = kwargs.pop("observer", None)
        state = self.initialize_state(inputs, config=config, **kwargs)
        output = self.continue_from_state(state, observer=observer)
        # Preserve the ordinary sampler's external RNG-consumption semantics.
        state.rng.restore()
        return output if state.config.return_dict else output.sequences

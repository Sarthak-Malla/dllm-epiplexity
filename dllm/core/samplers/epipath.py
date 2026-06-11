"""
EpiPath sampler with separable proposal generation and candidate scoring.

Run tests from the repository root with:
  source ~/.zshrc && conda activate ~/miniconda3/envs/dllm
  pytest /home/sarthak.malla/dllm-epiplexity/scripts/tests/test_epipath_sampler.py -v
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch

from dllm.core.samplers.base import BaseSamplerOutput
from dllm.core.samplers.epipath_logging import EpiPathJSONLLogger
from dllm.core.samplers.epipath_metrics import (
    compute_commit_metrics,
    compute_coverage_metrics,
    compute_lookahead_metrics,
    compute_state_metrics,
    compute_topk_metrics,
    score_candidates,
)
from dllm.core.samplers.epipath_proposals import CandidateReveal, EpiProposalGenerator
from dllm.core.samplers.mdlm import MDLMSampler, MDLMSamplerConfig
from dllm.core.samplers.utils import add_gumbel_noise, get_num_transfer_tokens


@dataclass
class EpiPathSamplerConfig(MDLMSamplerConfig):
    epipath_proposal_preset: str = "dynamic_light"
    epipath_proposals: Optional[str] = None
    epipath_score_mode: str = "epigain"
    epipath_force_proposal: Optional[str] = None
    epipath_margin_threshold: float = 0.2
    epipath_entropy_weight: float = 0.5
    epipath_margin_deficit_weight: float = 1.0
    epipath_lambda_commit: float = 0.5
    epipath_lambda_cluster: float = 0.1
    epipath_lambda_coverage: float = 0.1
    epipath_normalize_candidate_metrics: bool = True
    epipath_log_path: Optional[str] = None
    epipath_log_level: str = "candidate"
    epipath_run_id: str = "epipath"


class EpiPathSampler(MDLMSampler):
    """One-step EpiPath sampler with dynamic proposal families."""

    def __post_init__(self):
        super().__post_init__()
        self.proposal_generator = EpiProposalGenerator()
        self._epipath_sample_counter = 0

    @torch.no_grad()
    def sample(
        self,
        inputs: list[torch.Tensor | list],
        config: EpiPathSamplerConfig | None = None,
        **kwargs,
    ) -> BaseSamplerOutput | torch.Tensor:
        if config is None:
            config = EpiPathSamplerConfig()

        steps = _as_int(kwargs.get("steps", config.steps))
        max_new_tokens = _as_optional_int(
            kwargs.get("max_new_tokens", config.max_new_tokens)
        )
        max_length = _as_optional_int(kwargs.get("max_length", config.max_length))
        block_size = _as_int(kwargs.get("block_size", config.block_size))
        temperature = _as_float(kwargs.get("temperature", config.temperature))
        cfg_scale = _as_float(kwargs.get("cfg_scale", config.cfg_scale))
        cfg_keep_tokens = kwargs.get("cfg_keep_tokens", config.cfg_keep_tokens)
        remasking = kwargs.get("remasking", config.remasking)
        suppress_tokens = kwargs.get("suppress_tokens", config.suppress_tokens)
        stochastic_transfer = _as_bool(
            kwargs.get("stochastic_transfer", config.stochastic_transfer)
        )
        return_dict = _as_bool(kwargs.get("return_dict", config.return_dict))
        right_shift_logits = _as_bool(
            kwargs.get("right_shift_logits", config.right_shift_logits)
        )
        begin_suppress_tokens = kwargs.get(
            "begin_suppress_tokens", config.begin_suppress_tokens
        )

        proposal_preset = str(
            kwargs.get("epipath_proposal_preset", config.epipath_proposal_preset)
        )
        proposals = _none_if_string(
            kwargs.get("epipath_proposals", config.epipath_proposals)
        )
        score_mode = str(kwargs.get("epipath_score_mode", config.epipath_score_mode))
        force_proposal = _none_if_string(
            kwargs.get("epipath_force_proposal", config.epipath_force_proposal)
        )
        margin_threshold = _as_float(
            kwargs.get("epipath_margin_threshold", config.epipath_margin_threshold)
        )
        entropy_weight = _as_float(
            kwargs.get("epipath_entropy_weight", config.epipath_entropy_weight)
        )
        margin_deficit_weight = _as_float(
            kwargs.get(
                "epipath_margin_deficit_weight",
                config.epipath_margin_deficit_weight,
            )
        )
        lambda_commit = _as_float(
            kwargs.get("epipath_lambda_commit", config.epipath_lambda_commit)
        )
        lambda_cluster = _as_float(
            kwargs.get("epipath_lambda_cluster", config.epipath_lambda_cluster)
        )
        lambda_coverage = _as_float(
            kwargs.get("epipath_lambda_coverage", config.epipath_lambda_coverage)
        )
        normalize_candidate_metrics = _as_bool(
            kwargs.get(
                "epipath_normalize_candidate_metrics",
                config.epipath_normalize_candidate_metrics,
            )
        )
        log_path = _none_if_string(
            kwargs.get("epipath_log_path", config.epipath_log_path)
        )
        log_level = str(kwargs.get("epipath_log_level", config.epipath_log_level))
        run_id = str(kwargs.get("epipath_run_id", config.epipath_run_id))

        assert 1 <= block_size
        assert 1 <= steps
        mask_id = self.tokenizer.mask_token_id
        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id
        logger = EpiPathJSONLLogger(log_path=log_path, log_level=log_level)

        if right_shift_logits:
            inputs = [
                [bos_id] if isinstance(prompt, list) and len(prompt) == 0 else prompt
                for prompt in inputs
            ]

        if isinstance(inputs[0], list):
            inputs = [
                torch.as_tensor(prompt, dtype=torch.long, device=self.model.device)
                for prompt in inputs
            ]
        prompt_lens = [prompt.shape[0] for prompt in inputs]

        if max_new_tokens:
            max_length = max_new_tokens + max(prompt_lens)
        else:
            max_new_tokens = max_length - max(prompt_lens)

        batch_size = len(inputs)
        sequence_length = max_length
        x = torch.full(
            (batch_size, sequence_length),
            eos_id,
            dtype=torch.long,
            device=self.model.device,
        )
        generation_span_mask = torch.zeros_like(x, dtype=torch.bool)
        for row, prompt in enumerate(inputs):
            x[row, : prompt_lens[row]] = prompt
            start = prompt_lens[row]
            end = prompt_lens[row] + max_new_tokens
            x[row, start:end] = mask_id
            generation_span_mask[row, start:end] = True

        attention_mask = torch.zeros(
            (batch_size, sequence_length), dtype=torch.long, device=self.model.device
        )
        for row, prompt_len in enumerate(prompt_lens):
            valid_end = min(prompt_len + max_new_tokens, sequence_length)
            attention_mask[row, :valid_end] = 1

        unmasked_index = (x != mask_id) & attention_mask.bool()
        if not (cfg_keep_tokens is None or len(cfg_keep_tokens) == 0):
            keep_mask = torch.isin(
                x, torch.as_tensor(cfg_keep_tokens, device=self.model.device)
            )
            unmasked_index = unmasked_index & ~keep_mask

        num_blocks = math.ceil(max_new_tokens / block_size)
        steps = math.ceil(steps / num_blocks)
        histories = [x.clone()] if return_dict else None
        sample_id = self._epipath_sample_counter
        self._epipath_sample_counter += 1
        global_step_index = 0

        for block_index in range(num_blocks):
            block_mask_index = torch.zeros(
                (batch_size, block_size), dtype=torch.bool, device=x.device
            )
            block_span_mask = torch.zeros_like(x, dtype=torch.bool)

            for row in range(batch_size):
                start = prompt_lens[row] + block_index * block_size
                end = min(
                    start + block_size,
                    prompt_lens[row] + max_new_tokens,
                    sequence_length,
                )
                if start < end:
                    block_mask_index[row, : end - start] = x[row, start:end] == mask_id
                    block_span_mask[row, start:end] = True

            num_transfer_tokens = get_num_transfer_tokens(
                mask_index=block_mask_index,
                steps=steps,
                scheduler=self.scheduler,
                stochastic=stochastic_transfer,
            )

            effective_steps = num_transfer_tokens.size(1)
            for step_in_block in range(effective_steps):
                num_transfer = num_transfer_tokens[:, step_in_block]
                if torch.all(num_transfer == 0):
                    continue

                mask_index = x == mask_id
                current_block_mask = mask_index & block_span_mask

                logits = self._model_logits(
                    x=x,
                    attention_mask=attention_mask,
                    unmasked_index=unmasked_index,
                    cfg_scale=cfg_scale,
                    suppress_tokens=suppress_tokens,
                    right_shift_logits=right_shift_logits,
                )
                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)
                self._suppress_tokens_inplace(logits, begin_suppress_tokens)

                # EpiPath does not use MDLM's `remasking` score to rank transfer
                # positions. Position selection is controlled by proposal generation
                # (`epipath_proposal_preset` / `epipath_proposals`) and candidate scoring
                # (`epipath_score_mode`). We still accept the standard MDLM remasking values
                # for config compatibility, but `random` does not make selection random here.
                # Random behavior should be added explicitly as an EpiPath proposal family.
                if remasking in ("low_confidence", "random"):
                    pass
                else:
                    raise NotImplementedError(remasking)

                x0 = torch.where(mask_index, x0, x)
                base_metrics = compute_topk_metrics(
                    logits, margin_threshold=margin_threshold
                )
                selected_index = torch.zeros_like(mask_index, dtype=torch.bool)

                for row in range(batch_size):
                    k = int(num_transfer[row].item())
                    if k == 0 or not current_block_mask[row].any():
                        continue

                    revealed_mask = generation_span_mask[row] & (~mask_index[row])
                    row_metrics = {
                        key: value[row] for key, value in base_metrics.items()
                    }
                    candidates = self.proposal_generator.generate(
                        current_mask=current_block_mask[row],
                        revealed_mask=revealed_mask,
                        base_metrics=row_metrics,
                        token_ids=x0[row],
                        num_transfer=k,
                        proposal_preset=proposal_preset,
                        proposals=proposals,
                    )
                    if not candidates:
                        continue

                    state_before = compute_state_metrics(
                        row_metrics,
                        current_block_mask[row],
                        entropy_weight=entropy_weight,
                        margin_deficit_weight=margin_deficit_weight,
                    )
                    self._evaluate_candidates(
                        candidates=candidates,
                        row=row,
                        x=x,
                        x0=x0,
                        attention_mask=attention_mask,
                        unmasked_index=unmasked_index,
                        current_mask=current_block_mask[row],
                        revealed_positions=torch.where(revealed_mask)[0].tolist(),
                        base_metrics=row_metrics,
                        cfg_scale=cfg_scale,
                        suppress_tokens=suppress_tokens,
                        begin_suppress_tokens=begin_suppress_tokens,
                        right_shift_logits=right_shift_logits,
                        margin_threshold=margin_threshold,
                        entropy_weight=entropy_weight,
                        margin_deficit_weight=margin_deficit_weight,
                    )
                    self._score_candidates(
                        candidates=candidates,
                        score_mode=score_mode,
                        force_proposal=force_proposal,
                        normalize_candidate_metrics=normalize_candidate_metrics,
                        lambda_commit=lambda_commit,
                        lambda_cluster=lambda_cluster,
                        lambda_coverage=lambda_coverage,
                    )

                    winner = max(candidates, key=lambda candidate: candidate.score)
                    winner.selected = True
                    row_selected_index = torch.zeros_like(mask_index[row])
                    row_selected_index[winner.positions] = True
                    selected_index[row] = row_selected_index

                    logger.write(
                        {
                            "record_type": "step",
                            "run_id": run_id,
                            "example_id": f"sample_{sample_id}_row_{row}",
                            "batch_row": row,
                            "block_index": block_index,
                            "step_index": global_step_index,
                            "step_in_block": step_in_block,
                            "num_masked_before": int(
                                current_block_mask[row].sum().item()
                            ),
                            "num_transfer": int(k),
                            "score_mode": score_mode,
                            "proposal_preset": proposal_preset,
                            "selected_candidate_id": winner.candidate_id,
                            "selected_proposal_type": winner.proposal_type,
                            "state_metrics_before": state_before,
                            "state_metrics_after": winner.metadata.get(
                                "state_metrics_after", {}
                            ),
                            "candidates": [
                                candidate.to_dict() for candidate in candidates
                            ],
                        }
                    )

                x[selected_index] = x0[selected_index]
                if histories is not None:
                    histories.append(x.clone())
                global_step_index += 1

        if return_dict:
            return BaseSamplerOutput(sequences=x, histories=histories)
        return x

    def _model_logits(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        unmasked_index: torch.Tensor,
        cfg_scale: float,
        suppress_tokens: list[int] | None,
        right_shift_logits: bool,
    ) -> torch.Tensor:
        """Run the model and apply the shared sampler logit transforms."""
        if cfg_scale > 0.0:
            un_x = x.clone()
            un_x[unmasked_index] = self.tokenizer.mask_token_id
            x_ = torch.cat([x, un_x], dim=0)
            logits = self.model(
                x_, attention_mask=attention_mask.repeat(2, 1)
            ).logits
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
        else:
            logits = self.model(x, attention_mask=attention_mask).logits
        self._suppress_tokens_inplace(logits, suppress_tokens)
        if right_shift_logits:
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
        return logits

    def _evaluate_candidates(
        self,
        candidates: list[CandidateReveal],
        row: int,
        x: torch.Tensor,
        x0: torch.Tensor,
        attention_mask: torch.Tensor,
        unmasked_index: torch.Tensor,
        current_mask: torch.Tensor,
        revealed_positions: list[int],
        base_metrics: dict[str, torch.Tensor],
        cfg_scale: float,
        suppress_tokens: list[int] | None,
        begin_suppress_tokens: list[int] | None,
        right_shift_logits: bool,
        margin_threshold: float,
        entropy_weight: float,
        margin_deficit_weight: float,
    ) -> None:
        """Populate lookahead and commit metrics for candidates in-place."""
        for candidate in candidates:
            candidate_mask = torch.zeros_like(current_mask)
            candidate_mask[candidate.positions] = True
            candidate.metrics.update(compute_commit_metrics(base_metrics, candidate_mask))
            candidate.metrics.update(
                compute_coverage_metrics(
                    positions=candidate.positions,
                    revealed_positions=revealed_positions,
                    sequence_length=current_mask.numel(),
                )
            )

            lookahead_x = x.clone()
            row_candidate_mask = torch.zeros_like(x[row], dtype=torch.bool)
            row_candidate_mask[candidate.positions] = True
            lookahead_x[row, row_candidate_mask] = x0[row, row_candidate_mask]
            lookahead_logits = self._model_logits(
                x=lookahead_x,
                attention_mask=attention_mask,
                unmasked_index=unmasked_index,
                cfg_scale=cfg_scale,
                suppress_tokens=suppress_tokens,
                right_shift_logits=right_shift_logits,
            )
            self._suppress_tokens_inplace(lookahead_logits, begin_suppress_tokens)
            lookahead_metrics_all = compute_topk_metrics(
                lookahead_logits, margin_threshold=margin_threshold
            )
            lookahead_metrics = {
                key: value[row] for key, value in lookahead_metrics_all.items()
            }
            candidate.metrics.update(
                compute_lookahead_metrics(
                    base_metrics=base_metrics,
                    lookahead_metrics=lookahead_metrics,
                    current_mask=current_mask,
                    candidate_mask=candidate_mask,
                    entropy_weight=entropy_weight,
                    margin_deficit_weight=margin_deficit_weight,
                )
            )
            heldout_mask = current_mask & (~candidate_mask)
            candidate.metadata["state_metrics_after"] = compute_state_metrics(
                lookahead_metrics,
                heldout_mask,
                entropy_weight=entropy_weight,
                margin_deficit_weight=margin_deficit_weight,
            )

    def _score_candidates(
        self,
        candidates: list[CandidateReveal],
        score_mode: str,
        force_proposal: str | None,
        normalize_candidate_metrics: bool,
        lambda_commit: float,
        lambda_cluster: float,
        lambda_coverage: float,
    ) -> None:
        if score_mode == "force_proposal":
            if not force_proposal:
                raise ValueError(
                    "epipath_force_proposal must be set when score mode is force_proposal"
                )
            if not any(force_proposal in candidate.proposal_sources for candidate in candidates):
                raise ValueError(
                    f"Forced proposal {force_proposal!r} was not generated"
                )
            for candidate in candidates:
                candidate.score = (
                    1.0 if force_proposal in candidate.proposal_sources else 0.0
                )
            return
        score_candidates(
            candidates,
            score_mode=score_mode,
            normalize=normalize_candidate_metrics,
            lambda_commit=lambda_commit,
            lambda_cluster=lambda_cluster,
            lambda_coverage=lambda_coverage,
        )

    def _suppress_tokens_inplace(
        self,
        logits: torch.Tensor,
        token_ids: list[int] | None,
    ) -> None:
        if token_ids is None or len(token_ids) == 0:
            return
        for token_id in token_ids:
            logits[:, :, int(token_id)] = -torch.inf


def _as_int(value) -> int:
    return int(value)


def _as_optional_int(value) -> int | None:
    if value is None or value == "None":
        return None
    return int(value)


def _as_float(value) -> float:
    return float(value)


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "y")
    return bool(value)


def _none_if_string(value):
    if isinstance(value, str) and value.strip().lower() in ("", "none", "null"):
        return None
    return value

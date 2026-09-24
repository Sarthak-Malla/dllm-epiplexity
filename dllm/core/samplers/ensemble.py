"""Ensemble position selection during masked diffusion decoding.

Use EnsembleSampler(model, tokenizer).sample(
    inputs, EnsembleSamplerConfig(ensemble_policy="candidate_expansion")
).
Import both classes from dllm.core.samplers. Set ensemble_policy="majority_voting"
to use a strict majority instead of unanimous agreement.
"""

import math
from dataclasses import dataclass, replace
from itertools import combinations
from typing import Literal

import torch

from .base import BaseSamplerConfig, BaseSamplerOutput
from .mdlm import MDLMSampler
from .utils import add_gumbel_noise, get_num_transfer_tokens


@dataclass
class EnsembleSamplerConfig(BaseSamplerConfig):
    """Decode blocks using agreement or a scheduled union of proposals.

    Agreement policies propose ceil(candidate_fraction * remaining_masks)
    positions in the current block. The all policy instead uses the scheduler's
    per-step transfer counts and commits the union of strategy proposals.
    """

    max_new_tokens: int = 128
    max_length: int | None = None
    block_size: int | None = 128
    steps: int = 128
    stochastic_transfer: bool = False
    temperature: float = 0.0
    strategies: tuple[str, ...] = ("low_confidence", "min_entropy", "max_top2_prob")
    ensemble_policy: Literal["candidate_expansion", "majority_voting", "all"] = (
        "all"
    )
    candidate_fraction: float = 0.10
    cfg_scale: float = 0.0
    cfg_keep_tokens: list[int] | None = None
    suppress_tokens: list[int] | None = None
    begin_suppress_tokens: list[int] | None = None
    right_shift_logits: bool = False


class EnsembleSampler(MDLMSampler):
    """Share predictions across strategies and commit positions by agreement.

    min_entropy ranks by negative entropy; max_top2_prob ranks by the gap
    between the largest and second-largest token probabilities. Larger scores
    reveal earlier. Both use the full distribution from logits, independently
    of predicted_ids; low_confidence scores the proposed token instead.
    """

    def _score_positions(self, logits, predicted_ids, strategy):
        if strategy in {"low_confidence", "random"}:
            return super()._score_positions(logits, predicted_ids, strategy)

        if strategy not in {"min_entropy", "max_top2_prob"}:
            raise ValueError(f"Unknown strategy: {strategy}")

        p = logits.float().softmax(dim=-1)
        if strategy == "min_entropy":
            # xlogy defines 0 * log(0) as zero for suppressed tokens.
            return torch.special.xlogy(p, p).sum(dim=-1)
        else:
            top_two = p.topk(2, dim=-1).values
            return top_two[..., 0] - top_two[..., 1]

    def _select_positions(
        self,
        scores,
        eligible,
        ensemble_policy,
        candidate_fraction,
        strategies,
        metrics=None,
        proposal_k=None,
    ):
        """Return a Boolean commit mask from [strategy, batch, sequence] scores.

        Candidate expansion requires all strategies to agree; majority voting
        requires more than half. Both expand the common candidate count when
        needed, using the same scores and predictions without another forward.
        The all policy commits the union of scheduled per-strategy proposals.
        If supplied, metrics receives expansion counts, all-strategy and pairwise
        intersection counts, and each strategy's proposal count per active row.
        Counts are measured before and after empty-agreement expansion.
        """
        num_strategies = scores.shape[0]
        if ensemble_policy == "candidate_expansion":
            required_votes = num_strategies
        elif ensemble_policy == "all":
            required_votes = 1
        else:
            required_votes = num_strategies // 2 + 1

        transfer_index = torch.zeros_like(eligible)
        if metrics is not None:
            metrics["expanded_sequences"] = 0
            overlap_pairs = list(combinations(range(num_strategies), 2))
            overlap_names = ["all"] + [
                f"{strategies[first]}__{strategies[second]}"
                for first, second in overlap_pairs
            ]
            overlap_stages = ("before_expansion", "after_expansion")
            metrics["position_overlap"] = {
                f"{stage}/{name}": []
                for stage in overlap_stages
                for name in overlap_names
            }
            metrics["position_proposals"] = {
                f"{stage}/{strategy}": []
                for stage in overlap_stages
                for strategy in strategies
            }
        for j in range(eligible.shape[0]):
            positions = torch.where(eligible[j])[0]
            remaining = positions.numel()
            if remaining == 0:
                continue

            position_scores = scores[:, j, positions]
            if not torch.isfinite(position_scores).all():
                raise ValueError("Non-finite ensemble scores at eligible positions")

            # Rank only eligible positions. Stable ties favor earlier positions.
            order = position_scores.argsort(dim=-1, descending=True, stable=True)
            ranks = torch.empty_like(order)
            ranks.scatter_(
                -1,
                order,
                torch.arange(1, remaining + 1, device=scores.device).expand_as(order),
            )

            # The q-th smallest rank is the candidate count at which a position
            # gains q votes. Its minimum finds the first nonempty agreement
            # directly, equivalent to expanding top-k sets one position at a time.
            acceptance_rank = ranks.kthvalue(required_votes, dim=0).values
            if ensemble_policy == "all":
                if proposal_k is None:
                    raise ValueError("all requires scheduled proposal_k")
                candidate_k = min(int(proposal_k[j].item()), remaining)
            else:
                candidate_k = math.ceil(candidate_fraction * remaining)
            if ensemble_policy == "all":
                k = candidate_k
            else:
                k = max(candidate_k, int(acceptance_rank.min().item()))
            if metrics is not None:
                if k > candidate_k:
                    metrics["expanded_sequences"] += 1
                # Reuse the actual proposal ranks, including stable tie-breaking.
                # Count eligible positions only. Keep zeros before fallback and
                # distinguish them from the expanded agreement.
                cutoffs = [candidate_k] if k == candidate_k else [candidate_k, k]
                observations = []
                for cutoff in cutoffs:
                    candidates = ranks <= cutoff
                    counts = torch.stack([
                        candidates.all(dim=0).sum(),
                        *[
                            (candidates[first] & candidates[second]).sum()
                            for first, second in overlap_pairs
                        ],
                        *candidates.sum(dim=-1).unbind(),
                    ]).tolist()
                    observations.append(counts)
                for stage, counts in zip(
                    overlap_stages, (observations[0], observations[-1])
                ):
                    for name, count in zip(overlap_names, counts):
                        metrics["position_overlap"][f"{stage}/{name}"].append(count)
                    for strategy, count in zip(strategies, counts[len(overlap_names):]):
                        metrics["position_proposals"][f"{stage}/{strategy}"].append(count)
            transfer_index[j, positions[acceptance_rank <= k]] = True
        return transfer_index

    @torch.no_grad()
    def sample(
        self,
        inputs: list[torch.Tensor | list],
        config: EnsembleSamplerConfig | None = None,
        **kwargs,
    ) -> BaseSamplerOutput | torch.Tensor:
        """Append masks to prompts and reveal them by ensemble agreement.

        As in MDLMSampler, kwargs override config, prompts are right-padded with
        EOS, and blocks align with each prompt's generated tail. All strategies
        share the same predicted tokens, including any Gumbel noise.
        """
        config = replace(
            config if config is not None else EnsembleSamplerConfig(), **kwargs
        )
        max_new_tokens = config.max_new_tokens
        max_length = config.max_length
        mask_id = self.tokenizer.mask_token_id
        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id

        if not inputs:
            raise ValueError("inputs must contain at least one prompt")
        inputs = [
            torch.as_tensor(p, dtype=torch.long, device=self.model.device)
            for p in inputs
        ]
        if config.right_shift_logits:
            inputs = [p if p.numel() else p.new_tensor([bos_id]) for p in inputs]
        prompt_lens = [p.shape[0] for p in inputs]

        if max_new_tokens:
            max_length = max_new_tokens + max(prompt_lens)
        elif max_length is not None:
            max_new_tokens = max_length - max(prompt_lens)
        else:
            max_length = max(prompt_lens)
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be nonnegative and fit max_length")

        B = len(inputs)
        T = max_length
        x = torch.full((B, T), eos_id, dtype=torch.long, device=self.model.device)
        attention_mask = torch.zeros_like(x)
        region_ends = [pl + max_new_tokens for pl in prompt_lens]
        for j, p in enumerate(inputs):
            x[j, : prompt_lens[j]] = p
            x[j, prompt_lens[j] : region_ends[j]] = mask_id
            attention_mask[j, : region_ends[j]] = 1

        return self._decode_blocks(x, attention_mask, prompt_lens, region_ends, config)

    @torch.no_grad()
    def infill(
        self,
        inputs: list[torch.Tensor | list],
        config: EnsembleSamplerConfig | None = None,
        **kwargs,
    ) -> BaseSamplerOutput | torch.Tensor:
        """Fill existing masks by agreement, preserving known tokens and padding.

        Blocks use absolute sequence positions as in MDLMSampler.infill().
        block_size=None treats the entire padded sequence as one block.
        """
        config = replace(
            config if config is not None else EnsembleSamplerConfig(), **kwargs
        )
        if not inputs:
            raise ValueError("inputs must contain at least one sequence")
        inputs = [
            torch.as_tensor(p, dtype=torch.long, device=self.model.device)
            for p in inputs
        ]
        if config.right_shift_logits:
            inputs = [
                p if p.numel() else p.new_tensor([self.tokenizer.bos_token_id])
                for p in inputs
            ]
        seq_lens = [p.shape[0] for p in inputs]
        B = len(inputs)
        T = max(seq_lens)
        x = torch.full(
            (B, T),
            self.tokenizer.eos_token_id,
            dtype=torch.long,
            device=self.model.device,
        )
        attention_mask = torch.zeros_like(x)
        for j, p in enumerate(inputs):
            x[j, : seq_lens[j]] = p
            attention_mask[j, : seq_lens[j]] = 1

        return self._decode_blocks(x, attention_mask, [0] * B, seq_lens, config)

    def _decode_blocks(self, x, attention_mask, region_starts, region_ends, config):
        """Decode each region block until no masks remain, without scheduling."""
        steps = config.steps
        strategies = config.strategies
        supported = {"low_confidence", "random", "min_entropy", "max_top2_prob"}
        if isinstance(strategies, str) or not strategies:
            raise ValueError("strategies must be a nonempty sequence of strategy names")
        if len(set(strategies)) != len(strategies):
            raise ValueError("strategies must be unique so each strategy gets one vote")
        if any(strategy not in supported for strategy in strategies):
            raise ValueError(f"Unknown strategy in {strategies}")
        if config.ensemble_policy not in {
            "candidate_expansion", "majority_voting", "all"
        }:
            raise ValueError(f"Unknown ensemble policy: {config.ensemble_policy}")
        if config.ensemble_policy == "all" and config.steps < 1:
            raise ValueError("steps must be positive")
        if config.ensemble_policy != "all" and not 0 < config.candidate_fraction <= 1:
            raise ValueError("candidate_fraction must be in (0, 1]")

        if config.ensemble_policy == "all" and self.scheduler is None:
            super().__post_init__()

        block_size = (
            config.block_size if config.block_size is not None else max(1, x.shape[1])
        )
        if block_size < 1:
            raise ValueError("block_size must be positive")
        mask_id = self.tokenizer.mask_token_id
        histories = [x.clone()] if config.return_dict else None

        # CFG masks the originally known tokens, respecting cfg_keep_tokens.
        unmasked_index = (x != mask_id) & attention_mask.bool()
        if config.cfg_keep_tokens:
            keep_mask = torch.isin(
                x, torch.as_tensor(config.cfg_keep_tokens, device=x.device)
            )
            unmasked_index = unmasked_index & ~keep_mask

        positions = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        starts = torch.as_tensor(region_starts, device=x.device).unsqueeze(1)
        ends = torch.as_tensor(region_ends, device=x.device).unsqueeze(1)
        num_blocks = math.ceil(
            max(end - start for start, end in zip(region_starts, region_ends))
            / block_size
        )
        steps = math.ceil(steps / num_blocks)

        for b in range(num_blocks):
            # Both bounds are explicit: only this block's masks may be committed.
            block_start = starts + b * block_size
            block_end = torch.minimum(block_start + block_size, ends)
            block_mask_index = torch.zeros(
                (x.shape[0], block_size), dtype=torch.bool, device=x.device
            )
            for j in range(x.shape[0]):
                start = int(block_start[j].item())
                end = int(block_end[j].item())
                if start < end:
                    block_mask_index[j, : end - start] = x[j, start:end] == mask_id
            block_index = (
                (positions >= block_start)
                & (positions < block_end)
                & attention_mask.bool()
            )
            eligible = block_index & (x == mask_id)
            num_transfer_tokens = None
            if config.ensemble_policy == "all":
                num_transfer_tokens = get_num_transfer_tokens(
                    mask_index=block_mask_index,
                    steps=steps,
                    scheduler=self.scheduler,
                    stochastic=config.stochastic_transfer,
                )
            step = 0
            while eligible.any():
                # ----- Forward pass (+ optional CFG), matching MDLM -----
                if config.cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[unmasked_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = self.model(
                        x_, attention_mask=attention_mask.repeat(2, 1)
                    ).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (config.cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = self.model(x, attention_mask=attention_mask).logits

                # A committed prediction must reveal a token, never another mask.
                logits[:, :, mask_id] = -torch.inf
                if config.suppress_tokens:
                    logits[:, :, config.suppress_tokens] = -torch.inf
                if config.right_shift_logits:
                    logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
                if not torch.isfinite(logits[eligible]).any(dim=-1).all():
                    raise ValueError("No finite token logits at an eligible position")

                logits_with_noise = add_gumbel_noise(
                    logits, temperature=config.temperature
                )
                x0 = torch.argmax(logits_with_noise, dim=-1)
                if (x0[eligible] == mask_id).any():
                    raise ValueError(
                        "Token prediction produced a mask instead of a reveal"
                    )

                # MDLM applies begin_suppress_tokens to scoring after prediction.
                if config.begin_suppress_tokens:
                    logits[:, :, config.begin_suppress_tokens] = -torch.inf
                scores = torch.stack(
                    [
                        self._score_positions(logits, x0, strategy)
                        for strategy in strategies
                    ]
                )
                metrics = {} if self.step_callback is not None else None
                transfer_index = self._select_positions(
                    scores,
                    eligible,
                    config.ensemble_policy,
                    config.candidate_fraction,
                    strategies,
                    metrics=metrics,
                    proposal_k=(
                        num_transfer_tokens[:, step]
                        if num_transfer_tokens is not None else None
                    ),
                )

                x[transfer_index] = x0[transfer_index]
                if self.step_callback is not None:
                    self._report_step(eligible, x, block_index, mask_id, **metrics)
                if histories is not None:
                    histories.append(x.clone())
                eligible = block_index & (x == mask_id)
                step += 1

        if not config.return_dict:
            return x
        return BaseSamplerOutput(sequences=x, histories=histories)

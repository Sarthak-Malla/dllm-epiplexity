"""Heterogeneous two-model TSE sampling with CTCA.

Run the CPU integration tests with:
    pytest scripts/tests/test_ctca_sampler.py -v
"""

import math
import os
from collections.abc import Sequence
from concurrent.futures import Executor, ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from typing import Callable

import torch

from dllm.core.samplers.utils import get_num_transfer_tokens
from dllm.core.schedulers import LinearAlphaScheduler
from dllm.pipelines.tse.ctca import (
    CanvasRunCache,
    CrossTokenizerAligner,
    ModelCanvasView,
    build_model_canvas_view,
)
from dllm.pipelines.tse.ctca.projection import projection_token_mask

from .fusion import fuse_probabilities, logits_to_probabilities
from .scoring import baseline_candidates, distribution_consensus, fused_confidence
from .selection import commit_tokens, generation_mask, select_positions
from .utils import forward_token_logits, initialize_canvas, timer
from .weighting import pairwise_weights


class CTCATSESampler:
    """Run two masked-diffusion models on native tokenizer canvases."""

    def __init__(
        self,
        master_model: torch.nn.Module,
        auxiliary_model: torch.nn.Module,
        master_tokenizer,
        auxiliary_tokenizer,
        master_device: str,
        auxiliary_device: str,
        *,
        master_id: str,
        auxiliary_id: str,
        cache_dir: str | None = ".cache/ctca",
        force_rebuild: bool = True,
        anchor_temperature: float = 0.01,
        projection_temperature: float = 0.05,
        projection_chunk_size: int = 2500,
        projection_mode: str = "sparse_topk",
        projection_top_k: int = 64,
        num_anchors: int | str = "auto",
        min_anchors: int = 128,
        master_cache_id: str | None = None,
        auxiliary_cache_id: str | None = None,
        fusion_device: str | None = None,
    ) -> None:
        if master_id not in {"a", "b"} or auxiliary_id not in {"a", "b"}:
            raise ValueError("master_id and auxiliary_id must be 'a' or 'b'")
        if master_id == auxiliary_id:
            raise ValueError("master_id and auxiliary_id must be different")

        with timer("ctca.init.model_to_device"):
            self.master_model = master_model.to(master_device).eval()
            self.auxiliary_model = auxiliary_model.to(auxiliary_device).eval()
        self.master_tokenizer = master_tokenizer
        self.auxiliary_tokenizer = auxiliary_tokenizer
        self.master_device = torch.device(master_device)
        self.auxiliary_device = torch.device(auxiliary_device)
        self.master_id = master_id
        self.auxiliary_id = auxiliary_id
        self.scheduler = LinearAlphaScheduler()
        self.last_aligned_probabilities: list[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = []
        self._valid_token_id_masks: dict[tuple[int, int, str], torch.Tensor] = {}

        master_embeddings = self.master_model.get_input_embeddings().weight.detach()
        auxiliary_embeddings = (
            self.auxiliary_model.get_input_embeddings().weight.detach()
        )
        master_cache_id = _checkpoint_cache_id(master_model, master_cache_id)
        auxiliary_cache_id = _checkpoint_cache_id(auxiliary_model, auxiliary_cache_id)
        with timer("ctca.init.aligner"):
            self.aligner = CrossTokenizerAligner(
                master_tokenizer,
                master_embeddings,
                master_id=master_cache_id or master_id,
                cache_dir=cache_dir if master_cache_id and auxiliary_cache_id else None,
                force_rebuild=force_rebuild,
            )
            self.aligner.register_auxiliary_model(
                auxiliary_cache_id or auxiliary_id,
                auxiliary_tokenizer,
                auxiliary_embeddings,
                anchor_temperature=anchor_temperature,
                temperature=projection_temperature,
                chunk_size=projection_chunk_size,
                projection_mode=projection_mode,
                projection_top_k=projection_top_k,
                num_anchors=num_anchors,
                min_anchors=min_anchors,
            )
        self._aligner_auxiliary_id = auxiliary_cache_id or auxiliary_id
        self.fusion_device = torch.device(fusion_device or self.master_device)
        self.aligner.warm_projection_cache(self._aligner_auxiliary_id, self.fusion_device)
        self.ctca_projection_diagnostics = self.aligner.projection_diagnostics(
            self._aligner_auxiliary_id
        )

    def _valid_token_id_mask(
        self,
        tokenizer,
        vocabulary_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Return the IDs eligible for generation in this model vocabulary."""
        key = (id(tokenizer), vocabulary_size, str(device))
        cached = self._valid_token_id_masks.get(key)
        if cached is not None:
            return cached

        mask = projection_token_mask(tokenizer, vocabulary_size).to(device)
        self._valid_token_id_masks[key] = mask
        return mask

    def _model_probabilities(
        self,
        logits: torch.Tensor,
        tokenizer,
        temperature: float,
    ) -> torch.Tensor:
        """Convert native output logits over assigned non-mask IDs."""
        return logits_to_probabilities(
            logits,
            temperature,
            allowed_tokens=self._valid_token_id_mask(
                tokenizer, logits.shape[-1], logits.device
            ),
        )

    def _derive_auxiliary_prompt(self, master_prompt: torch.Tensor) -> torch.Tensor:
        text = self.master_tokenizer.decode(
            master_prompt.tolist(),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        encoded = self.auxiliary_tokenizer(text, add_special_tokens=False)
        input_ids = encoded["input_ids"]
        if input_ids and isinstance(input_ids[0], list):
            input_ids = input_ids[0]
        return torch.as_tensor(input_ids, dtype=torch.long)

    def _canvas_worker_count(self, batch_size: int) -> int:
        configured = os.environ.get("CTCA_CANVAS_WORKERS")
        if configured is not None:
            worker_count = int(configured)
        else:
            worker_count = min(batch_size, os.cpu_count() or 1, 8)
        return max(1, min(batch_size, worker_count))

    def _build_auxiliary_batch(
        self,
        canvas: torch.Tensor,
        prompt_lens: Sequence[int],
        auxiliary_prompts: Sequence[torch.Tensor],
        max_new_tokens: int,
        projection_range: range,
        auxiliary_run_caches: Sequence[CanvasRunCache] | None = None,
        canvas_executor: Executor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, list[ModelCanvasView]]:
        # Move one batch to the host before tokenizer work begins.
        cpu_canvas = canvas.cpu()

        def build_view(sample_index: int) -> ModelCanvasView:
            prompt_len = prompt_lens[sample_index]
            run_cache = (
                auxiliary_run_caches[sample_index]
                if auxiliary_run_caches is not None
                else None
            )
            generated = cpu_canvas[
                sample_index, prompt_len : prompt_len + max_new_tokens
            ]
            return build_model_canvas_view(
                auxiliary_prompts[sample_index],
                generated,
                master_tokenizer=self.master_tokenizer,
                model_tokenizer=self.auxiliary_tokenizer,
                master_mask_token_id=self.master_tokenizer.mask_token_id,
                model_mask_token_id=self.auxiliary_tokenizer.mask_token_id,
                run_cache=run_cache,
                master_range=projection_range,
            )

        with timer("ctca.auxiliary_batch.build_views"):
            sample_indices = range(len(prompt_lens))
            if canvas_executor is None:
                views = [build_view(sample_index) for sample_index in sample_indices]
            else:
                views = list(canvas_executor.map(build_view, sample_indices))

        with timer("ctca.auxiliary_batch.pad_and_copy"):
            pad_token_id = self.auxiliary_tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = self.auxiliary_tokenizer.eos_token_id
            input_ids = torch.nn.utils.rnn.pad_sequence(
                [view.input_ids for view in views],
                batch_first=True,
                padding_value=pad_token_id,
            ).to(self.auxiliary_device)
            attention_mask = torch.nn.utils.rnn.pad_sequence(
                [view.attention_mask for view in views], batch_first=True
            ).to(self.auxiliary_device)
        return input_ids, attention_mask, views

    @torch.no_grad()
    @timer()
    def _forward_aligned_probabilities(
        self,
        canvas: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_lens: Sequence[int],
        auxiliary_prompts: Sequence[torch.Tensor],
        max_new_tokens: int,
        *,
        auxiliary_run_caches: Sequence[CanvasRunCache] | None = None,
        canvas_executor: Executor | None = None,
        projection_range: range,
        temperature_master: float,
        temperature_auxiliary: float,
        fusion_device: torch.device,
        trace_callback: Callable[[str, dict], None] | None = None,
        trace_step: tuple[int, int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with timer("ctca.forward.master"):
            master_logits = forward_token_logits(
                self.master_model, canvas, attention_mask
            )
        with timer("ctca.forward.build_auxiliary_batch"):
            auxiliary_ids, auxiliary_mask, views = self._build_auxiliary_batch(
                canvas,
                prompt_lens,
                auxiliary_prompts,
                max_new_tokens,
                projection_range,
                auxiliary_run_caches=auxiliary_run_caches,
                canvas_executor=canvas_executor,
            )
        with timer("ctca.forward.auxiliary"):
            auxiliary_logits = forward_token_logits(
                self.auxiliary_model, auxiliary_ids, auxiliary_mask
            )

        with timer("ctca.forward.master_probabilities"):
            active_mask = generation_mask(
                canvas, prompt_lens, projection_range, self.master_tokenizer.mask_token_id
            )
            master_probabilities = self._model_probabilities(
                master_logits[active_mask.to(self.master_device)].to(fusion_device),
                self.master_tokenizer,
                temperature_master,
            )
        master_offsets = tuple((float(i), float(i + 1)) for i in projection_range)
        with timer("ctca.forward.project_auxiliary"):
            spatial_by_sample = []
            for sample_index, (prompt_len, view) in enumerate(zip(prompt_lens, views)):
                local_active = active_mask[
                    sample_index,
                    prompt_len + projection_range.start : prompt_len + projection_range.stop,
                ]
                generated_logits = auxiliary_logits[sample_index, view.generation_slice]
                auxiliary_probabilities = self._model_probabilities(
                    generated_logits.to(fusion_device),
                    self.auxiliary_tokenizer,
                    temperature_auxiliary,
                )
                spatial = self.aligner.spatial_warp_model_probabilities(
                    self._aligner_auxiliary_id,
                    auxiliary_probabilities,
                    view.offsets,
                    master_offsets,
                    overlap_matrix=view.overlap_matrix,
                )
                spatial_by_sample.append(spatial[local_active.to(fusion_device)])

            spatial_auxiliary = torch.cat(spatial_by_sample, dim=0)
            active_positions = active_mask.nonzero(as_tuple=False)
            # Count consecutive masks before the next committed slot or generation end.
            generation_masks = generation_mask(
                canvas,
                prompt_lens,
                range(max_new_tokens),
                self.master_tokenizer.mask_token_id,
            )
            columns = torch.arange(canvas.shape[1], device=canvas.device)
            blocked = torch.where(generation_masks, canvas.shape[1], columns)
            next_blocked = blocked.flip(1).cummin(dim=1).values.flip(1)
            masked_slots_to_right = next_blocked[active_mask] - active_positions[:, 1] - 1
            projected_auxiliary = self.aligner.project_spatial_probabilities(
                self._aligner_auxiliary_id,
                spatial_auxiliary,
                masked_slots_to_right=masked_slots_to_right,
            )
            if (projected_auxiliary.sum(dim=-1) <= 0).any():
                raise ValueError("CTCA projection produced an empty active distribution")
        if trace_callback is not None:
            trace_callback(
                "forward",
                {
                    "step": trace_step,
                    "canvas": canvas.detach().cpu(),
                    "prompt_lens": tuple(prompt_lens),
                    "active_positions": active_positions.detach().cpu(),
                    "masked_slots_to_right": masked_slots_to_right.cpu(),
                    "master_logits": master_logits.detach().cpu(),
                    "auxiliary_logits": auxiliary_logits.detach().cpu(),
                    "active_mask": active_mask.detach().cpu(),
                    "views": views,
                    "master_offsets": master_offsets,
                },
            )
        if master_probabilities.shape != projected_auxiliary.shape:
            raise ValueError(
                "CTCA projection did not align active probability shapes: "
                f"{tuple(master_probabilities.shape)} != "
                f"{tuple(projected_auxiliary.shape)}"
            )
        return master_probabilities, projected_auxiliary, active_positions

    @torch.no_grad()
    def sample(
        self,
        inputs: list[torch.Tensor | list],
        *,
        auxiliary_inputs: list[torch.Tensor | list] | None = None,
        max_new_tokens: int,
        steps: int,
        block_size: int,
        temperature: float = 0.0,
        remasking: str = "low_confidence",
        stochastic_transfer: bool = False,
        capture_logits: bool = False,
        baseline_model: str = "a",
        selection_mode: str = "tse",
        alpha: float = 0.5,
        temperature_a: float = 1.0,
        temperature_b: float = 1.0,
        epsilon: float = 1e-9,
        fusion_device: str | None = None,
        weighting_mode: str = "static",
        weight_temperature: float = 1.0,
        normalize_entropy: bool = True,
        trace_callback: Callable[[str, dict], None] | None = None,
    ) -> torch.Tensor:
        """Generate on the master grid while projecting the auxiliary model."""
        if not inputs:
            raise ValueError("CTCATSESampler.sample requires at least one input")
        if steps < 1 or block_size < 1 or max_new_tokens < 1:
            raise ValueError("steps, block_size, and max_new_tokens must be positive")
        if baseline_model not in {"a", "b"}:
            raise ValueError("baseline_model must be 'a' or 'b'")
        if selection_mode not in {"baseline", "tse"}:
            raise ValueError("selection_mode must be 'baseline' or 'tse'")
        if remasking not in {"low_confidence", "random"}:
            raise ValueError(f"Unsupported remasking strategy: {remasking}")
        if weighting_mode not in {"static", "online_entropy", "per_token_margin"}:
            raise ValueError(
                "weighting_mode must be 'static', 'online_entropy', "
                "or 'per_token_margin'"
            )

        with timer("ctca.sample.prepare_prompts"):
            prompts = [
                torch.as_tensor(
                    prompt, dtype=torch.long, device=self.master_device
                ).flatten()
                for prompt in inputs
            ]
            if auxiliary_inputs is None:
                auxiliary_prompts = [
                    self._derive_auxiliary_prompt(prompt.cpu()) for prompt in prompts
                ]
            else:
                if len(auxiliary_inputs) != len(prompts):
                    raise ValueError("auxiliary_inputs must align with inputs")
                auxiliary_prompts = [
                    torch.as_tensor(prompt, dtype=torch.long).flatten()
                    for prompt in auxiliary_inputs
                ]

        with timer("ctca.sample.initialize_canvas"):
            mask_id = self.master_tokenizer.mask_token_id
            master_pad_id = self.master_tokenizer.pad_token_id
            if master_pad_id is None:
                master_pad_id = self.master_tokenizer.eos_token_id
            canvas, attention_mask, prompt_lens = initialize_canvas(
                prompts, max_new_tokens, mask_id, master_pad_id
            )
        if trace_callback is not None:
            trace_callback(
                "canvas",
                {
                    "phase": "initial",
                    "canvas": canvas.detach().cpu(),
                    "prompt_lens": tuple(prompt_lens),
                    "auxiliary_prompts": tuple(
                        prompt.detach().cpu() for prompt in auxiliary_prompts
                    ),
                },
            )

        target_device = torch.device(fusion_device or self.fusion_device)
        temperature_master = temperature_a if self.master_id == "a" else temperature_b
        temperature_auxiliary = temperature_b if self.auxiliary_id == "b" else temperature_a
        master_static_weight = alpha if self.master_id == "a" else 1 - alpha
        self.last_aligned_probabilities = []
        auxiliary_run_caches: list[CanvasRunCache] = [{} for _ in prompts]
        num_blocks = math.ceil(max_new_tokens / block_size)
        steps_per_block = math.ceil(steps / num_blocks)
        canvas_worker_count = self._canvas_worker_count(len(prompts))
        executor_context = (
            ThreadPoolExecutor(max_workers=canvas_worker_count)
            if canvas_worker_count > 1
            else nullcontext()
        )
        all_slots = range(max_new_tokens)
        with executor_context as canvas_executor:
            for block_index, start in enumerate(range(0, max_new_tokens, block_size)):
                block = range(start, min(start + block_size, max_new_tokens))
                with timer("ctca.sample.transfer_schedule"):
                    transfer_counts = get_num_transfer_tokens(
                        generation_mask(canvas, prompt_lens, block, mask_id),
                        steps_per_block,
                        self.scheduler,
                        stochastic=stochastic_transfer,
                    )
                projection_range = all_slots if weighting_mode == "online_entropy" else block

                for step_index, counts in enumerate(transfer_counts.unbind(dim=1)):
                    master_probabilities, auxiliary_probabilities, active_positions = (
                        self._forward_aligned_probabilities(
                            canvas,
                            attention_mask,
                            prompt_lens,
                            auxiliary_prompts,
                            max_new_tokens,
                            auxiliary_run_caches=auxiliary_run_caches,
                            canvas_executor=canvas_executor,
                            projection_range=projection_range,
                            temperature_master=temperature_master,
                            temperature_auxiliary=temperature_auxiliary,
                            fusion_device=target_device,
                            trace_callback=trace_callback,
                            trace_step=(block_index, step_index),
                        )
                    )
                    if capture_logits:
                        self.last_aligned_probabilities.append(
                            (
                                active_positions.cpu(),
                                master_probabilities.detach().cpu(),
                                auxiliary_probabilities.detach().cpu(),
                            )
                        )

                    with timer("ctca.sample.score_candidates"):
                        if selection_mode == "tse":
                            weights_master = pairwise_weights(
                                master_probabilities,
                                auxiliary_probabilities,
                                mode=weighting_mode,
                                alpha=master_static_weight,
                                weight_temperature=weight_temperature,
                                epsilon=epsilon,
                                normalize_entropy=normalize_entropy,
                            )
                            fused = fuse_probabilities(
                                master_probabilities, auxiliary_probabilities, weights_master
                            )
                            scores, predicted_tokens = fused_confidence(fused)
                            if trace_callback is not None:
                                consensus = distribution_consensus(
                                    scores,
                                    master_probabilities,
                                    auxiliary_probabilities,
                                    weights_master,
                                    epsilon,
                                )
                        else:
                            selected_probabilities = (
                                master_probabilities
                                if baseline_model == self.master_id
                                else auxiliary_probabilities
                            )
                            scores, predicted_tokens = baseline_candidates(
                                selected_probabilities, temperature, remasking
                            )

                    if trace_callback is not None:
                        trace_callback(
                            "fusion",
                            {
                                "step": (block_index, step_index),
                                "active_positions": active_positions.detach().cpu(),
                                "predicted_tokens": predicted_tokens.detach().cpu(),
                                "scores": scores.detach().cpu(),
                                "consensus_scores": (
                                    consensus.detach().cpu()
                                    if selection_mode == "tse"
                                    else None
                                ),
                                "master_probabilities": master_probabilities.detach().cpu(),
                                "auxiliary_probabilities": (
                                    auxiliary_probabilities.detach().cpu()
                                ),
                                "fused_probabilities": (
                                    fused.detach().cpu()
                                    if selection_mode == "tse"
                                    else None
                                ),
                                "weights_a": (
                                    weights_master.detach().cpu()
                                    if selection_mode == "tse"
                                    else None
                                ),
                            },
                        )

                    with timer("ctca.sample.select_and_commit"):
                        in_block = generation_mask(canvas, prompt_lens, block, mask_id)[
                            active_positions[:, 0], active_positions[:, 1]
                        ]
                        selected_positions, selected_tokens = select_positions(
                            scores.to(self.master_device)[in_block],
                            predicted_tokens.to(self.master_device)[in_block],
                            active_positions[in_block],
                            counts,
                        )
                        canvas = commit_tokens(canvas, selected_positions, selected_tokens)
                    if trace_callback is not None:
                        trace_callback(
                            "canvas",
                            {
                                "phase": "next",
                                "step": (block_index, step_index),
                                "canvas": canvas.detach().cpu(),
                                "prompt_lens": tuple(prompt_lens),
                                "auxiliary_prompts": tuple(
                                    prompt.detach().cpu() for prompt in auxiliary_prompts
                                ),
                            },
                        )

        return canvas


def _checkpoint_cache_id(model: torch.nn.Module, checkpoint: str | None) -> str | None:
    """Use disk projection tables only for revision-identified Hub checkpoints."""
    config = getattr(model, "config", None)
    loaded_from = getattr(config, "_name_or_path", None)
    checkpoint = checkpoint or loaded_from
    revision = getattr(config, "_commit_hash", None)
    if not checkpoint or not revision:
        return None
    if any(
        path and Path(path).expanduser().exists() for path in (checkpoint, loaded_from)
    ):
        return None
    return f"{checkpoint}@{revision}"

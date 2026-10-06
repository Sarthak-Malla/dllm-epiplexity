"""Heterogeneous two-model TSE sampling with CTCA.

Run the CPU integration tests with:
    pytest scripts/tests/test_ctca_sampler.py -v
"""

import math
import os
from concurrent.futures import Executor, ThreadPoolExecutor
from collections.abc import Sequence
from typing import Callable

import torch

from dllm.core.samplers.utils import add_gumbel_noise, get_num_transfer_tokens
from dllm.core.schedulers import LinearAlphaScheduler
from dllm.pipelines.tse.ctca import (
    CanvasRunCache,
    CrossTokenizerAligner,
    build_model_canvas_view,
)
from dllm.pipelines.tse.divergence import agreement_factor, jensen_shannon_divergence
from dllm.pipelines.tse.fusion import fuse_probabilities, logits_to_probabilities
from dllm.pipelines.tse.scoring import consensus_scores, fused_confidence
from dllm.pipelines.tse.selection import commit_tokens, select_positions
from dllm.pipelines.tse.weighting import (
    online_entropy_weights,
    per_token_margin_weights,
    static_weights,
)
from dllm.pipelines.tse.utils import timer


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
        force_rebuild: bool = False,
        projection_temperature: float = 0.05,
        projection_chunk_size: int = 2500,
        projection_mode: str = "exact",
        projection_top_k: int = 64,
        num_anchors: int = 3000,
        min_anchors: int = 128,
        master_cache_id: str | None = None,
        auxiliary_cache_id: str | None = None,
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

        master_embeddings = self.master_model.get_input_embeddings().weight.detach()
        auxiliary_embeddings = (
            self.auxiliary_model.get_input_embeddings().weight.detach()
        )
        with timer("ctca.init.aligner"):
            self.aligner = CrossTokenizerAligner(
                master_tokenizer,
                master_embeddings,
                master_id=master_cache_id or master_id,
                cache_dir=cache_dir,
                force_rebuild=force_rebuild,
            )
            self.aligner.register_auxiliary_model(
                auxiliary_cache_id or auxiliary_id,
                auxiliary_tokenizer,
                auxiliary_embeddings,
                temperature=projection_temperature,
                chunk_size=projection_chunk_size,
                projection_mode=projection_mode,
                projection_top_k=projection_top_k,
                num_anchors=num_anchors,
                min_anchors=min_anchors,
            )
        self._aligner_auxiliary_id = auxiliary_cache_id or auxiliary_id

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

    def _generation_mask(
        self,
        canvas: torch.Tensor,
        prompt_lens: Sequence[int],
        max_new_tokens: int,
        *,
        block_index: int | None = None,
        block_size: int | None = None,
    ) -> torch.Tensor:
        mask = torch.zeros_like(canvas, dtype=torch.bool)
        for sample_index, prompt_len in enumerate(prompt_lens):
            generation_start = 0
            generation_end = max_new_tokens
            if block_index is not None:
                if block_size is None:
                    raise ValueError("block_size is required when block_index is set")
                generation_start = block_index * block_size
                generation_end = min(generation_start + block_size, max_new_tokens)
            generated = canvas[
                sample_index,
                prompt_len + generation_start : prompt_len + generation_end,
            ]
            mask[
                sample_index,
                prompt_len + generation_start : prompt_len + generation_end,
            ] = (
                generated == self.master_tokenizer.mask_token_id
            )
        return mask

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
        auxiliary_run_caches: Sequence[CanvasRunCache] | None = None,
        canvas_executor: Executor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, list]:
        def build_view(sample_index: int):
            prompt_len = prompt_lens[sample_index]
            run_cache = (
                auxiliary_run_caches[sample_index]
                if auxiliary_run_caches is not None
                else None
            )
            generated = canvas[
                sample_index, prompt_len : prompt_len + max_new_tokens
            ].detach().cpu()
            return build_model_canvas_view(
                auxiliary_prompts[sample_index],
                generated,
                master_tokenizer=self.master_tokenizer,
                model_tokenizer=self.auxiliary_tokenizer,
                master_mask_token_id=self.master_tokenizer.mask_token_id,
                model_mask_token_id=self.auxiliary_tokenizer.mask_token_id,
                run_cache=run_cache,
            )

        with timer("ctca.auxiliary_batch.build_views"):
            sample_indices = range(len(prompt_lens))
            if canvas_executor is None:
                views = [build_view(sample_index) for sample_index in sample_indices]
            else:
                views = list(canvas_executor.map(build_view, sample_indices))

        with timer("ctca.auxiliary_batch.pad_and_copy"):
            max_length = max(view.input_ids.numel() for view in views)
            pad_token_id = self.auxiliary_tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = self.auxiliary_tokenizer.eos_token_id
            input_ids = torch.full(
                (len(views), max_length),
                int(pad_token_id),
                dtype=torch.long,
                device=self.auxiliary_device,
            )
            attention_mask = torch.zeros_like(input_ids)
            for sample_index, view in enumerate(views):
                length = view.input_ids.numel()
                input_ids[sample_index, :length] = view.input_ids.to(self.auxiliary_device)
                attention_mask[sample_index, :length] = 1
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
        block_index: int | None = None,
        block_size: int | None = None,
        temperature_master: float,
        temperature_auxiliary: float,
        fusion_device: torch.device,
        trace_callback: Callable[[str, dict], None] | None = None,
        trace_step: tuple[int, int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with timer("ctca.forward.master"):
            master_logits = self.master_model(
                canvas.to(self.master_device),
                attention_mask=attention_mask.to(self.master_device),
            ).logits
        with timer("ctca.forward.build_auxiliary_batch"):
            auxiliary_ids, auxiliary_mask, views = self._build_auxiliary_batch(
                canvas,
                prompt_lens,
                auxiliary_prompts,
                max_new_tokens,
                auxiliary_run_caches=auxiliary_run_caches,
                canvas_executor=canvas_executor,
            )
        with timer("ctca.forward.auxiliary"):
            auxiliary_logits = self.auxiliary_model(
                auxiliary_ids,
                attention_mask=auxiliary_mask,
            ).logits

        with timer("ctca.forward.master_probabilities"):
            active_mask = self._generation_mask(
                canvas,
                prompt_lens,
                max_new_tokens,
                block_index=block_index,
                block_size=block_size,
            )
            master_probabilities = logits_to_probabilities(
                master_logits[active_mask.to(self.master_device)].to(fusion_device),
                temperature_master,
            )
        if block_index is None:
            projection_start = 0
            projection_end = max_new_tokens
        else:
            if block_size is None:
                raise ValueError("block_size is required when block_index is set")
            projection_start = block_index * block_size
            projection_end = min(projection_start + block_size, max_new_tokens)
        master_offsets = tuple(
            (float(index), float(index + 1))
            for index in range(projection_start, projection_end)
        )
        with timer("ctca.forward.project_auxiliary"):
            spatial_by_sample = []
            for sample_index, (prompt_len, view) in enumerate(zip(prompt_lens, views)):
                local_active = active_mask[
                    sample_index,
                    prompt_len + projection_start : prompt_len + projection_end,
                ]
                generated_logits = auxiliary_logits[sample_index, view.generation_slice]
                auxiliary_probabilities = logits_to_probabilities(
                    generated_logits.to(fusion_device), temperature_auxiliary
                )
                overlap_matrix = view.overlap_matrix if block_index is None else None
                spatial = self.aligner.spatial_warp_model_probabilities(
                    self._aligner_auxiliary_id,
                    auxiliary_probabilities,
                    view.offsets,
                    master_offsets,
                    overlap_matrix=overlap_matrix,
                )
                spatial_by_sample.append(spatial[local_active.to(fusion_device)])

            spatial_auxiliary = torch.cat(spatial_by_sample, dim=0)
            projected_auxiliary = self.aligner.project_spatial_probabilities(
                self._aligner_auxiliary_id,
                spatial_auxiliary,
            )
            active_positions = active_mask.nonzero(as_tuple=False)
        if trace_callback is not None:
            trace_callback(
                "forward",
                {
                    "step": trace_step,
                    "canvas": canvas.detach().cpu(),
                    "prompt_lens": tuple(prompt_lens),
                    "active_positions": active_positions.detach().cpu(),
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
            prompt_lens = [prompt.numel() for prompt in prompts]
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
            maximum_length = max(prompt_lens) + max_new_tokens
            master_pad_id = self.master_tokenizer.pad_token_id
            if master_pad_id is None:
                master_pad_id = self.master_tokenizer.eos_token_id
            canvas = torch.full(
                (len(prompts), maximum_length),
                int(master_pad_id),
                dtype=torch.long,
                device=self.master_device,
            )
            attention_mask = torch.zeros_like(canvas)
            for sample_index, prompt in enumerate(prompts):
                prompt_len = prompt.numel()
                canvas[sample_index, :prompt_len] = prompt
                canvas[
                    sample_index, prompt_len : prompt_len + max_new_tokens
                ] = self.master_tokenizer.mask_token_id
                attention_mask[sample_index, : prompt_len + max_new_tokens] = 1
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

        target_device = torch.device(fusion_device or self.master_device)
        temperature_master = temperature_a if self.master_id == "a" else temperature_b
        temperature_auxiliary = temperature_b if self.auxiliary_id == "b" else temperature_a
        master_static_weight = alpha if self.master_id == "a" else 1 - alpha
        self.last_aligned_probabilities = []
        auxiliary_run_caches: list[CanvasRunCache] = [{} for _ in prompts]
        num_blocks = math.ceil(max_new_tokens / block_size)
        steps_per_block = math.ceil(steps / num_blocks)
        canvas_worker_count = self._canvas_worker_count(len(prompts))
        canvas_executor = (
            ThreadPoolExecutor(max_workers=canvas_worker_count)
            if canvas_worker_count > 1
            else None
        )

        for block_index in range(num_blocks):
            with timer("ctca.sample.transfer_schedule"):
                block_mask = torch.zeros(
                    (len(prompts), block_size),
                    dtype=torch.bool,
                    device=self.master_device,
                )
                for sample_index, prompt_len in enumerate(prompt_lens):
                    start = prompt_len + block_index * block_size
                    end = min(start + block_size, prompt_len + max_new_tokens)
                    if start < end:
                        block_mask[sample_index, : end - start] = (
                            canvas[sample_index, start:end]
                            == self.master_tokenizer.mask_token_id
                        )
                transfer_counts = get_num_transfer_tokens(
                    block_mask,
                    steps_per_block,
                    self.scheduler,
                    stochastic=stochastic_transfer,
                )

            for step_index in range(transfer_counts.shape[1]):
                project_current_block_only = weighting_mode != "online_entropy"
                master_probabilities, auxiliary_probabilities, active_positions = (
                    self._forward_aligned_probabilities(
                        canvas,
                        attention_mask,
                        prompt_lens,
                        auxiliary_prompts,
                        max_new_tokens,
                        auxiliary_run_caches=auxiliary_run_caches,
                        canvas_executor=canvas_executor,
                        block_index=(
                            block_index if project_current_block_only else None
                        ),
                        block_size=block_size if project_current_block_only else None,
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
                        if weighting_mode == "static":
                            weighting = static_weights(
                                master_static_weight,
                                master_probabilities.shape[0],
                                target_device,
                            )
                        elif weighting_mode == "online_entropy":
                            weighting = online_entropy_weights(
                                master_probabilities,
                                auxiliary_probabilities,
                                weight_temperature,
                                epsilon,
                                normalize_entropy,
                            )
                        else:
                            weighting = per_token_margin_weights(
                                master_probabilities,
                                auxiliary_probabilities,
                                weight_temperature,
                            )
                        fused = fuse_probabilities(
                            master_probabilities,
                            auxiliary_probabilities,
                            weighting.weights_a,
                        )
                        divergence = jensen_shannon_divergence(
                            master_probabilities,
                            auxiliary_probabilities,
                            weighting.weights_a,
                            epsilon,
                        )
                        agreement = agreement_factor(
                            divergence, weighting.weights_a, epsilon
                        )
                        confidence, predicted_tokens = fused_confidence(fused)
                        scores = consensus_scores(confidence, agreement)
                    else:
                        selected_probabilities = (
                            master_probabilities
                            if baseline_model == self.master_id
                            else auxiliary_probabilities
                        )
                        if temperature == 0:
                            predicted_tokens = selected_probabilities.argmax(dim=-1)
                        else:
                            predicted_tokens = add_gumbel_noise(
                                selected_probabilities.clamp_min(epsilon).log(),
                                temperature,
                            ).argmax(dim=-1)
                        scores = selected_probabilities.gather(
                            -1, predicted_tokens.unsqueeze(-1)
                        ).squeeze(-1)
                        if remasking == "random":
                            scores = torch.rand_like(scores)

                if trace_callback is not None:
                    trace_callback(
                        "fusion",
                        {
                            "step": (block_index, step_index),
                            "active_positions": active_positions.detach().cpu(),
                            "predicted_tokens": predicted_tokens.detach().cpu(),
                            "scores": scores.detach().cpu(),
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
                                weighting.weights_a.detach().cpu()
                                if selection_mode == "tse"
                                else None
                            ),
                        },
                    )

                with timer("ctca.sample.select_and_commit"):
                    score_canvas = torch.full_like(
                        canvas, -torch.inf, dtype=torch.float32
                    )
                    token_canvas = canvas.clone()
                    token_canvas[active_positions[:, 0], active_positions[:, 1]] = (
                        predicted_tokens.to(self.master_device)
                    )
                    score_canvas[
                        active_positions[:, 0], active_positions[:, 1]
                    ] = scores.to(self.master_device)
                    for sample_index, prompt_len in enumerate(prompt_lens):
                        score_canvas[
                            sample_index,
                            prompt_len + (block_index + 1) * block_size :,
                        ] = -torch.inf

                    mask_index = torch.zeros_like(canvas, dtype=torch.bool)
                    if active_positions.numel():
                        mask_index[active_positions[:, 0], active_positions[:, 1]] = True
                    selected_positions, selected_tokens = select_positions(
                        score_canvas[mask_index],
                        token_canvas[mask_index],
                        mask_index.nonzero(as_tuple=False),
                        transfer_counts[:, step_index],
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

        if canvas_executor is not None:
            canvas_executor.shutdown(wait=True)

        return canvas

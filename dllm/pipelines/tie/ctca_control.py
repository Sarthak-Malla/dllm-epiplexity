"""Run the matched CTCA benchmark with python -m dllm.pipelines.tie.eval."""

import torch

from dllm.pipelines.tse.ctca import CrossTokenizerAligner, build_model_canvas_view


def create_ctca_control(master, dream):
    aligner = CrossTokenizerAligner(
        master.tokenizer, master.model.get_input_embeddings().weight.detach(),
        master_id=master.tokenizer.name_or_path, shared_token_policy="exact_shared",
    )
    peer_id = dream.tokenizer.name_or_path
    aligner.register_auxiliary_model(
        peer_id, dream.tokenizer, dream.model.get_input_embeddings().weight.detach(),
        anchor_temperature=0.03, temperature=0.10,
    )
    aligner.warm_projection_cache(peer_id, master.device)

    def choose(experiment, state, block_index, offset, canvas, ranked, tokens, confidence, count):
        prefix = state.prompt.numel()
        view = build_model_canvas_view(
            experiment.dream_prompt, canvas[prefix:],
            master_tokenizer=master.tokenizer, model_tokenizer=dream.tokenizer,
            master_mask_token_id=master.vocabulary.mask_id,
            model_mask_token_id=dream.vocabulary.mask_id,
        )
        native = view.input_ids.to(dream.device)
        active = (ranked + offset).tolist()
        peer = experiment.logp(dream, native, range(view.generation_start, len(native))).exp()
        projected = aligner.project_model_probabilities(
            peer_id, peer.to(master.device), view.offsets,
            [(pos - prefix, pos - prefix + 1) for pos in active],
        )
        base = experiment.logp(master, canvas, active).exp()
        projected.masked_fill_(~master._allowed_logits[(base.device, base.shape[-1])], 0)
        mass = projected.sum(-1, keepdim=True)
        projected = torch.where(mass > 0, projected / mass.clamp_min(1e-12), base)
        scores, values = ((base + projected) / 2).max(-1)
        order = scores.argsort(descending=True, stable=True)[:count]
        return ranked[order], values[order]

    return choose

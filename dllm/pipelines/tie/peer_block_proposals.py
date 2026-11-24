"""Run the peer block pilot with python -m dllm.pipelines.tie.eval --model tie."""

from dataclasses import asdict
from itertools import takewhile

import torch

from .relay import RelayError
from .shared_audit import clone_state


BLOCK_STARTS = (17, 65)
PEER_WIDTH = 24
BLOCK_STEPS = 8


def encode_block(owner, peer, owner_prefix, peer_prefix, proposed, width):
    prefix = owner.decode(owner_prefix)
    ordinary = list(takewhile(lambda token: token not in peer.special_ids, proposed))
    for end in range(1, len(ordinary) + 1):
        text = peer.decode(ordinary[:end])
        if not text.strip() or "\ufffd" in text:
            continue
        ids = owner.encode(text)
        if len(ids) != width or any(token in owner.special_ids for token in ids):
            continue
        if (owner.decode(owner_prefix + ids) == prefix + text
                and peer.decode(peer_prefix + ordinary[:end]) == prefix + text):
            return ids, text, end
    raise RelayError("no native Dream token prefix filled the LLaDA block exactly")


class PeerBlockProposals:
    """Compare Dream's next-block proposal with LLaDA's own resampling."""

    def __init__(self, sampler):
        self.sampler = sampler
        for step in BLOCK_STARTS:
            block, _ = sampler.schedule[step - 1]
            entries = sampler.schedule[step - 1:step + BLOCK_STEPS - 1]
            if (block == 0 or block == len(sampler.block_sizes) - 1
                    or sampler.schedule[step - 2][0] == block
                    or len(entries) != BLOCK_STEPS or any(index != block for index, _ in entries)
                    or sum(count for _, count in entries) != sampler.block_sizes[block]):
                raise ValueError(f"Peer block proposal step {step} requires a whole reasoning block")

    def propose(self, owner, template, block):
        source, target = owner.member.vocabulary, template.member.vocabulary
        owner_ids = torch.cat(owner.blocks[:block]).tolist()
        if any(token in source.special_ids for token in owner_ids):
            raise RelayError("owner prefix ended or contains special tokens")
        prefix = source.decode(owner_ids)
        foreign = target.encode(prefix)
        if not prefix or "\ufffd" in prefix or target.decode(foreign) != prefix:
            raise RelayError("owner prefix failed the Dream text round trip")
        if any(token in target.special_ids for token in foreign):
            raise RelayError("owner prefix encoded to a Dream special token")
        donor = clone_state(template)
        donor.blocks = [
            donor.prompt.new_tensor(foreign),
            donor.prompt.new_full((PEER_WIDTH,), target.mask_id),
            *donor.blocks[block + 1:],
        ]
        length = len(donor.prompt) + len(donor.marker) + sum(map(len, donor.blocks))
        if not donor.member.fits_context(length):
            raise RelayError("Dream proposal exceeds its context limit")
        for _ in range(BLOCK_STEPS):
            donor.step(1, PEER_WIDTH // BLOCK_STEPS)
        block_ids = donor.blocks[1].tolist()
        return owner_ids, foreign, block_ids, donor.processed_tokens

    def finish(self, state, step):
        for block, count in self.sampler.schedule[step:]:
            state.step(block, count)
        return asdict(state.completion(self.sampler.config.answer_marker))

    def probe(self, owner, template, step, block, doc_id):
        event = {
            "step": step, "block": block, "proposal_forwards": {"dream": 0, "llada": 0},
            "proposal_tokens": {"dream": 0, "llada": 0},
        }
        try:
            owner_prefix, peer_prefix, proposed, cost = self.propose(owner, template, block)
            event["proposal_forwards"]["dream"] = BLOCK_STEPS
            event["proposal_tokens"]["dream"] = cost
            ids, text, dream_width = encode_block(
                owner.member.vocabulary, template.member.vocabulary,
                owner_prefix, peer_prefix, proposed, self.sampler.block_sizes[block],
            )
        except RelayError as error:
            event["rejected"] = str(error)
            return event

        control = clone_state(owner)
        rng = torch.Generator(device=control.member.device).manual_seed(20261122 + doc_id * 128 + step)
        block_end = step + BLOCK_STEPS - 1
        for next_block, count in self.sampler.schedule[step - 1:block_end]:
            control.step(next_block, count, generator=rng)
        event["proposal_forwards"]["llada"] = BLOCK_STEPS
        event["proposal_tokens"]["llada"] = control.processed_tokens - owner.processed_tokens

        peer = clone_state(owner)
        peer.blocks[block] = owner.prompt.new_tensor(ids)
        event.update(
            proposed_text=text,
            dream_width=dream_width,
            llada_width=len(ids),
            owner=self.finish(control, block_end),
            peer=self.finish(peer, block_end),
            continuation_forwards_per_branch=len(self.sampler.schedule) - block_end,
        )
        return event

    @torch.inference_mode()
    def sample(self, prompts, doc_id):
        owner, template = self.sampler.initialize(prompts)
        events = []
        for step, (block, count) in enumerate(self.sampler.schedule, 1):
            if step in BLOCK_STARTS:
                events.append(self.probe(owner, template, step, block, doc_id))
            owner.step(block, count)
        return {
            "baseline": asdict(owner.completion(self.sampler.config.answer_marker)),
            "checkpoints": events,
        }

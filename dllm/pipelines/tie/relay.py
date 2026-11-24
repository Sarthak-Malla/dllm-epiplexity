"""Transfer partial masked sequences for `python -m dllm.pipelines.tie.eval`."""

from dataclasses import dataclass
from itertools import chain, groupby

from transformers import PreTrainedTokenizerBase


@dataclass(frozen=True)
class Vocabulary:
    tokenizer: PreTrainedTokenizerBase
    mask_id: int
    eos_id: int
    terminal_ids: frozenset[int]
    special_ids: frozenset[int]
    generation_ids: tuple[int, ...]

    @classmethod
    def from_tokenizer(cls, tokenizer: PreTrainedTokenizerBase) -> "Vocabulary":
        tokens = tokenizer.get_vocab()
        mask_id, eos_id = tokenizer.mask_token_id, tokenizer.eos_token_id
        if mask_id is None or eos_id is None or mask_id == eos_id:
            raise ValueError("TIE requires distinct mask and EOS token IDs")
        end_of_turn = tokens.get(getattr(tokenizer, "eot_token", None))
        terminal_ids = frozenset(
            token for token in (eos_id, end_of_turn, tokenizer.pad_token_id)
            if token is not None and token != mask_id
        )
        special_ids = frozenset(tokenizer.all_special_ids) | terminal_ids | {mask_id}
        generation_ids = tuple(sorted((set(tokens.values()) - special_ids) | terminal_ids))
        return cls(tokenizer, mask_id, eos_id, terminal_ids, special_ids, generation_ids)

    def kind(self, token: int) -> str:
        if token == self.mask_id:
            return "mask"
        if token in self.terminal_ids:
            return "terminal"
        if token in self.special_ids:
            raise RelayError(f"unsupported committed special token {token}")
        return "text"

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(
            ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)


class RelayError(ValueError):
    """A partial sequence cannot be faithfully represented by the receiver."""


def _regions(ids, vocabulary: Vocabulary):
    for kind, tokens in groupby(ids, key=vocabulary.kind):
        run = list(tokens)
        if kind == "text":
            text = vocabulary.decode(run)
            if not text or "\ufffd" in text:
                raise RelayError("committed run has empty or incomplete Unicode text")
            yield kind, text
        else:
            yield kind, len(run)


def transfer_blocks(
    blocks: list[list[int]],
    source: Vocabulary,
    target: Vocabulary,
) -> list[list[int]]:
    """Retokenize committed runs and preserve the number of masks in each gap.

    Blocks are logical scheduling regions. Their token lengths can change
    across tokenizers. End markers are relayed as native EOS tokens.
    """
    transferred = []
    for block in blocks:
        ids = []
        for kind, value in _regions(block, source):
            if kind == "mask":
                ids.extend([target.mask_id] * value)
            elif kind == "terminal":
                ids.extend([target.eos_id] * value)
            else:
                encoded = target.encode(value)
                if any(token in target.special_ids for token in encoded):
                    raise RelayError("retokenization introduced a special token")
                if target.decode(encoded) != value:
                    raise RelayError("committed text failed round-trip retokenization")
                ids.extend(encoded)
        transferred.append(ids)

    # Some tokenizers decode a run differently when joined to a preceding block.
    if list(_regions(chain.from_iterable(blocks), source)) != list(
        _regions(chain.from_iterable(transferred), target)
    ):
        raise RelayError("retokenization changed text across a block boundary")
    return transferred

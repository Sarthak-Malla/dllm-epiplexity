
"""Run TIE on LLaDA and Dream with `python -m dllm.pipelines.tie.eval`."""

from dataclasses import dataclass, field

import torch
from transformers import PreTrainedTokenizerBase

from dllm.core.samplers.utils import get_num_transfer_tokens
from dllm.core.schedulers import LinearAlphaScheduler

from .relay import RelayError, Vocabulary, transfer_blocks


@dataclass(frozen=True)
class TIESamplerConfig:
    max_new_tokens: int = 256
    answer_tokens: int = 16
    block_size: int = 16
    steps: int = 128
    relay_interval: int = 16
    answer_marker: str = "\nAnswer:"

    def __post_init__(self):
        if not 0 < self.answer_tokens < self.max_new_tokens:
            raise ValueError("answer_tokens must be between zero and max_new_tokens")
        if self.steps < 1 or self.block_size < 1 or self.relay_interval < 0:
            raise ValueError(
                "steps and block_size must be positive and relay_interval nonnegative"
            )
        if not self.answer_marker.strip():
            raise ValueError("answer_marker must contain visible text")


@dataclass
class TIEMember:
    name: str
    model: torch.nn.Module
    tokenizer: PreTrainedTokenizerBase
    right_shift_logits: bool = False
    probability_temperature: float = 1.0
    vocabulary: Vocabulary = field(init=False, repr=False)
    _allowed_logits: dict[tuple[torch.device, int], torch.Tensor] = field(
        init=False, default_factory=dict, repr=False
    )

    def __post_init__(self):
        if not self.probability_temperature > 0:
            raise ValueError("probability_temperature must be positive")
        self.vocabulary = Vocabulary.from_tokenizer(self.tokenizer)

    @property
    def device(self) -> torch.device:
        device = getattr(self.model, "device", None)
        return device if device is not None else next(self.model.parameters()).device

    def fits_context(self, length: int) -> bool:
        config = getattr(self.model, "config", None)
        return all(
            length <= limit
            for key in ("max_position_embeddings", "max_sequence_length")
            if (limit := getattr(config, key, None))
        )

    def _sampling_logits(
        self, canvas: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        batch = canvas.unsqueeze(0)
        logits = self.model(batch, attention_mask=torch.ones_like(batch)).logits
        if self.right_shift_logits:
            positions = (positions - 1).clamp_min(0)
        logits = logits[0].index_select(0, positions.to(logits.device)).float()
        key = (logits.device, logits.shape[-1])
        if key not in self._allowed_logits:
            size = logits.shape[-1]
            if self.vocabulary.mask_id >= size or self.vocabulary.eos_id >= size:
                raise ValueError(f"{self.name} tokenizer exceeds the model vocabulary")
            allowed_ids = [i for i in self.vocabulary.generation_ids if 0 <= i < size]
            allowed = torch.zeros(size, dtype=torch.bool, device=logits.device)
            allowed[torch.tensor(allowed_ids, device=logits.device, dtype=torch.long)] = True
            self._allowed_logits[key] = allowed
        logits.masked_fill_(~self._allowed_logits[key], -torch.inf)
        return logits / self.probability_temperature

    def log_probabilities(self, canvas, positions):
        return self._sampling_logits(canvas, positions).log_softmax(-1)

    def probabilities(self, canvas, positions):
        return self._sampling_logits(canvas, positions).softmax(-1)

    def predict(self, canvas, positions):
        confidence, tokens = self.probabilities(canvas, positions).max(-1)
        return confidence.to(canvas.device), tokens.to(canvas.device)


@dataclass(frozen=True)
class TIECompletion:
    member: str
    response: str
    reasoning_ids: tuple[int, ...]
    answer_ids: tuple[int, ...]
    token_changes: float
    answer_confidence: float | None
    forward_calls: int
    processed_tokens: int


@dataclass(frozen=True)
class RelayEvent:
    step: int
    leader: str | None
    applied: bool
    reason: str | None
    interval_token_changes: dict[str, float | None]
    before: dict[str, list[list[int]]] | None
    after: dict[str, list[list[int]]] | None


@dataclass(frozen=True)
class TIEOutput:
    completions: tuple[TIECompletion, ...]
    selected_index: int
    relays: tuple[RelayEvent, ...]

    @property
    def selected(self) -> TIECompletion:
        return self.completions[self.selected_index]


@dataclass
class _AnswerHistory:
    previous: dict[int, int] = field(default_factory=dict)
    interval_changes: float = 0.0
    total_changes: float = 0.0
    interval_confidence: float | None = None
    confidence: float | None = None

    def observe(self, hints: dict[int, tuple[int, float]]):
        current = {position: token for position, (token, _) in hints.items()}
        if self.previous:
            delta = sum(
                token != current[position]
                for position, token in self.previous.items()
                if position in current
            ) / len(self.previous)
            self.total_changes += delta
            if self.interval_confidence is not None:
                self.interval_changes += delta
        self.previous = current
        if hints:
            self.confidence = sum(score for _, score in hints.values()) / len(hints)
            self.interval_confidence = self.confidence

    def next_interval(self):
        self.interval_changes = 0.0
        self.interval_confidence = None

    def rank(self, *, final: bool) -> tuple[float, float]:
        confidence = self.confidence if final else self.interval_confidence
        changes = self.total_changes if final else self.interval_changes
        return (float("inf"), 0.0) if confidence is None else (changes, -confidence)


@dataclass
class _Trajectory:
    member: TIEMember
    prompt: torch.Tensor
    marker: torch.Tensor
    blocks: list[torch.Tensor]
    history: _AnswerHistory = field(default_factory=_AnswerHistory)
    forward_calls: int = 0
    processed_tokens: int = 0

    def step(self, block_index: int, count: int, choose=None, generator=None,
             *, include_probabilities=False):
        vocabulary = self.member.vocabulary
        answer = self.blocks[-1]
        block = self.blocks[block_index]
        answer_step = block_index == len(self.blocks) - 1
        answer_offset = self.prompt.numel() + sum(
            block.numel() for block in self.blocks[:-1]
        ) + self.marker.numel()
        block_offset = answer_offset if answer_step else self.prompt.numel() + sum(
            block.numel() for block in self.blocks[:block_index]
        )
        active = (block == vocabulary.mask_id).nonzero(as_tuple=True)[0]
        if count > active.numel():
            raise RuntimeError("relay did not preserve the scheduled mask count")
        answer_masks = active if answer_step else (
            answer == vocabulary.mask_id
        ).nonzero(as_tuple=True)[0]
        positions = active + block_offset
        if not answer_step:
            positions = torch.cat((positions, answer_masks + answer_offset))
        canvas = torch.cat((self.prompt, *self.blocks[:-1], self.marker, answer))
        if include_probabilities:
            if choose is None or generator is not None:
                raise ValueError("Commit probabilities require a deterministic chooser")
            probabilities = self.member.probabilities(canvas, positions)
            confidence, tokens = probabilities.max(-1)
        elif generator is None:
            confidence, tokens = self.member.predict(canvas, positions)
        else:
            logp = self.member.log_probabilities(canvas, positions)
            confidence, tokens = logp.exp().max(-1)
            rows = logp[:active.numel()]
            tokens[:active.numel()] = torch.multinomial(
                (rows / 0.3).softmax(-1), 1, generator=generator
            ).flatten()
        if not torch.isfinite(confidence).all():
            raise RuntimeError(f"{self.member.name} produced invalid token probabilities")
        self.forward_calls += 1
        self.processed_tokens += canvas.numel()
        answer_offset_in_predictions = 0 if answer_step else active.numel()
        hints = {
            position: (token, score)
            for position, token, score in zip(
                answer_masks.tolist(),
                tokens[answer_offset_in_predictions:].tolist(),
                confidence[answer_offset_in_predictions:].tolist(),
            )
            if token not in vocabulary.terminal_ids
        }
        self.history.observe(hints)
        ranked = confidence[:active.numel()].argsort(descending=True, stable=True)
        chosen = ranked[:count]
        slots, values = active[chosen], tokens[chosen]
        if choose is not None:
            extra = (
                {"probabilities": probabilities.index_select(0, ranked)}
                if include_probabilities else {}
            )
            slots, values = choose(
                self, block_index, block_offset, canvas, active[ranked],
                tokens[ranked], confidence[ranked], count, **extra,
            )
        block[slots] = values

    def snapshot(self) -> list[list[int]]:
        return [block.tolist() for block in self.blocks]

    def completion(self, answer_marker: str, *, reasoning_blocks: int | None = None) -> TIECompletion:
        vocabulary = self.member.vocabulary
        last_reasoning = -1 if reasoning_blocks is None else reasoning_blocks
        reasoning = torch.cat(self.blocks[:last_reasoning]).tolist()
        answer = self.blocks[-1].tolist()

        def decode_visible(ids):
            stop = next(
                (i for i, token in enumerate(ids) if token in vocabulary.terminal_ids),
                len(ids),
            )
            return vocabulary.decode(ids[:stop])

        reasoning_text = decode_visible(reasoning).rstrip()
        answer_text = decode_visible(answer).strip()
        response = f"{reasoning_text}{answer_marker} {answer_text}"
        return TIECompletion(
            self.member.name, response.lstrip(), tuple(reasoning), tuple(answer),
            self.history.total_changes, self.history.confidence,
            self.forward_calls, self.processed_tokens,
        )


def _relay(states: list[_Trajectory], step: int, capture: bool) -> RelayEvent:
    scores = {
        state.member.name: (
            state.history.interval_changes
            if state.history.interval_confidence is not None else None
        )
        for state in states
    }
    leader = min(states, key=lambda state: state.history.rank(final=False))
    if leader.history.interval_confidence is None:
        return RelayEvent(
            step, None, False, "no usable masked answer predictions", scores, None, None
        )
    donor_blocks = leader.snapshot()
    before = {state.member.name: state.snapshot() for state in states} if capture else None
    try:
        transfers = [
            donor_blocks if state is leader else transfer_blocks(
                donor_blocks, leader.member.vocabulary, state.member.vocabulary
            )
            for state in states
        ]
        for state, blocks in zip(states, transfers):
            length = state.prompt.numel() + state.marker.numel() + sum(map(len, blocks))
            if not state.member.fits_context(length):
                raise RelayError(f"relayed canvas exceeds {state.member.name}'s context limit")
    except RelayError as error:
        for state in states:
            state.history.next_interval()
        return RelayEvent(step, leader.member.name, False, str(error), scores, before, before)

    inherited_changes = leader.history.total_changes
    for state, blocks in zip(states, transfers):
        if state is not leader:
            state.blocks = [
                torch.tensor(ids, device=state.prompt.device, dtype=torch.long) for ids in blocks
            ]
        # The donor's history belongs to the shared prefix. Subsequent changes
        # must be measured in the receiving model's own token coordinates.
        state.history = _AnswerHistory(total_changes=inherited_changes)
    after = dict(zip((state.member.name for state in states), transfers)) if capture else None
    return RelayEvent(step, leader.member.name, True, None, scores, before, after)


class TIESampler:
    """Generate one shared question given prompt IDs in each member's tokenizer."""

    def __init__(
        self, members: list[TIEMember], config: TIESamplerConfig | None = None
    ):
        if not members or len({member.name for member in members}) != len(members):
            raise ValueError("TIE requires at least one member with a unique name")
        self.members = tuple(members)
        self.config = config or TIESamplerConfig()
        reasoning_tokens = self.config.max_new_tokens - self.config.answer_tokens
        self.block_sizes = [
            min(self.config.block_size, reasoning_tokens - start)
            for start in range(0, reasoning_tokens, self.config.block_size)
        ] + [self.config.answer_tokens]
        per_block, extra = divmod(self.config.steps, len(self.block_sizes))
        if per_block == 0:
            raise ValueError("steps must allow at least one denoising step per block")
        scheduler = LinearAlphaScheduler()
        self.schedule = [
            (index, int(count))
            for index, size in enumerate(self.block_sizes)
            for count in get_num_transfer_tokens(
                torch.ones((1, size), dtype=torch.bool), per_block + (index < extra),
                scheduler,
            )[0]
        ]

    def initialize(self, prompts):
        if len(prompts) != len(self.members):
            raise ValueError("provide a native prompt for each member in member order")
        states = []
        for member, ids in zip(self.members, prompts):
            vocabulary = member.vocabulary
            prompt = torch.as_tensor(ids, dtype=torch.long, device=member.device)
            marker = vocabulary.encode(self.config.answer_marker)
            if prompt.ndim != 1 or not prompt.numel():
                raise ValueError("each native prompt must be a nonempty token sequence")
            if not marker or any(token in vocabulary.special_ids for token in marker):
                raise ValueError(f"{member.name} cannot encode the answer marker as ordinary text")
            if not member.fits_context(prompt.numel() + len(marker) + self.config.max_new_tokens):
                raise ValueError(f"initial canvas exceeds {member.name}'s context limit")
            states.append(_Trajectory(
                member,
                prompt,
                prompt.new_tensor(marker),
                [prompt.new_full((size,), vocabulary.mask_id) for size in self.block_sizes],
            ))
        return states

    @torch.inference_mode()
    def sample(
        self,
        prompts: list[list[int] | torch.Tensor],
        *,
        capture_relay_canvases: bool = False,
    ) -> TIEOutput:
        states = self.initialize(prompts)
        relays = []
        interval = self.config.relay_interval if len(states) > 1 else 0
        for step, (block, count) in enumerate(self.schedule, start=1):
            for state in states:
                state.step(block, count)
            if interval and step < len(self.schedule) and step % interval == 0:
                relays.append(_relay(states, step, capture_relay_canvases))

        selected = min(
            range(len(states)), key=lambda i: states[i].history.rank(final=True)
        )
        return TIEOutput(
            tuple(state.completion(self.config.answer_marker) for state in states),
            selected,
            tuple(relays),
        )

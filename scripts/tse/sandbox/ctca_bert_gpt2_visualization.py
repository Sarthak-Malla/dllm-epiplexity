"""Run a standalone BERT/GPT-2 CTCA visualization demo.

Local usage:
    source ~/.zshrc
    conda activate ~/miniconda3/envs/dllm
    python /home/sarthak.malla/dllm-epiplexity/scripts/tse/sandbox/ctca_bert_gpt2_visualization.py

Google Colab usage (after uploading this file):
    !pip install -q torch transformers plotly
    !python /content/ctca_bert_gpt2_visualization.py --output /content/ctca_demo.html

The script extracts contextual sentence embeddings, fabricates a sparse skewed
full-vocabulary probability distribution at every native token position,
aligns auxiliary positions to master positions using character-span overlap,
and projects the auxiliary vocabulary into the master vocabulary using an
embedding-invariant relative-anchor space built from shared token strings.
"""

from __future__ import annotations

import argparse
import html
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import plotly.graph_objects as go
import plotly.io as pio
import torch
import torch.nn.functional as F
from plotly.subplots import make_subplots
from transformers import AutoModel, AutoTokenizer


DEFAULT_SENTENCE = "The underdeveloped protozoa is never malicious to human cells."
MODEL_IDS = {
    "bert": "bert-base-uncased",
    "gpt2": "gpt2",
}
Span = tuple[float, float]


@dataclass(frozen=True)
class TokenizedSentence:
    """One tokenizer's token IDs, offsets, labels, and contextual embeddings."""

    input_ids: torch.Tensor
    offsets: tuple[Span, ...]
    labels: tuple[str, ...]
    contextual_embeddings: torch.Tensor
    sentence_embedding: torch.Tensor


@dataclass(frozen=True)
class RelativeAnchorSelection:
    """Matching auxiliary and master token IDs used as relative anchors."""

    auxiliary_anchor_ids: torch.Tensor
    master_anchor_ids: torch.Tensor
    total_anchors: int


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI options while tolerating arguments injected by Jupyter/Colab."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sentence", default=DEFAULT_SENTENCE)
    parser.add_argument(
        "--master",
        choices=("bert", "gpt2"),
        default="bert",
        help="Tokenizer/model that defines the final token positions and vocabulary.",
    )
    parser.add_argument(
        "--num-anchors",
        default="auto",
        help="Number of shared anchors, or 'auto' to use every usable anchor.",
    )
    parser.add_argument("--min-anchors", type=int, default=128)
    parser.add_argument(
        "--anchor-temperature",
        type=float,
        default=0.01,
        help="Softmax temperature for token-to-anchor similarity profiles.",
    )
    parser.add_argument(
        "--distribution-support",
        type=int,
        default=128,
        help="Random vocabulary subset size used for each fabricated distribution.",
    )
    parser.add_argument(
        "--distribution-temperature",
        type=float,
        default=0.35,
        help="Softmax temperature controlling the fabricated distribution's skew.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Seed used to fabricate reproducible random probability weights.",
    )
    parser.add_argument(
        "--projection-source-top-k",
        type=int,
        default=64,
        help=(
            "Highest-probability auxiliary candidates per aligned position retained "
            "for the relative-anchor projection."
        ),
    )
    parser.add_argument(
        "--projection-top-k",
        type=int,
        default=16,
        help="Master vocabulary neighbors retained for each active auxiliary ID.",
    )
    parser.add_argument("--projection-temperature", type=float, default=0.08)
    parser.add_argument(
        "--projection-chunk-size",
        type=int,
        default=1024,
        help="Master vocabulary chunk size used during relative-anchor search.",
    )
    parser.add_argument("--plot-top-k", type=int, default=10)
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device. 'auto' uses CUDA when available, otherwise CPU.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path.cwd() / "ctca_bert_gpt2_demo.html",
        help="HTML dashboard path.",
    )
    parser.add_argument("--show", action="store_true", help="Open the output in a browser.")
    parser.add_argument("--local-files-only", action="store_true")
    raw_arguments = list(sys.argv[1:] if argv is None else argv)
    if "ipykernel" in sys.modules:
        cleaned_arguments: list[str] = []
        index = 0
        while index < len(raw_arguments):
            if raw_arguments[index] == "-f" and index + 1 < len(raw_arguments):
                print(
                    "Ignoring notebook kernel argument: "
                    f"-f {raw_arguments[index + 1]}"
                )
                index += 2
                continue
            cleaned_arguments.append(raw_arguments[index])
            index += 1
        raw_arguments = cleaned_arguments
    return parser.parse_args(raw_arguments)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def token_label(tokenizer, token_id: int) -> str:
    """Return a label that preserves tokenizer boundary markers."""
    token = tokenizer.convert_ids_to_tokens(int(token_id))
    decoded = tokenizer.decode(
        [int(token_id)],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if decoded and decoded != token:
        return f"{token} ({decoded!r})"
    return str(token)


def tokenize_and_embed(
    sentence: str,
    tokenizer,
    model,
    device: torch.device,
) -> TokenizedSentence:
    """Tokenize a sentence and run the model to obtain contextual embeddings."""
    encoded = tokenizer(
        sentence,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offsets = tuple(
        (float(start), float(end))
        for start, end in encoded.pop("offset_mapping")[0].tolist()
    )
    model_inputs = {key: value.to(device) for key, value in encoded.items()}
    model.eval()
    with torch.inference_mode():
        hidden = model(**model_inputs).last_hidden_state[0].float()
    input_ids = model_inputs["input_ids"][0]
    labels = tuple(token_label(tokenizer, token_id) for token_id in input_ids.tolist())
    sentence_embedding = hidden.mean(dim=0)
    return TokenizedSentence(
        input_ids=input_ids.detach(),
        offsets=offsets,
        labels=labels,
        contextual_embeddings=hidden.detach(),
        sentence_embedding=sentence_embedding.detach(),
    )


def normalize_token_text(token: str) -> str:
    """Normalize common word-boundary markers before matching anchors."""
    return token.replace("Ġ", "").replace("▁", "").replace("##", "").strip().casefold()


def collect_shared_anchors(
    auxiliary_tokenizer,
    master_tokenizer,
    auxiliary_vocab_size: int,
    master_vocab_size: int,
    num_anchors: int | str,
    min_anchors: int,
) -> RelativeAnchorSelection:
    """Collect deterministic, one-to-one shared normalized token strings."""

    def normalized_vocab(tokenizer, vocab_size: int) -> dict[str, int]:
        special_ids = set(tokenizer.all_special_ids)
        result: dict[str, int] = {}
        for token, token_id in sorted(tokenizer.get_vocab().items(), key=lambda item: item[1]):
            normalized = normalize_token_text(token)
            if normalized and 0 <= token_id < vocab_size and token_id not in special_ids:
                result.setdefault(normalized, int(token_id))
        return result

    auxiliary_vocab = normalized_vocab(auxiliary_tokenizer, auxiliary_vocab_size)
    master_vocab = normalized_vocab(master_tokenizer, master_vocab_size)
    all_shared = sorted(auxiliary_vocab.keys() & master_vocab.keys())
    if isinstance(num_anchors, str) and num_anchors.casefold() == "auto":
        selected_count = len(all_shared)
    else:
        try:
            selected_count = int(num_anchors)
        except (TypeError, ValueError) as error:
            raise ValueError("num-anchors must be a positive integer or 'auto'") from error
        if selected_count < 1:
            raise ValueError("num-anchors must be a positive integer or 'auto'")
        selected_count = min(selected_count, len(all_shared))
    shared = all_shared[:selected_count]
    if len(shared) < min_anchors:
        raise ValueError(
            f"Only {len(shared)} shared anchors were found; need {min_anchors}."
        )
    return RelativeAnchorSelection(
        auxiliary_anchor_ids=torch.tensor(
            [auxiliary_vocab[token] for token in shared], dtype=torch.long
        ),
        master_anchor_ids=torch.tensor(
            [master_vocab[token] for token in shared], dtype=torch.long
        ),
        total_anchors=len(all_shared),
    )


def relative_anchor_profiles(
    embeddings: torch.Tensor,
    anchor_embeddings: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    """Represent tokens by normalized similarities to in-model anchors."""
    if temperature <= 0:
        raise ValueError("anchor temperature must be positive")
    if embeddings.ndim != 2 or anchor_embeddings.ndim != 2:
        raise ValueError("embeddings and anchors must be two-dimensional")
    if embeddings.shape[1] != anchor_embeddings.shape[1]:
        raise ValueError("tokens and anchors must belong to the same embedding space")
    normalized_embeddings = F.normalize(embeddings.float(), dim=-1)
    normalized_anchors = F.normalize(anchor_embeddings.float(), dim=-1)
    scores = normalized_embeddings @ normalized_anchors.T
    profiles = torch.softmax(scores / temperature, dim=-1)
    return F.normalize(profiles, dim=-1)


def fabricate_probabilities(
    input_ids: torch.Tensor,
    vocabulary_size: int,
    special_ids: Sequence[int],
    *,
    support_size: int,
    temperature: float,
    seed: int,
) -> torch.Tensor:
    """Create random skewed distributions whose argmax is the observed token.

    A random vocabulary subset receives softmax-normalized random logits. The
    largest resulting probability is then swapped onto the observed token.
    These are deliberately synthetic distributions, not model logits.
    """
    if vocabulary_size < 2:
        raise ValueError("vocabulary size must be at least two")
    if support_size < 1 or support_size > vocabulary_size:
        raise ValueError("distribution support must be within the vocabulary size")
    if temperature <= 0:
        raise ValueError("distribution temperature must be positive")
    device = input_ids.device
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    selection_scores = torch.rand(
        (input_ids.numel(), vocabulary_size),
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    valid_special_ids = [
        int(token_id) for token_id in special_ids if 0 <= token_id < vocabulary_size
    ]
    if valid_special_ids:
        selection_scores[:, valid_special_ids] = -torch.inf
    available_tokens = vocabulary_size - len(set(valid_special_ids))
    if support_size > available_tokens:
        raise ValueError(
            "distribution support exceeds the number of non-special vocabulary tokens"
        )
    subset_ids = torch.topk(selection_scores, k=support_size, dim=-1).indices
    subset_logits = torch.randn(
        (input_ids.numel(), support_size),
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    subset_probabilities = torch.softmax(subset_logits / temperature, dim=-1)
    probabilities = torch.zeros_like(selection_scores)
    probabilities.scatter_(1, subset_ids, subset_probabilities)

    token_ids = input_ids.flatten().to(device)
    row_ids = torch.arange(token_ids.numel(), device=device)
    maximum_ids = probabilities.argmax(dim=-1)
    maximum_probabilities = probabilities[row_ids, maximum_ids].clone()
    observed_probabilities = probabilities[row_ids, token_ids].clone()
    probabilities[row_ids, token_ids] = maximum_probabilities
    probabilities[row_ids, maximum_ids] = observed_probabilities
    return probabilities


def build_overlap_matrix(
    auxiliary_offsets: Sequence[Span],
    master_offsets: Sequence[Span],
    device: torch.device,
) -> torch.Tensor:
    """Build the CTCA auxiliary-to-master character-overlap matrix."""
    overlap = torch.zeros(
        len(auxiliary_offsets), len(master_offsets), dtype=torch.float32, device=device
    )
    for auxiliary_index, (auxiliary_start, auxiliary_end) in enumerate(auxiliary_offsets):
        auxiliary_length = auxiliary_end - auxiliary_start
        if auxiliary_length <= 0:
            continue
        for master_index, (master_start, master_end) in enumerate(master_offsets):
            shared_length = max(
                0.0,
                min(auxiliary_end, master_end) - max(auxiliary_start, master_start),
            )
            overlap[auxiliary_index, master_index] = shared_length / auxiliary_length
    return overlap


def normalize_probability_rows(probabilities: torch.Tensor) -> torch.Tensor:
    mass = probabilities.sum(dim=-1, keepdim=True)
    return torch.where(mass > 0, probabilities / mass.clamp_min(1e-12), probabilities)


def spatially_align_probabilities(
    auxiliary_probabilities: torch.Tensor,
    overlap_matrix: torch.Tensor,
) -> torch.Tensor:
    """Move auxiliary distributions onto the master token positions."""
    warped = overlap_matrix.T @ auxiliary_probabilities
    return normalize_probability_rows(warped)


def project_auxiliary_vocabulary(
    spatial_auxiliary_probabilities: torch.Tensor,
    auxiliary_embeddings: torch.Tensor,
    auxiliary_anchor_embeddings: torch.Tensor,
    master_embeddings: torch.Tensor,
    master_anchor_embeddings: torch.Tensor,
    *,
    source_top_k: int,
    top_k: int,
    anchor_temperature: float,
    temperature: float,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project active probability mass using relative-anchor similarities."""
    if source_top_k < 1 or temperature <= 0 or chunk_size < 1:
        raise ValueError("projection temperature and chunk size must be positive")
    if auxiliary_anchor_embeddings.shape[0] != master_anchor_embeddings.shape[0]:
        raise ValueError("auxiliary and master anchor counts must match")
    retained_count = min(source_top_k, spatial_auxiliary_probabilities.shape[1])
    _, retained_ids_by_position = torch.topk(
        spatial_auxiliary_probabilities, k=retained_count, dim=-1
    )
    active_ids = torch.unique(retained_ids_by_position.flatten())
    retained_mass = spatial_auxiliary_probabilities.index_select(1, active_ids).sum(dim=-1)
    if active_ids.numel() == 0:
        return (
            spatial_auxiliary_probabilities.new_zeros(
                spatial_auxiliary_probabilities.shape[0], master_embeddings.shape[0]
            ),
            retained_mass,
        )

    auxiliary_profiles = relative_anchor_profiles(
        auxiliary_embeddings.index_select(0, active_ids),
        auxiliary_anchor_embeddings,
        temperature=anchor_temperature,
    )
    neighbor_count = min(top_k, master_embeddings.shape[0])
    neighbor_values = torch.full(
        (active_ids.numel(), neighbor_count),
        -torch.inf,
        dtype=torch.float32,
        device=auxiliary_embeddings.device,
    )
    neighbor_ids = torch.zeros(
        (active_ids.numel(), neighbor_count),
        dtype=torch.long,
        device=auxiliary_embeddings.device,
    )
    master_anchor_embeddings = master_anchor_embeddings.to(auxiliary_embeddings.device)
    for start in range(0, master_embeddings.shape[0], chunk_size):
        end = min(start + chunk_size, master_embeddings.shape[0])
        master_profiles = relative_anchor_profiles(
            master_embeddings[start:end].to(auxiliary_embeddings.device),
            master_anchor_embeddings,
            temperature=anchor_temperature,
        )
        similarities = auxiliary_profiles @ master_profiles.T
        local_count = min(neighbor_count, similarities.shape[1])
        local_values, local_ids = torch.topk(similarities, k=local_count, dim=-1)
        local_ids = local_ids + start
        candidate_values = torch.cat((neighbor_values, local_values), dim=-1)
        candidate_ids = torch.cat((neighbor_ids, local_ids), dim=-1)
        neighbor_values, order = torch.topk(
            candidate_values, k=neighbor_count, dim=-1
        )
        neighbor_ids = torch.gather(candidate_ids, 1, order)

    neighbor_weights = torch.softmax(neighbor_values / temperature, dim=-1)

    active_probabilities = spatial_auxiliary_probabilities.index_select(1, active_ids)
    contributions = active_probabilities.unsqueeze(-1) * neighbor_weights.unsqueeze(0)
    projected = spatial_auxiliary_probabilities.new_zeros(
        spatial_auxiliary_probabilities.shape[0], master_embeddings.shape[0]
    )
    projected.scatter_add_(
        1,
        neighbor_ids.flatten().unsqueeze(0).expand(projected.shape[0], -1),
        contributions.flatten(1),
    )
    return normalize_probability_rows(projected), retained_mass


def top_probability_data(
    probabilities: torch.Tensor,
    tokenizer,
    top_k: int,
) -> tuple[list[str], list[float]]:
    """Return non-zero top-k labels and probabilities in ascending bar order."""
    count = min(top_k, probabilities.numel())
    values, indices = torch.topk(probabilities, k=count)
    pairs = [
        (token_label(tokenizer, token_id), float(probability))
        for probability, token_id in zip(values.tolist(), indices.tolist())
        if probability > 0
    ]
    pairs.reverse()
    return [label for label, _ in pairs], [value for _, value in pairs]


def distribution_bar(
    probabilities: torch.Tensor,
    tokenizer,
    top_k: int,
    *,
    name: str,
    visible: bool,
) -> go.Bar:
    labels, values = top_probability_data(probabilities.detach().cpu(), tokenizer, top_k)
    return go.Bar(
        x=values,
        y=labels,
        orientation="h",
        name=name,
        visible=visible,
        hovertemplate="%{y}<br>p=%{x:.6f}<extra></extra>",
    )


def make_alignment_figure(
    overlap_matrix: torch.Tensor,
    auxiliary_labels: Sequence[str],
    master_labels: Sequence[str],
) -> go.Figure:
    figure = go.Figure(
        go.Heatmap(
            z=overlap_matrix.detach().cpu().tolist(),
            x=[f"M{i}: {label}" for i, label in enumerate(master_labels)],
            y=[f"A{i}: {label}" for i, label in enumerate(auxiliary_labels)],
            colorscale="Blues",
            zmin=0,
            zmax=1,
            colorbar={"title": "Aux span<br>fraction"},
            hovertemplate="%{y}<br>%{x}<br>overlap=%{z:.3f}<extra></extra>",
        )
    )
    figure.update_layout(
        title="CTCA spatial tokenizer alignment: auxiliary → master",
        xaxis_title="Master tokens",
        yaxis_title="Auxiliary tokens",
        height=max(500, 34 * len(auxiliary_labels)),
    )
    return figure


def make_native_distribution_figure(
    master_probabilities: torch.Tensor,
    auxiliary_probabilities: torch.Tensor,
    master_tokenizer,
    auxiliary_tokenizer,
    master_labels: Sequence[str],
    auxiliary_labels: Sequence[str],
    top_k: int,
) -> go.Figure:
    """Create a dropdown for every original master and auxiliary token."""
    figure = go.Figure()
    selections: list[tuple[str, str]] = []
    for model_name, probabilities, tokenizer, labels in (
        ("Master", master_probabilities, master_tokenizer, master_labels),
        ("Auxiliary", auxiliary_probabilities, auxiliary_tokenizer, auxiliary_labels),
    ):
        for position, label in enumerate(labels):
            selection = f"{model_name} token {position}: {label}"
            figure.add_trace(
                distribution_bar(
                    probabilities[position],
                    tokenizer,
                    top_k,
                    name=model_name,
                    visible=not selections,
                )
            )
            selections.append((selection, model_name))

    buttons = []
    for selected_index, (selection, model_name) in enumerate(selections):
        visible = [index == selected_index for index in range(len(selections))]
        buttons.append(
            {
                "label": selection,
                "method": "update",
                "args": [
                    {"visible": visible},
                    {"title": f"Fabricated native distribution — {selection}"},
                ],
            }
        )
    figure.update_layout(
        title=f"Fabricated native distribution — {selections[0][0]}",
        xaxis_title="Probability",
        yaxis_title="Vocabulary token",
        height=550,
        margin={"t": 150},
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "x": 0,
                "y": 1.18,
                "xanchor": "left",
                "yanchor": "top",
            }
        ],
    )
    return figure


def make_comparison_figure(
    master_probabilities: torch.Tensor,
    spatial_auxiliary_probabilities: torch.Tensor,
    projected_probabilities: torch.Tensor,
    master_tokenizer,
    auxiliary_tokenizer,
    master_labels: Sequence[str],
    auxiliary_labels: Sequence[str],
    overlap_matrix: torch.Tensor,
    top_k: int,
) -> go.Figure:
    """Compare master, spatially aligned auxiliary, and projected distributions."""
    figure = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            "Master distribution",
            "Aux distribution at master position",
            "Aux → master distribution",
        ),
        horizontal_spacing=0.11,
    )
    source_descriptions = []
    for master_position, master_label in enumerate(master_labels):
        visible = master_position == 0
        figure.add_trace(
            distribution_bar(
                master_probabilities[master_position],
                master_tokenizer,
                top_k,
                name="Master",
                visible=visible,
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            distribution_bar(
                spatial_auxiliary_probabilities[master_position],
                auxiliary_tokenizer,
                top_k,
                name="Spatial aux",
                visible=visible,
            ),
            row=1,
            col=2,
        )
        figure.add_trace(
            distribution_bar(
                projected_probabilities[master_position],
                master_tokenizer,
                top_k,
                name="Aux → master",
                visible=visible,
            ),
            row=1,
            col=3,
        )
        sources = [
            f"A{index}: {auxiliary_labels[index]}"
            for index in torch.nonzero(overlap_matrix[:, master_position] > 0).flatten().tolist()
        ]
        source_descriptions.append(
            f"M{master_position}: {master_label} | overlaps " + ", ".join(sources)
        )

    trace_count = 3 * len(master_labels)
    buttons = []
    for master_position, description in enumerate(source_descriptions):
        visible = [False] * trace_count
        visible[3 * master_position : 3 * master_position + 3] = [True, True, True]
        buttons.append(
            {
                "label": description,
                "method": "update",
                "args": [
                    {"visible": visible},
                    {"title": f"CTCA probability comparison — {description}"},
                ],
            }
        )

    figure.update_layout(
        title=f"CTCA probability comparison — {source_descriptions[0]}",
        height=620,
        margin={"t": 170},
        showlegend=False,
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "x": 0,
                "y": 1.22,
                "xanchor": "left",
                "yanchor": "top",
            }
        ],
    )
    figure.update_xaxes(title_text="Probability")
    return figure


def write_dashboard(
    output_path: Path,
    sentence: str,
    master_name: str,
    auxiliary_name: str,
    alignment_figure: go.Figure,
    native_figure: go.Figure,
    comparison_figure: go.Figure,
) -> Path:
    """Write all Plotly figures into one portable HTML dashboard."""
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sections = [
        pio.to_html(alignment_figure, full_html=False, include_plotlyjs="cdn"),
        pio.to_html(native_figure, full_html=False, include_plotlyjs=False),
        pio.to_html(comparison_figure, full_html=False, include_plotlyjs=False),
    ]
    document = f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>CTCA BERT/GPT-2 demo</title></head>
<body style="font-family: sans-serif; margin: 2rem;">
  <h1>CTCA BERT/GPT-2 visualization</h1>
  <p><strong>Sentence:</strong> {html.escape(sentence)}</p>
  <p><strong>Master:</strong> {html.escape(master_name)} &nbsp;
     <strong>Auxiliary:</strong> {html.escape(auxiliary_name)}</p>
  <p>Probabilities are synthetic; embeddings and tokenizer alignments are real.</p>
  {''.join(sections)}
</body>
</html>
"""
    output_path.write_text(document, encoding="utf-8")
    return output_path


def print_tokenization(name: str, tokenized: TokenizedSentence) -> None:
    print(f"\n{name} tokenization ({len(tokenized.labels)} tokens):")
    for index, (label, offset) in enumerate(zip(tokenized.labels, tokenized.offsets)):
        print(f"  {index:>2}: {label:<28} chars={offset}")
    print(f"  contextual embedding shape: {tuple(tokenized.contextual_embeddings.shape)}")
    print(f"  mean sentence embedding shape: {tuple(tokenized.sentence_embedding.shape)}")


def main() -> None:
    args = parse_args()
    if args.min_anchors < 1:
        raise ValueError("min-anchors must be positive")
    if args.projection_top_k < 1 or args.plot_top_k < 1:
        raise ValueError("projection-top-k and plot-top-k must be positive")
    if args.projection_source_top_k < 1:
        raise ValueError("projection-source-top-k must be positive")
    if args.distribution_support < 1:
        raise ValueError("distribution-support must be positive")
    if args.distribution_temperature <= 0:
        raise ValueError("distribution-temperature must be positive")
    if args.anchor_temperature <= 0 or args.projection_temperature <= 0:
        raise ValueError("anchor-temperature and projection-temperature must be positive")
    if args.projection_chunk_size < 1:
        raise ValueError("projection-chunk-size must be positive")
    device = resolve_device(args.device)
    auxiliary_key = "gpt2" if args.master == "bert" else "bert"
    master_model_id = MODEL_IDS[args.master]
    auxiliary_model_id = MODEL_IDS[auxiliary_key]

    print(f"Using device: {device}")
    print("Loading models and tokenizers...")
    master_tokenizer = AutoTokenizer.from_pretrained(
        master_model_id, local_files_only=args.local_files_only
    )
    auxiliary_tokenizer = AutoTokenizer.from_pretrained(
        auxiliary_model_id, local_files_only=args.local_files_only
    )
    master_model = AutoModel.from_pretrained(
        master_model_id, local_files_only=args.local_files_only
    ).to(device)
    auxiliary_model = AutoModel.from_pretrained(
        auxiliary_model_id, local_files_only=args.local_files_only
    ).to(device)

    master_sentence = tokenize_and_embed(
        args.sentence, master_tokenizer, master_model, device
    )
    auxiliary_sentence = tokenize_and_embed(
        args.sentence, auxiliary_tokenizer, auxiliary_model, device
    )
    print_tokenization(f"Master ({master_model_id})", master_sentence)
    print_tokenization(f"Auxiliary ({auxiliary_model_id})", auxiliary_sentence)

    master_embeddings = master_model.get_input_embeddings().weight.detach()
    auxiliary_embeddings = auxiliary_model.get_input_embeddings().weight.detach()
    anchors = collect_shared_anchors(
        auxiliary_tokenizer,
        master_tokenizer,
        auxiliary_embeddings.shape[0],
        master_embeddings.shape[0],
        args.num_anchors,
        args.min_anchors,
    )
    auxiliary_anchor_embeddings = auxiliary_embeddings.index_select(
        0, anchors.auxiliary_anchor_ids.to(auxiliary_embeddings.device)
    )
    master_anchor_embeddings = master_embeddings.index_select(
        0, anchors.master_anchor_ids.to(master_embeddings.device)
    )
    print(
        "\nRelative-anchor alignment selected "
        f"{len(anchors.auxiliary_anchor_ids)} of {anchors.total_anchors} shared anchors."
    )

    auxiliary_sentence_profile = relative_anchor_profiles(
        auxiliary_sentence.sentence_embedding.unsqueeze(0),
        auxiliary_anchor_embeddings,
        temperature=args.anchor_temperature,
    )
    master_sentence_profile = relative_anchor_profiles(
        master_sentence.sentence_embedding.unsqueeze(0),
        master_anchor_embeddings,
        temperature=args.anchor_temperature,
    )
    sentence_similarity = F.cosine_similarity(
        master_sentence_profile,
        auxiliary_sentence_profile,
        dim=-1,
    )
    print(
        "Mean-sentence relative-profile cosine similarity: "
        f"{float(sentence_similarity.item()):.4f}"
    )

    master_probabilities = fabricate_probabilities(
        master_sentence.input_ids,
        master_embeddings.shape[0],
        master_tokenizer.all_special_ids,
        support_size=args.distribution_support,
        temperature=args.distribution_temperature,
        seed=args.random_seed,
    )
    auxiliary_probabilities = fabricate_probabilities(
        auxiliary_sentence.input_ids,
        auxiliary_embeddings.shape[0],
        auxiliary_tokenizer.all_special_ids,
        support_size=args.distribution_support,
        temperature=args.distribution_temperature,
        seed=args.random_seed + 1,
    )
    if not torch.equal(master_probabilities.argmax(dim=-1), master_sentence.input_ids):
        raise RuntimeError("fabricated master distribution argmax check failed")
    if not torch.equal(
        auxiliary_probabilities.argmax(dim=-1), auxiliary_sentence.input_ids
    ):
        raise RuntimeError("fabricated auxiliary distribution argmax check failed")
    print(
        "Fabricated distribution shapes: "
        f"master={tuple(master_probabilities.shape)}, "
        f"auxiliary={tuple(auxiliary_probabilities.shape)}"
    )
    print(
        f"Random support size: {args.distribution_support}; "
        f"distribution temperature: {args.distribution_temperature}; "
        "argmax reproduces every native input token."
    )

    overlap_matrix = build_overlap_matrix(
        auxiliary_sentence.offsets, master_sentence.offsets, device
    )
    spatial_auxiliary_probabilities = spatially_align_probabilities(
        auxiliary_probabilities, overlap_matrix
    )
    projected_probabilities, retained_mass = project_auxiliary_vocabulary(
        spatial_auxiliary_probabilities,
        auxiliary_embeddings,
        auxiliary_anchor_embeddings,
        master_embeddings,
        master_anchor_embeddings,
        source_top_k=args.projection_source_top_k,
        top_k=args.projection_top_k,
        anchor_temperature=args.anchor_temperature,
        temperature=args.projection_temperature,
        chunk_size=args.projection_chunk_size,
    )
    print(
        "Aligned distribution shapes: "
        f"spatial auxiliary={tuple(spatial_auxiliary_probabilities.shape)}, "
        f"auxiliary-to-master={tuple(projected_probabilities.shape)}"
    )
    print(
        "Relative-anchor projection retained auxiliary source mass per master "
        f"position: min={float(retained_mass.min()):.4f}, "
        f"mean={float(retained_mass.mean()):.4f}."
    )

    alignment_figure = make_alignment_figure(
        overlap_matrix, auxiliary_sentence.labels, master_sentence.labels
    )
    native_figure = make_native_distribution_figure(
        master_probabilities,
        auxiliary_probabilities,
        master_tokenizer,
        auxiliary_tokenizer,
        master_sentence.labels,
        auxiliary_sentence.labels,
        args.plot_top_k,
    )
    comparison_figure = make_comparison_figure(
        master_probabilities,
        spatial_auxiliary_probabilities,
        projected_probabilities,
        master_tokenizer,
        auxiliary_tokenizer,
        master_sentence.labels,
        auxiliary_sentence.labels,
        overlap_matrix,
        args.plot_top_k,
    )
    output_path = write_dashboard(
        args.output,
        args.sentence,
        master_model_id,
        auxiliary_model_id,
        alignment_figure,
        native_figure,
        comparison_figure,
    )
    print(f"\nWrote interactive Plotly dashboard to: {output_path}")
    if args.show:
        comparison_figure.show()


if __name__ == "__main__":
    main()

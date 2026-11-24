"""Evaluate distribution-based position correspondence on one toy decoding step.

    source ~/.zshrc 2>/dev/null || source /apps/local/conda_init.sh
    conda activate dllm
    python /home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/sandbox/ctca_position_correspondence.py
"""

import argparse
from itertools import combinations
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from dllm.pipelines.tse.ctca.distribution_projection import build_relative_distribution_profiles
from dllm.pipelines.tse.fusion import fuse_probabilities


STUDY = "ctca-toy-position-correspondence-20261123"
GOLD = ("Ada", "packed", "red", "apples", "green", "pears", "blue", "plums")
ALTERNATIVES = ("Mia", "bought", "ripe", "mangoes", "yellow", "bananas", "purple", "grapes")
TOKENS = GOLD + ALTERNATIVES + ("filler",)
POSITIONS = tuple(range(len(GOLD)))
ANCHOR_TEMPERATURE = 0.1
MIN_SIMILARITY = 0.5
WINDOW = 4
ALPHA = 0.5


def make_distributions(gold_mass):
    """Put each slot's mass on its gold token and its own alternative."""
    masses = torch.tensor(gold_mass)
    positions = torch.arange(len(GOLD))
    rows = torch.zeros(len(GOLD), len(TOKENS))
    rows[positions, positions] = masses
    rows[positions, positions + len(GOLD)] = 1 - masses
    return rows


def anchor_profiles(token_order, basis):
    """Give each concept a shared anchor in a model's own embedding space."""
    vocabulary_size = len(TOKENS)
    anchors = basis[:vocabulary_size]
    embeddings = torch.cat([anchors[token_order], anchors])
    return build_relative_distribution_profiles(
        embeddings,
        torch.arange(vocabulary_size, 2 * vocabulary_size),
        temperature=ANCHOR_TEMPERATURE,
        chunk_size=vocabulary_size,
    )[:vocabulary_size].float()


def candidate_pairs(scores, radius=None):
    """Allow similar pairs and optionally restrict generation-relative distance."""
    allowed = scores > MIN_SIMILARITY
    if radius is not None:
        master_pos = torch.arange(scores.shape[0])[:, None]
        auxiliary_pos = torch.arange(scores.shape[1])
        allowed &= (master_pos - auxiliary_pos).abs() <= radius
    return allowed


def row_argmax(scores, allowed):
    """Pick each master's highest-scoring allowed auxiliary position."""
    pairs = []
    for master_pos, (row, mask) in enumerate(zip(scores, allowed)):
        if mask.any():
            auxiliary_pos = int(row.masked_fill(~mask, -torch.inf).argmax())
            pairs.append((master_pos, auxiliary_pos))
    return pairs


def monotonic_pairs(scores, allowed):
    """Maximize total similarity above threshold with skips on either side."""
    n_master, n_auxiliary = scores.shape
    rewards = (scores - MIN_SIMILARITY).masked_fill(~allowed, -torch.inf).tolist()
    values = [[0.0] * (n_auxiliary + 1) for _ in range(n_master + 1)]
    moves = [[None] * (n_auxiliary + 1) for _ in range(n_master + 1)]
    for i in range(1, n_master + 1):
        for j in range(1, n_auxiliary + 1):
            choices = (
                (values[i - 1][j], "skip_master"),
                (values[i][j - 1], "skip_auxiliary"),
                (values[i - 1][j - 1] + rewards[i - 1][j - 1], "match"),
            )
            values[i][j], moves[i][j] = max(choices, key=lambda choice: choice[0])

    pairs = []
    i, j = n_master, n_auxiliary
    while i and j:
        move = moves[i][j]
        if move == "match":
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif move == "skip_master":
            i -= 1
        else:
            j -= 1
    return pairs[::-1]


def measure(pairs, gold_pairs, master, auxiliary, alpha):
    """Score position matches and token decisions against separate gold labels."""
    predictions = master.clone()
    if pairs:
        master_pos, auxiliary_pos = zip(*pairs)
        master_pos = list(master_pos)
        predictions[master_pos] = fuse_probabilities(
            master[master_pos], auxiliary[list(auxiliary_pos)], alpha
        )
    token_ids = predictions.argmax(dim=-1)
    gold_ids = torch.arange(len(GOLD))
    correct = token_ids == gold_ids
    master_correct = master.argmax(dim=-1) == gold_ids
    true_positives = len(set(pairs) & set(gold_pairs))
    return {
        "pairs": pairs,
        "true_positives": true_positives,
        "precision": true_positives / len(pairs) if pairs else None,
        "recall": true_positives / len(gold_pairs),
        "crossings": sum(left > right for (_, left), (_, right) in combinations(pairs, 2)),
        "collisions": len(pairs) - len({j for _, j in pairs}),
        "correct_tokens": int(correct.sum()),
        "gains": int((correct & ~master_correct).sum()),
        "losses": int((~correct & master_correct).sum()),
        "mean_gold_nll": float(-predictions[gold_ids, gold_ids].log().mean()),
        "predicted_tokens": [TOKENS[index] for index in token_ids.tolist()],
    }


def run_toy():
    master = make_distributions([0.64, 0.42, 0.68, 0.64, 0.42, 0.64, 0.68, 0.42])
    peer = make_distributions([0.88, 0.88, 0.45, 0.88, 0.88, 0.88, 0.45, 0.88])
    overconfident_peer = make_distributions([0.10, 0.88, 0.10, 0.10, 0.88, 0.10, 0.45, 0.88])

    generator = torch.Generator().manual_seed(0)
    vocab_size = len(TOKENS)
    auxiliary_order = torch.randperm(vocab_size, generator=generator)
    auxiliary_basis = torch.linalg.qr(
        torch.randn(vocab_size + 2, vocab_size + 2, generator=generator)
    ).Q
    master_profiles = anchor_profiles(torch.arange(vocab_size), torch.eye(vocab_size))
    auxiliary_profiles = anchor_profiles(auxiliary_order, auxiliary_basis)
    master_relative = F.normalize(master @ master_profiles, dim=-1)

    results = {}
    cases = (
        ("clean", POSITIONS, peer),
        ("local_crossing", (0, "filler", 1, 2, "decoy", 3, 4, 5, 7), peer),
        ("drift_plus_five", ("filler",) * 5 + POSITIONS, peer),
        ("overconfident_peer", POSITIONS, overconfident_peer),
    )
    for name, layout, auxiliary_rows in cases:
        rows = dict(enumerate(auxiliary_rows))
        rows["filler"] = torch.zeros(vocab_size)
        rows["filler"][-1] = 1
        rows["decoy"] = master[1]
        auxiliary = torch.stack([rows[label] for label in layout])[:, auxiliary_order]
        auxiliary_relative = F.normalize(auxiliary @ auxiliary_profiles, dim=-1)
        scores = master_relative @ auxiliary_relative.T
        gold_pairs = sorted((label, j) for j, label in enumerate(layout) if isinstance(label, int))

        all_candidates = candidate_pairs(scores)
        band_candidates = candidate_pairs(scores, WINDOW)
        methods = {
            "master_only": [],
            "same_index": [(i, i) for i in range(min(scores.shape))],
            "row_argmax": row_argmax(scores, all_candidates),
            "row_argmax_window4": row_argmax(scores, band_candidates),
            "monotonic": monotonic_pairs(scores, all_candidates),
            "monotonic_window4": monotonic_pairs(scores, band_candidates),
            "oracle_fusion": gold_pairs,
            "oracle_peer_with_master_fallback": gold_pairs,
        }
        auxiliary_in_master_vocab = auxiliary[:, auxiliary_order.argsort()]
        results[name] = {
            "auxiliary_layout": layout,
            "gold_pairs": gold_pairs,
            "gold_pairs_in_window": sum(abs(i - j) <= WINDOW for i, j in gold_pairs),
            "cosine_similarity": scores.tolist(),
            "methods": {
                method: measure(
                    pairs, gold_pairs, master, auxiliary_in_master_vocab,
                    0.0 if method == "oracle_peer_with_master_fallback" else ALPHA,
                )
                for method, pairs in methods.items()
            },
        }
    return {
        "study_id": STUDY,
        "gold_tokens": GOLD,
        "master_vocab": TOKENS,
        "auxiliary_vocab": [TOKENS[index] for index in auxiliary_order.tolist()],
        "anchor_temperature": ANCHOR_TEMPERATURE,
        "minimum_cosine_exclusive": MIN_SIMILARITY,
        "window_radius": WINDOW,
        "master_fusion_weight": ALPHA,
        "cases": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / ".logs" / STUDY / "report.json")
    args = parser.parse_args()
    report = run_toy()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{'Case':22} {'Method':32} {'Matches':>8} {'Correct':>8} {'Gain':>5} {'Loss':>5}")
    for case, result in report["cases"].items():
        for method, stats in result["methods"].items():
            matches = f"{stats['true_positives']}/{len(result['gold_pairs'])}"
            correct = f"{stats['correct_tokens']}/{len(GOLD)}"
            print(f"{case:22} {method:32} {matches:>8} {correct:>8} {stats['gains']:5} {stats['losses']:5}")
    print(output)


if __name__ == "__main__":
    main()

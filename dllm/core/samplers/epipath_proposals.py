"""
Candidate proposal generation for the EpiPath sampler.

Run tests from the repository root with:
  source ~/.zshrc && conda activate ~/miniconda3/envs/dllm
  pytest /home/sarthak.malla/dllm-epiplexity/scripts/tests/test_epipath_proposals.py -v
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class CandidateReveal:
    """A candidate set of positions and token IDs to commit."""

    candidate_id: str
    proposal_type: str
    positions: list[int]
    token_ids: list[int]
    proposal_sources: list[str]
    metrics: dict[str, float] = field(default_factory=dict)
    score: float | None = None
    selected: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "candidate_id": self.candidate_id,
            "proposal_type": self.proposal_type,
            "positions": list(self.positions),
            "token_ids": list(self.token_ids),
            "proposal_sources": list(self.proposal_sources),
            "metrics": dict(self.metrics),
            "score": self.score,
            "selected": self.selected,
            "metadata": dict(self.metadata),
        }


class EpiProposalGenerator:
    """Generate deterministic reveal-set proposals for one batch row."""

    PRESETS = {
        "minimal": ["greedy", "spaced_0", "spaced_1"],
        "dynamic_light": [
            "greedy",
            "spaced_0",
            "spaced_1",
            "coverage_anti_cluster",
            "frontier_conf_q60",
            "high_margin",
        ],
    }

    def resolve_proposals(
        self,
        proposal_preset: str = "dynamic_light",
        proposals: str | list[str] | None = None,
    ) -> list[str]:
        """Resolve a preset or comma-separated proposal override."""
        if proposals is not None:
            if isinstance(proposals, str):
                return [item.strip() for item in proposals.split(",") if item.strip()]
            return list(proposals)
        if proposal_preset not in self.PRESETS:
            raise ValueError(f"Unknown EpiPath proposal preset: {proposal_preset}")
        return list(self.PRESETS[proposal_preset])

    def generate(
        self,
        current_mask: torch.Tensor,
        revealed_mask: torch.Tensor,
        base_metrics: dict[str, torch.Tensor],
        token_ids: torch.Tensor,
        num_transfer: int,
        proposal_preset: str = "dynamic_light",
        proposals: str | list[str] | None = None,
    ) -> list[CandidateReveal]:
        """Generate deduplicated candidates for a single batch row."""
        proposal_names = self.resolve_proposals(proposal_preset, proposals)
        valid_positions = torch.where(current_mask)[0].tolist()
        if not valid_positions:
            return []
        k = max(0, min(int(num_transfer), len(valid_positions)))
        if k == 0:
            return []

        raw_candidates: list[tuple[str, list[int], dict[str, Any]]] = []
        for proposal_name in proposal_names:
            positions, metadata = self._generate_one(
                proposal_name,
                valid_positions,
                revealed_mask,
                base_metrics,
                k,
                sequence_length=current_mask.numel(),
            )
            positions = self._normalize_size(
                positions=positions,
                valid_positions=valid_positions,
                base_metrics=base_metrics,
                k=k,
                metadata=metadata,
            )
            raw_candidates.append((proposal_name, positions, metadata))

        by_key: dict[tuple[int, ...], CandidateReveal] = {}
        for proposal_name, positions, metadata in raw_candidates:
            key = tuple(sorted(positions))
            candidate = by_key.get(key)
            if candidate is not None:
                if proposal_name not in candidate.proposal_sources:
                    candidate.proposal_sources.append(proposal_name)
                candidate.metadata.setdefault("proposal_sources", []).append(
                    proposal_name
                )
                continue

            candidate_token_ids = [int(token_ids[pos].item()) for pos in positions]
            by_key[key] = CandidateReveal(
                candidate_id="",
                proposal_type=proposal_name,
                positions=list(positions),
                token_ids=candidate_token_ids,
                proposal_sources=[proposal_name],
                metadata={
                    **metadata,
                    "proposal_sources": [proposal_name],
                },
            )

        candidates = list(by_key.values())
        for index, candidate in enumerate(candidates):
            candidate.candidate_id = f"cand_{index:03d}_{candidate.proposal_type}"
        return candidates

    def _generate_one(
        self,
        proposal_name: str,
        valid_positions: list[int],
        revealed_mask: torch.Tensor,
        base_metrics: dict[str, torch.Tensor],
        k: int,
        sequence_length: int,
    ) -> tuple[list[int], dict[str, Any]]:
        if proposal_name == "greedy":
            return (
                self._rank_top(valid_positions, base_metrics["top1_prob"], k, True),
                {},
            )
        if proposal_name in ("spaced_0", "spaced_1"):
            rho = 0.0 if proposal_name == "spaced_0" else 0.5
            return self._spaced(valid_positions, k, rho), {"rho": rho}
        if proposal_name == "coverage_anti_cluster":
            return (
                self._coverage_anti_cluster(
                    valid_positions,
                    revealed_mask,
                    base_metrics,
                    k,
                    sequence_length,
                ),
                {},
            )
        if proposal_name == "frontier_conf_q60":
            return (
                self._frontier(valid_positions, base_metrics, k, quantile=0.60),
                {"quantile": 0.60},
            )
        if proposal_name == "high_margin":
            return (
                self._rank_top(valid_positions, base_metrics["margin"], k, True),
                {},
            )
        raise ValueError(f"Unknown EpiPath proposal: {proposal_name}")

    def _normalize_size(
        self,
        positions: list[int],
        valid_positions: list[int],
        base_metrics: dict[str, torch.Tensor],
        k: int,
        metadata: dict[str, Any],
    ) -> list[int]:
        """Deduplicate, truncate, and confidence-fill a proposal to size k."""
        seen = set()
        normalized: list[int] = []
        for position in positions:
            if position in seen or position not in valid_positions:
                continue
            normalized.append(position)
            seen.add(position)
            if len(normalized) == k:
                break

        if len(normalized) < k:
            fill_positions = [
                pos
                for pos in self._rank_top(
                    valid_positions,
                    base_metrics["top1_prob"],
                    k=len(valid_positions),
                    descending=True,
                )
                if pos not in seen
            ][: k - len(normalized)]
            normalized.extend(fill_positions)
            metadata["filled_positions"] = fill_positions
        return normalized[:k]

    def _rank_top(
        self,
        positions: list[int],
        values: torch.Tensor,
        k: int,
        descending: bool,
    ) -> list[int]:
        ranked = sorted(
            positions,
            key=lambda pos: (float(values[pos].item()), -pos),
            reverse=descending,
        )
        if not descending:
            ranked = sorted(positions, key=lambda pos: (float(values[pos].item()), pos))
        return ranked[:k]

    def _spaced(self, valid_positions: list[int], k: int, rho: float) -> list[int]:
        n = len(valid_positions)
        step_float = n / k
        selected = []
        for index in range(k):
            raw = int(index * step_float + rho * step_float)
            selected.append(valid_positions[min(n - 1, raw)])
        return selected

    def _coverage_anti_cluster(
        self,
        valid_positions: list[int],
        revealed_mask: torch.Tensor,
        base_metrics: dict[str, torch.Tensor],
        k: int,
        sequence_length: int,
    ) -> list[int]:
        selected: list[int] = []
        revealed_positions = torch.where(revealed_mask)[0].tolist()

        while len(selected) < k:
            anchors = selected + revealed_positions
            best_position = None
            best_score = None
            for position in valid_positions:
                if position in selected:
                    continue
                if anchors:
                    coverage_score = min(abs(position - anchor) for anchor in anchors)
                else:
                    coverage_score = min(position, sequence_length - 1 - position)
                tie_conf = float(base_metrics["top1_prob"][position].item())
                score = (coverage_score, tie_conf, -position)
                if best_score is None or score > best_score:
                    best_score = score
                    best_position = position
            if best_position is None:
                break
            selected.append(best_position)
        return selected

    def _frontier(
        self,
        valid_positions: list[int],
        base_metrics: dict[str, torch.Tensor],
        k: int,
        quantile: float,
    ) -> list[int]:
        confidences = sorted(
            float(base_metrics["top1_prob"][pos].item()) for pos in valid_positions
        )
        if len(confidences) == 1:
            target_conf = confidences[0]
        else:
            raw_index = quantile * (len(confidences) - 1)
            low_index = int(raw_index)
            high_index = min(len(confidences) - 1, low_index + 1)
            fraction = raw_index - low_index
            target_conf = (
                confidences[low_index] * (1.0 - fraction)
                + confidences[high_index] * fraction
            )
        ranked = sorted(
            valid_positions,
            key=lambda pos: (
                -abs(float(base_metrics["top1_prob"][pos].item()) - target_conf),
                float(base_metrics["margin"][pos].item()),
                -pos,
            ),
            reverse=True,
        )
        return ranked[:k]

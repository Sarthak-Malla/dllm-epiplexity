"""Run with: pytest scripts/tests/test_tse_loader.py -v"""

from types import SimpleNamespace

import pytest
import torch

from dllm.pipelines.tse.loader import _validate_compatibility


class FakeTokenizer:
    def __init__(self, mask_token_id):
        self.mask_token_id = mask_token_id

    def __call__(self, text, **kwargs):
        return {"input_ids": [0], "offset_mapping": [(0, len(text))]}


class FakeModel:
    def __init__(self, vocab_size, embedding_size=None):
        self.config = SimpleNamespace(vocab_size=vocab_size)
        self.embedding = torch.nn.Embedding(
            embedding_size if embedding_size is not None else vocab_size,
            3,
        )

    def get_input_embeddings(self):
        return self.embedding

    def get_output_embeddings(self):
        return None


def test_homogeneous_loading_still_rejects_unequal_vocabularies():
    with pytest.raises(ValueError, match="equal config vocab sizes"):
        _validate_compatibility(
            FakeModel(4),
            FakeModel(5),
            FakeTokenizer(3),
            FakeTokenizer(4),
            ctca_enabled=False,
        )


def test_ctca_loading_accepts_unequal_vocabularies():
    _validate_compatibility(
        FakeModel(4),
        FakeModel(5),
        FakeTokenizer(3),
        FakeTokenizer(4),
        ctca_enabled=True,
    )


def test_ctca_requires_native_mask_ids_for_both_models():
    with pytest.raises(ValueError, match="Model A tokenizer must define mask_token_id"):
        _validate_compatibility(
            FakeModel(4),
            FakeModel(5),
            FakeTokenizer(None),
            FakeTokenizer(4),
            ctca_enabled=True,
        )


def test_ctca_rejects_input_embedding_vocab_mismatch():
    with pytest.raises(ValueError, match="input vocabulary"):
        _validate_compatibility(
            FakeModel(4, embedding_size=3),
            FakeModel(5),
            FakeTokenizer(2),
            FakeTokenizer(4),
            ctca_enabled=True,
        )

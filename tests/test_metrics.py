"""Tests for evaluation metrics."""

from __future__ import annotations


import torch

from obliteratus.evaluation.metrics import accuracy, f1_score_metric, perplexity


class TestPerplexity:
    def test_perfect_prediction(self):
        # Create logits that strongly predict the correct next token
        vocab_size = 10
        seq_len = 5
        batch_size = 1

        labels = torch.tensor([[0, 1, 2, 3, 4]])
        logits = torch.full((batch_size, seq_len, vocab_size), -100.0)
        # Set high logit for the correct next token
        for t in range(seq_len - 1):
            logits[0, t, labels[0, t + 1]] = 100.0

        ppl = perplexity(logits, labels)
        assert ppl < 2.0, f"Expected near-1 perplexity, got {ppl}"

    def test_random_prediction_higher(self):
        vocab_size = 100
        seq_len = 20
        batch_size = 2

        torch.manual_seed(42)
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        ppl = perplexity(logits, labels)
        assert ppl > 10, f"Random logits should yield high perplexity, got {ppl}"


class TestAccuracy:
    def test_perfect(self):
        assert accuracy([1, 2, 3], [1, 2, 3]) == 1.0

    def test_zero(self):
        assert accuracy([1, 2, 3], [4, 5, 6]) == 0.0

    def test_partial(self):
        assert accuracy([1, 2, 3, 4], [1, 2, 0, 0]) == 0.5

    def test_empty(self):
        assert accuracy([], []) == 0.0


class TestF1:
    def test_perfect(self):
        assert f1_score_metric([0, 1, 0, 1], [0, 1, 0, 1]) == 1.0

    def test_zero(self):
        score = f1_score_metric([0, 0, 0, 0], [1, 1, 1, 1])
        assert score == 0.0

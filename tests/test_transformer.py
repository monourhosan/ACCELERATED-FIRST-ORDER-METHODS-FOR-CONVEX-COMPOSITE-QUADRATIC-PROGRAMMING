"""
Unit tests for Transformer modules in transformer_scratch.py.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformer_scratch import (
    MultiHeadAttention,
    PositionwiseFeedForward,
    PositionalEncoding,
    subsequent_mask,
    make_transformer,
)


def test_multi_head_attention_shapes():
    batch_size = 2
    seq_len = 5
    d_model = 64
    n_heads = 4

    mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
    q = torch.randn(batch_size, seq_len, d_model)
    k = torch.randn(batch_size, seq_len, d_model)
    v = torch.randn(batch_size, seq_len, d_model)

    out = mha(q, k, v)
    assert out.shape == (batch_size, seq_len, d_model)


def test_causal_subsequent_mask():
    size = 5
    mask = subsequent_mask(size)
    assert mask.shape == (1, size, size)
    # Lower triangular should be 1, strictly upper triangular should be 0
    for i in range(size):
        for j in range(size):
            if j <= i:
                assert mask[0, i, j].item() == 1
            else:
                assert mask[0, i, j].item() == 0


def test_full_transformer_forward():
    src_vocab = 20
    tgt_vocab = 20
    model = make_transformer(src_vocab=src_vocab, tgt_vocab=tgt_vocab, N=2, d_model=64, d_ff=128, h=4)

    batch_size = 2
    src_len = 6
    tgt_len = 5

    src = torch.randint(1, src_vocab, (batch_size, src_len))
    tgt = torch.randint(1, tgt_vocab, (batch_size, tgt_len))

    src_mask = (src != 0).unsqueeze(-2)
    tgt_mask = (tgt != 0).unsqueeze(-2) & subsequent_mask(tgt_len)

    logits = model(src, tgt, src_mask, tgt_mask)
    assert logits.shape == (batch_size, tgt_len, tgt_vocab)


if __name__ == "__main__":
    test_multi_head_attention_shapes()
    test_causal_subsequent_mask()
    test_full_transformer_forward()
    print("All Transformer unit tests passed successfully!")

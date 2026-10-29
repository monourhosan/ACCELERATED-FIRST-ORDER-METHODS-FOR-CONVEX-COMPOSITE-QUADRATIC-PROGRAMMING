import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# 1. Positional Encoding
# =============================================================================
class PositionalEncoding(nn.Module):
    """
    Implements the sinusoidal positional encoding from Section 3.5:
    PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
    """
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute positional encodings once in log space
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (persistent state not considered a parameter to optimize)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        """
        # Add positional encoding up to the sequence length of x
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)
        return self.dropout(x)


# =============================================================================
# 2. Scaled Dot-Product Attention & Multi-Head Attention
# =============================================================================
def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
              mask: torch.Tensor = None, dropout: nn.Dropout = None):
    """
    Compute Scaled Dot-Product Attention:
    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
    
    Args:
        query: (batch_size, n_heads, seq_len_q, d_k)
        key:   (batch_size, n_heads, seq_len_k, d_k)
        value: (batch_size, n_heads, seq_len_v, d_k)
        mask:  (batch_size, 1, seq_len_q, seq_len_k) or broadcastable
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        # Ensure mask broadcasts cleanly across heads (dim 1)
        if mask.dim() == 3:
            # (batch_size, seq_len_q, seq_len_k) or (batch_size, 1, seq_len_k) -> (batch_size, 1, seq_len_q, seq_len_k)
            mask = mask.unsqueeze(1)
        elif mask.dim() == 2:
            # (batch_size, seq_len_k) -> (batch_size, 1, 1, seq_len_k)
            mask = mask.unsqueeze(1).unsqueeze(2)
        # Mask where mask == 0 with a very large negative value so softmax yields 0
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    output = torch.matmul(p_attn, value)
    return output, p_attn


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention (Section 3.2):
    Projects Q, K, V h times with learned linear projections, performs attention
    in parallel, concatenates results, and applies a final linear projection.
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Linear projections for Q, K, V and final output
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: torch.Tensor = None):
        batch_size = query.size(0)

        # 1) Linear projections and reshape to (batch_size, n_heads, seq_len, d_k)
        q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # 2) Apply scaled dot-product attention across all heads
        x, attn_weights = attention(q, k, v, mask=mask, dropout=self.dropout)

        # 3) Concatenate heads and apply final linear layer
        # Transpose back: (batch_size, seq_len, n_heads, d_k) -> contiguous -> (batch_size, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.w_o(x)


# =============================================================================
# 3. Position-wise Feed-Forward Network
# =============================================================================
class PositionwiseFeedForward(nn.Module):
    """
    FFN(x) = max(0, x W_1 + b_1) W_2 + b_2
    Two linear transformations with a ReLU activation in between.
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


# =============================================================================
# 4. Residual Connection & Layer Normalization
# =============================================================================
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm:
    LayerNorm(x + Sublayer(x))  [Post-LN as in Vaswani et al.]
    """
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, sublayer):
        return self.norm(x + self.dropout(sublayer(x)))


# =============================================================================
# 5. Encoder and Decoder Layers
# =============================================================================
class EncoderLayer(nn.Module):
    """
    Encoder is composed of 2 sub-layers:
    1) Multi-Head Self-Attention
    2) Position-wise Feed-Forward Network
    """
    def __init__(self, d_model: int, self_attn: MultiHeadAttention, 
                 feed_forward: PositionwiseFeedForward, dropout: float = 0.1):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayers = nn.ModuleList([SublayerConnection(d_model, dropout) for _ in range(2)])

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        # Sublayer 1: Self-Attention
        x = self.sublayers[0](x, lambda x: self.self_attn(x, x, x, mask))
        # Sublayer 2: Feed Forward
        x = self.sublayers[1](x, self.feed_forward)
        return x


class Encoder(nn.Module):
    """
    Core encoder is a stack of N identical EncoderLayers.
    """
    def __init__(self, layer: EncoderLayer, N: int):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.self_attn.d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    """
    Decoder is composed of 3 sub-layers:
    1) Masked Multi-Head Self-Attention (prevents positions from attending to future positions)
    2) Multi-Head Cross-Attention (queries from decoder, keys/values from encoder output)
    3) Position-wise Feed-Forward Network
    """
    def __init__(self, d_model: int, self_attn: MultiHeadAttention, 
                 cross_attn: MultiHeadAttention, feed_forward: PositionwiseFeedForward, 
                 dropout: float = 0.1):
        super().__init__()
        self.self_attn = self_attn
        self.cross_attn = cross_attn
        self.feed_forward = feed_forward
        self.sublayers = nn.ModuleList([SublayerConnection(d_model, dropout) for _ in range(3)])

    def forward(self, x: torch.Tensor, memory: torch.Tensor, 
                src_mask: torch.Tensor, tgt_mask: torch.Tensor):
        # 1) Masked Self-Attention
        x = self.sublayers[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        # 2) Cross-Attention with Encoder Output (memory)
        x = self.sublayers[1](x, lambda x: self.cross_attn(query=x, key=memory, value=memory, mask=src_mask))
        # 3) Feed-Forward
        x = self.sublayers[2](x, self.feed_forward)
        return x


class Decoder(nn.Module):
    """
    Core decoder is a stack of N identical DecoderLayers.
    """
    def __init__(self, layer: DecoderLayer, N: int):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.self_attn.d_model)

    def forward(self, x: torch.Tensor, memory: torch.Tensor, 
                src_mask: torch.Tensor, tgt_mask: torch.Tensor):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


# =============================================================================
# 6. Embeddings and Output Generator
# =============================================================================
class Embeddings(nn.Module):
    """
    Token Embedding scaled by sqrt(d_model) as specified in Section 3.4.
    """
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.lut = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lut(x) * math.sqrt(self.d_model)


class Generator(nn.Module):
    """
    Linear projection followed by log-softmax to predict next-token distributions.
    """
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(self.proj(x), dim=-1)


# =============================================================================
# 7. Complete Transformer Architecture
# =============================================================================
class Transformer(nn.Module):
    """
    Full Encoder-Decoder Transformer model as described in Vaswani et al. (2017).
    """
    def __init__(self, encoder: Encoder, decoder: Decoder, 
                 src_embed: nn.Sequential, tgt_embed: nn.Sequential, 
                 generator: Generator):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, 
               src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, 
                src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        memory = self.encode(src, src_mask)
        output = self.decode(tgt, memory, src_mask, tgt_mask)
        return self.generator(output)


def make_transformer(src_vocab: int, tgt_vocab: int, N: int = 2, 
                     d_model: int = 64, d_ff: int = 256, h: int = 4, 
                     dropout: float = 0.1) -> Transformer:
    """
    Factory function to construct a Transformer instance with given hyperparameters.
    Default hyperparameters are sized for fast and reliable demonstration on CPU/GPU.
    """
    c = copy.deepcopy
    attn = MultiHeadAttention(d_model, h, dropout)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout=dropout)

    model = Transformer(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(src_vocab, d_model), c(position)),
        nn.Sequential(Embeddings(tgt_vocab, d_model), c(position)),
        Generator(d_model, tgt_vocab)
    )

    # Initialize parameters with Xavier uniform
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
            
    return model


# =============================================================================
# 8. Masking Utilities
# =============================================================================
def subsequent_mask(size: int) -> torch.Tensor:
    """
    Mask out subsequent positions (causal / look-ahead mask).
    Returns a lower triangular matrix of shape (1, size, size) with 1s on/below diagonal.
    """
    attn_shape = (1, size, size)
    mask = torch.tril(torch.ones(attn_shape, dtype=torch.uint8))
    return mask == 1

def make_std_mask(tgt: torch.Tensor, pad: int = 0) -> torch.Tensor:
    """
    Create a mask to hide padding tokens and future tokens.
    """
    tgt_mask = (tgt != pad).unsqueeze(-2)
    tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask)
    return tgt_mask

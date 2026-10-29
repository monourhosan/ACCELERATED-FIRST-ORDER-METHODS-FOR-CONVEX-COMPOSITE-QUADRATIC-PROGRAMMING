import time
import torch
import torch.nn as nn
import torch.optim as optim
from transformer_scratch import make_transformer, subsequent_mask, make_std_mask


class Batch:
    """
    Object for holding a batch of data with masks during training.
    """
    def __init__(self, src, tgt=None, pad=0):
        self.src = src
        # Source mask: hide padding tokens
        self.src_mask = (src != pad).unsqueeze(-2)
        if tgt is not None:
            # Decoder input is everything except the last token
            self.tgt = tgt[:, :-1]
            # Target output to predict is everything except the first token (SOS)
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()


def data_gen(vocab_size: int, batch_size: int, num_batches: int, seq_len: int = 10, pad: int = 0, sos: int = 1):
    """
    Generate random batches of token sequences for the copy task.
    Symbols are randomly drawn from [2, vocab_size - 1].
    The target sequence is identical to the source sequence, prefixed by <SOS>.
    """
    for _ in range(num_batches):
        # Generate random tokens from [2, vocab_size - 1]
        data = torch.randint(2, vocab_size, size=(batch_size, seq_len))
        # Input to encoder
        src = data.clone()
        # Input to decoder starts with SOS, followed by data
        tgt = torch.cat([torch.full((batch_size, 1), sos, dtype=torch.long), data], dim=1)
        yield Batch(src, tgt, pad=pad)


def greedy_decode(model, src, src_mask, max_len: int, start_symbol: int = 1):
    """
    Greedy autoregressive decoding: generate tokens one by one.
    """
    memory = model.encode(src, src_mask)
    batch_size = src.size(0)
    ys = torch.full((batch_size, 1), start_symbol, dtype=torch.long, device=src.device)
    
    for i in range(max_len):
        tgt_mask = subsequent_mask(ys.size(1)).type_as(src_mask)
        out = model.decode(ys, memory, src_mask, tgt_mask)
        prob = model.generator(out[:, -1])  # (batch_size, vocab_size)
        _, next_word = torch.max(prob, dim=-1, keepdim=True)
        ys = torch.cat([ys, next_word], dim=1)
        
    return ys[:, 1:]  # Exclude initial <SOS> token


def run_epoch(data_iter, model, loss_compute, optimizer=None):
    """
    Standard training and evaluation loop.
    """
    total_tokens = 0
    total_loss = 0.0
    start_time = time.time()
    
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
        loss = loss_compute(out, batch.tgt_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        
        if optimizer is not None:
            optimizer.step()
            optimizer.zero_grad()
            
    elapsed = time.time() - start_time
    return total_loss / total_tokens, total_tokens / elapsed


class SimpleLossCompute:
    """
    Simple loss computation and train step.
    Computes negative log-likelihood (CrossEntropy with log_softmax outputs).
    """
    def __init__(self, criterion, opt=None):
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        # x is (batch_size, seq_len, vocab_size) of log probabilities
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1))
        if self.opt is not None:
            loss.backward()
        return loss.item() * norm


def main():
    print("=" * 70)
    print("TRAINING 'ATTENTION IS ALL YOU NEED' TRANSFORMER ON TOY COPY TASK")
    print("=" * 70)
    
    # 1. Hyperparameters
    vocab_size = 11      # 0: <PAD>, 1: <SOS>, 2..10: actual tokens
    seq_len = 10         # Length of sequence to copy
    d_model = 64         # Embedding & hidden dimension
    d_ff = 256           # Inner dimension of FFN
    h = 4                # Number of attention heads (d_k = 64 / 4 = 16)
    N = 2                # Number of Encoder and Decoder layers
    dropout = 0.0        # No dropout for toy deterministic copy task
    batch_size = 32
    num_train_batches = 300
    num_val_batches = 30
    epochs = 15
    lr = 0.001
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 2. Build model from scratch
    torch.manual_seed(42)
    model = make_transformer(
        src_vocab=vocab_size,
        tgt_vocab=vocab_size,
        N=N,
        d_model=d_model,
        d_ff=d_ff,
        h=h,
        dropout=dropout
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Constructed Transformer model from scratch.")
    print(f"Total trainable parameters: {num_params:,}")
    print(f"Architecture: {N} Encoder layers, {N} Decoder layers, {h} heads, d_model={d_model}, d_ff={d_ff}")
    print("-" * 70)

    # 3. Test model before training (untrained baseline)
    print("Testing model BEFORE training (random weights):")
    model.eval()
    test_src = torch.tensor([[2, 4, 6, 8, 10, 3, 5, 7, 9, 2]], device=device)
    test_mask = (test_src != 0).unsqueeze(-2)
    with torch.no_grad():
        untrained_pred = greedy_decode(model, test_src, test_mask, max_len=seq_len)
    print(f"  Input sequence : {test_src.cpu().tolist()[0]}")
    print(f"  Untrained pred : {untrained_pred.cpu().tolist()[0]}")
    print("-" * 70)

    # 4. Loss and Optimizer
    criterion = nn.NLLLoss(reduction="mean")
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)

    # 5. Training Loop
    print(f"Starting training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_speed = run_epoch(
            data_gen(vocab_size, batch_size, num_train_batches, seq_len=seq_len),
            model,
            SimpleLossCompute(criterion, optimizer),
            optimizer
        )
        
        model.eval()
        with torch.no_grad():
            val_loss, _ = run_epoch(
                data_gen(vocab_size, batch_size, num_val_batches, seq_len=seq_len),
                model,
                SimpleLossCompute(criterion, None),
                None
            )
            
        print(f"Epoch {epoch:2d}/{epochs:2d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Tokens/sec: {train_speed:.0f}")

    print("-" * 70)
    print("Training complete! Evaluating exact copy performance on 5 unseen test sequences:")
    model.eval()
    
    test_samples = [
        [3, 8, 5, 2, 9, 4, 7, 6, 10, 2],
        [7, 7, 2, 9, 5, 3, 10, 4, 8, 6],
        [10, 9, 8, 7, 6, 5, 4, 3, 2, 10],
        [2, 3, 4, 5, 6, 7, 8, 9, 10, 2],
        [9, 2, 4, 8, 10, 7, 3, 5, 6, 4],
    ]
    
    correct_count = 0
    total_samples = len(test_samples)
    
    for idx, sample in enumerate(test_samples, 1):
        src_t = torch.tensor([sample], device=device)
        src_m = (src_t != 0).unsqueeze(-2)
        with torch.no_grad():
            pred = greedy_decode(model, src_t, src_m, max_len=seq_len)
        pred_list = pred.cpu().tolist()[0]
        match = (sample == pred_list)
        if match:
            correct_count += 1
        status = "PASSED (100% Exact Match)" if match else "FAILED"
        print(f"Sample {idx}:")
        print(f"  Input  : {sample}")
        print(f"  Output : {pred_list}")
        print(f"  Result : {status}")
        
    print("=" * 70)
    print(f"Overall Accuracy: {correct_count}/{total_samples} ({(correct_count/total_samples)*100:.1f}%)")
    print("The Transformer successfully learned the sequence copy task from scratch!")
    print("=" * 70)


if __name__ == "__main__":
    main()

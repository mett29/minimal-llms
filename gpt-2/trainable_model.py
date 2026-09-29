import math

import torch
from model import HParams
from torch import Tensor, nn


class GPT2(nn.Module):
    def __init__(self, hparams: HParams):
        super().__init__()
        self.hparams = hparams

        # torch.randn draws random numbers. Multiplying by 0.02 makes
        # their initial size small; e.g. 0.5 becomes 0.01.
        # nn.Parameter makes each tensor a trainable model weight.
        self.wte = nn.Parameter(torch.randn(hparams.n_vocab, hparams.n_embd) * 0.02)
        self.wpe = nn.Parameter(torch.randn(hparams.n_ctx, hparams.n_embd) * 0.01)

    def forward(self, token_ids: Tensor) -> Tensor:
        if token_ids.dim() != 2:
            raise ValueError("token_ids must have shape [batch_size, seq_length]")

        sequence_length = token_ids.size(1)
        if sequence_length > self.hparams.n_ctx:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds model context size {self.hparams.n_ctx}"
            )

        positions = torch.arange(sequence_length, device=token_ids.device).unsqueeze(
            0
        )  # Shape: [1, seq_length]
        return self.wte[token_ids] + self.wpe[positions]


class CausalSelfAttention(nn.Module):
    def __init__(self, hparams: HParams):
        super().__init__()  # Register weights belonging to this module.

        if hparams.n_embd % hparams.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")

        self.n_head = hparams.n_head
        self.head_dim = hparams.n_embd // hparams.n_head
        width = hparams.n_embd

        # torch.randn draws random numbers. Multiplying by 0.02 makes
        # their initial size small; e.g. 0.5 becomes 0.01.
        # nn.Parameter makes each tensor a trainable model weight.
        self.qkv_weight = nn.Parameter(torch.randn(width, 3 * width) * 0.02)
        self.qkv_bias = nn.Parameter(torch.zeros(3 * width))
        self.out_weight = nn.Parameter(torch.randn(width, width) * 0.02)
        self.out_bias = nn.Parameter(torch.zeros(width))

    def split_heads(self, tensor: Tensor) -> Tensor:
        """Turn [B, T, D] into [B, H, T, D/H]."""
        batch, sequence, _ = tensor.shape

        # With D=4 and H=2, split each four-number token vector in two:
        # token 0: [1, 2, 3, 4] -> [[1, 2], [3, 4]] (head 0, head 1)
        # token 1: [5, 6, 7, 8] -> [[5, 6], [7, 8]] (head 0, head 1)
        # reshape changes [B, T, D] into [B, T, H, D/H].
        grouped = tensor.reshape(batch, sequence, self.n_head, self.head_dim)

        # transpose swaps the token and head axes: [B, T, H, D/H]
        # becomes [B, H, T, D/H]. For the numbers above:
        # head 0 has tokens [[1, 2], [5, 6]]
        # head 1 has tokens [[3, 4], [7, 8]]
        # To sum up:
        # - we have 2 tokens, e.g., "the" and "cat"
        # - each token has 4 features, e.g., [1, 2, 3, 4]
        # - we have 2 heads, each with 2 features
        # - thanks to the transpose, head 0 sees the first 2 features of each token (same for head 1 with the last 2 features)
        # - each head thus operates on a matrix representing the entire sequence
        # - but each head sees only a subset of the features, so they can learn different aspects of the sequence
        # - head 0 now owns both [1, 2] and [5, 6], and it can use @ to make "the" and "cat" interact, same for head 1 with [3, 4] and [7, 8]
        return grouped.transpose(1, 2)

    def forward(self, x: Tensor) -> Tensor:
        # For the toy size, x has shape [1, 2, 4]:
        # one batch, two tokens, four features per token
        # e.g., x = [[[1, 2, 3, 4], [5, 6, 7, 8]]]
        batch, sequence, width = x.shape

        # @ is matrix multiplication. It transforms each D-number
        # token vector into 3D numbers, then adds the bias.
        # Example: x = [[[1, 2, 3, 4], [5, 6, 7, 8]]]
        # Assuming n_embd=4, the weight matrix has shape [4, 12] and the bias has shape [12]
        # The result is a tensor of shape [1, 2, 4] @ [4, 12] + [12] = [1, 2, 12]
        # Note: matmul takes the last 2 dimensions of the tensors and performs matrix multiplication on them,
        # while broadcasting the other dimensions. In this case, it multiplies each token vector (of size 4)
        # by the weight matrix (of size 4x12) to produce a new vector of size 12 for each token.
        qkv = x @ self.qkv_weight + self.qkv_bias

        # chunk(3) cuts the final axis into three equal parts.
        # In our example, qkv has shape [1, 2, 12], so chunk(3) produces three tensors of shape [1, 2, 4].
        # Example: qkv = [[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]]]
        # The first chunk is q = [[[1, 2, 3, 4], [13, 14, 15, 16]]]
        # the second is k = [[[5, 6, 7, 8], [17, 18, 19, 20]]]
        # and the third is v = [[[9, 10, 11, 12], [21, 22, 23, 24]]]
        # So:
        # - q (query) indicates what each token is looking for
        # - k (key) indicates what each token has to offer
        # - v (value) indicates what each token will contribute to the output
        q, k, v = qkv.chunk(3, dim=-1)

        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)
        # Each now has shape [B, H, T, D/H] = [1, 2, 2, 2].

        # For ONE illustrative head, from the example above, we have:
        # q = [[[1, 2], [13, 14]],      # head 0
        #      [[3, 4], [15, 16]]]      # head 1
        # k = [[[5, 6], [17, 18]],      # head 0
        #      [[7, 8], [19, 20]]]      # head 1
        # v = [[[9, 10], [21, 22]],     # head 0
        #      [[11, 12], [23, 24]]]    # head 1
        # k.transpose(-2, -1) turns its [tokens, features] matrix
        # into [features, tokens], so q @ kᵀ computes the dot product of each query with each key.
        # PyTorch does this separately for every batch row and head.
        scores = q @ k.transpose(-2, -1)

        # Divide by sqrt(head_dim).
        # scores shape: [B, H, T, T] = [1, 2, 2, 2].
        scores = scores / math.sqrt(self.head_dim)

        # allowed shape: [T, T] = [2, 2]. It is a boolean matrix indicating which tokens can attend to which other tokens.
        allowed = torch.ones(sequence, sequence, dtype=torch.bool, device=x.device)

        # Keep the lower triangle:
        # [[True, False],
        #  [True,  True]]
        # Thus token 0 sees only token 0; token 1 sees tokens 0 and 1.
        allowed = allowed.tril()

        # ~allowed marks the False positions. masked_fill puts -infinity
        # there. This ensures that when we apply softmax, those positions will have zero probability.
        # scores shape: [B, H, T, T] = [1, 2, 2, 2].
        scores = scores.masked_fill(~allowed, float("-inf"))

        # softmax turns each score row into probabilities summing to 1.
        # The last axis lists the keys a query token may attend to.
        # weights shape: [B, H, T, T] = [1, 2, 2, 2].
        weights = torch.softmax(scores, dim=-1)

        # weights shape: [B, H, T, T] = [1, 2, 2, 2].
        # v shape: [B, H, T, D/H] = [1, 2, 2, 2].
        # context shape: [B, H, T, D/H] = [1, 2, 2, 2].
        # In our example "the cat":
        # For head 0:
        # - token 0 sees only itself, so its output is [9, 10].
        # - token 1 sees both tokens, so its output is a weighted sum of [9, 10] and [21, 22].
        # For head 1:
        # - token 0 sees only itself, so its output is [11, 12].
        # - token 1 sees both tokens, so its output is a weighted sum of [11, 12] and [23, 24].
        context = weights @ v

        # Put tokens before heads again: [B, H, T, D/H] -> [B, T, H, D/H].
        context = context.transpose(1, 2)

        # Join each token's head vectors.
        # For example, token 0's head outputs [[9, 10], [11, 12]] become [9, 10, 11, 12].
        # Shape: [B, T, H, D/H] -> [B, T, D].
        context = context.reshape(batch, sequence, width)

        # Another learned matrix multiplication mixes the head outputs.
        # The returned shape is [B, T, D], matching the input shape.
        return context @ self.out_weight + self.out_bias


if __name__ == "__main__":
    hparams = HParams(n_vocab=5, n_ctx=8, n_embd=4, n_head=2, n_layer=2)
    model = GPT2(hparams)
    tokens = torch.tensor([[1, 2, 3]], dtype=torch.long)

    hidden = model(tokens)
    assert torch.equal(hidden, model(tokens))
    hidden.sum().backward()

    print(f"hidden.shape: {hidden.shape}")
    print(
        f"Total parameters: {sum(parameter.numel() for parameter in model.parameters())}"
    )
    print(f"wte.grad is not None: {model.wte.grad is not None}")
    print(f"wpe.grad is not None: {model.wpe.grad is not None}")

    attention = CausalSelfAttention(hparams)
    original = attention(model(tokens))

    changed_tokens = tokens.clone()
    changed_tokens[0, -1] = 4
    changed = attention(model(changed_tokens))

    print(f"original.shape: {original.shape}")
    print(
        f"torch.allclose(original[:, :-1], changed[:, :-1]): {torch.allclose(original[:, :-1], changed[:, :-1])}"
    )

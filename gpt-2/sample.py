import sys
from contextlib import redirect_stdout
from io import StringIO
from typing import TextIO

import model
import torch
from model import HParams
from torch import Tensor


def top_k_logits(logits: Tensor, k: int):
    """
    Truncate the logits by keeping only the top-k highest
    values, while setting the rest to a very low value.

    Args:
        :param logits: Logits of shape [batch, vocab_size].
        :param k: Number of top values to retain.

    Returns:
        torch.Tensor: Logits with only the top-k values retained, others set to -1e10.
    """
    if k == 0:
        # no truncation
        return logits

    def _top_k():
        values, _ = torch.topk(logits, k=k, dim=-1)
        # Get the smallest value among the top-k values in each row
        min_values = values[:, -1].unsqueeze(-1)
        return torch.where(
            logits < min_values,
            torch.ones_like(logits, dtype=logits.dtype) * -1e10,
            logits,
        )

    return _top_k()

def top_p_logits(logits: Tensor, p: float):
    """
    Nucleus sampling.
    In nucleus sampling, instead of selecting the top-k logits,
    we select the smallest subset of logits whose cumulative probability
    adds up to a predefined threshold p (typically around 0.9 or 0.95).
    This ensures that a dynamic number of tokens (instead of a fixed k)
    are selected for sampling based on their cumulative probabilities.

    Args:
        :param logits: Logits of shape [batch, vocab_size].
        :param p: Cumulative probability threshold for nucleus sampling.

    Returns:
        torch.Tensor: Logits with only top-p elements retained, others set to -1e10.
    """
    if not 0 < p <= 1:
        raise ValueError('top_p must be in (0, 1]')
    if p == 1:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    # Keep the first token that takes the cumulative mass past p.
    remove = cumulative_probs >= p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    remove = torch.zeros_like(remove).scatter(-1, sorted_indices, remove)
    return logits.masked_fill(remove, -1e10)

def sample_sequence(
    *, hparams: HParams, length: int, start_token: int = None, batch_size: int = None,
    context: Tensor = None, temperature: float = 1, top_k: int = 0, top_p: float = 1,
    trace: TextIO | None = None,
):
    """
    Generates a sequence of tokens using the model and the given hyperparameters.
    
    Args:
        :param hparams: Model hyperparameters.
        :param length: The length of the sequence to generate.
        :param start_token: The token to start the sequence. Specify this or 'context'.
        :param batch_size: The batch size for generation.
        :param context: Initial context of tokens.
        :param temperature: Temperature for sampling.
        :param top_k: If > 0, only the top k tokens will be considered.
        :param top_p: If < 1, use top-p (nucleus) sampling.
        :param trace: Text stream for an annotated record of each generation step.
    
    Returns:
        torch.Tensor: Generated sequence of tokens.
    """
    if start_token is None:
        assert context is not None, 'Specify exactly one of start_token and context!'
        if batch_size is None:
            batch_size = context.size(0)
        elif batch_size != context.size(0):
            raise ValueError('batch_size must match the number of context rows')
    else:
        assert context is None, 'Specify exactly one of start_token and context!'
        context = torch.full([batch_size, 1], start_token, dtype=torch.long)

    if length < 1:
        raise ValueError('length must be at least 1')
    if temperature <= 0:
        raise ValueError('temperature must be positive')

    log = trace if trace is not None else sys.stdout
    print(f"**Initial token IDs:** `{context.tolist()}`  ", file=log)
    print(f"**Plan:** generate {length} new token(s) for {batch_size} sequence(s).", file=log)

    def step(hparams, tokens, past=None):
        print("\n#### Model values", file=log)
        print("For large tensors, the trace shows the first eight values in flattened order and the full range.\n", file=log)

        def record_tensor(name: str, value: Tensor):
            flat = value.detach().reshape(-1)
            shown = min(8, flat.numel())
            numbers = ', '.join(f'{float(item):.6g}' for item in flat[:shown])
            label = 'all values' if shown == flat.numel() else 'first eight values'
            summary = f'- **{name}** · shape `{list(value.shape)}` · {label} `[{numbers}]`'
            if shown < flat.numel():
                summary += f' · min `{float(flat.min()):.6g}`, max `{float(flat.max()):.6g}`'
            print(summary, file=log)

        diagnostics = StringIO()
        with redirect_stdout(diagnostics):
            lm_output = model.model(hparams=hparams, X=tokens, past=past, trace=record_tensor)
        print('\n<details><summary>Original model shape diagnostics</summary>\n', file=log)
        print('```text', file=log)
        print(diagnostics.getvalue().rstrip(), file=log)
        print('```\n</details>\n', file=log)

        logits = lm_output['logits'][:, :, :hparams.n_vocab]
        presents = lm_output['present']
        sequence_length = tokens.size(1)
        past_shape = model.past_shape(hparams=hparams, batch_size=batch_size, sequence=sequence_length)
        presents = presents.reshape(past_shape)
        print(f"The model returned logits shaped `{list(logits.shape)}` and new cache values shaped `{list(presents.shape)}`.\n", file=log)
        return {
            'logits': logits,
            'presents': presents,
        }

    def body(step_number, past, prev, output):
        past_length = 0 if past is None else past.size(-2)
        print(f"\n### Step {step_number}/{length}\n", file=log)
        print(f"Input IDs: `{prev.tolist()}`. Cached positions: `{past_length}`. "
              f"Current position indices: `{list(range(past_length, past_length + prev.size(1)))}`.", file=log)
        next_outputs = step(hparams, prev, past=past)
        last_logits = next_outputs['logits'][:, -1, :]
        highest_logits, highest_ids = torch.topk(last_logits, k=min(5, last_logits.size(-1)), dim=-1)
        for row in range(batch_size):
            scores = [(int(token_id), round(float(score), 6)) for token_id, score in zip(highest_ids[row], highest_logits[row])]
            print(f"- Row {row}, highest raw `(token ID, logit)` pairs: `{scores}`", file=log)
        logits = last_logits / temperature
        print(f"\n#### Choose the next token\n\nUse the last position's logits and divide by temperature `{temperature}`.", file=log)
        logits = top_k_logits(logits, k=top_k)
        print(f"Top-k filter: {'off' if top_k == 0 else f'keep up to {top_k} highest-scoring IDs'}.", file=log)
        logits = top_p_logits(logits, p=top_p)
        print(f"Top-p filter: {'off' if top_p == 1 else f'keep the smallest probability prefix reaching {top_p}'}.\n", file=log)
        probabilities = torch.softmax(logits, dim=-1)
        top_probs, top_ids = torch.topk(probabilities, k=min(5, probabilities.size(-1)), dim=-1)
        for row in range(batch_size):
            candidates = [(int(token_id), round(float(probability), 6)) for token_id, probability in zip(top_ids[row], top_probs[row])]
            print(f"- Row {row}, highest `(token ID, probability)` pairs: `{candidates}`", file=log)
        samples = torch.multinomial(probabilities, num_samples=1)
        updated_past = next_outputs['presents'] if past is None else torch.cat([past, next_outputs['presents']], dim=-2)
        updated_output = torch.cat([output, samples], dim=1)
        print(f"\nSampled IDs: `{samples.squeeze(-1).tolist()}`.  ", file=log)
        print(f"Cache after this step: shape `{list(updated_past.shape)}`; accumulated IDs: `{updated_output.tolist()}`.", file=log)
        return [
            updated_past,
            samples,
            updated_output,
        ]

    past, prev, tokens = body(1, None, context, context)

    for step_number in range(2, length + 1):
        past, prev, tokens = body(step_number, past, prev, tokens)

    print(f"\n**Final token IDs:** `{tokens.tolist()}`.", file=log)
    return tokens

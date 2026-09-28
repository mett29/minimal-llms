import json
import os
import re
from datetime import datetime
from pathlib import Path

import encoder
import model
import numpy as np
import sample
import torch
from model import HParams


def sample_model(
    seed=42,
    nsamples=0,
    batch_size=1,
    length=None,
    temperature=1,
    top_k=0,
    top_p=1,
    config_dir='config',
    output_dir=None,
):
    """
    Run the sample_model
    :param seed=None: Integer seed for random number generators, fix seed to 
        reproduce results
    :param nsamples=0: Number of samples to return, if 0, continues to
        generate samples indefinately.
    :param batch_size=1: Number of batches (only affects speed/memory).
    :param length=None: Number of tokens in generated text, if None (default), is
        determined by model hyperparameters
    :param temperature=1: Float value controlling randomness in boltzmann
        distribution. Lower temperature results in less random completions. As the
        temperature approaches zero, the model will become deterministic and
        repetitive. Higher temperature results in more random completions.
    :param top_k=0: Integer value controlling diversity. 1 means only 1 word is
        considered for each step (token), resulting in deterministic completions,
        while 40 means 40 words are considered at each step. 0 (default) is a
        special setting meaning no restrictions. 40 generally is a good value.
    :param config_dir: path to parent folder containing model subfolders
    :param output_dir: folder for the annotated .md trace (defaults to gpt-2/output)
    """
    config_dir = os.path.expanduser(os.path.expandvars(config_dir))
    enc = encoder.get_encoder(config_dir)
    hparams = model.default_hparams()
    with open(os.path.join(config_dir, 'hparams.json')) as f:
        hparams = HParams(**json.load(f))

    if length is None:
        length = hparams.n_ctx
    elif length > hparams.n_ctx:
        raise ValueError(f"Can't get samples longer than window size: {hparams.n_ctx}")

    np.random.seed(seed)
    torch.manual_seed(seed)

    output_dir = Path(output_dir) if output_dir is not None else Path(__file__).resolve().parent / 'output'
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / f"generation_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.md"
    print(f'Trace writing to {trace_path}')

    generated = 0
    with trace_path.open('w', encoding='utf-8', buffering=1) as trace:
        print('# GPT-2 generation trace', file=trace)
        print('\n## Initial state\n', file=trace)
        print(f'- Configuration: `{config_dir}`', file=trace)
        print(f'- Model dimensions: vocabulary `{hparams.n_vocab}`, context `{hparams.n_ctx}`, '
              f'width `{hparams.n_embd}`, heads `{hparams.n_head}`, blocks `{hparams.n_layer}`', file=trace)
        print(f'- Random seed: `{seed}`; start ID: `{enc.encoder["<|endoftext|>"]}`', file=trace)
        print(f'- New tokens per sample: `{length}`; batch size: `{batch_size}`', file=trace)
        print(f'- Temperature: `{temperature}`; top-k: `{top_k}`; top-p: `{top_p}`', file=trace)
        print('\n> This educational model redraws untrained embeddings on every forward call. '
              'Tensor values show the calculation, not pretrained GPT-2 behavior.', file=trace)
        try:
            while nsamples == 0 or generated < nsamples:
                print(f'\n## Batch {generated // batch_size + 1}\n', file=trace)
                output = sample.sample_sequence(
                    hparams=hparams, length=length,
                    start_token=enc.encoder['<|endoftext|>'],
                    batch_size=batch_size,
                    temperature=temperature, top_k=top_k, top_p=top_p,
                    trace=trace,
                )[:, 1:]
                print('\n### Decode the generated IDs\n\nDrop the start ID, then map the generated IDs back to text.', file=trace)
                for i in range(min(batch_size, nsamples - generated) if nsamples else batch_size):
                    generated += 1
                    text = enc.decode(output[i])
                    header = "=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40
                    print(f'\n#### Sample {generated}\n', file=trace)
                    fence = '`' * max(3, max((len(run) for run in re.findall(r'`+', text)), default=0) + 1)
                    print(f'{fence}text\n{text}\n{fence}', file=trace)
                    print(header)
                    print(text)
        except KeyboardInterrupt:
            print('\n> Generation was interrupted. The trace contains every completed step up to this point.', file=trace)
        print(f'\n## Last state\n\nCompleted samples: `{generated}`.', file=trace)
    print(f'Trace saved to {trace_path}')
    return trace_path


if __name__ == '__main__':
    """
    Generate
    """
    sample_model(
        config_dir='gpt-2/config',
        top_k=40,
        nsamples=1,
        length=1,
    )

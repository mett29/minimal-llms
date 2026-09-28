# GPT-2 generation trace

## Initial state

- Configuration: `gpt-2/config`
- Model dimensions: vocabulary `50257`, context `1024`, width `768`, heads `12`, blocks `2`
- Random seed: `42`; start ID: `50256`
- New tokens per sample: `1`; batch size: `1`
- Temperature: `1`; top-k: `40`; top-p: `1`

> This educational model redraws untrained embeddings on every forward call. Tensor values show the calculation, not pretrained GPT-2 behavior.

## Batch 1

**Initial token IDs:** `[[50256]]`  
**Plan:** generate 1 new token(s) for 1 sequence(s).

### Step 1/1

Input IDs: `[[50256]]`. Cached positions: `0`. Current position indices: `[0]`.

#### Model values
For large tensors, the trace shows the first eight values in flattened order and the full range.

- **input token IDs** · shape `[1, 1]` · all values `[50256]`
- **position IDs** · shape `[1, 1]` · all values `[0]`
- **token embeddings** · shape `[1, 1, 768]` · first eight values `[-0.0368522, 0.0146036, 0.0138254, -0.0153772, -0.0432263, 0.0498494, -0.00878573, 0.0106393]` · min `-0.0614996`, max `0.0677667`
- **position embeddings** · shape `[1, 1, 768]` · first eight values `[0.0192691, 0.0148728, 0.00900717, -0.0210552, 0.00678418, -0.0123454, -0.000430675, -0.0160467]` · min `-0.0279363`, max `0.0302505`
- **combined embeddings** · shape `[1, 1, 768]` · first eight values `[-0.0175831, 0.0294764, 0.0228326, -0.0364324, -0.0364421, 0.037504, -0.00921641, -0.00540736]` · min `-0.0651121`, max `0.0714572`
- **block 1: normalized input to attention** · shape `[1, 1, 768]` · first eight values `[-0.781069, 1.31146, 1.01604, -1.61922, -1.61965, 1.66841, -0.409041, -0.239669]` · min `-2.89447`, max `3.17816`
- **block 1: queries** · shape `[1, 12, 1, 64]` · first eight values `[1.64522e-07, 1.64522e-07, 1.64522e-07, 1.64522e-07, 1.64522e-07, 1.64522e-07, 1.64522e-07, 1.64522e-07]` · min `1.64522e-07`, max `1.64522e-07`
- **block 1: new keys** · shape `[1, 12, 1, 64]` · first eight values `[1.64522e-07, 1.64522e-07, 1.64522e-07, 1.64522e-07, 1.64522e-07, 1.64522e-07, 1.64522e-07, 1.64522e-07]` · min `1.64522e-07`, max `1.64522e-07`
- **block 1: new values** · shape `[1, 12, 1, 64]` · first eight values `[1.64522e-07, 1.64522e-07, 1.64522e-07, 1.64522e-07, 1.64522e-07, 1.64522e-07, 1.64522e-07, 1.64522e-07]` · min `1.64522e-07`, max `4.61936e-07`
- **block 1: masked attention scores** · shape `[1, 12, 1, 1]` · first eight values `[2.1654e-13, 2.1654e-13, 2.1654e-13, 2.1654e-13, 2.1654e-13, 2.1654e-13, 2.1654e-13, 2.1654e-13]` · min `2.1654e-13`, max `2.1654e-13`
- **block 1: attention weights** · shape `[1, 12, 1, 1]` · first eight values `[1, 1, 1, 1, 1, 1, 1, 1]` · min `1`, max `1`
- **block 1: attention output** · shape `[1, 1, 768]` · first eight values `[4.0498e-06, 4.0498e-06, 4.0498e-06, 4.0498e-06, 4.0498e-06, 4.0498e-06, 4.0498e-06, 4.0498e-06]` · min `4.0498e-06`, max `4.04982e-06`
- **block 1: after attention residual** · shape `[1, 1, 768]` · first eight values `[-0.017579, 0.0294805, 0.0228366, -0.0364284, -0.0364381, 0.037508, -0.00921236, -0.00540331]` · min `-0.065108`, max `0.0714613`
- **block 1: normalized input to MLP** · shape `[1, 1, 768]` · first eight values `[-0.781069, 1.31146, 1.01604, -1.61922, -1.61965, 1.66841, -0.409041, -0.239669]` · min `-2.89447`, max `3.17816`
- **block 1: MLP expanded projection** · shape `[1, 1, 3072]` · first eight values `[1.4217e-07, 1.4217e-07, 1.4217e-07, 1.4217e-07, 1.4217e-07, 1.4217e-07, 1.4217e-07, 1.4217e-07]` · min `1.4217e-07`, max `1.4217e-07`
- **block 1: MLP after GELU** · shape `[1, 1, 3072]` · first eight values `[7.10852e-08, 7.10852e-08, 7.10852e-08, 7.10852e-08, 7.10852e-08, 7.10852e-08, 7.10852e-08, 7.10852e-08]` · min `7.10852e-08`, max `7.10852e-08`
- **block 1: MLP output** · shape `[1, 1, 768]` · first eight values `[4.36747e-06, 4.36747e-06, 4.36747e-06, 4.36747e-06, 4.36747e-06, 4.36747e-06, 4.36747e-06, 4.36747e-06]` · min `4.36747e-06`, max `4.36749e-06`
- **block 1: after MLP residual** · shape `[1, 1, 768]` · first eight values `[-0.0175746, 0.0294849, 0.022841, -0.036424, -0.0364337, 0.0375124, -0.00920799, -0.00539895]` · min `-0.0651037`, max `0.0714657`
- **block 2: normalized input to attention** · shape `[1, 1, 768]` · first eight values `[-0.781069, 1.31146, 1.01604, -1.61922, -1.61965, 1.66841, -0.409041, -0.239669]` · min `-2.89447`, max `3.17816`
- **block 2: queries** · shape `[1, 12, 1, 64]` · first eight values `[-2.08007e-07, -2.08007e-07, -2.08007e-07, -2.08007e-07, -2.08007e-07, -2.08007e-07, -2.08007e-07, -2.08007e-07]` · min `-2.08007e-07`, max `-2.08007e-07`
- **block 2: new keys** · shape `[1, 12, 1, 64]` · first eight values `[-2.08007e-07, -2.08007e-07, -2.08007e-07, -2.08007e-07, -2.08007e-07, -2.08007e-07, -2.08007e-07, -2.08007e-07]` · min `-2.08007e-07`, max `-2.08007e-07`
- **block 2: new values** · shape `[1, 12, 1, 64]` · first eight values `[-2.08007e-07, -2.08007e-07, -2.08007e-07, -2.08007e-07, -2.08007e-07, -2.08007e-07, -2.08007e-07, -2.08007e-07]` · min `-2.08007e-07`, max `1.04308e-07`
- **block 2: masked attention scores** · shape `[1, 12, 1, 1]` · first eight values `[3.46135e-13, 3.46135e-13, 3.46135e-13, 3.46135e-13, 3.46135e-13, 3.46135e-13, 3.46135e-13, 3.46135e-13]` · min `3.46135e-13`, max `3.46135e-13`
- **block 2: attention weights** · shape `[1, 12, 1, 1]` · first eight values `[1, 1, 1, 1, 1, 1, 1, 1]` · min `1`, max `1`
- **block 2: attention output** · shape `[1, 1, 768]` · first eight values `[-1.59593e-06, -1.59593e-06, -1.59593e-06, -1.59593e-06, -1.59593e-06, -1.59593e-06, -1.59593e-06, -1.59593e-06]` · min `-1.59594e-06`, max `-1.59593e-06`
- **block 2: after attention residual** · shape `[1, 1, 768]` · first eight values `[-0.0175762, 0.0294833, 0.0228394, -0.0364256, -0.0364353, 0.0375108, -0.00920959, -0.00540054]` · min `-0.0651052`, max `0.0714641`
- **block 2: normalized input to MLP** · shape `[1, 1, 768]` · first eight values `[-0.781069, 1.31146, 1.01604, -1.61922, -1.61965, 1.66841, -0.409041, -0.239669]` · min `-2.89447`, max `3.17816`
- **block 2: MLP expanded projection** · shape `[1, 1, 3072]` · first eight values `[1.4217e-07, 1.4217e-07, 1.4217e-07, 1.4217e-07, 1.4217e-07, 1.4217e-07, 1.4217e-07, 1.4217e-07]` · min `1.4217e-07`, max `1.4217e-07`
- **block 2: MLP after GELU** · shape `[1, 1, 3072]` · first eight values `[7.10852e-08, 7.10852e-08, 7.10852e-08, 7.10852e-08, 7.10852e-08, 7.10852e-08, 7.10852e-08, 7.10852e-08]` · min `7.10852e-08`, max `7.10852e-08`
- **block 2: MLP output** · shape `[1, 1, 768]` · first eight values `[4.36747e-06, 4.36747e-06, 4.36747e-06, 4.36747e-06, 4.36747e-06, 4.36747e-06, 4.36747e-06, 4.36747e-06]` · min `4.36747e-06`, max `4.36749e-06`
- **block 2: after MLP residual** · shape `[1, 1, 768]` · first eight values `[-0.0175719, 0.0294876, 0.0228438, -0.0364212, -0.0364309, 0.0375152, -0.00920522, -0.00539617]` · min `-0.0651009`, max `0.0714684`
- **new cache keys and values** · shape `[1, 2, 2, 12, 1, 64]` · first eight values `[1.64522e-07, 1.64522e-07, 1.64522e-07, 1.64522e-07, 1.64522e-07, 1.64522e-07, 1.64522e-07, 1.64522e-07]` · min `-2.08007e-07`, max `4.61936e-07`
- **final normalized hidden states** · shape `[1, 1, 768]` · first eight values `[-0.781069, 1.31146, 1.01604, -1.61922, -1.61965, 1.66841, -0.409041, -0.239669]` · min `-2.89447`, max `3.17816`
- **vocabulary logits** · shape `[1, 1, 50257]` · first eight values `[-0.141749, 0.467544, 0.110795, 0.939843, -0.0761153, -0.822295, -0.402036, 0.187952]` · min `-2.57081`, max `13.2716`

<details><summary>Original model shape diagnostics</summary>

```text
Shape of X: torch.Size([1, 1])
Max index in X: 50256
Shape of wte: torch.Size([50257, 768])
Shape of wpe: torch.Size([1024, 768])
Shape of h: torch.Size([1, 1, 768])
x shape at the start of attn(): torch.Size([1, 1, 768])
c shape: torch.Size([1, 1, 2304])
q shape: torch.Size([1, 12, 1, 64])
k shape: torch.Size([1, 12, 1, 64])
v shape: torch.Size([1, 12, 1, 64])
q shape in multihead_attn: torch.Size([1, 12, 1, 64])
k.mT shape in multihead_attn: torch.Size([1, 12, 64, 1])
w shape in multihead_attn: torch.Size([1, 12, 1, 1])
x shape at the start of attn(): torch.Size([1, 1, 768])
c shape: torch.Size([1, 1, 2304])
q shape: torch.Size([1, 12, 1, 64])
k shape: torch.Size([1, 12, 1, 64])
v shape: torch.Size([1, 12, 1, 64])
q shape in multihead_attn: torch.Size([1, 12, 1, 64])
k.mT shape in multihead_attn: torch.Size([1, 12, 64, 1])
w shape in multihead_attn: torch.Size([1, 12, 1, 1])
```
</details>

The model returned logits shaped `[1, 1, 50257]` and new cache values shaped `[1, 2, 2, 12, 1, 64]`.

- Row 0, highest raw `(token ID, logit)` pairs: `[(50256, 13.27157), (38530, 2.239048), (22469, 2.195926), (42136, 2.163901), (25711, 2.157997)]`

#### Choose the next token

Use the last position's logits and divide by temperature `1`.
Top-k filter: keep up to 40 highest-scoring IDs.
Top-p filter: off.

- Row 0, highest `(token ID, probability)` pairs: `[(50256, 0.999555), (38530, 1.6e-05), (22469, 1.5e-05), (42136, 1.5e-05), (25711, 1.5e-05)]`

Sampled IDs: `[50256]`.  
Cache after this step: shape `[1, 2, 2, 12, 1, 64]`; accumulated IDs: `[[50256, 50256]]`.

**Final token IDs:** `[[50256, 50256]]`.

### Decode the generated IDs

Drop the start ID, then map the generated IDs back to text.

#### Sample 1

```text
<|endoftext|>
```

## Last state

Completed samples: `1`.

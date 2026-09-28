# GPT-2, worked through from numbers

Open a Markdown preview of this file on the right and the linked source file on the left. Follow the sections in order; each **Read in code** link points to the function that implements that step. The numbered example uses deliberately tiny, invented embeddings so you can check the arithmetic by hand. The real 124M configuration is much larger, and this repository does not load pretrained weights.

| Step | Read in code | What to watch |
| --- | --- | --- |
| 1. Text → IDs | [Encoder.encode and Encoder.decode](encoder.py#L129) | Byte-level BPE and the reverse mapping |
| 2. IDs → vectors | [positions_for and model](model.py#L308) | Shapes `[B,T]` → `[B,T,D]` |
| 3. Normalize | [norm](model.py#L52) | One mean and variance per token |
| 4. Attend | [attn](model.py#L143) | Q, K, V; heads; causal mask |
| 5. Complete a block | [block](model.py#L263) | Attention and MLP residuals |
| 6. Score the vocabulary | [model output](model.py#L402) | Final norm and tied embedding matrix |
| 7. Generate | [sample_sequence](sample.py#L70) | Last-position logits, filters, cache |

Suppose a tokenizer has turned a two-token prompt into **X = [[1, 2]]**. The outer brackets are one batch row; the inner values are token IDs. We will use a tiny, invented model so every vector fits on the page:

| Quantity | Value | Meaning |
| --- | ---: | --- |
| Vocabulary size V | 3 | IDs 0, 1, and 2 exist |
| Context length C | 4 | Positions 0 through 3 exist |
| Embedding width D | 2 | Each token has two features |
| Attention heads H | 1 | One head uses both features |
| Transformer blocks L | 1 | The hidden states pass through one block |

For the arithmetic, imagine these embedding tables in place of the random values drawn by **model.py**:

~~~text
wte: token ID → vector       wpe: position → vector
0 → [0, 0]                   0 → [0, 0]
1 → [1, 0]                   1 → [0, 0]
2 → [0, 1]                   2 → [0, 0]
                             3 → [0, 0]
~~~

The model will take the two IDs, compute a three-number score for the next ID at each position, and use the **last** set of scores to continue the prompt. Here is the whole route; we will calculate each stage below.

~~~mermaid
flowchart LR
    A["IDs [1, 2]"] --> B["Token + position<br/>embeddings"]
    B --> C["Transformer block:<br/>attention, then MLP"]
    C --> D["Final normalization"]
    D --> E["Vocabulary scores<br/>(logits)"]
    E --> F["Choose next ID"]
    F --> G["Decode IDs to text"]
~~~

## 1. From text to token IDs

**Read in code:** [Encoder.encode and Encoder.decode](encoder.py#L129).

A language model predicts *tokens*, which can be whole words, word pieces, spaces, or punctuation. A token ID is an integer indexing the vocabulary; ID 2 does not mean “twice” ID 1. An embedding *vector* is just a list of numbers used to represent an ID during calculation.

In [encoder.py](encoder.py), **Encoder.encode** splits text with a regular expression, converts each piece to UTF-8 bytes, then maps those bytes to reversible Unicode symbols. Its byte pair encoding (**bpe**) repeatedly merges the adjacent pair with the best (lowest) rank in **vocab.bpe**. **encoder.json** maps the remaining groups to IDs. **decode** reverses the token lookup and byte mapping.

For example, with an *invented* merge table, “word” could start as (“w”, “o”, “r”, “d”) and merge (“o”, “r”) to become (“w”, “or”, “d”). The real downloaded tables determine its actual IDs. Our **[1, 2]** is simply an assumed tokenizer output; its two IDs are not tied to particular English words.

## 2. Add position information

**Read in code:** [positions_for](model.py#L308), then [model](model.py#L332).

The same token can mean something different at different points in a sequence. GPT-2 therefore adds a *position vector* to each token vector.

The forward function is **model(hparams, X, past=None)**. Here **hparams** contains the sizes in our opening table; **X** is a two-dimensional tensor of token IDs, with one row per sequence; and **past** holds cached attention data from earlier calls, or is **None** on the first call. In our example, **X = [[1, 2]]** has one row (**B=1**) and two IDs in that row (**T=2**).

The helper is defined as **positions_for(tokens, past_length)**. Its first argument supplies the *shape* of the current token batch: it uses the number of rows and the number of tokens per row, not the ID values. Its second argument is how many tokens precede this batch in the cache. On the first call **past=None**, so **model()** sets **past_length=0**. Substituting our values makes the call and result:

~~~text
positions_for(tokens=X, past_length=0)
tokens = X = [[1, 2]]     # one row, two tokens
past_length = 0           # no cached tokens precede them

arange(2) = [0, 1]        # positions within this call
0 + [0, 1] = [0, 1]       # offset by the number of earlier tokens
repeat for 1 row → [[0, 1]]
~~~

If two tokens had already been cached, the *same* X shape with **past_length=2** would produce **[[2, 3]]**. If X had two batch rows, the positions would be repeated for both rows. Thus **positions_for** answers “where do these new tokens sit in each sequence?” rather than “which token IDs are they?”

**model()** creates a token table **wte** of shape [V, D] and a position table **wpe** of shape [C, D]. Indexing **wte[X]** selects a vector by token ID; indexing **wpe[positions]** selects a vector by position. Both lookups have shape [B, T, D]. GPT-2 adds them feature by feature to give each position one starting vector that carries both *what* token is there and *where* it is. In a trained GPT-2, the embedding tables and later weights are learned together so these combined vectors become useful for prediction. For our first call, the addition is:

~~~text
h = wte[X] + wpe[positions_for(X, 0)]

token 1 at position 0: [1, 0] + [0, 0] = [1, 0]
token 2 at position 1: [0, 1] + [0, 0] = [0, 1]

h = [[[1, 0], [0, 1]]]                  shape [batch=1, tokens=2, features=2]
~~~

The result **h** contains one vector for each input token. **model()** passes it to the first transformer block; section 3 follows the first operation on those vectors, **norm(h)**.

We set the toy position vectors to zero to keep the sums simple. Normally they give positions distinct representations. In this port, both embedding tables are drawn randomly **inside every call** to **model()**, rather than loaded as trained parameters.

In general, X is [B, T] and the resulting hidden state **h** is [B, T, D].

## 3. Normalize before the attention layer

**Read in code:** [norm](model.py#L52), then [block](model.py#L263).

A transformer block in [model.py](model.py) receives the current hidden state as **x**; in our first block, **x = h** from step 2. The block begins with **norm(x)**. For each token independently, it subtracts the mean of that token's D features and divides by the square root of their variance plus ε = 1e-5. A scale of ones and bias of zeros follow the normalization.

Normalization keeps the size of the features going into attention and the MLP more consistent across tokens and layers. This helps the projections and residual connections behave predictably during training; in this code, it also determines the values passed to each sublayer.

For our first vector [1, 0], the mean is 0.5 and the variance is 0.25:

~~~text
norm([1, 0]) = ([1, 0] − 0.5) / sqrt(0.25 + 0.00001)
             ≈ [0.99998, −0.99998]

norm([0, 1]) ≈ [−0.99998, 0.99998]
~~~

The normalized features of each token sum to zero. That fact will explain the actual output of this port in step 5. The scale and bias here are fixed tensors created by **norm**, not learned parameters.

## 4. Understand causal self-attention

**Read in code:** [attn](model.py#L143), especially [the causal mask](model.py#L184) and [Q/K/V projection](model.py#L220).

*Self-attention* lets each position gather information from positions in the same sequence. Each token is projected into three vectors:

- A **query** asks what information this position needs.
- A **key** describes what a position offers.
- A **value** is the information carried forward.

The **x** passed to **attn** is the normalized hidden state from step 3, shaped [B, T, D]. **attn** calls **conv1d(x, 3D)**: the second argument asks for 3D output features, enough to split evenly into q, k, and v. In our example D=2, so the call requests 6 output features, split into three 2-feature vectors per token. Despite the name, this **conv1d** is a matrix multiplication across features. For H heads, each [B, T, D] tensor is reshaped to [B, H, T, d], where d = D/H. Different heads can compare different feature groups.

For each head, a query–key *dot product* multiplies matching features and adds the products to make one score. Dividing by √d keeps scores from growing too large as d grows. A causal mask hides future positions, so a prediction cannot peek at the token it is meant to predict:

~~~text
scores = q @ kᵀ / sqrt(d)

                   key at position
                   0       1
query at 0         ✓       hidden
query at 1         ✓       ✓
~~~

Softmax turns allowed scores into attention weights. It assigns each allowed score sᵢ the weight exp(sᵢ) / Σⱼ exp(sⱼ), so each query row sums to 1. Multiplying those weights by values combines information from visible positions. The code subtracts the row maximum before exponentiating for numerical stability; this leaves the probabilities unchanged.

**Numerical attention example.** To see the mechanism without the port's constant projections, imagine that q₀ = k₀ = [1, 0], q₁ = k₁ = [0, 1], v₀ = [2, 0], and v₁ = [0, 4]. These are *hypothetical* projections, separate from our running forward pass. For query 1, d = 2, so the scores are **[0, 1/√2] ≈ [0, 0.707]**. Both keys are visible. A standard softmax gives approximately **[0.330, 0.670]**, yielding **0.330[2,0] + 0.670[0,4] = [0.660, 2.680]**. Query 0 can see only key 0, so its output is [2, 0].

After attention, **merge_heads** returns to [B, T, D], and another **conv1d** projects back to D features.

## 5. Follow what this implementation actually computes

**Read in code:** [conv1d](model.py#L88), [mlp](model.py#L248), and [block](model.py#L263).

Return to our running **X = [[1, 2]]**. The input to the attention projection is the two normalized vectors from step 3. **conv1d** fills every weight with **0.02** and every bias with **0**. Every output feature is therefore:

~~~text
0.02 × sum(input features) = 0.02 × 0 = 0
~~~

Consequently q, k, and v are all approximately zero, with shape **[1, 1, 2, 2]**. The mask still applies, but combining zero values gives a zero attention result. The output projection is zero too.

A *residual connection* adds the layer's input back to its output. It lets information continue through a block even if a sublayer contributes little. Here the first addition is **x + attention = h + 0 = h**.

The block then normalizes again and runs an MLP: one projection expands D to 4D, **gelu** applies a smooth nonlinear activation, and another projection reduces 4D back to D. Its normalized input again sums to zero; the constant weights produce zeros; **gelu(0) = 0**. The second residual addition is **h + 0 = h**. Thus this one block returns approximately:

~~~text
h = [[[1, 0], [0, 1]]]    shape [1, 2, 2]
~~~

The same reasoning applies to every block in this port: the combination of feature normalization and identical projection columns makes attention and MLP outputs approximately zero. A trained GPT-2 has distinct learned weights, so its blocks can change the hidden states. The code still follows the architecture's data flow, but this behavior is essential when interpreting its output.

## 6. Convert hidden states into next-token scores

**Read in code:** [final normalization and logits](model.py#L402).

After the last block, **model()** normalizes **h** once more. It multiplies each token vector by the transpose of **wte**. Reusing the input token table for output scores is called *weight tying*:

~~~text
final vectors ≈ [[ 0.99998, −0.99998],
                 [−0.99998,  0.99998]]

wteᵀ = [[0, 1, 0],
        [0, 0, 1]]

logits ≈ [[[0,  0.99998, −0.99998],
           [0, −0.99998,  0.99998]]]   shape [1, 2, 3]
~~~

A *logit* is a score, not a probability. The first row scores the token after ID 1; the second scores the token after the full prompt [1, 2]. To continue that prompt, generation uses only the second row: **ID 0 → 0**, **ID 1 → −0.99998**, **ID 2 → 0.99998**. Softmax turns those scores into probabilities, so higher logits make an ID more likely.

## 7. Sample a token, then repeat

**Read in code:** [filters and generation loop](sample.py#L12), then [sample_model](generate.py#L15).

[sample.py](sample.py) takes the last-position logits, divides by **temperature**, applies **top_k_logits** and **top_p_logits**, then uses **torch.multinomial** to choose an ID. Temperature changes how strongly score differences matter; top-k keeps only the k highest scores; top-p aims to keep a set whose probability mass reaches p.

For our last row, **top_k=1** changes the logits to approximately **[−1e10, −1e10, 0.99998]**. Softmax runs across the vocabulary axis, so ID 2 has essentially all the probability. With **top_p < 1**, the code sorts IDs by probability and keeps the smallest prefix whose cumulative mass reaches or exceeds p; for example, probabilities [0.6, 0.3, 0.1] with p=0.8 retain the first two IDs.

Once an ID is chosen, **sample_sequence** appends it and runs the model again. Its **length** argument counts *new* IDs: with a one-ID start and length 2, the returned tensor contains three IDs; with our two-ID prompt [1, 2] and length 2, it would contain four. [generate.py](generate.py) uses a one-ID start, drops that ID, and calls **Encoder.decode** on the new IDs.

### Why the model returns a cache

**Read in code:** [present and past in attn](model.py#L231), then [the sampling loop](sample.py#L144).

Without a cache, generating the third token would require recomputing keys and values for the first two. Let **P** be the number of earlier tokens already in that cache. **attn** returns the new keys and values as **present**, shaped [B, 2, H, T, d] for one layer; the 2 is key and value. **model** stacks all L layers into [B, L, 2, H, T, d]. On the next call, it supplies each layer with its previous keys and values as **past** and concatenates the new ones along the sequence axis:

~~~mermaid
flowchart LR
    P["Past keys + values<br/>P positions"] --> K["Concatenate along<br/>sequence axis"]
    N["New keys + values<br/>T positions"] --> K
    K --> A["Current queries attend to<br/>P + T positions"]
~~~

**positions_for** starts the new tokens at position P. The causal mask allows a new query to see earlier cached positions and earlier positions in the current call.

For our two-token example, B=1, L=1, H=1, T=2, and d=2, so **present** is **[1, 1, 2, 1, 2, 2]**. If one new ID is passed on the next call, P=2 and T=1. Its position is 2, its queries are [1, 1, 1, 2], and after concatenation its keys are [1, 1, 3, 2]. The attention scores therefore cover three visible positions and have shape [1, 1, 1, 3]. This is the same calculation as before, extended with the cached context.

In this port, each call draws new embeddings while reusing a cache from earlier calls, so the cache is not based on a consistent parameter set.

## Locate the code and run it

| File | Main things to follow |
| --- | --- |
| [encoder.py](encoder.py) | byte mapping, BPE merges, encode and decode |
| [model.py](model.py) | embeddings, normalization, attention, MLP, logits, cache |
| [sample.py](sample.py) | generation loop, temperature, top-k, top-p, sampling |
| [generate.py](generate.py) | configuration, start token, decoding |
| [download_model_config.py](download_model_config.py) | download tokenizer files and 124M hyperparameters |

From the repository root:

~~~bash
uv sync --frozen
uv run python gpt-2/download_model_config.py
uv run python gpt-2/generate.py
~~~

The download provides **encoder.json**, **vocab.bpe**, and **hparams.json**, but no trained weights. The 124M configuration uses V=50,257, C=1,024, D=768, H=12, and L=12. Running `generate.py` saves an annotated `.md` trace in `gpt-2/output/`: initial settings, representative tensor values and shapes at each model stage, each sampling and cache update, and the final decoded text. The terminal shows the generated text and trace path.

**Implementation limit:** **conv1d**, **norm**, **wte**, and **wpe** create values during forward calls, so this port has no persistent trainable parameters or optimizer. The tokenizer and hyperparameters match GPT-2, but the code cannot reproduce pretrained GPT-2 output. Its generated text is useful for tracing the computation, not as a language sample. Because each forward call redraws embeddings, cached keys and values come from a different parameter draw than the current token. A faithful generation implementation would keep one parameter set for every call and load trained weights.

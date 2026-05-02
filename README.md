# Assignment 08: Softmax, Autoregressive RNNs, and NER

**Related to**: Week 9 - RNN, LSTM, GRU, sequence labeling, and generation

---

## Goal

In this assignment you will connect three important ideas:

- how softmax turns model logits into probabilities,
- how an autoregressive RNN predicts the next token step by step,
- how RNNs can also be used for named entity recognition, where we predict one tag per input token.

By the end, you should be able to explain the difference between:

```text
many-to-one:      one sequence -> one label
many-to-many:     one sequence -> one label per token
autoregressive:   previous tokens -> next token -> next token -> ...
```

---

## What You Will Submit

Submit one notebook or Python script that runs end-to-end and includes:

- a softmax calculation and explanation,
- an autoregressive character-level RNN,
- generated text from your model,
- a small NER model or NER experiment,
- answers to the follow-up questions.

---

## Part A: Softmax From Logits to Probabilities

A neural network usually outputs raw scores called logits. Softmax converts these logits into probabilities.

Use this vector:

```python
logits = torch.tensor([2.0, 1.0, 0.1])
```

### A1. Manual Softmax

Compute softmax by hand or with basic tensor operations:

```text
softmax(logit_i) = exp(logit_i) / sum(exp(all logits))
```

Print:

- `exp(logits)`,
- the denominator,
- the final probabilities,
- the sum of the probabilities.

Then compare your result with:

```python
torch.softmax(logits, dim=0)
```

### A2. Numerically Stable Softmax

Now use larger logits:

```python
large_logits = torch.tensor([1000.0, 999.0, 998.0])
```

Implement the stable version:

```python
shifted = large_logits - large_logits.max()
probs = torch.exp(shifted) / torch.exp(shifted).sum()
```

Answer:

1. Why do we subtract the maximum logit?
2. Does subtracting the same number from every logit change the final probabilities?
3. Which class has the highest probability, and why?

### A3. Softmax and Cross-Entropy

For a three-class prediction:

```python
logits = torch.tensor([[2.0, 1.0, 0.1]])
true_class = torch.tensor([0])
```

Compute:

```python
loss = nn.CrossEntropyLoss()(logits, true_class)
```

Answer:

1. Why does `CrossEntropyLoss` expect logits instead of already-softmaxed probabilities?
2. What happens to the loss when the model assigns higher probability to the true class?
3. What happens to the loss when the model is confident but wrong?

---

## Part B: Autoregressive Character RNN

In this part you will train a small model to predict the next character.

Autoregressive modeling means:

```text
previous characters -> predict the next character
```

During training, the correct previous characters are given to the model. During generation, the model must feed its own prediction back as the next input.

### Dataset



Clean the data StateGrants.csv:

- keep only non-empty strings,
- lowercase the text,
- remove rows that are too short or clearly invalid.

Build a character vocabulary with:

```text
<PAD>
<S>
<F>
<UNK>
```

Example:

```python
chars = sorted(set("".join(names)))
char2idx = {"<PAD>": 0, "<S>": 1, "<F>": 2, "<UNK>": 3}
for ch in chars:
    if ch not in char2idx:
        char2idx[ch] = len(char2idx)
idx2char = {i: ch for ch, i in char2idx.items()}
```

### B1. Create Input and Target Sequences

For each name:

```text
input:  <S> + name
target: name + <F>
```

Example:

```text
name:   "anna"
input:  ["<S>", "a", "n", "n", "a"]
target: ["a", "n", "n", "a", "<F>"]
```

Pad input and target sequences in each batch.

Required shapes:

```text
x_batch shape: (batch_size, max_seq_len)
y_batch shape: (batch_size, max_seq_len)
```

### B2. Build the Model

Use an RNN, GRU, or LSTM:

```python
class CharRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.rnn = nn.GRU(embed_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded, hidden)
        logits = self.fc(output)
        return logits, hidden
```

Important shape:

```text
logits shape: (batch_size, max_seq_len, vocab_size)
```

### B3. Train With Next-Character Prediction

Use:

```python
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
```

Flatten predictions and targets:

```python
logits, _ = model(x_batch)
loss = criterion(
    logits.reshape(-1, vocab_size),
    y_batch.reshape(-1)
)
```

Train for 10-30 epochs, or fewer if your dataset is very small.

Report:

- training loss per epoch,
- validation loss or validation next-character accuracy,
- one example showing input characters, target characters, and predicted characters.

---

## Part C: Generate Text Autoregressively

Implement:

```python
generate(start_text="", max_len=30, temperature=1.0)
```

The function should:

1. Start with `<S>`.
2. Feed characters one by one into the RNN.
3. If `start_text` is provided, use it as a prefix.
4. Convert the final-step logits to probabilities using softmax.
5. Sample the next character from the probability distribution.
6. Append the sampled character to the output.
7. Stop when the model outputs `<F>` or reaches `max_len`.

Temperature:

```python
probs = torch.softmax(logits / temperature, dim=-1)
```

Generate at least 5 examples for each setting:

- greedy decoding, using `argmax`,
- sampling with `temperature=0.5`,
- sampling with `temperature=1.0`,
- sampling with `temperature=1.5`.

Use at least three prefixes, for example:

```python
"a", "ai", "nur"
```

Answer:

1. What changes when temperature increases?
2. Why does greedy decoding often repeat safe or common characters?
3. What kinds of mistakes does the model make when it feeds its own predictions back into itself?
4. How is generation different from training with the correct previous characters?

---

## Part D: Try Named Entity Recognition

Now try a many-to-many RNN task.

NER input:

```text
Apple opened an office in Almaty
```

NER output:

```text
B-ORG O O O O B-LOC
```

### Dataset

Use one of these options:

1. a small instructor-provided NER dataset,
2. a manually created dataset of 100-300 sentences,
3. a public CoNLL-style NER dataset from Hugging Face (recommended: `eriktks/conll2003`).

Use BIO tags:

```text
B-PER = beginning of person name
I-PER = inside person name
B-LOC = beginning of location
I-LOC = inside location
B-ORG = beginning of organization
I-ORG = inside organization
O     = outside any named entity
```

Example:

```python
tokens = ["Apple", "opened", "an", "office", "in", "Almaty"]
tags   = ["B-ORG", "O",      "O",  "O",      "O",  "B-LOC"]
```

### D1. Prepare the Data

Build:

- `word2idx`,
- `tag2idx`,
- `idx2tag`.

Reserve:

```python
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_TAG = "<PAD>"
```

Convert tokens and tags to integer IDs.

Pad both tokens and tags in each batch.

Required shapes:

```text
x_batch shape:   (batch_size, max_seq_len)
tag_batch shape: (batch_size, max_seq_len)
```

### D2. Build a Token Tagger

Use a bidirectional LSTM or GRU:

```python
class BiRNNNER(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_tags, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.rnn = nn.LSTM(
            embed_dim,
            hidden_size,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_size * 2, num_tags)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.rnn(embedded)
        logits = self.fc(output)
        return logits
```

Important shape:

```text
logits shape: (batch_size, max_seq_len, num_tags)
```

Train with:

```python
criterion = nn.CrossEntropyLoss(ignore_index=pad_tag_id)
```

Flatten predictions and targets:

```python
logits = model(x_batch)
loss = criterion(
    logits.reshape(-1, num_tags),
    tag_batch.reshape(-1)
)
```

Report:

- training loss,
- validation token accuracy ignoring padding,
- at least 5 validation sentences with token, true tag, and predicted tag.

Example output:

```text
Apple      B-ORG   B-ORG
opened     O       O
an         O       O
office     O       O
in         O       O
Almaty     B-LOC   B-LOC
```

---

## Part E: Compare the Two RNN Uses

Fill this table:

| Task | Input | Target | Output shape | Uses softmax over | Decoding method |
| --- | --- | --- | --- | --- | --- |
| Character generation | previous characters | next character | | | |
| NER | sentence tokens | tag for each token | | | |

Answer:

1. Why is character generation autoregressive?
2. Why is NER not usually generated autoregressively in this assignment?
3. Why is the final hidden state not enough for NER?
4. Why do both tasks use cross-entropy loss?
5. In both tasks, what does softmax help us interpret?

---

## Bonus

Choose one:

- compare `nn.RNN`, `nn.LSTM`, and `nn.GRU` for character generation,
- plot training loss for both the character model and the NER model,
- compute precision, recall, and F1 score per entity type,
- test NER on sentences about Kazakhstan, universities, companies, or cities,
- save and load the character model using `torch.save`.

---

## Submission

Submit:

- softmax calculation and explanation,
- character vocabulary and sequence preparation code,
- autoregressive RNN model code,
- generation function and generated examples,
- NER preprocessing code,
- NER model code and validation examples,
- comparison table,
- answers to all explanation questions.

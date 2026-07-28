# How a Mixture-of-Experts Model Actually Works

### Every number, worked out by hand

Mixtral 8×7B holds about **46.7 billion** parameters but generates text at roughly the speed of
a **13 billion** parameter model. That is not a compression trick or a quantization trick. It is
an architectural one, and it is called **Mixture of Experts**.

Most explanations of MoE stop at a block diagram with arrows. This post does the opposite. We
will build a complete, working language model — small enough that every single number fits on
the page — and push one sentence through it from raw text to a probability distribution over
the next word. Nothing is hidden, nothing is hand-waved, and every matrix is printed in full.

By the end you will have seen, in actual numbers:

- how a word becomes a vector,
- how attention lets one word read another,
- how a **router** decides which experts a token gets sent to,
- why that decision is what makes a 47B model run like a 13B one,
- and how the whole thing turns into a prediction.


---

## Table of contents

- [Who this post is for](#who-this-post-is-for)
- [The 60-second version](#the-60-second-version)
- [How to read the matrices in this post](#how-to-read-the-matrices-in-this-post)
- [The running example](#the-running-example)

**PART 0 — THE FIVE IDEAS YOU NEED FIRST**

- [0.1 — A language model is a next-word guesser in a loop](#01--a-language-model-is-a-next-word-guesser-in-a-loop)
- [0.2 — Words become vectors, because computers do arithmetic, not language](#02--words-become-vectors-because-computers-do-arithmetic-not-language)
- [0.3 — The residual stream is a shared workspace](#03--the-residual-stream-is-a-shared-workspace)
- [0.4 — "Weights" are just numbers that were learned, then frozen](#04--weights-are-just-numbers-that-were-learned-then-frozen)
- [0.5 — Softmax turns any list of numbers into probabilities](#05--softmax-turns-any-list-of-numbers-into-probabilities)

**PART I — ATTENTION: HOW WORDS READ EACH OTHER**

- [A0. Where attention sits](#a0-where-attention-sits)
- [A1 — Tokenize and embed](#a1--tokenize-and-embed)
- [A2 — RMSNorm](#a2--rmsnorm)
- [A3 — Q, K, V projections](#a3--q-k-v-projections)
- [A4 — Split into heads](#a4--split-into-heads)
- [A5 — RoPE positional rotation](#a5--rope-positional-rotation)
- [A6 — Attention scores](#a6--attention-scores)
- [A7 — Causal mask](#a7--causal-mask)
- [A8 — Softmax over keys](#a8--softmax-over-keys)
- [A9 — Context, concat, output projection, residual](#a9--context-concat-output-projection-residual)
- [✅ What Part I accomplished](#-what-part-i-accomplished)

**PART II — THE MoE SUB-LAYER**

- [M0. The one idea behind MoE](#m0-the-one-idea-behind-moe)
- [M0b. The MoE configuration](#m0b-the-moe-configuration)
- [M1 — RMSNorm](#m1--rmsnorm)
- [M2 — Router logits](#m2--router-logits)
- [M3 — Softmax over experts](#m3--softmax-over-experts)
- [M4 — Top-k selection (k = 2)](#m4--top-k-selection-k--2)
- [M5 — Renormalize the surviving gates](#m5--renormalize-the-surviving-gates)
- [M6 — The expert FFNs](#m6--the-expert-ffns)
- [M7 — Combine: applying G to the expert outputs](#m7--combine-applying-g-to-the-expert-outputs)
- [M8 — Residual add, then RMSNorm](#m8--residual-add-then-rmsnorm)
- [✅ What Part II accomplished](#-what-part-ii-accomplished)

**PART III — PREDICTING THE NEXT TOKEN**

- [P1 — Unembedding to vocabulary logits](#p1--unembedding-to-vocabulary-logits)
- [P2 — Only the last row matters](#p2--only-the-last-row-matters)
- [P3 — Softmax over the vocabulary](#p3--softmax-over-the-vocabulary)
- [P4 — Sampling](#p4--sampling)
- [P5 — Append and repeat: the KV cache](#p5--append-and-repeat-the-kv-cache)

**What MoE buys you**


**What real MoEs change**


**Capacity and dropping**


**Scaling to real Mixtral**


**Glossary**


**The whole model on one page**


---

## Who this post is for

You should be comfortable with **matrix multiplication** — if `(4 × 2) · (2 × 4) = (4 × 4)`
looks normal to you, you have enough.

You do **not** need to have read a transformer paper, know what an embedding is, or have any
prior exposure to attention or MoE. Every ML-specific idea is defined the first time it appears,
and there is a [glossary](#glossary) at the end.

There is no code to run and no framework to install. Just arithmetic.

---

## The 60-second version

If you read nothing else, read this.

A language model is a machine that does one thing over and over: **look at some text, and guess
the next word.** To do that, it converts each word into a list of numbers (a *vector*), pushes
those vectors through a stack of identical processing blocks, and converts the result back into
a probability for every word it knows.

Each block does two jobs, in order:

1. **Attention** — lets each word look at the other words and pull in information. This is the
   only place in the entire model where words communicate.
2. **A feed-forward network (FFN)** — processes each word on its own, applying the model's
   stored knowledge. This is where most of the parameters live.

The problem: the FFN is enormous, and **every single word pays the full cost of it**, every
time.

Mixture of Experts changes exactly one thing. It replaces that one big FFN with several smaller
ones, called **experts**, plus a tiny **router** that looks at each word and picks just two of
them.

The consequence is the whole point of the architecture:

> **Total parameters grow with the number of experts. Compute per token does not.**

You store a huge model but only ever run a small slice of it. That is the trick. Everything
below is the detail.

---

## How to read the matrices in this post

Every intermediate result is printed in full. A quick key so nothing is ambiguous:

**Shapes** are written `(rows × columns)`. So `(4 × 2)` is 4 rows, 2 columns.

**Rows are almost always tokens (words), columns are almost always features.** Our sentence has
4 words, so most matrices have 4 rows labelled `t1`–`t4`. The model's internal vectors are 4
numbers wide, so columns are labelled `d0`–`d3`.

```
              d0         d1         d2         d3     ← feature index
   t1      [  2.000000  -1.000000   0.000000   1.000000 ]   ← the word "the"
   t2      [ -1.000000   2.000000   1.000000   0.000000 ]   ← the word "cat"
   t3      [  0.500000   0.500000  -2.000000   1.000000 ]   ← the word "sat"
   t4      [  1.000000   1.000000   1.000000   1.000000 ]   ← the word "on"
```

**Symbols:**

| Symbol | Means |
|---|---|
| `·` | matrix multiplication |
| `⊙` | elementwise multiplication (multiply matching positions) |
| `ᵀ` | transpose (flip rows and columns) |
| `Σ` | "add up all of these" |
| `−inf` | negative infinity — a value used to switch something off |

**Everything is shown to 6 decimal places.** Not because that precision matters, but so you can
check any step yourself and land on exactly the printed number.

**Weight matrices are round numbers** like `0.5` and `-1.0`. That is artificial — I chose them
so the arithmetic is followable. In a real model every weight is an ugly decimal learned from
data. The one place I kept ugly decimals is the embedding table, because there it is
*pedagogically true*: embeddings really are arbitrary learned values.

---

## The running example

We will push this sentence through the model:

```
   "the cat sat on"
```

and watch it predict what comes next. (Spoiler: it says `"the"`, with `"mat"` in second place.)

Our toy model is a real model in every structural sense — it just has small numbers:

| | Our model | Mixtral 8×7B |
|---|---|---|
| Words in the prompt | 4 | up to 32,768 |
| Vector width (`d_model`) | 4 | 4096 |
| Attention heads | 2 | 32 |
| Blocks stacked | 1 | 32 |
| Experts | 4 | 8 |
| Experts used per token | 2 | 2 |
| Vocabulary | 6 | 32,000 |

Every operation you are about to see is the operation Mixtral performs. Only the sizes differ.

---
---

# PART 0 — THE FIVE IDEAS YOU NEED FIRST

If you already know what embeddings, the residual stream, and softmax are, skip to
[Part I](#part-i--attention-how-words-read-each-other). Otherwise, these five ideas are all the
background required.

## 0.1 — A language model is a next-word guesser in a loop

That is genuinely all it does. Given `"the cat sat on"`, it outputs a probability for every word
it knows:

```
   "the"    72%
   "mat"    23%
   "cat"     2%
   ...
```

Pick one, stick it on the end, and feed the whole thing back in. `"the cat sat on the"` →
predict again → `"mat"`. Repeat until done. Long, fluent paragraphs come from running this loop
hundreds of times.

So our job in this post is to trace **one turn of that loop**, completely.

## 0.2 — Words become vectors, because computers do arithmetic, not language

A model cannot multiply the word "cat". So step one is always to replace each word with a list
of numbers.

This happens in two stages:

1. **Tokenize** — split the text and look up each piece in a fixed dictionary, getting an
   integer. `"cat"` → `1`.
2. **Embed** — use that integer as a row index into a big table of numbers. Row 1 might be
   `[-1.588275, 2.284264, 0.256386, 0.969211]`.

That list of 4 numbers *is* the word, as far as the model is concerned. It is called an
**embedding**, and its width (`d_model = 4` here, 4096 in Mixtral) is fixed for the whole model.

Why a vector and not a single number? Because a vector has a **direction**, and directions can
encode relationships. Words with similar meanings end up pointing in similar directions, which
is what lets the model generalize. The specific numbers are *learned during training* — nobody
writes them by hand, which is exactly why they look random.

> **Beginner note.** You will see the phrase "hidden state" a lot. It just means "the vector
> currently representing this token at this point in the network". It starts as the embedding
> and gets modified as it flows through.

## 0.3 — The residual stream is a shared workspace

Here is the single most useful mental model for reading any transformer.

Picture a **conveyor belt** carrying one vector per token. Every component in the model does the
same three things:

1. **Read** the belt,
2. compute something,
3. **Add** its result back onto the belt.

Nothing ever replaces what is on the belt. Components only ever *add corrections* to it. That
belt is called the **residual stream**, and the additions are called **residual connections**.

```
   Emb ──┬────────────────────────────┬──▶ ... the belt keeps flowing
         │                            │
         ▼                            │
     [ Attention ]                    │
         │                            │
         └───────── + ────────────────┘   ← its output is ADDED, not substituted
```

Two reasons this matters, and both will show up in the numbers:

- **Information is never destroyed.** A token's identity survives all the way from the embedding
  table to the final prediction.
- **It is why deep models train at all.** During training, the learning signal can flow
  backwards through the `+` without passing through any matrix, so it does not fade away over
  dozens of layers.

## 0.4 — "Weights" are just numbers that were learned, then frozen

You will see matrices named `W_Q`, `W_r`, `W1[0]`, and so on. Every one of them is a grid of
numbers that was adjusted during **training** and is now **fixed**. When you chat with a model,
none of these change.

The only thing that varies is the input. Same weights, different sentence, different answer.

So when this post says a matrix is "learned", read it as: *these numbers were tuned over
billions of examples; we are just using them now.*

## 0.5 — Softmax turns any list of numbers into probabilities

Softmax appears **three times** in this post, doing three different jobs, so it is worth 30
seconds now.

Give it any list of numbers, positive or negative. It gives back a list that is all positive and
sums to exactly 1 — in other words, a probability distribution.

```
  softmax(z)ᵢ  =  e^(zᵢ) / Σⱼ e^(zⱼ)
```

In words: exponentiate everything (which makes it positive), then divide by the total (which
makes it sum to 1).

The one behaviour to internalize: **softmax exaggerates differences.** Because it exponentiates
first, a modest gap in the inputs becomes a large ratio in the outputs.

```
  input   [ 3.5,   2.4,   0.2  ]    ← gap of only 1.1 between the top two
  output  [ 0.730, 0.243, 0.027 ]    ← but the winner takes 3× the runner-up
```

Whenever you see a softmax below, ask: *normalized over what axis?* That is the whole meaning of
the operation, and it differs each time:

| Where | Normalized over | Answers the question |
|---|---|---|
| A8, attention | keys (other tokens) | "where should this word look?" |
| M3, router | experts | "which experts should handle this word?" |
| P3, output | vocabulary | "which word comes next?" |

---
---


# PART I — ATTENTION: HOW WORDS READ EACH OTHER

Attention is the only machinery in the entire model that lets one word see another. Everything
else — including the whole MoE half of this post — processes each word in total isolation.

Nine steps. We start from raw text and finish with an updated conveyor belt.

## A0. Where attention sits

A Mixtral-style block is two sub-layers, each wrapped in a pre-norm and a residual:

```
  ids → Embed
          │
          ├──────────────────────────┐  residual
          ▼                          │
      RMSNorm → Multi-Head Attn ──▶ (+) ──▶  X
                                     │
          ┌──────────────────────────┘
          ├──────────────────────────┐  residual
          ▼                          │
      RMSNorm → MoE Feed-Forward ──▶(+) ──▶ X + Y
                                     │
                                     ▼
                              RMSNorm → Unembed → probs
```

The two sub-layers divide the labour cleanly:

- **Attention moves information *between* positions.** It is the only operation in the whole
  model where token 4 can read token 1. It mixes across the sequence but applies the *same*
  transformation to every position.
- **The FFN (here, the MoE) processes each position *independently*.** It never looks sideways.
  It is where the model's stored knowledge lives.

MoE only ever replaces the second one. Attention in a Mixtral block is byte-for-byte identical
to attention in a dense Llama block — which is why the walkthrough below has nothing
MoE-specific in it.

**Config for this example**

| Symbol | Meaning | Value |
|---|---|---|
| `T` | tokens in the prompt | 4 |
| `d_model` | residual stream width | 4 |
| `n_heads` | attention heads | 2 |
| `d_head` | width per head | 2 |
| `V` | vocabulary size | 6 |
| RoPE base | `θ₀` | 1.0 |
| masking | | causal |

Real Mixtral uses `d_head = 128` and RoPE base 1,000,000. With `d_head = 2` there is exactly
**one** rotation pair per head, which makes RoPE a single 2-D rotation you can picture — that's
the whole reason for the small numbers.

---

## A1 — Tokenize and embed

**In plain English:** turn the words into numbers by looking them up in a table.

This is the very first thing any language model does, and it is nothing more than a lookup. The
model keeps one big table with one row per word it knows. To embed a word, you find its row and
copy it out. No arithmetic at all.

The prompt is split into tokens and each is looked up in the embedding table.

```
  "the cat sat on"  →  ids [0, 1, 2, 3]  →  positions m = 0,1,2,3
```

```
  Embedding table (V=6 × d_model=4)
              d0         d1         d2         d3
   0 "the" [  1.920781  -1.184273  -1.343296   2.527569 ]
   1 "cat" [ -1.588275   2.284264   0.256386   0.969211 ]
   2 "sat" [  0.821115  -0.095264  -2.565433   1.748479 ]
   3 "on"  [  0.364150   0.581870   1.120565   1.920648 ]
   4 "mat" [  0.500000  -1.500000   1.000000   0.500000 ]
   5 "eos" [ -2.000000   0.500000   0.500000  -1.000000 ]
```

**Why these look arbitrary.** Embeddings are *learned lookup values* — there is no formula
behind them, so ugly decimals are exactly right. (The weight matrices later in this document
are hand-picked round numbers purely so you can follow the arithmetic; in a real model every
one of them would look like the table above.)

Gathering rows 0–3 gives the initial residual stream:

```
  Emb (4 × 4)
              d0         d1         d2         d3
   t1 "the" [  1.920781  -1.184273  -1.343296   2.527569 ]
   t2 "cat" [ -1.588275   2.284264   0.256386   0.969211 ]
   t3 "sat" [  0.821115  -0.095264  -2.565433   1.748479 ]
   t4 "on"  [  0.364150   0.581870   1.120565   1.920648 ]
```

Note what is *not* here: no position information at all. Row t1 would be identical if "the"
appeared at the end of the sentence. Position enters in step A5, not here.

---

## A2 — RMSNorm

**In plain English:** rescale each token's vector so they are all roughly the same size.

Some tokens naturally end up with big numbers and some with small ones. That is a problem,
because the next step compares tokens using dot products, and a token with large numbers would
win every comparison just for being loud rather than relevant.

Normalization fixes this by shrinking or stretching each row until they are all a comparable
length. It keeps the *direction* of each vector (which carries the meaning) and discards the
*magnitude* (which carries noise).

> **Why "RMS"?** RMS stands for root-mean-square — square the numbers, average them, take the
> square root. It is a measure of "typical size" for a list of numbers. Dividing by it makes the
> typical size equal to 1.

> **Pre-norm vs post-norm.** Notice we normalize a *copy* and feed that to the sub-layer, while
> the original un-normalized vector stays on the conveyor belt to be added back later. That is
> called a **pre-norm** architecture, and every modern model uses it because it trains more
> stably.

**Formula:** `h = x / sqrt(mean(x²) + ε) ⊙ g`, gain `g = [1,1,1,1]`.

Same operation, same reasoning as before the MoE: attention compares tokens via dot products,
and dot products are scale-sensitive. Normalizing first means a token's *direction* drives the
comparison, not how large its vector happens to have grown. This is a **pre-norm** architecture:
the norm feeds the sub-layer, and the un-normalized `Emb` is what gets added back at the end.

```
  H_pre (4 × 4)
              d0         d1         d2         d3
   t1      [  1.053969  -0.649833  -0.737092   1.386925 ]
   t2      [ -1.074141   1.544834   0.173392   0.655472 ]
   t3      [  0.511156  -0.059304  -1.597020   1.088454 ]
   t4      [  0.312955   0.500066   0.963027   1.650628 ]
```

---

## A3 — Q, K, V projections

**In plain English:** each word produces three different vectors, so that words can search each
other like a database.

This is the heart of attention and the part most worth slowing down for. Think of every word in
the sentence as simultaneously being a **searcher** and a **filing cabinet**:

| Vector | Role | Think of it as |
|---|---|---|
| **Query** (`Q`) | what this word is looking for | the search box you type into |
| **Key** (`K`) | what this word advertises about itself | the label on a filing cabinet |
| **Value** (`V`) | what this word will actually give you | the contents of that cabinet |

The mechanism is then exactly what you would expect from a search engine: compare every query
against every key to get a relevance score, then use those scores to decide how much of each
value to take.

Splitting into three separate vectors matters. A word can advertise something different from
what it is looking for — the word "sat" might advertise "I am a verb" while searching for "who
is my subject?" One vector could not express both.

Each of `Q`, `K`, `V` is made the same way: multiply the normalized input by a learned weight
matrix.

**Formula:** `Q = H_pre · W_Q`, `K = H_pre · W_K`, `V = H_pre · W_V` — three independent
`(4×4)·(4×4)` products.

**The intuition that makes attention click.** Each token emits three different vectors:

- **Query** — "here is what I am looking for."
- **Key** — "here is what I offer to anyone looking."
- **Value** — "here is what I will actually hand over if you pick me."

Attention then matches every query against every key, and the match scores decide how much of
each value to take. Q and K only ever meet inside a dot product; V is the payload.

```
  W_Q (4 × 4)                W_K (4 × 4)
      [  1.0  0.0  0.0  0.5 ]    [  0.0  1.0  0.5  0.0 ]
      [  0.0  1.0  0.5  0.0 ]    [  1.0  0.0  0.0  0.5 ]
      [  0.0  0.5  1.0  0.0 ]    [  0.5  0.0  0.0  1.0 ]
      [  0.5  0.0  0.0  1.0 ]    [  0.0  0.5  1.0  0.0 ]

  W_V (4 × 4)                W_O (4 × 4)
      [  1.0  0.0 -0.5  0.0 ]    [  0.5  0.0  0.5  0.0 ]
      [  0.0  1.0  0.0 -0.5 ]    [  0.0  0.5  0.0  0.5 ]
      [ -0.5  0.0  1.0  0.0 ]    [  0.5  0.0 -0.5  0.0 ]
      [  0.0 -0.5  0.0  1.0 ]    [  0.0  0.5  0.0 -0.5 ]
```

```
  Q (4 × 4)
              q0         q1         q2         q3
   t1      [  1.747432  -1.018379  -1.062009   1.913910 ]
   t2      [ -0.746405   1.631531   0.945810   0.118401 ]
   t3      [  1.055383  -0.857813  -1.626671   1.344031 ]
   t4      [  1.138269   0.981579   1.213060   1.807105 ]

  K (4 × 4)
              k0         k1         k2         k3
   t1      [ -1.018379   1.747432   1.913910  -1.062009 ]
   t2      [  1.631531  -0.746405   0.118401   0.945810 ]
   t3      [ -0.857813   1.055383   1.344031  -1.626671 ]
   t4      [  0.981579   1.138269   1.807105   1.213060 ]

  V (4 × 4)
              v0         v1         v2         v3
   t1      [  1.422515  -1.343296  -1.264077   1.711842 ]
   t2      [ -1.160837   1.217099   0.710463  -0.116946 ]
   t3      [  1.309666  -0.603530  -1.852598   1.118105 ]
   t4      [ -0.168558  -0.325248   0.806549   1.400595 ]
```

---

## A4 — Split into heads

**In plain English:** run several independent attention operations side by side, each on a slice
of the vector.

Here is the problem one attention operation has. In a moment we will make each word distribute a
fixed budget of attention across the other words — a budget that must add up to 1. So if a word
spends most of its attention on the subject of the sentence, it has almost nothing left for the
verb. It can only track **one** relationship at a time.

The fix is cheap: chop the vector into `n_heads` pieces and run the whole attention machinery
separately on each piece. Each **head** gets its own budget, so each can track a different
relationship — one following grammatical structure, another tracking what a pronoun refers to,
and so on.

The cost is zero, because the pieces are smaller in exactly the proportion that there are more
of them. Two heads of width 2 is the same total work as one head of width 4.

This step itself is pure bookkeeping — no arithmetic happens, we are just deciding which columns
belong to which head.

**Formula:** slice the last axis into `n_heads` contiguous chunks of width `d_head`.
Head `h` takes columns `[h·d_head : (h+1)·d_head]`. No arithmetic — pure reshaping.

```
  head 0 ← columns 0,1        head 1 ← columns 2,3
```

**Why bother?** One head can only express one similarity pattern per position — its softmax
row must sum to 1, so attending strongly to the subject means attending weakly to everything
else. Splitting the same budget into `n_heads` independent low-dimensional subspaces lets the
model run several relations in parallel (one head tracking syntax, another tracking the
antecedent of a pronoun, and so on) at zero extra cost, since `n_heads · d_head = d_model`.

```
  Q head 0 (4×2)        Q head 1 (4×2)
   t1 [  1.747432 -1.018379 ]   t1 [ -1.062009  1.913910 ]
   t2 [ -0.746405  1.631531 ]   t2 [  0.945810  0.118401 ]
   t3 [  1.055383 -0.857813 ]   t3 [ -1.626671  1.344031 ]
   t4 [  1.138269  0.981579 ]   t4 [  1.213060  1.807105 ]

  K head 0 (4×2)        K head 1 (4×2)
   t1 [ -1.018379  1.747432 ]   t1 [  1.913910 -1.062009 ]
   t2 [  1.631531 -0.746405 ]   t2 [  0.118401  0.945810 ]
   t3 [ -0.857813  1.055383 ]   t3 [  1.344031 -1.626671 ]
   t4 [  0.981579  1.138269 ]   t4 [  1.807105  1.213060 ]

  V head 0 (4×2)        V head 1 (4×2)
   t1 [  1.422515 -1.343296 ]   t1 [ -1.264077  1.711842 ]
   t2 [ -1.160837  1.217099 ]   t2 [  0.710463 -0.116946 ]
   t3 [  1.309666 -0.603530 ]   t3 [ -1.852598  1.118105 ]
   t4 [ -0.168558 -0.325248 ]   t4 [  0.806549  1.400595 ]
```

From here, heads 0 and 1 proceed **completely independently** through steps A5–A9 and only
meet again at the concatenation.

---

## A5 — RoPE positional rotation

**In plain English:** stamp each word with its position by rotating its vector — further along
the sentence means a bigger turn.

Look back at the embeddings in A1. Nothing in them says *where* in the sentence a word sits. The
row for "the" is the same row whether it is the first word or the last. But position obviously
matters: "the cat sat on the mat" and "the mat sat on the cat" use identical words.

**RoPE** (Rotary Position Embedding) injects that missing information. Instead of adding a
position vector, it *rotates* each token's query and key vectors by an angle proportional to
their position. Word 0 is not turned at all, word 1 is turned a little, word 2 more, and so on.

That sounds like an odd choice until you see the payoff: when two rotated vectors are compared,
their absolute rotations cancel and only the *difference* survives. The model ends up sensing
**how far apart** two words are rather than where each one sits — which is usually what actually
matters, and which lets the model handle sentences longer than any it saw during training.

> This is the most mathematically involved step in the post. If the trigonometry is unfamiliar,
> it is safe to take the "after RoPE" matrices on faith and continue — nothing later depends on
> deriving them yourself.

### What the operation actually is

RoPE takes each row of `Q` and `K` and **rotates it**, by an angle that depends on that row's
position in the sequence:

```
  q'_m = R_m · q_m        k'_m = R_m · k_m        v_m  unchanged
```

**`R_m` is not a weight matrix.** This is the point that most often trips people up. `W_Q`,
`W_K`, `W_V`, `W_O` are learned, and each is applied identically to every row. `R_m` is the
opposite on both counts: it is **computed from the position index `m`** by a fixed formula with
no parameters, and it is **different for every row**. So RoPE is not `Q · R`; it is a per-row
operation, row `m` getting its own `R_m`.

Three steps: pair up the dimensions, build `R_m`, apply it row by row.

---

### Step 1 — Pair up the dimensions and assign each pair a frequency

RoPE treats a `d_head`-dimensional vector as `d_head / 2` independent 2-D planes and spins each
plane at its own rate. Pair `i` gets frequency

```
  θ_i = base^( −2i / d_head )        for i = 0 … d_head/2 − 1
```

For a realistic head (`d_head = 8`, `base = 10000`) that table would read:

```
  pair (d0,d1)   θ_0 = 10000^(−0/8) = 1.000000    ← fast: turns a full radian per token
  pair (d2,d3)   θ_1 = 10000^(−2/8) = 0.100000
  pair (d4,d5)   θ_2 = 10000^(−4/8) = 0.010000
  pair (d6,d7)   θ_3 = 10000^(−6/8) = 0.001000    ← slow: barely moves over 1000 tokens
```

Fast pairs resolve nearby tokens precisely; slow pairs stay nearly still over long spans and so
carry coarse, long-range position. Together they act like the hands of a clock at different
speeds — the full set of angles pins down the position unambiguously.

**In our example `d_head = 2`**, so there is exactly **one** pair, `(d0, d1)`, and
`θ_0 = base^(−0/2) = base⁰ = 1` — the base cancels entirely. One plane, one frequency, angle
equal to the position index in radians:

```
  position m → angle  m · θ_0 = m radians
```

*(A note on conventions: this document pairs adjacent dimensions, `(d0,d1)`, `(d2,d3)`, … The
HuggingFace Llama/Mixtral implementation instead pairs `(d0, d0+d_head/2)` via its `rotate_half`
helper. The two are related by a permutation of columns and are mathematically equivalent — but
weights are not interchangeable between them.)*

---

### Step 2 — Build `R_m` for each position

For one pair, `R_m` is the standard 2-D rotation matrix by angle `m·θ`:

```
             ┌                          ┐
     R_m  =  │  cos(mθ)     −sin(mθ)    │
             │  sin(mθ)      cos(mθ)    │
             └                          ┘
```

For `d_head > 2` you stack one of these per pair down the diagonal, giving a block-diagonal
`(d_head × d_head)` matrix — still no parameters, still fixed by `m`.

Our four positions, with `θ = 1`, so the angle is `m · θ = m` radians. Each entry below is
literally the cosine or sine *of that angle* — nothing is being looked up or learned:

```
   m    angle = m·θ        cos(m·θ)                sin(m·θ)
  ───────────────────────────────────────────────────────────────
   0    0 rad  (  0.0°)    cos(0) =  1.000000      sin(0) =  0.000000
   1    1 rad  ( 57.3°)    cos(1) =  0.540302      sin(1) =  0.841471
   2    2 rad  (114.6°)    cos(2) = -0.416147      sin(2) =  0.909297
   3    3 rad  (171.9°)    cos(3) = -0.989992      sin(3) =  0.141120
```

**These are radians, not degrees** — the usual source of confusion here. That is why `cos`
turns negative from `m = 2` onward: 2 radians is about 114.6°, already past the quarter turn.
By `m = 3` (171.9°, nearly half a turn) the vector points almost exactly backwards, which is
why `cos(3) ≈ −0.99` and `sin(3) ≈ 0.14`.

```
  R_0 = [  1.000000   0.000000 ]     R_1 = [  0.540302  -0.841471 ]
        [  0.000000   1.000000 ]           [  0.841471   0.540302 ]

  R_2 = [ -0.416147  -0.909297 ]     R_3 = [ -0.989992  -0.141120 ]
        [  0.909297  -0.416147 ]           [  0.141120  -0.989992 ]
```

`R_0` is the identity — position 0 is not rotated at all, which is why row t1 always comes out
unchanged.

---

### Step 3 — Apply it, row by row

Written out, `R_m · z` is:

```
  z'₀ = z₀·cos(mθ) − z₁·sin(mθ)
  z'₁ = z₀·sin(mθ) + z₁·cos(mθ)
```

Here is **every substitution** for `Q head 0`, whose unrotated rows were
`t1 [1.747432, −1.018379]`, `t2 [−0.746405, 1.631531]`, `t3 [1.055383, −0.857813]`,
`t4 [1.138269, 0.981579]`:

```
  t1  (m=0, cos 1.000000, sin 0.000000)
    z₀' = 1.747432(1.000000) − (−1.018379)(0.000000)
        =  1.747432 + 0.000000                        =  1.747432
    z₁' = 1.747432(0.000000) + (−1.018379)(1.000000)
        =  0.000000 − 1.018379                        = −1.018379

  t2  (m=1, cos 0.540302, sin 0.841471)
    z₀' = (−0.746405)(0.540302) − (1.631531)(0.841471)
        = −0.403284 − 1.372886                        = −1.776170
    z₁' = (−0.746405)(0.841471) + (1.631531)(0.540302)
        = −0.628078 + 0.881520                        =  0.253442

  t3  (m=2, cos −0.416147, sin 0.909297)
    z₀' = (1.055383)(−0.416147) − (−0.857813)(0.909297)
        = −0.439194 + 0.780007                        =  0.340813
    z₁' = (1.055383)(0.909297) + (−0.857813)(−0.416147)
        =  0.959657 + 0.356976                        =  1.316633

  t4  (m=3, cos −0.989992, sin 0.141120)
    z₀' = (1.138269)(−0.989992) − (0.981579)(0.141120)
        = −1.126878 − 0.138521                        = −1.265398
    z₁' = (1.138269)(0.141120) + (0.981579)(−0.989992)
        =  0.160633 − 0.971756                        = −0.811124
```

The same four-line recipe runs on `K head 0`, `Q head 1`, and `K head 1`, giving:

```
  Q head 0 AFTER RoPE (4×2)     K head 0 AFTER RoPE (4×2)
   t1 [  1.747432 -1.018379 ]    t1 [ -1.018379  1.747432 ]
   t2 [ -1.776170  0.253442 ]    t2 [  1.509598  0.969601 ]
   t3 [  0.340813  1.316633 ]    t3 [ -0.602680 -1.219202 ]
   t4 [ -1.265398 -0.811124 ]    t4 [ -1.132389 -0.988357 ]

  Q head 1 AFTER RoPE (4×2)     K head 1 AFTER RoPE (4×2)
   t1 [ -1.062009  1.913910 ]    t1 [  1.913910 -1.062009 ]
   t2 [  0.411392  0.859844 ]    t2 [ -0.731899  0.610654 ]
   t3 [ -0.545190 -2.038443 ]    t3 [  0.919814  1.899058 ]
   t4 [ -1.455939 -1.617834 ]    t4 [ -1.960208 -0.945901 ]
```

**Lengths are preserved**, because rotations preserve norm — RoPE injects position without
changing magnitude:

```
                |q| before RoPE    |q| after RoPE
   t1              2.022527           2.022527
   t2              1.794161           1.794161
   t3              1.360028           1.360028
   t4              1.503048           1.503048
```

---

### V is not touched at all

`V` passes through this step byte for byte:

```
  V head 0, before AND after RoPE (4×2)   ← identical
   t1 [  1.422515 -1.343296 ]
   t2 [ -1.160837  1.217099 ]
   t3 [  1.309666 -0.603530 ]
   t4 [ -0.168558 -0.325248 ]
```

**Why.** `Q` and `K` exist only to be *compared* — they meet inside a dot product, and that is
where a rotation can encode relative distance. `V` is the *payload* that gets retrieved once the
comparison is done. Rotating it would spin the content itself by an amount depending on where
the token happened to sit, corrupting what gets copied into the residual stream. Position
belongs in the matching, not in the cargo.

---

## A6 — Attention scores

**In plain English:** score how strongly every word wants to read from every other word.

This is the comparison step. Every query meets every key, and the result is a grid of relevance
scores — one number for each (asking word, answering word) pair.

The comparison itself is a **dot product**: multiply the two vectors position by position and
add up the results. It is worth knowing why that measures relevance. A dot product is large and
positive when two vectors point the same way, near zero when they are unrelated, and negative
when they point opposite ways. So "does this query match this key?" becomes "do these two
vectors point in a similar direction?"

The output is a square grid, `T × T` — every word against every word.

**Formula:** `S_h = Q_h · K_hᵀ / √d_head`, computed **separately and independently for each
head**. Two heads means two score matrices, and they never interact.

### Which `Q` and `K`, and what shape

This document has used the names `Q` and `K` for two different things, so to be exact — the
matrices entering A6 are the **per-head slices, after RoPE**, from A5. Not the full `(4×4)`
projections from A3:

| | Object | Shape | Source |
|---|---|---|---|
| ✗ not this | `Q`, `K` full | `(T × d_model)` = `(4 × 4)` | A3 |
| ✓ this | `Q_h`, `K_h` rotated | `(T × d_head)` = `(4 × 2)` | A5 |

```
   Q_h          ·        K_hᵀ         =        S_h
 (T × d_head)        (d_head × T)          (T × T)
   (4 × 2)      ·      (2 × 4)        =      (4 × 4)
                          ↑
                 d_head is CONTRACTED AWAY
```

Note what this means: **`d_head` vanishes from the output shape.** The score matrix is `(T × T)`
no matter how wide the head is — 2 here, 128 in Mixtral. Every head produces a same-shaped
`(T × T)` attention pattern; the head width only controls how much room the model has to
*express* that pattern, not its size.

The scaling is `√d_head`, **not** `√d_model` — `√2 = 1.414214` here, `√128 = 11.31` in Mixtral.

**Why divide by it at all.** The dot product of two `d_head`-dimensional vectors has variance
proportional to `d_head`, so without the scaling the scores would grow with head width, the
softmax would saturate into a near-one-hot, and gradients would vanish.

### The two score matrices

```
  S head 0 (4 × 4)          key→   t1         t2         t3         t4
   query t1              [ -2.516661   1.167078   0.133266  -0.687484 ]
   query t2              [  1.592183  -1.722205   0.538438   1.245091 ]
   query t3              [  1.381438   1.266499  -1.280318  -1.193057 ]
   query t4              [ -0.091025  -1.906861   1.238536   1.580103 ]

  S head 1 (4 × 4)          key→   t1         t2         t3         t4
   query t1              [ -2.874514   1.376044   1.879332   0.191900 ]
   query t2              [ -0.088950   0.158371   1.422202  -1.145330 ]
   query t3              [  0.792949  -0.598042  -3.091892   2.119094 ]
   query t4              [ -0.755460   0.054916  -3.119439   3.100136 ]
```

---

## A7 — Causal mask

**In plain English:** stop each word from peeking at words that come after it.

The model is trained to predict the next word. If, while processing "sat", it were allowed to
look ahead at "on", the task would be trivial and it would learn nothing useful — it would just
copy the answer.

So before we turn the scores into attention weights, we switch off every score that points
forward in the sentence. The switch-off value is negative infinity, which will become exactly
zero in the next step.

The result is a triangular pattern: word 1 sees only itself, word 2 sees words 1–2, and so on.
This is what "causal" or "autoregressive" means, and it is why these models can generate text
one word at a time.

**Formula:** set `S[i,j] = −∞` for all `j > i`.

**Why.** The model is trained to predict token `i+1` from tokens `1…i`. If position 2 could see
position 3, it would be reading the answer. Masking with `−∞` (rather than deleting) keeps the
matrix rectangular so it stays one GEMM, and `exp(−∞) = 0` makes those keys contribute exactly
zero after the softmax.

```
  S masked, head 0 (4 × 4)
                        t1         t2         t3         t4
   t1                [ -2.516661       -inf       -inf       -inf ]
   t2                [  1.592183  -1.722205       -inf       -inf ]
   t3                [  1.381438   1.266499  -1.280318       -inf ]
   t4                [ -0.091025  -1.906861   1.238536   1.580103 ]

  S masked, head 1 (4 × 4)
                        t1         t2         t3         t4
   t1                [ -2.874514       -inf       -inf       -inf ]
   t2                [ -0.088950   0.158371       -inf       -inf ]
   t3                [  0.792949  -0.598042  -3.091892       -inf ]
   t4                [ -0.755460   0.054916  -3.119439   3.100136 ]
```

Everything is now lower-triangular. Row t1 has a single live entry: the first token can only
ever attend to itself.

---

## A8 — Softmax over keys

**In plain English:** turn the raw scores into percentages that say how much attention to pay
where.

Raw scores can be any size, positive or negative — you cannot take a weighted average with
those. Softmax converts each row into a set of weights that are all positive and sum to exactly
1, which is precisely what a weighted average needs.

**Watch the axis.** We normalize across the *keys* — across the row. Each word gets one unit of
attention to spend and must divide it among the words it is allowed to look at. Spending more on
one means spending less on another.

Every score we set to `−inf` in A7 becomes `e^(−inf) = 0`, so blocked positions get exactly zero
weight. That is why `−inf` was the right switch-off value.

**Formula:** `A[i,j] = exp(S[i,j]) / Σ_j exp(S[i,j])`, taken **along the key axis** (each row
sums to 1). Note the axis: this normalizes over *where to look*, so each query spends exactly
one unit of attention distributed across the positions it is allowed to see.

```
  exp(S masked), head 0                                  row sum
   t1  [  0.080729   0.000000   0.000000   0.000000 ]    0.080729
   t2  [  4.914464   0.178672   0.000000   0.000000 ]    5.093136
   t3  [  3.980623   3.548408   0.277949   0.000000 ]    7.806980
   t4  [  0.912995   0.148546   3.450557   4.855454 ]    9.367553

  A head 0 (4 × 4)   ← attention weights, rows sum to 1
                        t1         t2         t3         t4
   t1                [  1.000000   0.000000   0.000000   0.000000 ]
   t2                [  0.964919   0.035081   0.000000   0.000000 ]
   t3                [  0.509880   0.454517   0.035603   0.000000 ]
   t4                [  0.097464   0.015857   0.368352   0.518327 ]
```

```
  exp(S masked), head 1                                  row sum
   t1  [  0.056444   0.000000   0.000000   0.000000 ]    0.056444
   t2  [  0.914891   1.171600   0.000000   0.000000 ]    2.086492
   t3  [  2.209903   0.549887   0.045416   0.000000 ]    2.805206
   t4  [  0.469794   1.056452   0.044182  22.200960 ]   23.771389

  A head 1 (4 × 4)   ← attention weights, rows sum to 1
                        t1         t2         t3         t4
   t1                [  1.000000   0.000000   0.000000   0.000000 ]
   t2                [  0.438483   0.561517   0.000000   0.000000 ]
   t3                [  0.787786   0.196024   0.016190   0.000000 ]
   t4                [  0.019763   0.044442   0.001859   0.933936 ]
```

**Read the two heads side by side at t4.** Head 0 splits its attention between itself (0.518)
and "sat" (0.368). Head 1 is almost entirely self-focused (0.934). This is exactly the
specialization multi-head buys: one head gathering context, one holding the current token.
Row t1 is `[1,0,0,0]` in both — forced, not learned, since there is nothing else to look at.

---

## A9 — Context, concat, output projection, residual

**In plain English:** actually fetch the information, stitch the heads back together, and add
the result onto the conveyor belt.

Three things happen here and it is worth naming them separately:

1. **Fetch.** Multiply the attention weights by the value vectors. Because each row of weights
   sums to 1, each output row is a *weighted average* of the values — a blend of the other
   words, mixed according to how much attention was paid to each.
2. **Stitch and mix.** Glue the heads' outputs back into one full-width vector, then multiply by
   `W_O`. That second part matters more than it looks: without it, each head's findings would be
   stuck in its own slice of the vector, unable to influence the others.
3. **Add.** Put the result back on the residual stream, on top of the original embeddings.

After this step, attention is completely finished. Every word now carries information gathered
from the words before it.

**Step 1 — Context.** `C_h = A_h · V_h`, shapes `(4×4)·(4×2) = (4×2)`. Each output row is a
weighted average of the value vectors, with weights from the attention row.

```
  C head 0 (4 × 2)          C head 1 (4 × 2)
   t1 [  1.422515 -1.343296 ]   t1 [ -1.264077  1.711842 ]
   t2 [  1.331889 -1.253475 ]   t2 [ -0.155339  0.684947 ]
   t3 [  0.244319 -0.153214 ]   t3 [ -0.886548  1.343743 ]
   t4 [  0.515285 -0.502519 ]   t4 [  0.756415  1.338778 ]
```

Row t1 of `C head 0` equals row t1 of `V head 0` exactly — with attention `[1,0,0,0]` the
weighted average is just the one value vector it was allowed to see.

**Step 2 — Concatenate the heads** back to full width, `(4×2) ‖ (4×2) → (4×4)`:

```
  C (4 × 4)
              c0         c1         c2         c3
   t1      [  1.422515  -1.343296  -1.264077   1.711842 ]
   t2      [  1.331889  -1.253475  -0.155339   0.684947 ]
   t3      [  0.244319  -0.153214  -0.886548   1.343743 ]
   t4      [  0.515285  -0.502519   0.756415   1.338778 ]
```

**Step 3 — Output projection.** `AttnOut = C · W_O`, shape `(4×4)`.

Concatenation alone would leave the two heads in separate, non-interacting slices of the vector.
`W_O` is what lets the model *blend* them — every output dimension can draw on both heads.

```
  AttnOut (4 × 4)
              d0         d1         d2         d3
   t1      [  0.079219   0.184273   1.343296  -1.527569 ]
   t2      [  0.588275  -0.284264   0.743614  -0.969211 ]
   t3      [ -0.321115   0.595264   0.565433  -0.748479 ]
   t4      [  0.635850   0.418130  -0.120565  -0.920648 ]
```

**Step 4 — Residual.** `X = Emb + AttnOut`, adding the **un-normalized** embeddings:

```
  Emb (4 × 4)
   t1 [  1.920781  -1.184273  -1.343296   2.527569 ]
   t2 [ -1.588275   2.284264   0.256386   0.969211 ]
   t3 [  0.821115  -0.095264  -2.565433   1.748479 ]
   t4 [  0.364150   0.581870   1.120565   1.920648 ]

                          +

  AttnOut (4 × 4)
   t1 [  0.079219   0.184273   1.343296  -1.527569 ]
   t2 [  0.588275  -0.284264   0.743614  -0.969211 ]
   t3 [ -0.321115   0.595264   0.565433  -0.748479 ]
   t4 [  0.635850   0.418130  -0.120565  -0.920648 ]

                          =

  X (4 × 4)   ← input to the MoE sub-layer
              d0         d1         d2         d3
   t1      [  2.000000  -1.000000   0.000000   1.000000 ]
   t2      [ -1.000000   2.000000   1.000000   0.000000 ]
   t3      [  0.500000   0.500000  -2.000000   1.000000 ]
   t4      [  1.000000   1.000000   1.000000   1.000000 ]
```

**Why the residual is not optional.** The `+` means attention only ever writes a *correction*
onto a stream that already carries the token's identity. Gradients reach the embeddings through
the `+` without passing through any matrix, which is what makes deep stacks trainable. Think of
the residual stream as a shared bus that each sub-layer reads from and adds to.

This `X` is exactly the input assumed by Part II.


---

## ✅ What Part I accomplished

Worth pausing to consolidate before the MoE half.

We started with four independent word vectors that knew nothing about each other or about their
positions. We finished with four vectors that each carry information gathered from the words
before them.

The chain was: **look up embeddings → normalize → project to Q, K, V → split into heads → rotate
for position → score every pair → mask the future → softmax into weights → fetch values → mix
heads → add to the residual stream.**

Three things to carry forward:

1. **`X` is the output**, and it is the input to everything in Part II. Attention's job is done.
2. **The `(T × T)` score grid is the expensive part.** It grows with the *square* of the sentence
   length, which is why long context windows are hard and why so much research targets this one
   matrix.
3. **Nothing here is MoE-specific.** This is byte-for-byte the attention in a dense Llama model.
   Mixtral changes what comes next, not this.

---
---

---
---

# PART II — THE MoE SUB-LAYER

The residual stream `X` produced by A9 now enters the second sub-layer. Attention has finished
moving information between positions; from here every token is processed **independently**.

```
  X (4 × 4)   ← carried over from A9
              d0         d1         d2         d3
   t1      [  2.000000  -1.000000   0.000000   1.000000 ]
   t2      [ -1.000000   2.000000   1.000000   0.000000 ]
   t3      [  0.500000   0.500000  -2.000000   1.000000 ]
   t4      [  1.000000   1.000000   1.000000   1.000000 ]
```

## M0. The one idea behind MoE

In the dense block of A0, the second sub-layer is a single FFN. It holds most of the parameters
(~2/3 of a dense Transformer), and **every token pays for every one of them**. MoE makes one
change: replace that single FFN with `E` parallel FFNs ("experts") plus a tiny **router** that
picks `k` of them per token.

```
                      ┌→ Expert 0 ┐
 X → RMSNorm → Router ┼→ Expert 1 ┼→ weighted sum → (+X) → out
      (picks k of E)  ├→ Expert 2 ┤
                      └→ Expert 3 ┘
```

**Total parameters scale with `E`; compute per token scales with `k`.** A big model that runs
like a small one. Part I is untouched by any of this — attention in a Mixtral block is identical
to attention in a dense Llama block.

Routing is decided **per token, not per sequence**. Two tokens in the same sentence can go to
entirely different experts.

---

## M0b. The MoE configuration

Extending the config from A0:

| Symbol | Meaning | Value |
|---|---|---|
| `E` | number of experts | 4 |
| `k` | experts per token (top-k) | 2 |
| `d_ff` | expert hidden width | 4 |
| activation | | ReLU |
| biases | | none |

**Router weight** `W_r`, shape `(d_model, E)`. Rows are input dims, **columns are experts**:

```
  W_r (4 × 4)
          e0     e1     e2     e3
    d0 [  1.0    0.0    0.0   -0.5  ]
    d1 [  0.0    1.0    0.5    0.0  ]
    d2 [  0.0    0.5   -1.0    0.0  ]
    d3 [  0.5    0.0    0.0    1.0  ]
```

**Expert weights.** Each expert `e` has `W1[e]` of shape `(d_model, d_ff)` and `W2[e]` of shape
`(d_ff, d_model)`. Each expert's pair is restated inside its own section below so you never have
to scroll back. In a real model `d_ff ≈ 4·d_model`; here they're equal only for readability.

**Parameter inventory for this sub-layer:** 4 experts × (16 + 16) = **128 expert params**, + 16
router params = **144 total**. (Attention adds a further 64 in `W_Q`, `W_K`, `W_V`, `W_O`, and
the unembedding another 24 — but the MoE sub-layer alone outweighs both, which is the usual
ratio at any scale.)

---

## M1 — RMSNorm

**In plain English:** same rescaling as A2, done again before the second sub-layer.

Every sub-layer in the model gets its own fresh normalization of the conveyor belt. Attention
just added something to it, so the numbers have grown and need re-scaling before the router
compares them.

Note carefully which vector goes where: the **normalized** version feeds the router and the
experts, while the **original un-normalized `X`** waits to be added back at the end.

**Formula:** `h = x / sqrt(mean(x²) + ε) ⊙ g`, gain `g = [1,1,1,1]`.

**Why:** the router is a linear map into a softmax, and softmax is scale-sensitive — doubling a
token's magnitude doubles all its logits and sharpens routing for no good reason. Normalizing
first makes the router decide on the token's *direction* (what it's about), not its *magnitude*
(how loud it is). Note that the normalized `H` feeds the experts, while the **un-normalized `X`**
is what we add back at the residual.

```
  H (4 × 4)
          d0         d1         d2         d3
    t1 [  1.632993  -0.816497   0.000000   0.816497 ]
    t2 [ -0.816497   1.632993   0.816497   0.000000 ]
    t3 [  0.426401   0.426401  -1.705606   0.852803 ]
    t4 [  1.000000   1.000000   1.000000   1.000000 ]
```

Token 4 is unchanged because its RMS was already exactly 1 — normalization is a per-row rescale,
nothing more.

---

## M2 — Router logits

**In plain English:** the router scores how well each token matches each expert.

This is the one genuinely new component in an MoE model, and it is startlingly small: a single
matrix with one column per expert.

Each column is a learned "profile" of the kind of token that expert handles well. To score a
token against an expert, take the dot product of the token with that expert's column — the same
similarity measure used in attention. A high score means "this expert is a good match for this
token."

Notice how cheap this is. The router here is 16 numbers deciding how to spend 128 numbers' worth
of expert compute. In Mixtral the router is about 32,000 parameters deciding the fate of 45
billion. **The decision is essentially free; the thing it controls is not.** That asymmetry is
the entire economic case for MoE.

**Formula:** `L = H · W_r`, shapes `(4×4) · (4×4) = (4×4)`. Entry `L[t,e]` is the affinity of
token `t` for expert `e`.

```
  H (4 × 4)
          d0         d1         d2         d3
    t1 [  1.632993  -0.816497   0.000000   0.816497 ]
    t2 [ -0.816497   1.632993   0.816497   0.000000 ]
    t3 [  0.426401   0.426401  -1.705606   0.852803 ]
    t4 [  1.000000   1.000000   1.000000   1.000000 ]

                        ·

  W_r (4 × 4)
          e0     e1     e2     e3
    d0 [  1.0    0.0    0.0   -0.5  ]
    d1 [  0.0    1.0    0.5    0.0  ]
    d2 [  0.0    0.5   -1.0    0.0  ]
    d3 [  0.5    0.0    0.0    1.0  ]

                        =

  L (4 × 4)
          e0         e1         e2         e3
    t1 [  2.041241  -0.816497  -0.408248   0.000000 ]
    t2 [ -0.816497   2.041241   0.000000   0.408248 ]
    t3 [  0.852803  -0.426401   1.918806   0.639602 ]
    t4 [  1.500000   1.500000  -0.500000   0.500000 ]
```

**Reading it.** The router is *one matrix* — 16 numbers versus 128 in the experts. It is
deliberately the cheapest thing in the layer. Column `e` of `W_r` is a learned "prototype
direction"; the logit is the inner product of the token with that prototype. t1 aligns with e0
(2.04), t2 with e1 (2.04) — the mirror symmetry we predicted. t3 lights up e2 (1.92). t4 is
perfectly ambiguous between e0 and e1.

---

## M3 — Softmax over experts

**In plain English:** turn the router's raw scores into percentages across the experts.

Second appearance of softmax, new axis. In attention we normalized across other tokens; here we
normalize across **experts**. Each row of the result answers "how should this token's processing
be split among the four experts?"

Two reasons we need probabilities rather than raw scores. First, we are about to blend expert
outputs together, and blending weights should be positive and sum to 1 so the result does not
blow up. Second — and this is the subtle one — softmax is *smooth*, which is what allows the
router to be trained at all. If we simply picked the highest-scoring expert with no probabilities
involved, there would be no gradient and the router could never improve.

**Formula:** `P[t,e] = exp(L[t,e]) / Σⱼ exp(L[t,j])`, taken **along the expert axis**.

**Why:** we need weights that are positive, form a convex combination (so output magnitude stays
controlled), and are *differentiable* — this is the only path by which gradients reach `W_r`. A
hard argmax would give the router no learning signal at all.

```
  exp(L)                                        row sum
          e0         e1         e2         e3
    t1 [  7.700163   0.441977   0.664814   1.000000 ]   9.806954
    t2 [  0.441977   7.700163   1.000000   1.504181 ]  10.646321
    t3 [  2.346214   0.652854   6.812822   1.895727 ]  11.707617
    t4 [  4.481689   4.481689   0.606531   1.648721 ]  11.218630
```

Divide each row by its own sum:

```
  P (4 × 4)      ← each row sums to 1.000000
          e0         e1         e2         e3
    t1 [  0.785174   0.045068   0.067790   0.101968 ]
    t2 [  0.041515   0.723270   0.093929   0.141286 ]
    t3 [  0.200401   0.055763   0.581914   0.161922 ]
    t4 [  0.399486   0.399486   0.054065   0.146963 ]
```

Check t1: `7.700163 / 9.806954 = 0.785174` ✓. Note that a logit gap of 2.86 becomes a probability
ratio of 17×. Softmax exponentiates differences — small logit changes cause large routing
changes, which is why MoE training is unstable and why balancing losses exist.

---

## M4 — Top-k selection (k = 2)

**In plain English:** keep only the top 2 experts per token and throw the rest away.

**This single step is what makes the model sparse. Everything else is ordinary neural network
machinery.**

The router just produced a probability for all 4 experts. We now discard all but the largest 2
for each token. Those discarded experts will not run for that token, will not consume any
compute, and will not contribute anything to the output.

Here that saves 50%. In Mixtral (2 of 8) it saves 75%. In DeepSeek-V3 (8 of 256) it saves about
97%. **The fraction you throw away is the speedup.**

The cost of this trick is that the choice is *discrete* — an expert is either in or out, with
nothing in between. That makes the model's behaviour jump abruptly when a token sits near a
decision boundary, as one of ours does.

Keep the 2 largest entries of each row of `P`:

```
  t1:  e0 = 0.785174 ,  e3 = 0.101968    (e1, e2 discarded)
  t2:  e1 = 0.723270 ,  e3 = 0.141286    (e0, e2 discarded)
  t3:  e2 = 0.581914 ,  e0 = 0.200401    (e1, e3 discarded)
  t4:  e0 = 0.399486 ,  e1 = 0.399486    ← exact tie
```

**This is the sparsity.** 8 of 16 (token, expert) pairs are now dead — 50% here, 75% in Mixtral
(8 experts, top-2), ~97% in DeepSeek-V3 (256 routed, top-8). That discarded fraction *is* the
efficiency gain.

The tie at t4 is real, not contrived — it happens whenever a token sits on a decision boundary.
Frameworks break it deterministically by lower index. Note the **discontinuity**: an
infinitesimal change to `X[4]` would flip t4 from e1 to e3 and change the output by a finite
amount. Top-k is not continuous in its input, and this is an accepted wart of the architecture.

---

## M5 — Renormalize the surviving gates

**In plain English:** rescale the two surviving weights so they add to 1 again.

When we deleted the losing experts we also deleted their share of the probability. Token t1's
kept experts add up to 0.887 rather than 1.0, and token t3's only reach 0.782 — each token lost a
different amount.

If we left it there, every token's output would be scaled by a different arbitrary factor
depending on how much probability happened to leak away. So we divide by what remains, restoring
a clean split that sums to 1.

The result is the **gate matrix `G`**, which is worth watching for the rest of Part II. It is
mostly zeros — one zero for every (token, expert) pair we are skipping — and those zeros are the
sparsity, made concrete.

**Formula:** `ĝ[t,e] = P[t,e] / Σ_{j ∈ top-k(t)} P[t,j]` for selected `e`, else 0.

**Why:** without it, t1's gates would sum to 0.887 and t3's to 0.782 — every token's FFN
contribution would be scaled by an arbitrary factor determined by how much probability leaked to
unselected experts. Renormalizing makes each token's mixture a proper convex combination.

```
  G (4 × 4)      ← sparse gate matrix, rows sum to 1.000000
          e0         e1         e2         e3
    t1 [  0.885060   0.000000   0.000000   0.114940 ]
    t2 [  0.000000   0.836579   0.000000   0.163421 ]
    t3 [  0.256164   0.000000   0.743836   0.000000 ]
    t4 [  0.500000   0.500000   0.000000   0.000000 ]
```

Reading `G` **column-wise** inverts the mapping and tells each expert which rows of `H` it must
process:

```
  Expert 0 ← t1, t3, t4      (3 tokens)
  Expert 1 ← t2, t4          (2 tokens)
  Expert 2 ← t3              (1 token)
  Expert 3 ← t1, t2          (2 tokens)
                             ───────────
                             8 = T × k dispatch slots
```

---

## M6 — The expert FFNs

**In plain English:** each expert is an ordinary small neural network, and it only sees the
tokens routed to it.

Here is the thing that surprises most people: **there is nothing special about an expert.** It
is a bog-standard two-layer feed-forward network, exactly what a dense model would have. The
word "expert" describes its *role*, not its architecture.

Each one does three things:

```
    widen  ─▶  filter  ─▶  narrow
    × W1       ReLU        × W2
```

- **Widen** (`× W1`) projects into a bigger internal space where features can be teased apart.
- **Filter** (`ReLU`) sets every negative number to zero. Without this the two matrices would
  collapse into one and the whole layer could only do linear algebra.
- **Narrow** (`× W2`) projects back to the original width so the result fits on the conveyor
  belt.

The only unusual part is the *input*: instead of all 4 tokens, each expert receives only its
assigned rows. Read the gate matrix `G` down its columns and you get the assignment list. This is
why the four expert sections below have different heights — expert 0 got 3 tokens, expert 2 got
just 1.

> Watch for expert 2. It is about to demonstrate something genuinely counter-intuitive.

Every expert runs the same three-stage pipeline. Only the weights and the set of input rows
differ:

```
  H_e  ──[× W1[e]]──▶  A  ──[ReLU]──▶  Z  ──[× W2[e]]──▶  O_e
 (n×4)                (n×4)           (n×4)              (n×4)
       expand into              kill negative      project back
       "feature" space          features           to model space
```

Two things to hold on to:

- **`n` varies per expert** (3, 2, 1, 2 here) because the router decides how many rows each one
  gets. This raggedness is what makes MoE kernels awkward to write.
- **The width never varies.** `H_e` is always `(n × d_model)` and `W1[e]` is always
  `(d_model × d_ff)`, so the inner dimension always matches. The router changes *how many* rows
  an expert sees, never *how wide* they are.

Column labels: `d0…d3` are model dimensions, `f0…f3` are the expert's internal feature units.

---

### ▸ EXPERT 0 — tokens {t1, t3, t4} — shapes (3×4) → (3×4)

**Stage 0: gather the assigned rows of `H`**

```
  H₀ (3 × 4)
          d0         d1         d2         d3
    t1 [  1.632993  -0.816497   0.000000   0.816497 ]
    t3 [  0.426401   0.426401  -1.705606   0.852803 ]
    t4 [  1.000000   1.000000   1.000000   1.000000 ]
```

**Stage 1: up-projection, `A = H₀ · W1[0]`**

```
  W1[0] (4 × 4)
          f0     f1     f2     f3
    d0 [  1.0    0.0    0.5    0.0  ]
    d1 [  0.0    1.0    0.0    0.5  ]
    d2 [  0.5    0.0   -1.0    0.0  ]
    d3 [  0.0   -0.5    0.0    1.0  ]

  A (3 × 4)
          f0         f1         f2         f3
    t1 [  1.632993  -1.224745   0.816497   0.408248 ]
    t3 [ -0.426401   0.000000   1.918806   1.066004 ]
    t4 [  1.500000   0.500000  -0.500000   1.500000 ]
```

**Stage 2: `Z = ReLU(A)`** — clamp negatives to zero. This is the layer's only nonlinearity;
without it `W1` and `W2` would collapse into a single matrix.

```
  Z (3 × 4)
          f0         f1         f2         f3
    t1 [  1.632993   0.000000   0.816497   0.408248 ]
    t3 [  0.000000   0.000000   1.918806   1.066004 ]
    t4 [  1.500000   0.500000   0.000000   1.500000 ]
```

**Stage 3: down-projection, `O₀ = Z · W2[0]`**

```
  W2[0] (4 × 4)
          d0     d1     d2     d3
    f0 [  1.0    0.0    0.0    0.5  ]
    f1 [  0.0    1.0    0.5    0.0  ]
    f2 [  0.5    0.0    1.0    0.0  ]
    f3 [  0.0    0.5    0.0    1.0  ]

  O₀ (3 × 4)   ← OUTPUT
          d0         d1         d2         d3
    t1 [  2.041241   0.204124   0.816497   1.224745 ]
    t3 [  0.959403   0.533002   1.918806   1.066004 ]
    t4 [  1.500000   1.250000   0.250000   2.250000 ]
```

*Takeaway:* three tokens in, three out, width preserved. This is just an ordinary MLP — the only
"expert" thing about it is which rows it saw.

---

### ▸ EXPERT 1 — tokens {t2, t4} — shapes (2×4) → (2×4)

**Stage 0**

```
  H₁ (2 × 4)
          d0         d1         d2         d3
    t2 [ -0.816497   1.632993   0.816497   0.000000 ]
    t4 [  1.000000   1.000000   1.000000   1.000000 ]
```

**Stage 1: `A = H₁ · W1[1]`**

```
  W1[1] (4 × 4)
          f0     f1     f2     f3
    d0 [  0.0    1.0    0.0   -0.5  ]
    d1 [  1.0    0.0   -0.5    0.0  ]
    d2 [  0.0    0.5    1.0    0.0  ]
    d3 [  0.5    0.0    0.0    1.0  ]

  A (2 × 4)
          f0         f1         f2         f3
    t2 [  1.632993  -0.408248   0.000000   0.408248 ]
    t4 [  1.500000   1.500000   0.500000   0.500000 ]
```

**Stage 2: `Z = ReLU(A)`**

```
  Z (2 × 4)
          f0         f1         f2         f3
    t2 [  1.632993   0.000000   0.000000   0.408248 ]
    t4 [  1.500000   1.500000   0.500000   0.500000 ]
```

**Stage 3: `O₁ = Z · W2[1]`**

```
  W2[1] (4 × 4)
          d0     d1     d2     d3
    f0 [  0.5    0.0    1.0    0.0  ]
    f1 [  0.0    0.5    0.0    1.0  ]
    f2 [  1.0    0.0    0.5    0.0  ]
    f3 [  0.0    1.0    0.0    0.5  ]

  O₁ (2 × 4)   ← OUTPUT
          d0         d1         d2         d3
    t2 [  0.816497   0.408248   1.632993   0.204124 ]
    t4 [  1.250000   1.250000   1.750000   1.750000 ]
```

*Takeaway:* t4 appears here **and** in expert 0 — it was routed to both, and will receive a 50/50
blend at combine time.

---

### ▸ EXPERT 2 — token {t3} — shapes (1×4) → (1×4)

**Stage 0**

```
  H₂ (1 × 4)
          d0         d1         d2         d3
    t3 [  0.426401   0.426401  -1.705606   0.852803 ]
```

**Stage 1: `A = H₂ · W1[2]`**

```
  W1[2] (4 × 4)
          f0     f1     f2     f3
    d0 [ -1.0    0.5    0.0    0.0  ]
    d1 [  0.5   -1.0    0.0    0.0  ]
    d2 [  0.0    0.0    1.0    0.5  ]
    d3 [  0.0    0.0    0.5    1.0  ]

  A (1 × 4)
          f0         f1         f2         f3
    t3 [ -0.213201  -0.213201  -1.279204   0.000000 ]
```

**Stage 2: `Z = ReLU(A)`** — every entry was ≤ 0, so the entire hidden layer dies:

```
  Z (1 × 4)
          f0         f1         f2         f3
    t3 [  0.000000   0.000000   0.000000   0.000000 ]
```

**Stage 3: `O₂ = Z · W2[2]`** — zero times anything is zero:

```
  O₂ (1 × 4)   ← OUTPUT, entirely zero
          d0         d1         d2         d3
    t3 [  0.000000   0.000000   0.000000   0.000000 ]
```

*Takeaway — worth pausing on.* Expert 2 was t3's **highest-confidence** choice (gate 0.7438) and
it contributes **nothing**. Being routed to an expert guarantees compute, not contribution. This
"dead expert" pattern is one reason modern MoEs prefer SwiGLU/GELU over ReLU: smooth activations
never hard-zero an entire block, so gradient keeps flowing.

---

### ▸ EXPERT 3 — tokens {t1, t2} — shapes (2×4) → (2×4)

**Stage 0**

```
  H₃ (2 × 4)
          d0         d1         d2         d3
    t1 [  1.632993  -0.816497   0.000000   0.816497 ]
    t2 [ -0.816497   1.632993   0.816497   0.000000 ]
```

**Stage 1: `A = H₃ · W1[3]`**

```
  W1[3] (4 × 4)
          f0     f1     f2     f3
    d0 [  0.5    0.5    0.5    0.5  ]
    d1 [ -0.5    0.5   -0.5    0.5  ]
    d2 [  0.5   -0.5    0.5   -0.5  ]
    d3 [  0.5    0.5   -0.5   -0.5  ]

  A (2 × 4)
          f0         f1         f2         f3
    t1 [  1.632993   0.816497   0.816497   0.000000 ]
    t2 [ -0.816497   0.000000  -0.816497   0.000000 ]
```

**Stage 2: `Z = ReLU(A)`** — t2's row dies entirely:

```
  Z (2 × 4)
          f0         f1         f2         f3
    t1 [  1.632993   0.816497   0.816497   0.000000 ]
    t2 [  0.000000   0.000000   0.000000   0.000000 ]
```

**Stage 3: `O₃ = Z · W2[3]`**

```
  W2[3] (4 × 4)
          d0     d1     d2     d3
    f0 [  1.0    0.0   -1.0    0.0  ]
    f1 [  0.0    1.0    0.0   -1.0  ]
    f2 [ -1.0    0.0    1.0    0.0  ]
    f3 [  0.0   -1.0    0.0    1.0  ]

  O₃ (2 × 4)   ← OUTPUT
          d0         d1         d2         d3
    t1 [  0.816497   0.816497  -0.816497  -0.816497 ]
    t2 [  0.000000   0.000000   0.000000   0.000000 ]
```

*Takeaway:* dead activations are per-row, not per-expert. t1 got a real output from the same
weights that gave t2 nothing.

---

### M6 summary

| Expert | Tokens | `H_e` shape | `O_e` shape | Non-zero rows out |
|---|---|---|---|---|
| 0 | t1, t3, t4 | (3 × 4) | (3 × 4) | 3 of 3 |
| 1 | t2, t4 | (2 × 4) | (2 × 4) | 2 of 2 |
| 2 | t3 | (1 × 4) | (1 × 4) | 0 of 1 |
| 3 | t1, t2 | (2 × 4) | (2 × 4) | 1 of 2 |

---

## M7 — Combine: applying G to the expert outputs

**In plain English:** blend each token's expert outputs together, weighted by the router's
confidence.

Each token was processed by two experts and now has two candidate results. We mix them using the
gate values from M5 — an expert the router was 88% confident about contributes 88% of the blend.

The bookkeeping step to notice is **scatter**. Each expert produced a short, compact output
covering only its own tokens. Before we can add them, we spread each one back out into a
full-height matrix with zero rows wherever that expert was not involved. Then the whole thing is
just addition.

Those zero rows are the sparsity showing up one final time: they mark every place we chose not to
spend compute.

**Per token:** `y_t = Σ_{e ∈ top-k(t)} ĝ[t,e] · O_e(h_t)`

**At matrix level:** scatter each `O_e` back into a full `(T × d_model)` matrix `Õ_e` with zero
rows for tokens that expert never saw, then

```
  Y = Σ_e  diag(G[:,e]) · Õ_e     ≡     Σ_e  G[:,e] ⊙ Õ_e
```

**The gate matrix in play:**

```
  G (4 × 4)
          e0         e1         e2         e3
    t1 [  0.885060   0.000000   0.000000   0.114940 ]
    t2 [  0.000000   0.836579   0.000000   0.163421 ]
    t3 [  0.256164   0.000000   0.743836   0.000000 ]
    t4 [  0.500000   0.500000   0.000000   0.000000 ]
```

**The four scattered outputs**, each now `(4 × 4)`:

```
  Õ₀ (4 × 4)
          d0         d1         d2         d3
    t1 [  2.041241   0.204124   0.816497   1.224745 ]
    t2 [  0.000000   0.000000   0.000000   0.000000 ]
    t3 [  0.959403   0.533002   1.918806   1.066004 ]
    t4 [  1.500000   1.250000   0.250000   2.250000 ]

  Õ₁ (4 × 4)
          d0         d1         d2         d3
    t1 [  0.000000   0.000000   0.000000   0.000000 ]
    t2 [  0.816497   0.408248   1.632993   0.204124 ]
    t3 [  0.000000   0.000000   0.000000   0.000000 ]
    t4 [  1.250000   1.250000   1.750000   1.750000 ]

  Õ₂ (4 × 4)
          d0         d1         d2         d3
    t1 [  0.000000   0.000000   0.000000   0.000000 ]
    t2 [  0.000000   0.000000   0.000000   0.000000 ]
    t3 [  0.000000   0.000000   0.000000   0.000000 ]
    t4 [  0.000000   0.000000   0.000000   0.000000 ]

  Õ₃ (4 × 4)
          d0         d1         d2         d3
    t1 [  0.816497   0.816497  -0.816497  -0.816497 ]
    t2 [  0.000000   0.000000   0.000000   0.000000 ]
    t3 [  0.000000   0.000000   0.000000   0.000000 ]
    t4 [  0.000000   0.000000   0.000000   0.000000 ]
```

**Scale each by its gate column** — `G[:,e]` acts as a per-row multiplier:

```
  G[:,0] = [0.885060, 0.000000, 0.256164, 0.500000]ᵀ
  diag(G[:,0]) · Õ₀
          d0         d1         d2         d3
    t1 [  1.806620   0.180662   0.722648   1.083972 ]
    t2 [  0.000000   0.000000   0.000000   0.000000 ]
    t3 [  0.245764   0.136536   0.491529   0.273072 ]
    t4 [  0.750000   0.625000   0.125000   1.125000 ]

  G[:,1] = [0.000000, 0.836579, 0.000000, 0.500000]ᵀ
  diag(G[:,1]) · Õ₁
          d0         d1         d2         d3
    t1 [  0.000000   0.000000   0.000000   0.000000 ]
    t2 [  0.683064   0.341532   1.366128   0.170766 ]
    t3 [  0.000000   0.000000   0.000000   0.000000 ]
    t4 [  0.625000   0.625000   0.875000   0.875000 ]

  G[:,2] = [0.000000, 0.000000, 0.743836, 0.000000]ᵀ
  diag(G[:,2]) · Õ₂
          d0         d1         d2         d3
    t1 [  0.000000   0.000000   0.000000   0.000000 ]
    t2 [  0.000000   0.000000   0.000000   0.000000 ]
    t3 [  0.000000   0.000000   0.000000   0.000000 ]
    t4 [  0.000000   0.000000   0.000000   0.000000 ]

  G[:,3] = [0.114940, 0.163421, 0.000000, 0.000000]ᵀ
  diag(G[:,3]) · Õ₃
          d0         d1         d2         d3
    t1 [  0.093848   0.093848  -0.093848  -0.093848 ]
    t2 [  0.000000   0.000000   0.000000   0.000000 ]
    t3 [  0.000000   0.000000   0.000000   0.000000 ]
    t4 [  0.000000   0.000000   0.000000   0.000000 ]
```

**Sum the four:**

```
  Y (4 × 4)
          d0         d1         d2         d3
    t1 [  1.900469   0.274510   0.628800   0.990124 ]
    t2 [  0.683064   0.341532   1.366128   0.170766 ]
    t3 [  0.245764   0.136536   0.491529   0.273072 ]
    t4 [  1.375000   1.250000   1.000000   2.000000 ]
```

Look at t3's row against the others. 74% of its routing weight went to an expert that returned
zero, so its update lands at roughly a quarter strength. The token was routed, was computed, and
got almost nothing.

---

## M8 — Residual add, then RMSNorm

**In plain English:** add the result back onto the conveyor belt, then rescale one last time.

The MoE sub-layer is finished. Its output gets added to `X` — the same residual pattern from A9
— and then normalized.

Notice the shape of the result. It is `(4 × 4)`, exactly what came in. **An MoE layer is a
drop-in replacement for an ordinary FFN**, which is why you can convert an existing dense model
into an MoE by copying its FFN a few times and bolting a router on the front. That trick is
called *upcycling*.

If our model had more blocks, this output would flow straight into the next block's attention and
the whole cycle would repeat. Mixtral does this 32 times.

**Formula:** `out = RMSNorm(X + Y)`, using the **original un-normalized `X`**.

```
  X + Y (4 × 4)                                        RMS
          d0         d1         d2         d3
    t1 [  3.900469  -0.725490   0.628800   1.990124 ]  2.241427
    t2 [ -0.316936   2.341532   2.366128   0.170766 ]  1.674137
    t3 [  0.745764   0.636536  -1.508471   1.273072 ]  1.101991
    t4 [  2.375000   2.250000   2.000000   3.000000 ]  2.434293

  RMSNorm(X + Y) (4 × 4)   ← final output
          d0         d1         d2         d3
    t1 [  1.740172  -0.323673   0.280535   0.887883 ]
    t2 [ -0.189313   1.398650   1.413342   0.102002 ]
    t3 [  0.676743   0.577623  -1.368859   1.155247 ]
    t4 [  0.975643   0.924293   0.821594   1.232391 ]
```

**Shape in = shape out.** The MoE layer is a drop-in replacement for a dense FFN, which is why
you can convert a dense model to MoE ("upcycling") by cloning its FFN `E` times and bolting on a
router. In a pre-norm Transformer this final normalization is literally the first operation of
the *next* block; in post-norm it belongs to this one. Either way `X + Y` is what propagates
forward and the norm is what the next attention layer consumes.

---


---

## ✅ What Part II accomplished

This was the MoE half, so it is worth being precise about what was actually different.

Steps M1 and M8 — normalize, and add to the residual stream — are identical to what a dense model
does. **Everything genuinely new happened in M2 through M7**, and it amounts to: score the
experts, keep the best 2, run only those, blend the results.

Four things to take away:

1. **An expert is just an FFN.** No special architecture. A dense model has one; we have four and
   use two.
2. **The router is tiny and the savings are large.** 16 numbers chose how to spend 128 numbers'
   worth of compute. That ratio only gets more extreme at scale.
3. **The zeros in `G` are the whole point.** Every zero is compute not spent.
4. **The output is shape-identical to a dense FFN's**, which is why MoE is a drop-in replacement.

And one honest observation the numbers handed us: expert 2 was token t3's *most confident* choice
and returned nothing but zeros. Routing a token somewhere guarantees it gets processed, not that
it gets helped.

---
---

---
---

# PART III — PREDICTING THE NEXT TOKEN

We now hold `RMSNorm(X + Y)`, the final hidden states. In a real model this is the output of
**final_norm** after the last of N blocks; in our 1-block model it is step M8's output directly.

```
  Final hidden states (4 × 4)
              d0         d1         d2         d3
   t1      [  1.740172  -0.323673   0.280535   0.887883 ]
   t2      [ -0.189313   1.398650   1.413342   0.102002 ]
   t3      [  0.676743   0.577623  -1.368859   1.155247 ]
   t4      [  0.975643   0.924293   0.821594   1.232391 ]
```

---

## P1 — Unembedding to vocabulary logits

**In plain English:** score every word in the vocabulary against the final hidden state.

We have pushed the sentence all the way through the model and hold a 4-number vector for each
token. Now we need to convert that back into words.

The tool is one more matrix, the **unembedding matrix** or "LM head", with one column per word in
the vocabulary. Each column is a learned direction meaning "evidence for this particular word."
Take the dot product of the hidden state against each column and you get a score for every word.

These raw scores are called **logits**. They are not probabilities yet — they can be negative,
and they do not sum to anything in particular. Turning them into probabilities is P3.

**Formula:** `logits = final · W_U`, shapes `(T × d_model) · (d_model × V) = (T × V)`.

`W_U` (the "LM head") is a single linear map from the residual stream into vocabulary space.
Column `v` is a learned direction meaning "evidence for token `v`", and the logit is the inner
product of the hidden state with that direction. Mixtral keeps `W_U` **untied** from the
embedding table — they are separate parameter blocks.

```
  W_U (4 × 6)
             "the"   "cat"   "sat"    "on"   "mat"   "eos"
       d0 [   1.0     0.5    -1.0     0.0     0.5    -1.0  ]
       d1 [   1.0    -1.0     0.5    -0.5     0.5    -1.0  ]
       d2 [   0.5     0.0     0.5     1.0     1.0    -1.0  ]
       d3 [   1.0     0.5     0.0    -1.0     0.5    -0.5  ]
```

```
  logits (4 × 6)
             "the"      "cat"      "sat"       "on"      "mat"      "eos"
   t1  [  2.444650   1.637701  -1.761741  -0.445511   1.432726  -2.140976 ]
   t2  [  2.018010  -1.442305   1.595309   0.612014   2.069011  -2.673680 ]
   t3  [  1.725183   0.338371  -1.072361  -2.812918  -0.164053  -0.463130 ]
   t4  [  3.543124   0.179724  -0.102699  -0.872943   2.387757  -3.337725 ]
```

This is the **widest** matrix in the model and, in real systems, by far the most expensive
single matrix multiply: `d_model × V` with `V` in the 32k–256k range. Mixtral's `W_U` alone is
`4096 × 32000 ≈ 131 M` parameters.

---

## P2 — Only the last row matters

**In plain English:** we computed a prediction at every position, but only the last one is
useful right now.

This trips up nearly everyone the first time. We just produced 4 rows of logits, one per input
word. Why keep only one?

Because of the causal mask from A7. Row 2 was built from a hidden state that could only see "the
cat" — so it is a prediction of the *third* word. We already know the third word. The same goes
for rows 1 and 3. Only row 4, which has seen the whole prompt, predicts something we do not
already have.

The discarded rows are not wasted during **training**, though, and this is a genuinely important
detail: there, every row is compared against the word that actually followed it, so a 4-word
sentence yields 4 separate learning signals from a single pass. That technique is called
**teacher forcing**, and it is a large part of why training language models is practical at all.

All four rows were computed, but for generating the next token we use **only row t4**.

Row `i` predicts what follows tokens `1…i`. That is what the causal mask in A7 guaranteed —
row t2 saw only "the cat", so its distribution is a prediction of the third token, which we
already know is "sat".

- **During training**, all `T` rows are used at once: each is scored against the token that
  actually followed. This is *teacher forcing*, and it is why one sequence gives `T` training
  signals for the price of one forward pass.
- **During generation**, rows 1…T−1 predict tokens we already have. They are discarded.

```
  z = logits[t4] =
      [  3.543124   0.179724  -0.102699  -0.872943   2.387757  -3.337725 ]
```

---

## P3 — Softmax over the vocabulary

**In plain English:** turn the vocabulary scores into a proper probability distribution.

Third and final softmax, third different axis — this time across the **vocabulary**. Same formula
as always, and the output is what a language model fundamentally produces: a probability for
every word it knows, summing to 1.

**Formula:** `p_v = exp(z_v) / Σ_u exp(z_u)`. Note the axis has changed for the third time in
this document — A8 normalized over *keys*, M3 over *experts*, and here we normalize over the
*vocabulary*. Same function, entirely different meaning each time.

```
  exp(z)
   "the"    34.574746
   "cat"     1.196887
   "sat"     0.902398
   "on"      0.417720
   "mat"    10.889044
   "eos"     0.035518
   ─────────────────
   sum      48.016314
```

```
  Next-token distribution
   token     logit        probability
   "the"    3.543124      0.720062   ██████████████
   "mat"    2.387757      0.226778   ████
   "cat"    0.179724      0.024927   ▌
   "sat"   -0.102699      0.018794   ▎
   "on"    -0.872943      0.008700   ▏
   "eos"   -3.337725      0.000740
                          ────────
                          1.000000
```

**The model predicts "the"**, with "mat" as the runner-up. Read against the prompt
`"the cat sat on"`, that is the sensible completion: `"on the mat"` needs the article first,
and "mat" is the plausible-but-premature alternative.

Be honest about what this shows: `W_U` here was hand-picked to make the outcome legible. A
*trained* model earns this distribution from data. What the example does show faithfully is the
**mechanism** — how a hidden state becomes a calibrated distribution over a vocabulary.

Note also the compression: a logit gap of 1.16 between "the" and "mat" became a probability
ratio of 3.2×, and the gap of 6.9 down to "eos" became a ratio of 973×. Softmax is exponential
in the differences, which is why logits are the natural space to manipulate in the next step.

---

## P4 — Sampling

**In plain English:** actually choose a word from the distribution.

Having a distribution is not the same as having an answer. Always taking the highest-probability
word (**greedy decoding**) is an option, but it makes the model deterministic and, in practice,
repetitive — it is a known cause of models falling into loops.

So real systems reshape the distribution before drawing from it. The two standard knobs are
below. Both are things you have probably seen as settings in an API call.

Greedy decoding takes `argmax(p)` → **"the"** and is deterministic. Real systems reshape the
distribution first.

**Temperature.** Divide the logits by `T` *before* the softmax: `p_v ∝ exp(z_v / T)`.
Because it acts on logits, it stretches or compresses the gaps:

```
  probability of →      "the"     "mat"     "cat"     "sat"      "on"     "eos"
   T = 0.7  (sharper)  0.828168  0.158966  0.006783  0.004531  0.001508  0.000045
   T = 1.0  (as-is)    0.720062  0.226778  0.024927  0.018794  0.008700  0.000740
   T = 1.5  (flatter)  0.581416  0.269137  0.061757  0.051158  0.030613  0.005919
```

`T → 0` converges to greedy; `T → ∞` converges to uniform. Note that at `T = 1.5` the
implausible "eos" gains 8× probability — temperature cannot distinguish "creative" from "wrong",
it just flattens everything, which is why it is usually paired with a truncation method.

**Top-p (nucleus).** Sort descending, keep the shortest prefix whose cumulative mass reaches
`p`, renormalize, sample within it.

```
  sorted:      "the"     "mat"     "cat"     "sat"      "on"     "eos"
  prob:      0.720062  0.226778  0.024927  0.018794  0.008700  0.000740
  cumulative:0.720062  0.946841  0.971767  0.990561  0.999260  1.000000
                                  ▲
                       p = 0.95 cut here → nucleus = {"the", "mat", "cat"}
```

Everything outside the nucleus is assigned probability zero. The point is *adaptivity*: when the
model is confident the nucleus is one or two tokens, and when it is uncertain the nucleus widens
automatically — unlike top-k, which uses the same fixed count either way.

---

## P5 — Append and repeat: the KV cache

**In plain English:** stick the new word on the end and run again — but skip almost all the work
the second time.

We have one word. To get a sentence we repeat the entire process with the prompt now one word
longer. Doing that naively would be enormously wasteful, and the fix explains most of what you
hear about inference performance.

The insight comes straight from the causal mask. Because no word can see forward, **adding a word
to the end cannot change anything computed for the earlier words.** Their keys and values are
exactly what they were. So we store them and reuse them. That store is the **KV cache**, and it
is the single most important optimization in language model serving.

Say we sampled **"the"** (id 0). Append it and the sequence becomes `["the","cat","sat","on","the"]`
at positions `m = 0…4`. Now run the whole model again — but almost none of it needs recomputing.

**What changes with a 5th token:**

| Object | Grows? | Recomputed? |
|---|---|---|
| `K`, `V` for positions 0–3 | no | **no — read from cache** |
| `K`, `V` for position 4 | new row | yes (1 row) |
| `Q` for positions 0–3 | — | no, never needed again |
| `Q` for position 4 | new row | yes (1 row) |
| Score row for position 4 | 1 × 5 | yes |
| MoE routing for position 4 | 1 token | yes |

Rows 0–3 of `K` and `V` are **provably identical** to what we computed in A3 — the causal mask
means no earlier position ever depends on a later one, so appending a token cannot change them.
Caching them turns generation from `O(T²)` work per token into `O(T)`.

```
  KV cache, head 0 (grows by one row per generated token)
       K                          V
   t1 [ -1.018379  1.747432 ]   [  1.422515 -1.343296 ]  ┐
   t2 [  1.509598  0.969601 ]   [ -1.160837  1.217099 ]  │ cached
   t3 [ -0.602680 -1.219202 ]   [  1.309666 -0.603530 ]  │ from
   t4 [ -1.132389 -0.988357 ]   [ -0.168558 -0.325248 ]  ┘ prefill
   t5 [    ...        ...    ]   [    ...       ...    ]  ← appended now
```

This splits inference into two regimes with very different bottlenecks:

- **Prefill** — process the whole prompt in one pass. Large matrices, arithmetic-bound. All `T`
  positions run through the MoE together, so most experts get work.
- **Decode** — one token at a time. Tiny matrices, **memory-bandwidth-bound**: the model reads
  gigabytes of weights to produce a single token.

**This is exactly where MoE pays off, and exactly where it hurts.** During decode a dense 47B
model must stream all 47B parameters per token. Mixtral streams only the ~13B its router
selected — a ~3.6× bandwidth saving on the operation that dominates generation latency. But at
batch size 1 a *single* token activates only 2 of 8 experts, so the other 6 sit in VRAM doing
nothing: you pay full memory cost for a fraction of the compute. MoE is a much better deal at
large batch, where the union of many tokens' choices keeps every expert busy.

Loop back to A1 with the extended sequence and repeat until `"eos"` is sampled or a length
limit is hit.


---
---

---
---

# What MoE buys you

| | This toy layer | Mixtral 8×7B | DeepSeek-V3 |
|---|---|---|---|
| Experts / top-k | 4 / 2 | 8 / 2 | 256 + 1 shared / 8 |
| Total params | 144 | ~46.7 B | ~671 B |
| Active per token | 80 | ~12.9 B | ~37 B |
| Active fraction | 55% | ~28% | ~5.5% |

Our toy is a poor advertisement because `E` is tiny; the ratio improves as `E` grows with `k`
fixed. Per-token FLOPs are `k · 2 · d_model · d_ff · 2` regardless of `E` — **you can grow
knowledge capacity almost for free in compute terms.** What you pay instead:

- **Memory.** All 144 params (all 671 B) must be resident even though only a slice is used. MoE
  trades FLOPs for VRAM and bandwidth.
- **Communication.** An `all-to-all` step must physically gather each expert's rows into one
  contiguous buffer so it runs as a single dense GEMM. In distributed training experts live on
  different GPUs, making this a network operation — usually the bottleneck.
- **Instability.** Routing collapse, and dropped tokens (§13).

---

# What real MoEs change

**SwiGLU instead of ReLU.** Production MoEs use a gated FFN:
`O = (SiLU(h·W_gate) ⊙ (h·W_up)) · W_down`, with `SiLU(z) = z·σ(z)`. Three matrices per expert,
not two. For expert 0 / token 1 with `W_gate = W1[0]` and a plausible `W_up`:

```
  a          = [ 1.632993  -1.224745   0.816497   0.408248 ]
  u          = [-0.408248   1.632993   0.408248   0.816497 ]
  SiLU(a)    = [ 1.366116  -0.278093   0.566186   0.245177 ]
  SiLU(a)⊙u  = [-0.557670  -0.454149   0.231164   0.200201 ]
```

Compare against ReLU's `[1.632993, 0, 0.816497, 0.408248]`. SiLU passes a small negative through
instead of hard-zeroing — no dead experts like our expert 2.

**Shared experts (DeepSeek-MoE).** One expert every token always uses, plus the top-k routed
ones. Common patterns ("this is English", "attend to syntax") would otherwise be redundantly
relearned inside every expert; isolating them frees the routed experts to specialize.
Numerically it's just an extra term with gate 1.0 in the Step 7 sum.

**Load balancing.** Routing has a positive feedback loop: an expert that gets more tokens trains
faster, attracts higher logits, and gets even more, until the router collapses onto one or two
experts and you've paid for a big model and got a small one. Classic fix is an auxiliary loss
`L_aux = E · Σ_e f_e · P̄_e` (`f_e` = fraction of dispatch slots, `P̄_e` = mean router
probability), minimized at 1.0 under perfect balance; for our `G` it evaluates to ~1.079, i.e.
~8% imbalanced. DeepSeek-V3 instead adds a per-expert bias to the logits *for selection only*,
nudged up or down by observed load — balance without a loss term fighting the main objective.

**Expert-choice routing.** Invert the argmax: each expert picks its top-`C` tokens rather than
each token picking its top-`k` experts. Perfect balance and zero drops by construction, but a
token can receive zero experts and it isn't causal — so it's used for encoders and training, not
autoregressive decoding.

---

# Capacity and dropping

GPU kernels need fixed-size buffers, so each expert gets a fixed capacity:

```
  capacity = ceil( (T · k / E) · capacity_factor )
           = ceil( (4 · 2 / 4) · 1.0 )
           = 2
```

Recall the column loads from `G`: expert 0 got **3** tokens, experts 1 and 3 got 2, expert 2 got
1. Expert 0 is over by one.

With `capacity_factor = 1.0`, the third token in sequence order — **t4** — is **dropped**. It
skips expert 0 entirely and keeps only expert 1's contribution, un-renormalized:

```
  y₄ with drop = 0.5 · [1.25, 1.25, 1.75, 1.75]
               = [0.625, 0.625, 0.875, 0.875]
  X[4] + y₄    = [1.625, 1.625, 1.875, 1.875]

  vs. without drop:
  X[4] + y₄    = [2.375, 2.250, 2.000, 3.000]
```

A materially different vector — the token silently received half the FFN update it was routed
for. Hence `capacity_factor` of 1.25–2.0 in practice, hence load balancing being non-optional,
and hence "dropless" implementations (block-sparse kernels handling ragged expert batches, e.g.
MegaBlocks).

---

# Scaling to real Mixtral

Everything above holds shape-for-shape at production scale. Only the numbers change:

| | This walkthrough | Mixtral 8×7B |
|---|---|---|
| Blocks | 1 | 32 |
| `d_model` | 4 | 4096 |
| `d_ff` per expert | 4 | 14336 |
| Attention heads | 2 | 32 query / 8 KV (GQA) |
| `d_head` | 2 | 128 |
| RoPE base | 1 | 1,000,000 |
| Experts | 4 | 8 |
| Top-k | 2 | 2 |
| Vocabulary | 6 | 32000 |
| Context | 4 | 32768 |
| Total params | 144 | ~46.7 B |
| Active per token | 80 | ~12.9 B |

**Three differences worth naming:**

**Grouped-query attention (GQA).** Mixtral uses 32 query heads but only 8 key/value heads —
each KV head is shared by 4 query heads. This shrinks the KV cache 4×, which matters enormously
at 32k context, where the cache can otherwise exceed the weights in size. The math of A6–A9 is
unchanged; you just index into a shared `K`/`V` instead of a private one.

**SwiGLU experts.** Each expert is `(SiLU(h·W_gate) ⊙ (h·W_up)) · W_down` — three matrices, not
two, and no dead-ReLU pathology like our expert 2.

**Every block has its own router.** With 32 blocks, a token is routed 32 times independently.
Its expert path through the network is a *sequence* of 32 top-2 choices, not one. The number of
distinct paths is astronomically large, which is a good intuition for where MoE's extra capacity
actually lives.

Router placement, gate renormalization, causal masking, RoPE, the residual structure, and the
final unembedding are all **identical** to what you computed by hand above.

---

---
---

# Glossary

Every term this post uses, in one place.

**Attention** — the mechanism letting each token read from other tokens. The only place in the
model where positions communicate.

**Attention head** — one independent copy of the attention machinery, operating on a slice of the
vector. Multiple heads let the model track several relationships simultaneously.

**Autoregressive** — generating one token at a time, each conditioned on everything before it.

**Capacity factor** — how much headroom each expert's fixed-size buffer gets beyond its fair
share. Below ~1.25 tokens start getting dropped.

**Causal mask** — the triangular pattern of `−inf` values preventing a token from seeing tokens
that come after it.

**`d_ff`** — the width of an expert's internal hidden layer. Typically about 4× `d_model`.

**`d_head`** — the width of one attention head. `n_heads × d_head = d_model`.

**`d_model`** — the width of the residual stream. 4 here, 4096 in Mixtral. Fixed throughout the
model.

**Decode** — the generation phase, producing one token at a time. Bottlenecked by memory
bandwidth, not arithmetic.

**Dense model** — a conventional model where every parameter is used for every token. The
opposite of sparse/MoE.

**Dispatch** — physically gathering each expert's assigned tokens into one contiguous buffer so
the expert can run as a single matrix multiply.

**Dot product** — multiply two vectors position by position and sum. Large when the vectors point
the same way; the model's universal similarity measure.

**Dropping** — silently skipping a token that arrived at an expert already at capacity. It
receives less processing than it was routed for.

**Embedding** — the vector representing a token. Looked up from a table by token id.

**Expert** — one FFN inside an MoE layer. Ordinary architecture; only its *selective use* is
unusual.

**FFN (feed-forward network)** — the two-layer network processing each token independently.
Holds most of a transformer's parameters.

**Gate** — the weight applied to an expert's output when blending, from the router's
renormalized probability.

**GQA (grouped-query attention)** — sharing each key/value head across several query heads.
Shrinks the KV cache substantially.

**Greedy decoding** — always taking the highest-probability token. Deterministic; prone to
repetition.

**Hidden state** — the vector representing a token at some point in the network.

**KV cache** — stored keys and values from previous positions, reused during generation so
earlier tokens are never recomputed.

**LM head** — the final matrix mapping a hidden state to one score per vocabulary token. Also
called the unembedding matrix.

**Load balancing** — techniques keeping the router from collapsing onto a few favoured experts
and leaving the rest idle.

**Logit** — a raw, unnormalized score. Becomes a probability after softmax.

**MoE (Mixture of Experts)** — replacing one FFN with several, plus a router selecting a few per
token.

**Nucleus sampling (top-p)** — restricting sampling to the smallest set of tokens whose
probabilities reach `p`, then drawing from it.

**Prefill** — processing the whole prompt in one pass before generation begins. Arithmetic-bound.

**Pre-norm** — normalizing the input to each sub-layer while leaving the residual stream itself
un-normalized. Standard in modern models.

**ReLU** — an activation that replaces negatives with zero. Provides the nonlinearity in our
experts.

**Residual connection** — adding a sub-layer's output back onto its input rather than replacing
it.

**Residual stream** — the running vector per token that every sub-layer reads from and adds to.
The model's shared workspace.

**RMSNorm** — normalization dividing each vector by its root-mean-square. Equalizes magnitudes
while preserving direction.

**RoPE (rotary position embedding)** — encoding position by rotating query and key vectors by an
angle proportional to position.

**Router** — the small matrix scoring each token against each expert. The only genuinely new
component in an MoE layer.

**Scatter** — spreading an expert's compact output back into a full-height matrix with zeros
where it was not involved.

**Softmax** — converts any list of numbers into positive values summing to 1. Used three times
here, over three different axes.

**Sparsity** — using only a fraction of available parameters per token. The source of MoE's
efficiency.

**SwiGLU** — a gated FFN variant used in production models instead of ReLU. Avoids dead units.

**Teacher forcing** — training on all positions at once, each against the token that actually
followed. Yields `T` learning signals per sequence.

**Temperature** — dividing logits before the softmax to sharpen (`<1`) or flatten (`>1`) the
distribution.

**Token** — the unit of text the model operates on. Usually a subword piece, not a whole word.

**Top-k routing** — keeping only the `k` highest-scoring experts per token.

**Transpose (`ᵀ`)** — flipping a matrix's rows and columns.

**Unembedding** — see LM head.

**Upcycling** — converting a trained dense model into an MoE by cloning its FFN and adding a
router.

**Vocabulary** — the full set of tokens a model knows. 6 here, 32,000 in Mixtral.

# The whole model on one page

```
  PART I — ATTENTION  (mixes information ACROSS positions)
  ids            (T,)          ──embed──▶   Emb   (T, d_model)
  Emb                          ──RMSNorm▶   H_pre (T, d_model)
  H_pre · W_Q/W_K/W_V          ─────────▶   Q,K,V (T, d_model)
  split n_heads                ─────────▶   per head (T, d_head)
  RoPE on Q,K only             ─────────▶   position encoded
  Q·Kᵀ / √d_head               ─────────▶   S     (T, T)   ← O(T²)
  causal mask (−∞ above diag)  ─────────▶   S'
  softmax over keys            ─────────▶   A     (T, T)
  A · V                        ─────────▶   C_h   (T, d_head)
  concat heads · W_O           ─────────▶   AttnOut (T, d_model)
  residual: Emb + AttnOut      ─────────▶   X     (T, d_model)

  PART II — MoE  (processes EACH position independently)
  X                            ──RMSNorm▶   H     (T, d_model)
  H · W_r                      ─────────▶   L     (T, E)
  softmax over experts         ─────────▶   P     (T, E)
  top-k                        ─────────▶   k of E   ← sparsity
  renormalize                  ─────────▶   G     (T, E)
  dispatch                     ─────────▶   E ragged batches
    per expert: O_e = ReLU(H_e · W1[e]) · W2[e]
    scatter → Õ_e              (T, d_model)
  Y = Σ_e diag(G[:,e]) · Õ_e   ─────────▶   Y     (T, d_model)
  residual: X + Y              ─────────▶   (T, d_model)
  final RMSNorm                ─────────▶   (T, d_model)

  (repeat both parts × n_blocks)

  PART III — PREDICTION
  final · W_U                  ─────────▶   logits (T, V)
  take last row                ─────────▶   z      (V,)
  softmax over vocabulary      ─────────▶   p      (V,)
  temperature / top-p / sample ─────────▶   next token id
  append, cache K & V, repeat
```

Attention is the only place tokens talk to each other. The MoE is the only place the model
consults its stored knowledge, and the only place it does so *selectively*. Everything above
`dispatch` in Part II costs `d_model × E` FLOPs per token — essentially nothing. Everything
below costs `k` experts' worth. That asymmetry — a nearly-free decision unlocking a
nearly-free-to-scale parameter store — is the whole architecture.

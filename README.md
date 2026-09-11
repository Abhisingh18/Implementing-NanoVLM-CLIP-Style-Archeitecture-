Complete VLM Experiment — One Flow
                  SYNTHETIC DATASET
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
            IMAGE                  TEXT
              │                     │
              │              "a red square
              │               is on the left"
              │                     │
              ▼                     ▼
       VISION ENCODER          TOKENIZER
              │                     │
              ▼                     ▼
       VISUAL FEATURES         TEXT TOKENS
              │                     │
              ▼                     ▼
       IMAGE EMBEDDING         TEXT ENCODER
              │                     │
              │                     ▼
              │                TEXT EMBEDDING
              │                     │
              └──────────┬──────────┘
                         ▼
                  CONTRASTIVE LOSS
                         │
                         ▼
                    BACKPROP
                         │
                         ▼
                  MODEL LEARNS
                  IMAGE ↔ TEXT

Ye tumhara first/baseline model hai.

STEP 1 — Synthetic Dataset

Hum manually images generate kar rahe hain.

Teen attributes hain:

colours = [
    'red', 'green', 'yellow', 'purple',
    'orange', 'pink', 'brown', 'gray'
]

shapes = [
    'square', 'circle', 'triangle'
]

positions = [
    'left', 'centre', 'right',
    'top', 'bottom',
    'top-left', 'top-right'
]

Isse:

$$ 8 \times 3 \times 7 = 168 $$

image-text pairs milte hain.

Example:

IMAGE:

┌────────────────┐
│                │
│ 🟥             │
│                │
└────────────────┘

TEXT:

"a red square is on the left"

Important: Image aur text same semantic information represent karte hain.

STEP 2 — Image Generate

draw_sample() function:

colour + shape + position
          ↓
      PIL Image
          ↓
       32 × 32

Example:

draw_sample(
    'red',
    'square',
    'left'
)

Output:

32 × 32 RGB image
STEP 3 — Text Generate

Image ke attributes se caption banate hain:

create_caption(
    'red',
    'square',
    'left'
)

Output:

"a red square is on the left"

Ab hamare paas:

Image  ↔  Caption
STEP 4 — Dataset Class

PyTorch Dataset image + text pair return karta hai:

Dataset
   │
   ├── Image
   │
   └── Text

Example:

image, text, caption = dataset[0]
STEP 5 — DataLoader

Dataset ko batches mein convert karte hain.

Agar:

BATCH_SIZE = 32

toh:

Batch
 │
 ├── 32 Images
 │
 └── 32 Texts
STEP 6 — Vision Encoder 👁️

Image ko neural network mein dete hain.

Tumhare current experiment mein CNN use ho raha hai.

32×32×3 Image
       ↓
     Conv2D
       ↓
     ReLU
       ↓
    MaxPool
       ↓
     Conv2D
       ↓
     ReLU
       ↓
    MaxPool
       ↓
     Conv2D
       ↓
Global Average Pool
       ↓
     Linear
       ↓
64-D Image Embedding

Output:

$$ I \in R^{64} $$

Matlab:

Image → [0.12, 0.87, ..., 0.31]
              64 values
STEP 7 — Text Tokenizer 📝

Caption:

"a red square is on the left"

ko tokens mein todte hain:

[a] [red] [square] [is] [on] [the] [left]

Phir token IDs:

[2, 5, 12, 8, 4, 7, 19]

Padding ke baad fixed length:

[2,5,12,8,4,7,19,0,0,0]
STEP 8 — Text Encoder

Ab token IDs ko embeddings mein convert karte hain:

Token IDs
    ↓
Embedding
    ↓
Transformer Encoder
    ↓
Mean Pooling
    ↓
Linear Projection
    ↓
64-D Text Embedding

Output:

$$ T \in R^{64} $$

Ab:

IMAGE → 64-D vector
TEXT  → 64-D vector

Important: Dono same embedding space mein hain.

STEP 9 — Normalize

Hum dono embeddings ko normalize karte hain:

image_features = F.normalize(
    image_features,
    dim=-1
)

text_features = F.normalize(
    text_features,
    dim=-1
)

Iska purpose similarity calculation ko stable banana hai.

STEP 10 — Similarity Matrix 🔥

Ab image aur text embeddings ka dot product:

logits = (
    image_features @ text_features.T
) / TEMPERATURE

Agar batch size 4:

             TEXT
          T1   T2   T3   T4

IMAGE I1  8.2  1.2  0.4  0.8
      I2  0.7  7.9  0.9  0.2
      I3  0.3  1.1  8.4  0.5
      I4  0.6  0.4  0.8  8.0

Correct pair:

I1 ↔ T1
I2 ↔ T2
I3 ↔ T3
I4 ↔ T4

isliye diagonal high honi chahiye.

STEP 11 — Temperature

Tumhare code mein:

TEMPERATURE = 0.07

Similarity ko sharpen karta hai:

$$ logits = \frac{I T^T}{\tau} $$

where:

$$ \tau=0.07 $$
STEP 12 — Labels

Batch ke andar correct matching pair same index par hai:

labels = torch.arange(
    batch_size,
    device=logits.device
)

For batch 4:

labels = [0, 1, 2, 3]

Meaning:

Image 0 → Text 0
Image 1 → Text 1
Image 2 → Text 2
Image 3 → Text 3
STEP 13 — Contrastive Loss
Image → Text
loss_i2t = F.cross_entropy(
    logits,
    labels
)

Model se pooch rahe hain:

"Is image ka correct text kaunsa hai?"

Text → Image
loss_t2i = F.cross_entropy(
    logits.T,
    labels
)

Ab pooch rahe hain:

"Is text ka correct image kaunsa hai?"

Final Loss
loss = (
    loss_i2t + loss_t2i
) / 2

Yaani bidirectional alignment.

STEP 14 — Backpropagation

Ab:

loss.backward()

gradients calculate hote hain.

Then:

optimizer.step()

model ke weights update hote hain.

Complete:

Image + Text
     ↓
Encoders
     ↓
Embeddings
     ↓
Similarity
     ↓
Contrastive Loss
     ↓
Backward
     ↓
Weight Update

Repeat:

Epoch 1
Epoch 2
Epoch 3
...
Epoch 20
🔥 Ab Second Model — Cross Attention

Ab yahin se tumhara actual VLM fusion experiment interesting hota hai.

Baseline mein:

IMAGE → Image Embedding
                 ↕
             Similarity
                 ↕
TEXT  → Text Embedding

Image aur text internally interact nahi kar rahe.

Cross-attention mein hum bolte hain:

Text ko image ke visual tokens ke saath interact karne do.

STEP 15 — Image → Visual Tokens

Instead of:

Image → one 64-D vector

hum karte hain:

Image
 ↓
Vision Encoder
 ↓
Feature Map
 ↓
4 × 4
 ↓
16 Visual Tokens

Output:

$$ [B,16,64] $$

Example:

Image Tokens:

V1 V2 V3 V4
V5 V6 V7 V8
V9 V10 V11 V12
V13 V14 V15 V16

Ye image ke different spatial regions represent karte hain.

STEP 16 — Text → Text Tokens

Is baar mean pooling immediately nahi karenge.

Instead:

"a red square is on the left"

        ↓

T1 T2 T3 T4 T5 T6 T7

Output:

$$ [B,7,64] $$
STEP 17 — Cross Attention 🔥🔥

Ab:

TEXT → Query (Q)

IMAGE → Key (K)
IMAGE → Value (V)

Mathematically:

$$ Q = TW_Q $$ $$ K = VW_K $$ $$ V'=VW_V $$

Then:

$$ Attention = softmax \left( \frac{QK^T}{\sqrt{d_k}} \right)V' $$
STEP 18 — Intuition

Text token:

"red"

Query banata hai.

Phir image ke 16 visual tokens ke saath similarity calculate hoti hai:

             Visual Tokens

red →       V1 V2 V3 V4 ... V16
            ↓  ↓  ↓
attention   .  .  █  .

Model seekhta hai ki:

"red" ke liye image ka kaunsa region useful hai?

Similarly:

"square" → shape-related region

"left" → left-side region
STEP 19 — Cross-Attention Output

Ab text representation mein visual information aa gayi:

Text Tokens
     +
Visual Information
     ↓
Fused Text Tokens

Shape:

[B, Text_Length, 64]
STEP 20 — Transformer Block

Cross-attention ke baad:

Cross Attention
      ↓
Residual Connection
      ↓
LayerNorm
      ↓
FFN
      ↓
Residual
      ↓
LayerNorm

FFN:

Linear
  ↓
GELU
  ↓
Linear
STEP 21 — Fused Representation

Ab text tokens ko mean pool karte hain:

Fused Text Tokens
       ↓
Mean Pooling
       ↓
64-D Fused Embedding

Yaani:

Image
  ↓
Visual Tokens
  ↓
      ┌──────────────┐
      │Cross Attention│
      └──────────────┘
             ↑
             │
        Text Tokens
             ↓
      Fused Embedding
STEP 22 — Contrastive Objective

Ab fused representation ko text representation ke saath align kar sakte hain.

Image
 ↓
Vision Encoder
 ↓
Visual Tokens
 ↓
Cross Attention ← Text
 ↓
Fused Embedding
       ↕
Contrastive Loss
       ↕
Text Embedding
STEP 23 — Evaluation

Training ke baad hum poochte hain:

Image → Text Retrieval
Image
 ↓
Embedding
 ↓
Compare with all captions
 ↓
Rank captions

Example:

Image:
red square left

Model:

1. a red square is on the left     0.94 ✓
2. a red square is in the centre   0.72
3. a green square is on the left  0.61
4. a red circle is on the left    0.55
STEP 24 — Recall@1

Agar correct caption first position par hai:

Correct → ✓

then:

$$ Recall@1 = \frac{correct\ top1}{total} $$

For example:

100 queries
87 correct at #1

Recall@1 = 87%
STEP 25 — Recall@5

Agar correct caption top 5 mein kahin bhi hai:

1. wrong
2. wrong
3. correct ✓
4. wrong
5. wrong

then successful.

$$ Recall@5 = \frac{correct\ top5}{total} $$
🎯 Final Experiment

Tumhara final research pipeline:

                    SAME DATASET
                         │
             ┌───────────┼───────────┐
             │           │           │
             ▼           ▼           ▼
          BASELINE    EARLY/LATE   CROSS
             │          FUSION     ATTENTION
             │           │           │
             ▼           ▼           ▼
          Embedding   Fusion      Q-K-V
             │           │           │
             └───────────┼───────────┘
                         ▼
                  Contrastive Loss
                         │
                         ▼
                    Training
                         │
                         ▼
                  Retrieval Test
                         │
                ┌────────┴────────┐
                ▼                 ▼
             Recall@1          Recall@5

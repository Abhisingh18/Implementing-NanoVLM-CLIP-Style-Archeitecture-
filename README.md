Vision-Language Fusion Ablation Study

A controlled study comparing Early Fusion, Late Fusion, and Cross-Attention Fusion strategies for Vision-Language Models using synthetic image-text data and contrastive learning.

🔬 Research Question

How does the choice of multimodal fusion strategy affect image-text representation alignment and retrieval performance?

🎯 Objectives

Build a controlled Vision-Language learning setup.

Compare different multimodal fusion strategies.

Train the models using contrastive learning.

Evaluate image-text retrieval performance.

Analyze cross-modal interactions using attention visualization.

🧠 Methodology

The system consists of:

Vision Encoder

Text Encoder

Multimodal Fusion Module

Contrastive Learning Objective

Image ──→ Vision Encoder ──→ Visual Representation ──┐
                                                     ├──→ Fusion ──→ Joint Representation
Text  ──→ Text Encoder  ──→ Text Representation ─────┘

🗂️ Synthetic Dataset

The experiment uses procedurally generated images containing simple:

Colors

Shapes

Spatial positions

The corresponding text caption describes the image.

Example:

Image: red square on the left
Caption: "a red square is on the left"

The current configuration contains:

8 colors

3 shapes

7 positions

168 unique image-text pairs

🔀 Fusion Strategies

1. Early Fusion

Image and text representations are combined at an early stage.

Image → Vision Encoder ──┐
                         ├→ Concatenation → Multimodal Network
Text  → Text Encoder ────┘

2. Late Fusion

The two modalities are encoded independently and combined at a later stage.

Image → Vision Encoder → Image Embedding ──┐
                                           ├→ Fusion → Joint Embedding
Text  → Text Encoder  → Text Embedding ────┘

3. Cross-Attention Fusion

Text tokens attend dynamically to visual tokens.

Text Tokens  → Query (Q)
Image Tokens → Key (K), Value (V)

             ↓

      Cross-Attention

             ↓

   Fused Representation

The attention operation is:

$$
Attention(Q,K,V)=softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

🔗 Contrastive Learning

The objective is to bring matching image-text pairs closer in the embedding space and push mismatched pairs apart.

For a batch of size $N$:

Image₁ ↔ Caption₁  Positive
Image₁ ↔ Caption₂  Negative
Image₁ ↔ Caption₃  Negative
...

The similarity matrix is:

$$
S = \frac{I T^T}{\tau}
$$

where:

$I$ = normalized image embeddings

$T$ = normalized text embeddings

$\tau$ = temperature

The final loss is:

$$
L = \frac{L_{image\rightarrow text}+L_{text\rightarrow image}}{2}
$$

🏗️ Models

Baseline Contrastive VLM

Independent vision and text encoders produce global embeddings, which are trained with symmetric contrastive loss.

Cross-Attention VLM

The vision encoder produces visual tokens and the text encoder produces text tokens. Multi-head cross-attention allows text tokens to attend to relevant visual tokens before generating the final representation.

⚙️ Experimental Setup

Parameter

Value

Image Size

32 × 32

Embedding Dimension

64

Attention Heads

4

Temperature

0.07

Optimizer

AdamW

Learning Rate

3e-4

Weight Decay

1e-4

Batch Size

32

Epochs

20

All fusion strategies should use comparable training conditions for a fair ablation study.

📊 Evaluation

The experiments use:

Contrastive Loss

Recall@1

Recall@5

Image-to-Text Retrieval

Text-to-Image Retrieval

Similarity Matrix

Cross-Attention Visualization

🧪 Ablation Study

The main comparison is:

Model

Fusion Strategy

Cross-Modal Interaction

Baseline

Independent

Low

Early Fusion

Early

Joint representation

Late Fusion

Late

High-level

Cross-Attention

Dynamic

Token-level

Results

Results will be added after all models are trained under the same evaluation protocol.

Model

Loss

Image→Text R@1

Image→Text R@5

Text→Image R@1

Text→Image R@5

Baseline

TBD

TBD

TBD

TBD

TBD

Early Fusion

TBD

TBD

TBD

TBD

TBD

Late Fusion

TBD

TBD

TBD

TBD

TBD

Cross-Attention

TBD

TBD

TBD

TBD

TBD

🔎 Qualitative Analysis

Cross-attention maps are visualized to investigate which visual tokens receive attention from different text tokens.

This provides an interpretable view of text-to-image interaction.

⚠️ Limitations

The dataset is synthetic and very small.

The task is intentionally controlled.

Results should not be interpreted as representative of large-scale real-world VLMs.

Training and evaluation must be separated for a meaningful generalization study.

🚀 Future Work

Implement complete Early Fusion and Late Fusion baselines.

Create train/validation/test splits.

Test compositional generalization.

Use larger synthetic datasets.

Introduce real image-text datasets.

Compare different embedding dimensions.

Study the effect of temperature and attention heads.

Add pretrained vision and language encoders.

Extend the experiment toward larger VLM architectures.

📦 Installation

pip install torch torchvision numpy pillow matplotlib

▶️ Usage

The main experiment can be run from:

notebooks/vlm_fusion_ablation.ipynb

The notebook contains dataset generation, model definitions, training, retrieval evaluation, and visualization.

📚 References

The project is inspired by research on:

Vision-Language Models

Contrastive Learning

CLIP-style image-text representation learning

Transformer architectures

Multi-Head Cross-Attention

📄 License

This project is intended for research and educational purposes.

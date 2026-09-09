# Related work and implementation boundaries

This file exists partly to prevent a recurring documentation error: an HPM component may be
**inspired by** a published mechanism without being an implementation-equivalent reproduction.

## Selective state-space models

- Tri Dao and Albert Gu, *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality* (Mamba-2 / SSD), 2024: https://arxiv.org/abs/2405.21060

HPM's blockwise recurrent path uses selective state-space ideas but is not the official Mamba-2
SSD algorithm or kernel.

## Delta-rule / linear-attention memory

- Gated DeltaNet, ICLR 2025: https://openreview.net/pdf?id=r8H7xhYPwz
- Kimi Linear / Kimi Delta Attention, 2025: https://arxiv.org/abs/2510.26692
- Gated DeltaNet-2, 2026 preprint: https://arxiv.org/abs/2605.22791

The current HPM fast-weight path uses a delta correction and input-dependent channel-wise decay.
It still couples erase and write/correction strength through one scalar `beta`; Gated DeltaNet-2
explicitly decouples those controls. HPM should therefore not be described as Gated DeltaNet-2.

## Softmax gating theory

- Huy Nguyen, Nhat Ho, Alessandro Rinaldo, *On Least Square Estimation in Softmax Gating Mixture of Experts*, ICML 2024: https://proceedings.mlr.press/v235/nguyen24f.html
- Huy Nguyen et al., *A General Theory for Softmax Gating Multinomial Logistic Mixture of Experts*, ICML 2024: https://proceedings.mlr.press/v235/nguyen24b.html

These papers provide estimation/identifiability theory for softmax-gated MoEs. They do not
already prove HPM's dynamic necessary-support preservation question; HPM's four paths are also
structurally heterogeneous rather than interchangeable experts.

## Differentiable Top-K

- Xie et al., *Differentiable Top-k with Optimal Transport*, NeurIPS 2020: https://proceedings.neurips.cc/paper/2020/hash/ec24a54d62ce57ba93a531b460fa8d18-Abstract.html

Hard Top-K membership is discontinuous. Entropic optimal-transport relaxations provide a smooth
training surrogate at positive regularization, which motivates HPM's relaxed-backward read/write
experiments. It does not make the deployed hard operator globally Lipschitz.

## JEPA

- Assran et al., *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture*, 2023: https://arxiv.org/abs/2301.08243

JEPA's core context/target predictive idea and EMA target encoder motivate HPM's optional
predictive auxiliary. HPM's token/block predictive modules are adapted research components,
not a reproduction of I-JEPA's image masking/training setup.

# NERD
Normalized Effective Rank Distribution

NER (Normalized Effective Rank) quantifies dimensional utilization across layers using entropy analysis of singular value distributions. NER calculation involves Singular Value Decomposition (SVD) of the weight matrix A. Singular values form a probability distribution through normalization. The entropy H of this distribution indicates how evenly the singular values are spread, yielding the Effective Rank (ERank) as 2^H. Normalizing by the maximum possible entropy H_max produces a value between 0 and 1, measuring how efficiently the layer uses its dimensions.

**Why Entropy as a Measure**
Entropy is used here because it captures the spread or uniformity of the singular value distribution. A higher entropy value indicates that the singular values are more evenly distributed, meaning the layer uses all available dimensions more effectively. In contrast, lower entropy suggests that only a few dimensions dominate, leading to inefficient dimensional utilization. Thus, entropy provides an intuitive and mathematically grounded way to quantify how well a layer’s representational capacity is being utilized.

Run the script with:
    python mastermerge.py --config mastermerge_config.yaml (optional)

The script loads the configuration, processes each model by downloading, loading weights, normalizing layers, and calculating NER for each layer. It then selects the optimal layer based on NER and creates a composite model with the highest NER in each layer, aiming to maximize information flow and improve model performance.

**Unified Theoretical Analysis of NER in Neural Networks**

Given a neural network layer W ∈ ℝᵐˣⁿ with SVD W = UΣV*:

1. **Capacity and Information Flow**

The fundamental relationships:
• SVD components: Σ = diag(σ₁,...,σᵣ), σ₁ ≥ σ₂ ≥ ... ≥ σᵣ ≥ 0
• Probability distribution: pᵢ = σᵢ/∑σⱼ
• Entropy: H(W) = -∑pᵢlog(pᵢ)
• NER: NER(W) = H(W)/log(r)

Capacity bounds:
C(W) ≤ min{I(x;y), log(r)}
C_eff(W) = exp(H(W))  // effective dimensions
η(W) = exp(H(W))/r    // utilization efficiency

Subject to constraints:
tr(WW^T) ≤ P  // power constraint
E[||x||²] ≤ σ²  // input constraint

Optimal capacity:
C_opt = (r/2)log(1 + P/rσ²)
C_actual ≈ NER(W) * C_opt

2. **Information Bottleneck Analysis**

For network path x → t → y:
• Data Processing Inequality: I(x;t) ≥ I(t;y)
• IB Lagrangian: L(W) = I(t;y) - βI(x;t)
• Optimal solution requires: ∂L/∂W = 0

**Intuitive Explanation**:
Higher NER implies that a layer can transmit more information without significant loss. Since entropy measures how well the information is spread across dimensions, a high NER indicates efficient information transfer, minimizing bottlenecks. In practical terms, this means that layers with higher NER are better at preserving and passing along the complexity of the input data, which is crucial for deep learning models that rely on rich feature representations.

Information flow properties:
I(x;t) ≈ log(exp(H(W))) = H(W)
I(t;y) ≤ H(W)  // information ceiling

NER implications:
Higher NER → Better information preservation
Lower NER → Information bottleneck

3. **Optimization Landscape Properties**

Gradient and conditioning analysis:
• Condition number: κ(W) = σ₁/σₘᵢₙ
• Gradient bound: ||∇L||₂ ≤ κ(W)||∇L*||₂
• NER relationship: κ(W) ∝ 1/NER(W)

**Practical Example**:
Consider a layer with poor conditioning (high condition number). It results in inefficient gradient flow, where small updates in some directions are overwhelmed by large updates in others. Higher NER leads to better conditioning, meaning gradients propagate more uniformly, leading to smoother optimization and faster convergence. This is especially important for deep models where gradient flow can be a significant challenge.

Hessian properties:
• λₘₐₓ(H)/λₘᵢₙ(H) ≈ κ(W)²
• Higher NER → Better conditioning

Optimization implications:
• Smoother loss landscape
• Better gradient flow
• Faster convergence

4. **Statistical Learning Theory**

Generalization bounds:
Rademacher complexity:
R_n(F_W) ≤ √(∑σᵢ²/n)

Generalization bound:
|L_test - L_train| ≤ 2R_n(F_W) + √(log(1/δ)/2n)

With probability 1-δ:
E[L_test] ≤ L_train + O(√(1/NER(W) * log(r)/n))

**Practical Takeaways**:
- **Layer Selection for Fine-Tuning**: Use NER to identify which layers are most efficient in terms of information utilization. These layers can be prioritized during fine-tuning or targeted for modifications to improve model performance.
- **Model Design**: When designing or merging models, consider maximizing NER across critical layers to ensure robust information flow and efficient training dynamics.

**Unified Performance Bound**:

Performance P(W) bounded by:
P(W) ≤ O(1/NER(W)) * [
    1/√n +                     // sample complexity
    κ(W) +                     // optimization difficulty
    exp(-I(t;y)/β) +          // information loss
    √(tr(WW^T)/C_opt)         // capacity utilization
]

**License**
Use, modify, and distribute as you see fit. Good luck with that shit.
Copyright 2024, nobody. No rights reserved.

// Computes expert affinity scores, correction biases, top-k selections, and normalized routing weights for learned MoE layers.
// Its backward path differentiates selected routing weights and records expert-load statistics without differentiating through top-k indices.

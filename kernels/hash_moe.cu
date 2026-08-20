// Implements deterministic token-ID-to-expert assignment for the early hash-routed MoE layers used by DeepSeek-V4.
// Expert indices remain non-differentiable while routing weights, expert parameters, and token representations receive gradients.

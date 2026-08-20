// Executes grouped matrix multiplications for variable-sized batches of tokens assigned to different MoE experts.
// Its backward path computes expert-weight and token-input gradients while preserving the dispatch layout and expert offsets.

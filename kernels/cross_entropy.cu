// Computes token-level cross-entropy loss from vocabulary logits using a numerically stable softmax reduction.
// Its backward path produces logit gradients without materializing unnecessary intermediate probability tensors.

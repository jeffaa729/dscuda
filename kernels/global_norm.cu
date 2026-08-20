// Computes the global L2 norm of model gradients using numerically stable block reductions with FP32 accumulation.
// The resulting norm is used for gradient clipping, overflow detection, and training diagnostics before optimizer updates.

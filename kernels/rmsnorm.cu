// Implements RMSNorm by scaling activations with their inverse root-mean-square and a learned per-channel weight.
// Its backward path computes activation and scale gradients using reduction kernels with FP32 accumulation.

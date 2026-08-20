// Implements the SwiGLU feed-forward activation by multiplying a SiLU-activated gate projection with an up projection.
// Its backward path computes gradients for both branches and supports the clamped activation variant used by DeepSeek-V4 experts.

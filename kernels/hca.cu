// Implements Heavily Compressed Attention by aggregating large token windows and attending densely over the shortened KV sequence.
// Its backward path differentiates through dense attention and the learned token compressor while respecting causal window availability.

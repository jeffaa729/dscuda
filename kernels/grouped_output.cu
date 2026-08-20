// Implements the grouped low-rank attention-output projection used to reduce the cost of wide V4 attention outputs.
// Its backward path accumulates gradients through both low-rank projection stages and into the attention-head groups.

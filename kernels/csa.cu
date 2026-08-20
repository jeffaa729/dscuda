// Implements V4 Compressed Sparse Attention by combining compressed long-range KV entries selected by the Lightning Indexer with a local sliding window.
// Its backward path propagates gradients through the selected attention entries, grouped output projection, and token-compression inputs.

// Scores compressed KV entries for each query and selects the top candidates consumed by Compressed Sparse Attention.
// Backpropagation updates the differentiable query and index-key projections, while the discrete top-k indices are treated as fixed selections.

// Dispatches tokens to selected MoE experts by counting assignments, computing expert offsets, and permuting token representations into contiguous expert batches.
// The reverse path combines weighted expert outputs and scatters their gradients back to the original token order.

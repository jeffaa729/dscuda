// Implements causal attention restricted to a fixed local sliding window using shared or multi-head key and value layouts.
// Its backward path computes query, key, and value gradients without constructing attention scores outside the active window.

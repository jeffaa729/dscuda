// Implements dense linear projections using interchangeable custom CUDA and cuBLAS-backed matrix-multiplication paths.
// Its backward path computes input, weight, and optional bias gradients for attention, feed-forward, router, and output layers.

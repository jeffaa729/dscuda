// Implements token-embedding lookup for the forward pass and scatter-accumulation into the embedding-weight gradients during backpropagation.
// The same weight storage can be shared with the language-model output projection when tied embeddings are enabled.

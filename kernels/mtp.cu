// Implements the Multi-Token Prediction module that adds auxiliary predictions for future tokens beyond the standard next-token objective.
// Its backward path accumulates auxiliary-loss gradients into the shared transformer states and MTP-specific parameters.

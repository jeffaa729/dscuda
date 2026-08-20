// Compresses consecutive hidden-state or KV windows into learned summary entries for CSA and HCA long-context attention.
// Its backward path distributes gradients through the normalized compression weights, positional terms, and source tokens.

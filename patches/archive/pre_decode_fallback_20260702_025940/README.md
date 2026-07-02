Archive of the Sparge Celerity patch before adding the decode fallback.

Reason:
The original Sparge patch worked for long-context prefill timing, but fails during autoregressive generation because SpargeAttention requires query sequence length >= 128, while decode steps usually have q_len=1.

This archive preserves the prior patch files and modeling_celerity.py copies before adding:

    if query.size(-2) < 128:
        return None

which falls back to original Celerity attention during short-query decode.

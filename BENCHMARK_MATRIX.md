# Benchmark Matrix

Target hardware: a local RTX 4060 (SM89) and a rented H100 (SM90). Tables are transposed so that experiment properties are rows and the two GPUs are the only result columns.
## Repository scope

Only the following six experiment families belong in this repository. Training code, standalone transformer operators, optimizers, HCA, and CSA are out of scope.

| Family | Retained operations | Current implementation |
| :-- | :-- | :-- |
| GEMM | FP32/BF16 row-major NN | SM89 CUDA + cuBLAS reference |
| MoE | route, dispatch, combine, BF16 grouped NN GEMM | SM89 CUDA + PyTorch/cuBLAS references |
| FlashAttention | causal BF16 forward/backward, D=128 | SM89 CUDA + official FlashAttention reference |
| MLA | dense C512/R64 prefill forward/backward and paged decode | SM89 CUDA + PyTorch; FlashMLA decode adapter for SM90 |
| KDA | BF16 forward/backward | FLA reference + recurrent PyTorch oracle; custom CUDA pending |
| DSA | FP8 indexer, sparse prefill/decode, complete sparse pipeline | SM90 CUDA and official-reference benchmark pending |

A benchmark is part of the final matrix only when custom and reference backends receive the same logical inputs, use the same dtype/layout contract, and return the same required outputs. Reference-only KDA remains useful while its custom kernel is developed; DSA is not exposed by the runner until its FP8 contract exists.

## Shared timing configuration

| Setting | RTX 4060 | H100 |
| :-- | :-- | :-- |
| Random seed | 2026 | 2026 |
| TF32 | Disabled | Disabled |
| CUDA Graph operations per replay | 10 | 10 |
| Warmup | 1000 ms | 1000 ms |
| Minimum sample duration | 20 ms | 20 ms |
| Trials | 9 | 9 |
| Primary output | custom time, reference time, reference percentage | custom time, reference time, reference percentage |

Correctness checks run before timing. The same common shape must be used for cross-GPU scaling; H100-only production shapes are reported separately.

## 1. GEMM

| Setting | RTX 4060 | H100 |
| :-- | :-- | :-- |
| Custom kernel | `kernels/gemm/matmul/sm89.cu` | SM89 fallback; `kernels/gemm/matmul/sm90.cu` is currently empty |
| Operation | Row-major NN GEMM | Row-major NN GEMM |
| Dtypes | FP32 and BF16; BF16 is primary | FP32 and BF16; BF16 is primary |
| Quick shape | `M=N=K=2048` | `M=N=K=2048` |
| Full shapes | `M=N=K=2048,4096,8192` | `M=N=K=2048,4096,8192` |
| Performance reference | cuBLAS `cublasGemmEx` | cuBLAS `cublasGemmEx`; planned DeepGEMM `deep_gemm.bf16_gemm_nn` after BF16-output parity |
| Correctness oracle | PyTorch FP32 matmul | PyTorch FP32 matmul |

## 2. MoE

| Setting | RTX 4060 | H100 |
| :-- | :-- | :-- |
| Custom kernels | `kernels/moe/expert_dispatch/sm89.cu`, `kernels/moe/grouped_gemm/sm89.cu` | SM89 fallback; corresponding `sm90.cu` files are currently empty |
| Routing quick shape | `tokens=512,D=512,E=8,topk=2` | Same |
| Routing full shapes | `4096,512,64,8`; `8192,1024,128,8` | Same |
| Routing operations | route, dispatch, combine | route, dispatch, combine |
| Routing reference | PyTorch | PyTorch |
| Grouped GEMM quick shape | `M=512,N=512,K=256,E=8` | Same |
| Grouped GEMM full shapes | `M=4096,N=1536,K=512,E=8`; `M=8192,N=1536,K=512,E=16` | Same |
| Expert distributions | uniform, hot, empty | uniform, hot, empty |
| Grouped GEMM dtype | BF16 | BF16 |
| Grouped GEMM reference | Per-expert cuBLAS loop | Per-expert cuBLAS loop; planned DeepGEMM `deep_gemm.m_grouped_bf16_gemm_nn_contiguous` after BF16-output parity |
| Correctness oracle | PyTorch grouped GEMM and routing | PyTorch grouped GEMM and routing |

## 3. Dense attention / FlashAttention

| Setting | RTX 4060 | H100 |
| :-- | :-- | :-- |
| Custom kernel | `kernels/attention/flash_attention/sm89.cu` | SM89 fallback; `kernels/attention/flash_attention/sm90.cu` is currently empty |
| Operations | Causal forward and backward | Causal forward and backward |
| Dtype and output | BF16 input/output, FP32 LSE | Same |
| Head dimension | `D=128` | `D=128` |
| Quick shape | `B=1,T=512,H=8,D=128` | Same |
| Full shapes | `B={1,4},T={128,256,512,1024,2048},H={4,8},D=128` | Same |
| Attention settings | `causal=True`, `dropout=0`, scale `1/sqrt(128)` | Same |
| Performance reference | Official `flash_attn.flash_attn_func` | Official `flash_attn.flash_attn_func` |
| Correctness oracle | Materialized PyTorch attention | Materialized PyTorch attention |

This is the normal dense-attention kernel comparison. It is separate from dense MLA and DSA.

## 4. MLA

| Setting | RTX 4060 | H100 |
| :-- | :-- | :-- |
| Custom kernel | `kernels/attention/mla/sm89.cu` | SM89 fallback; `kernels/attention/mla/sm90.cu` is currently empty |
| Geometry | `C=512,RoPE=64,QK=576,V=512` | Same |
| Common prefill shape | `B=1,Q=KV=128,H=8` | Same |
| Common prefill operations | Forward and backward | Forward and backward |
| Common decode shape | `B=2,Q=1,KV=1024,H=16` | Same |
| H100 official decode shapes | N/A | `(B,KV,H)=(1,1024,64),(4,4096,64),(4,4096,128),(8,8192,128)`; `Q=1,page=64,splits=8` |
| Prefill reference | PyTorch | PyTorch |
| Decode reference | PyTorch | PyTorch and FlashMLA |
| Exact FlashMLA path | N/A | `get_mla_metadata` and `flash_mla_with_kvcache(indices=None,is_fp8_kvcache=False)` |
| Correctness oracle | PyTorch MLA | PyTorch MLA |

FlashMLA cannot run on SM89. Its dense prefill forward/backward path is an SM100 MHA kernel, not the H100 SM90 absorbed-MLA contract used here. FlashMLA is therefore used only for H100 dense MLA decode.

## 5. KDA

| Setting | RTX 4060 | H100 |
| :-- | :-- | :-- |
| Custom kernel | Not implemented yet | Not implemented yet |
| Operations | Forward and backward | Forward and backward |
| Prepared-input dtype | BF16 Q/K/V; FP32 decay, beta, and state | Same |
| Quick shape | `B=1,T=128,H=4,HV=4,K=128,V=128` | Same |
| Full shapes | Same geometry with `T=256,512,1024` | Same |
| Performance reference | FLA `fla.ops.kda.chunk_kda` | FLA `fla.ops.kda.chunk_kda` |
| Correctness oracle | PyTorch recurrent KDA | PyTorch recurrent KDA |

## 6. DSA

DSA follows the official DeepSeek-V3.2 precision and layout contract and is skipped completely on the RTX 4060.

| Setting | RTX 4060 | H100 |
| :-- | :-- | :-- |
| Custom kernels | Skipped | Planned `kernels/attention/dsa/indexer_sm90.cu` and `sparse_mla_sm90.cu`; both are currently empty |
| Indexer precision | N/A | FP8 E4M3 query/key, FP32 key scales and head weights, FP32 logits |
| Indexer geometry | N/A | `Hi=64,Di=128`, with query/KV lengths matched to the attention case |
| Indexer reference | N/A | DeepGEMM `fp8_fp4_mqa_logits` for prefill and `fp8_fp4_paged_mqa_logits` for decode |
| Sparse prefill production shape | N/A | `B=1,Q=4096,Hq=128,Hkv=1,Dqk=576,Dv=512,topk=2048`; `KV=8192,32768,65536,98304,131072` |
| Sparse prefill reference | N/A | FlashMLA `flash_mla_sparse_fwd` / `sparse_attn_fwd` |
| Sparse decode geometry | N/A | `Q=1,Hq=128,Hkv=1,Dqk=576,Dv=512,page=64` |
| Sparse decode production cases `(B,KV,topk)` | N/A | `(64,8192,128),(128,8192,128),(64,16384,512),(128,16384,512),(64,32768,2048),(128,32768,2048),(256,16384,2048)` |
| Sparse decode cache | N/A | FlashMLA FP8 KV cache: FP8 NoPE values and scales, BF16 RoPE |
| Sparse decode reference | N/A | `flash_mla_with_kvcache(indices=...,is_fp8_kvcache=True)` / `sparse_decode_fwd` |
| Component benchmark | N/A | Give custom and FlashMLA the same prepared top-k indices |
| End-to-end benchmark | N/A | custom indexer + top-k + custom sparse MLA versus DeepGEMM indexer + top-k + FlashMLA |
| Correctness oracle | N/A | PyTorch adapted to identical FP8 quantization, masking, top-k, and cache layouts |

FP8 DSA does not mean every tensor is FP8. The official sparse decode stores the NoPE cache in FP8, retains the RoPE part in BF16, performs BF16 MMA, and accumulates in FP32.

## Comparison rules

1. Compare custom FlashAttention only with official FlashAttention using identical dense-attention inputs.
2. Compare custom dense MLA decode only with FlashMLA dense decode.
3. Compare custom DSA sparse attention with FlashMLA sparse attention using identical prepared top-k indices.
4. Compare complete custom DSA with the complete official-component pipeline; include indexer and top-k time on both sides.
5. Use dense MLA versus DSA only as a separate algorithm-level scaling experiment, not as the primary kernel-quality percentage.
## Canonical commands

RTX 4060 correctness and quick runtime:

```bash
uv sync --locked
DSCUDA_CUDA_ARCH=89 bash scripts/test.sh all
DSCUDA_CUDA_ARCH=89 bash scripts/benchmark.sh all quick
```

Run one full SM89 family at a time:

```bash
DSCUDA_CUDA_ARCH=89 bash scripts/benchmark.sh matmul full
DSCUDA_CUDA_ARCH=89 bash scripts/benchmark.sh grouped_gemm full
DSCUDA_CUDA_ARCH=89 bash scripts/benchmark.sh expert_dispatch full
DSCUDA_CUDA_ARCH=89 bash scripts/benchmark.sh flash_attention full --reference flash_attention
DSCUDA_CUDA_ARCH=89 bash scripts/benchmark.sh mla full --reference pytorch
DSCUDA_CUDA_ARCH=89 bash scripts/benchmark.sh kda full --reference fla
```

The H100 suite uses `DSCUDA_CUDA_ARCH=90`. FlashMLA is selected only for MLA decode with `bash scripts/benchmark.sh mla h100 --reference both`. DSA is intentionally absent from the launcher until its FP8 custom and official backends implement the exact contract above.

Runtime tables overwrite `profiles/runtime/<family>.md` and `.csv`; raw trial samples overwrite `profiles/runtime/<family>_samples.json`. Nsight Compute is optional and uses `scripts/profile.sh` for GEMM, grouped GEMM, FlashAttention, and MLA.

## Completion checklist

- [x] One benchmark/test entry point and one table format.
- [x] SM89 GEMM, grouped GEMM, routing, FlashAttention, and dense MLA kernels.
- [x] cuBLAS, PyTorch, official FlashAttention, FLA, and FlashMLA adapter boundaries.
- [x] HCA, CSA, training, optimizer, and unrelated transformer references removed.
- [ ] Make BF16 GEMM and grouped-GEMM output storage match DeepGEMM before enabling that reference.
- [ ] Add DeepGEMM BF16 NN and grouped-NN H100 adapters.
- [ ] Implement and benchmark dedicated SM90 kernels rather than compiling the SM89 sources for H100.
- [ ] Implement KDA CUDA forward/backward and compare with FLA.
- [ ] Implement the FP8 DSA indexer, sparse prefill/decode, PyTorch FP8 oracle, and official-component pipeline.
- [ ] Run the completed H100 matrix and record environment/version provenance with its result files.

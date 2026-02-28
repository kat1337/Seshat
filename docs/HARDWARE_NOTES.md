# Hardware Notes: AMD AI MAX 395+ (Strix Halo)

## Specifications

- **APU**: AMD AI MAX 395+ (Strix Halo)
- **CPU**: Zen 5 cores
- **GPU**: RDNA 3.5 integrated graphics (40 CUs)
- **Memory**: 128 GB unified (shared CPU+GPU)
- **Memory Bandwidth**: ~256 GB/s (LPDDR5X)

## What This Means for ML

### The Good
- 128 GB unified memory is enormous — most consumer GPUs max out at 24 GB VRAM
- Can load models that won't fit on a single RTX 4090
- Unified memory means no CPU↔GPU memory transfer overhead
- Runs Qwen 3 80B quantized (Q4) for inference comfortably

### The Constraints
- **Not a datacenter GPU**: The iGPU has ~40 RDNA 3.5 CUs vs hundreds/thousands of CUDA cores on an A100
- **Compute-limited, not memory-limited**: Training is slow per step, but you can fit large models
- **ROCm support is evolving**: As of 2025, ROCm on RDNA 3.5 iGPU (gfx1151) has partial support
- **Memory bandwidth**: ~256 GB/s vs ~2 TB/s on H100. This is the main bottleneck for LLM inference

## Practical Implications

### For Training (QLoRA)
| Model Size | 4-bit QLoRA Memory | Feasibility | Notes |
|---|---|---|---|
| 8B | ~8-12 GB | ✅ Easy | Fast training, good for iteration |
| 14B | ~15-25 GB | ✅ Comfortable | **Recommended starting point** |
| 32B | ~30-50 GB | ⚠️ Feasible | May need reduced seq length |
| 80B | ~65-90 GB | ⚠️ Tight | Needs careful memory management |
| 80B (seq 512) | ~55-70 GB | ✅ Feasible | Short context but workable |

### For Inference
| Model Size | GGUF Quant | Memory | Tokens/sec (est.) |
|---|---|---|---|
| 8B | Q4_K_M | ~5 GB | 20-40 |
| 14B | Q4_K_M | ~9 GB | 12-25 |
| 32B | Q4_K_M | ~20 GB | 5-15 |
| 80B | Q4_K_M | ~45 GB | 2-8 |
| 80B | Q5_K_M | ~55 GB | 2-6 |

*Note: Inference speeds are rough estimates. Actual performance depends on ROCm/Vulkan support and llama.cpp optimizations for this specific APU.*

## ROCm Setup

### Check Compatibility
```bash
# Check if ROCm detects the GPU
rocminfo 2>/dev/null | grep "Name:" || echo "ROCm not installed or GPU not detected"

# Check PyTorch ROCm support
python -c "import torch; print(f'ROCm available: {torch.cuda.is_available()}')"
```

### If ROCm Doesn't Work
This is expected for some RDNA 3.5 iGPU configurations. Fallback options:

1. **CPU training with PyTorch**: Slow but works. QLoRA on 14B is feasible on CPU with 128 GB RAM
2. **Vulkan backend via llama.cpp**: For inference, llama.cpp's Vulkan backend often works when ROCm doesn't
3. **ONNX Runtime**: May have better AMD iGPU support for inference
4. **Cloud training**: Train on cloud (Lambda Labs, Vast.ai, AWS), deploy locally for inference

### Docker with ROCm
```bash
# Use the ROCm PyTorch Docker image
docker run -it --device=/dev/kfd --device=/dev/dri \
  --group-add video \
  -v $(pwd):/workspace \
  rocm/pytorch:latest
```

## Recommended Workflow

1. **Start with 14B model** — trains comfortably, fast iteration
2. **Validate pipeline end-to-end** before scaling up
3. **Try 32B** if 14B results are promising
4. **80B inference only locally** — train on cloud if needed, deploy locally
5. **Use llama.cpp for inference** — best compatibility with AMD APUs

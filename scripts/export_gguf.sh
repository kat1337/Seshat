#!/bin/bash
# Export trained LoRA adapter: merge with base model and convert to GGUF
#
# Prerequisites:
#   - Trained adapter in models/adapters/
#   - llama.cpp installed (git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp && make)
#
# Usage:
#   bash scripts/export_gguf.sh [adapter_dir] [base_model] [quant_type]

set -euo pipefail

ADAPTER_DIR="${1:-models/adapters/qwen3-14b-seshat}"
BASE_MODEL="${2:-Qwen/Qwen3-14B}"
QUANT_TYPE="${3:-Q4_K_M}"
OUTPUT_DIR="models/merged"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-../llama.cpp}"

mkdir -p "$OUTPUT_DIR"

echo "=== Step 1: Merge LoRA adapter with base model ==="
python -c "
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

print('Loading base model...')
model = AutoModelForCausalLM.from_pretrained('${BASE_MODEL}', trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained('${BASE_MODEL}', trust_remote_code=True)

print('Loading adapter...')
model = PeftModel.from_pretrained(model, '${ADAPTER_DIR}')

print('Merging...')
model = model.merge_and_unload()

print('Saving merged model...')
model.save_pretrained('${OUTPUT_DIR}/merged-hf')
tokenizer.save_pretrained('${OUTPUT_DIR}/merged-hf')
print('Done!')
"

echo "=== Step 2: Convert to GGUF ==="
python "${LLAMA_CPP_DIR}/convert_hf_to_gguf.py" \
    "${OUTPUT_DIR}/merged-hf" \
    --outfile "${OUTPUT_DIR}/seshat-merged-f16.gguf" \
    --outtype f16

echo "=== Step 3: Quantize to ${QUANT_TYPE} ==="
"${LLAMA_CPP_DIR}/llama-quantize" \
    "${OUTPUT_DIR}/seshat-merged-f16.gguf" \
    "${OUTPUT_DIR}/seshat-${QUANT_TYPE}.gguf" \
    "${QUANT_TYPE}"

echo "=== Done! ==="
echo "Quantized model: ${OUTPUT_DIR}/seshat-${QUANT_TYPE}.gguf"
echo ""
echo "Test with:"
echo "  ${LLAMA_CPP_DIR}/llama-cli -m ${OUTPUT_DIR}/seshat-${QUANT_TYPE}.gguf -p 'Translate to Spanish:' -n 128"

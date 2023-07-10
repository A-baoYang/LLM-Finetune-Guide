COMBINED_MODEL=multi-ee-llama-1024-3e-4-combined

./quantize ./zh-models/$COMBINED_MODEL/ggml-model-f16.bin ./zh-models/$COMBINED_MODEL/ggml-model-q4_0.bin 2
BASE_MODEL=/workspace/.cache/huggingface/hub/models--hfl--chinese-llama-2-7b/snapshots/c40cf9ac38b789d542b582f842a9f62511fa3bf1
OUTPUT_DIR=/workspace/Code/LLM-Finetune-Guide/efficient-finetune/lora/finetuned
LORA_MODEL=${OUTPUT_DIR}/sft-dcardwom-llama2_7b-lora-llama2_7b-lora-64-128-0.05-1e-4-1-5-8-/sft_lora_model

python inference_hf.py --base_model ${BASE_MODEL} \
    --lora_model ${LORA_MODEL} \
    --with_prompt \
    --interactive
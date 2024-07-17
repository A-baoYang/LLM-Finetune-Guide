BASE_MODEL=hfl/chinese-llama-2-7b
OUTPUT_DIR=/workspace/Code/LLM-Finetune-Guide/efficient-finetune/Chinese-LLaMA-2/lora/finetuned
LORA_MODEL=${OUTPUT_DIR}/sft-dcard-wom-zhcn-llama2_7b-lora-llama2_7b-lora-64-128-0.05-1e-4-1-2048-10-8/sft_lora_model
cache_dir=/workspace/Code/models/huggingface

python gradio_demo.py --base_model ${BASE_MODEL} \
    --lora_model ${LORA_MODEL} \
    --cache_dir ${cache_dir} \
    --gpus 0
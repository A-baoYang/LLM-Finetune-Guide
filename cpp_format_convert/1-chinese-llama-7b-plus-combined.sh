BASE_MODEL=/home/jovyan/gpt/model/combined/chinese-llama-7b-plus-combined
LORA_MODEL=/home/jovyan/gpt/open_gpt/LLM-Finetune-Guide/efficient-finetune/lora/finetuned/multi-ee-llama-1024-3e-4/lora_model
COMBINED_MODEL=/home/jovyan/stock-pred/temp_model/multi-ee-llama-1024-3e-4-combined

python merge_llama_with_chinese_lora.py \
    --base_model $BASE_MODEL \
    --lora_model $LORA_MODEL \
    --output_dir $COMBINED_MODEL \
    --output_type huggingface \
    --max_shard_size 500MB
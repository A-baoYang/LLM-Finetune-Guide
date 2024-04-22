# 运行脚本前请仔细阅读wiki(https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/wiki/sft_scripts_zh)
# Read the wiki(https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/wiki/sft_scripts_zh) carefully before running the script
lr=1e-4
lora_rank=64
lora_alpha=128
lora_trainable="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
modules_to_save="embed_tokens,lm_head"
lora_dropout=0.05

# pretrained_model=hfl/chinese-llama-2-7b
pretrained_model=/root/.cache/huggingface/hub/models--hfl--chinese-llama-2-7b/snapshots/c40cf9ac38b789d542b582f842a9f62511fa3bf1
chinese_tokenizer_path=/workspace/Code/Chinese-LLaMA-Alpaca-2/scripts/tokenizer
dataset_dir=/workspace/Code/LLM-Finetune-Guide/instruction-datasets/dcard-wom/train
per_device_train_batch_size=1
per_device_eval_batch_size=1
gradient_accumulation_steps=8
num_train_epochs=5
max_seq_length=512
output_dir=/workspace/Code/LLM-Finetune-Guide/efficient-finetune/lora/finetuned/sft-dcardwom-llama2_7b-lora-llama2_7b-lora-${lora_rank}-${lora_alpha}-${lora_dropout}-${lr}-${per_device_train_batch_size}-${num_train_epochs}-${gradient_accumulation_steps}-${block_size}
validation_file=/workspace/Code/LLM-Finetune-Guide/instruction-datasets/dcard-wom/val/dev.json

deepspeed_config_file=/workspace/Code/LLM-Finetune-Guide/efficient-finetune/lora/ds_zero2_no_offload.json

# resume_checkpoint=checkpoint-3800

CUDA_VISIBLE_DEVICES=1 torchrun --nnodes 1 --nproc_per_node 1 --master_port=29051 run_clm_sft_with_peft.py \
    --deepspeed ${deepspeed_config_file} \
    --model_name_or_path ${pretrained_model} \
    --tokenizer_name_or_path ${chinese_tokenizer_path} \
    --dataset_dir ${dataset_dir} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --do_train \
    --do_eval \
    --seed $RANDOM \
    --fp16 \
    --num_train_epochs ${num_train_epochs} \
    --lr_scheduler_type cosine \
    --learning_rate ${lr} \
    --warmup_ratio 0.03 \
    --weight_decay 0 \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --save_total_limit 2 \
    --evaluation_strategy steps \
    --eval_steps 1000 \
    --save_steps 1000 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --preprocessing_num_workers 8 \
    --max_seq_length ${max_seq_length} \
    --output_dir ${output_dir} \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --lora_rank ${lora_rank} \
    --lora_alpha ${lora_alpha} \
    --trainable ${lora_trainable} \
    --lora_dropout ${lora_dropout} \
    --modules_to_save ${modules_to_save} \
    --torch_dtype float16 \
    --validation_file ${validation_file} \
    --load_in_kbits 16 \
    --save_safetensors False \
    --gradient_checkpointing \
    --ddp_find_unused_parameters False
    # --resume_from_checkpoint ${output_dir}/${resume_checkpoint}
# 运行脚本前请仔细阅读wiki(https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/wiki/sft_scripts_zh)
# Read the wiki(https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/wiki/sft_scripts_zh) carefully before running the script
lr=1e-4
lora_rank=64
lora_alpha=128
lora_trainable="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
modules_to_save="embed_tokens,lm_head"
lora_dropout=0.05

pretrained_model=/workspace/Code/models/huggingface/Breeze-7B-Instruct-v0_1
model_name=breeze_7b_instruct_v0_1
cache_dir=/workspace/Code/models/huggingface
dataset_tag=dcard-wom-zhtw
# chinese_tokenizer_path=/workspace/Code/models/huggingface/Llama3-TAIDE-LX-8B-Chat-Alpha1  # 官方提示要用 Chinese-LLaMA-2 的 tokenizer, 但那是簡中的
chinese_tokenizer_path=/workspace/Code/models/huggingface/Breeze-7B-Instruct-v0_1
dataset_dir=/workspace/Code/LLM-Finetune-Guide/instruction-datasets/${dataset_tag}/train
per_device_train_batch_size=4
per_device_eval_batch_size=4
gradient_accumulation_steps=8
num_train_epochs=10
max_seq_length=2048
checkpoint_step=10
output_dir=/workspace/Code/LLM-Finetune-Guide/efficient-finetune/Chinese-LLaMA-2/lora/finetuned/sft-${dataset_tag}-${model_name}-lora-${model_name}-lora-${lora_rank}-${lora_alpha}-${lora_dropout}-${lr}-${per_device_train_batch_size}-${max_seq_length}-${num_train_epochs}-${gradient_accumulation_steps}
validation_file=/workspace/Code/LLM-Finetune-Guide/instruction-datasets/${dataset_tag}/val/dev.json

deepspeed_config_file=/workspace/Code/LLM-Finetune-Guide/efficient-finetune/Chinese-LLaMA-2/lora/ds_zero2_no_offload.json
# resume_checkpoint=checkpoint-3800

CUDA_VISIBLE_DEVICES=0 torchrun --nnodes 1 --nproc_per_node 1 --master_port=29051 run_clm_sft_with_peft.py \
    --deepspeed ${deepspeed_config_file} \
    --cache_dir ${cache_dir} \
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
    --logging_steps ${checkpoint_step} \
    --save_strategy steps \
    --save_total_limit 2 \
    --evaluation_strategy steps \
    --eval_steps ${checkpoint_step} \
    --save_steps ${checkpoint_step} \
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
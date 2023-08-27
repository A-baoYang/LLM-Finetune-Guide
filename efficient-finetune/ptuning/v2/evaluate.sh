MODEL_PATH=THUDM/chatglm-6b
MODEL_TYPE=chatglm-6b
MAX_LENGTH=512
PRE_SEQ_LEN=512
BATCH_SIZE=2
STEP=3000
LR=2e-2
DATATAG=ec-product-custom-tag-no-instruction
CACHE_DIR=/home/jovyan/gpt/model/huggingface

CUDA_VISIBLE_DEVICES=2,3 accelerate launch --config_file /home/jovyan/gpt/open_gpt/LLM-Finetune-Guide/efficient-finetune/config/use_deepspeed.yaml finetune.py --do_predict \
    --test_file /home/jovyan/gpt/open_gpt/LLM-Finetune-Guide/instruction-datasets/$DATATAG/ood.json \
    --overwrite_cache \
    --prompt_column input \
    --response_column output \
    --output_file ood-generated_prediction.json \
    --model_name_or_path $MODEL_PATH \
    --ptuning_checkpoint finetuned/$DATATAG-$MODEL_TYPE-pt-$PRE_SEQ_LEN-$LR/checkpoint-$STEP \
    --cache_dir $CACHE_DIR \
    --output_dir finetuned/$DATATAG-$MODEL_TYPE-pt-$PRE_SEQ_LEN-$LR \
    --overwrite_output_dir \
    --max_source_length $MAX_LENGTH \
    --max_target_length $MAX_LENGTH \
    --per_device_eval_batch_size $BATCH_SIZE \
    --predict_with_generate \
    --pre_seq_len $PRE_SEQ_LEN \
    --quantization_bit 4
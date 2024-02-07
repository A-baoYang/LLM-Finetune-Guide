MODEL_PATH=THUDM/chatglm2-6b
MODEL_NAME=chatglm2-6b
MAX_LENGTH=768
PRE_SEQ_LEN=768
LR=2e-2
BATCH_SIZE=6
EPOCHS=5
MAX_STEPS=3000
SAVE_STEPS=10
DATATAG=multi-ee-no-instruction
CACHE_DIR=/home/abaoyang/model/huggingface

CUDA_VISIBLE_DEVICES=0 python finetune.py \
    --do_train \
    --train_file ../../../instruction-datasets/$DATATAG/train.json \
    --validation_file ../../../instruction-datasets/$DATATAG/dev.json \
    --prompt_column input \
    --response_column output \
    --overwrite_cache \
    --model_name_or_path $MODEL_PATH \
    --output_dir finetuned/$DATATAG-$MODEL_NAME-pt-$PRE_SEQ_LEN-$LR \
    --cache_dir $CACHE_DIR \
    # --resume_from_checkpoint finetuned/$DATATAG-$MODEL_NAME-pt-$PRE_SEQ_LEN-$LR/checkpoint-100 \
    --overwrite_output_dir \
    --max_source_length $MAX_LENGTH \
    --max_target_length $MAX_LENGTH \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps 2 \
    --predict_with_generate \
    --num_train_epochs $EPOCHS \
    # --max_steps $MAX_STEPS \
    --logging_steps $SAVE_STEPS \
    --save_steps $SAVE_STEPS \
    --save_total_limit 1 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN \
    --quantization_bit 4

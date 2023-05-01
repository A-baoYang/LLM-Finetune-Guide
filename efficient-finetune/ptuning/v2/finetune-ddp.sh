MODEL_PATH=/home/jovyan/stock-pred/chatglm-6b
MODEL_TYPE=chatglm-6b
MAX_LENGTH=512
PRE_SEQ_LEN=512
LR=2e-2
BATCH_SIZE=2
EPOCHS=50
MAX_STEPS=3000
SAVE_STEPS=100
DATATAG=ee-no-instruction

accelerate launch --config_file ../../config/use_deepspeed.yaml finetune.py --do_train \
    --train_file ../../../instruction-datasets/$DATATAG/train.json \
    --validation_file ../../../instruction-datasets/$DATATAG/dev.json \
    --prompt_column input \
    --response_column output \
    --overwrite_cache \
    --model_name_or_path $MODEL_PATH \
    --output_dir finetuned/$DATATAG-$MODEL_TYPE-pt-$PRE_SEQ_LEN-$LR \
    # --resume_from_checkpoint finetuned/$DATATAG-$MODEL_TYPE-pt-$PRE_SEQ_LEN-$LR/qcheckpoint-100 \
    --overwrite_output_dir \
    --max_source_length $MAX_LENGTH \
    --max_target_length $MAX_LENGTH \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps 2 \
    --predict_with_generate \
    --num_train_epochs $EPOCHS \
    --max_steps $MAX_STEPS \
    --logging_steps 10 \
    --save_steps $SAVE_STEPS \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN \
    --quantization_bit 4

deepspeed --num_nodes 1 --num_gpus 2 finetune.py \
    --deepspeed ../../config/zero_stage3_offload_config.json \
    --do_train \
    --train_file ../../../instruction-datasets/$DATATAG/train.json \
    --validation_file ../../../instruction-datasets/$DATATAG/dev.json \
    --prompt_column input \
    --response_column output \
    --overwrite_cache \
    --model_name_or_path $MODEL_PATH \
    --output_dir finetuned/$DATATAG-$MODEL_TYPE-pt-$PRE_SEQ_LEN-$LR \
    # --resume_from_checkpoint finetuned/$DATATAG-$MODEL_TYPE-pt-$PRE_SEQ_LEN-$LR/qcheckpoint-100 \
    --overwrite_output_dir \
    --max_source_length $MAX_LENGTH \
    --max_target_length $MAX_LENGTH \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps 2 \
    --predict_with_generate \
    --num_train_epochs $EPOCHS \
    --max_steps $MAX_STEPS \
    --logging_steps 10 \
    --save_steps $SAVE_STEPS \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN \
    --quantization_bit 4
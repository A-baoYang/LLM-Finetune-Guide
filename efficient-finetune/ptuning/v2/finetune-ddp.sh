MODEL_PATH=THUDM/chatglm-6b
MODEL_TYPE=chatglm-6b
MAX_LENGTH=512
PRE_SEQ_LEN=512
LR=2e-2
BATCH_SIZE=2
EPOCHS=10
MAX_STEPS=10000
SAVE_STEPS=100
DATATAG=ner-no-instruction


# torchrun --standalone --nnodes=1  --nproc_per_node=2 finetune.py --do_train --do_eval \
#     --train_file ../../../instruction-datasets/$DATATAG/train.json \
#     --validation_file ../../../instruction-datasets/$DATATAG/dev.json \
#     --prompt_column input \
#     --response_column output \
#     --overwrite_cache \
#     --model_name_or_path $MODEL_PATH \
#     --output_dir finetuned/$DATATAG-$MODEL_TYPE-pt-$PRE_SEQ_LEN-$LR \
#     --resume_from_checkpoint finetuned/$DATATAG-$MODEL_TYPE-pt-$PRE_SEQ_LEN-$LR/checkpoint-3300 \
#     --overwrite_output_dir \
#     --max_source_length $MAX_LENGTH \
#     --max_target_length $MAX_LENGTH \
#     --per_device_train_batch_size $BATCH_SIZE \
#     --per_device_eval_batch_size $BATCH_SIZE \
#     --gradient_accumulation_steps 2 \
#     --predict_with_generate \
#     --num_train_epochs $EPOCHS \
#     --max_steps $MAX_STEPS \
#     --logging_steps 10 \
#     --save_steps $SAVE_STEPS \
#     --save_total_limit 3 \
#     --learning_rate $LR \
#     --pre_seq_len $PRE_SEQ_LEN \
#     --quantization_bit 4

CUDA_VISIBLE_DEVICES=2,3 accelerate launch --main_process_port 29052 --config_file ../../config/use_deepspeed.yaml finetune.py --do_train --do_eval \
    --train_file ../../../instruction-datasets/$DATATAG/train.json \
    --validation_file ../../../instruction-datasets/$DATATAG/dev.json \
    --prompt_column input \
    --response_column output \
    --overwrite_cache \
    --model_name_or_path $MODEL_PATH \
    --output_dir finetuned/$DATATAG-$MODEL_TYPE-pt-$PRE_SEQ_LEN-$LR \
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
    --save_total_limit 3 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN \
    --quantization_bit 4

# deepspeed --num_nodes 1 --num_gpus 2 finetune.py \
#     --deepspeed ../../config/zero_stage3_offload_config.json \
#     --do_train \
#     --train_file ../../../instruction-datasets/$DATATAG/train.json \
#     --validation_file ../../../instruction-datasets/$DATATAG/dev.json \
#     --prompt_column input \
#     --response_column output \
#     --overwrite_cache \
#     --model_name_or_path $MODEL_PATH \
#     --output_dir finetuned/$DATATAG-$MODEL_TYPE-pt-$PRE_SEQ_LEN-$LR \
#     # --resume_from_checkpoint finetuned/$DATATAG-$MODEL_TYPE-pt-$PRE_SEQ_LEN-$LR/qcheckpoint-100 \
#     --overwrite_output_dir \
#     --max_source_length $MAX_LENGTH \
#     --max_target_length $MAX_LENGTH \
#     --per_device_train_batch_size $BATCH_SIZE \
#     --per_device_eval_batch_size $BATCH_SIZE \
#     --gradient_accumulation_steps 2 \
#     --predict_with_generate \
#     --num_train_epochs $EPOCHS \
#     --max_steps $MAX_STEPS \
#     --logging_steps 10 \
#     --save_steps $SAVE_STEPS \
#     --save_total_limit 3 \
#     --learning_rate $LR \
#     --pre_seq_len $PRE_SEQ_LEN \
#     --quantization_bit 4
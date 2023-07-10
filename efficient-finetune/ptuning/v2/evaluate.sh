MODEL_PATH=THUDM/chatglm-6b
MODEL_TYPE=chatglm-6b
MAX_LENGTH=512
PRE_SEQ_LEN=512
BATCH_SIZE=4
STEP=10000
LR=2e-2
DATATAG=ner-no-instruction

CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file ../../config/use_deepspeed.yaml finetune.py --do_predict \
    --validation_file ../../../instruction-datasets/$DATATAG/dev.json \
    --test_file ../../../instruction-datasets/$DATATAG/test.json \
    --overwrite_cache \
    --prompt_column input \
    --response_column output \
    --output_file generated-prediction.json \
    --model_name_or_path $MODEL_PATH \
    --ptuning_checkpoint finetuned/$DATATAG-$MODEL_TYPE-pt-$PRE_SEQ_LEN-$LR/checkpoint-$STEP \
    --output_dir finetuned/$DATATAG-$MODEL_TYPE-pt-$PRE_SEQ_LEN-$LR \
    --overwrite_output_dir \
    --max_source_length $MAX_LENGTH \
    --max_target_length $MAX_LENGTH \
    --per_device_eval_batch_size $BATCH_SIZE \
    --predict_with_generate \
    --pre_seq_len $PRE_SEQ_LEN \
    --quantization_bit 4

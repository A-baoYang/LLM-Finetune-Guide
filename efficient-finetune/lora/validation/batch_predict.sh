MODEL_PATH=/home/jovyan/gpt/model/bigscience/bloomz-7b1-mt
MODEL_TYPE=bloom
DATATAG=multi-ee
LR=3e-4
MICRO_BATCH_SIZE=1
MAX_LENGTH=1024

CUDA_VISIBLE_DEVICES=0,1 python batch_generate.py \
    --model_name_or_path $MODEL_PATH \
    --lora_weights ../finetuned/$DATATAG-$MODEL_TYPE-$MAX_LENGTH-$LR \
    --dev_file ../../../instruction-datasets/$DATATAG/dev.json \
    --dev_batch_size $MICRO_BATCH_SIZE \
    --max_length $MAX_LENGTH \
    --output_file ../finetuned/$DATATAG-$MODEL_TYPE-$MAX_LENGTH-$LR/generate_predictions.json

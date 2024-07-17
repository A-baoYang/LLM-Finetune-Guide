# MODEL_PATH=/home/jovyan/gpt/model/chinese-llama-7b-plus-combined
MODEL_PATH=bigscience/bloomz-7b1-mt
MODEL_TYPE=bloom
DATATAG=multi-ee
LR=3e-4
MICRO_BATCH_SIZE=1
MAX_LENGTH=1024

CUDA_VISIBLE_DEVICES=2,3 python batch_generate.py \
    --model_name_or_path $MODEL_PATH \
    --finetuned_dir ../finetuned/$DATATAG-$MODEL_TYPE-$MAX_LENGTH-$LR \
    --dev_file ../../../instruction-datasets/$DATATAG/test.json \
    --dev_batch_size $MICRO_BATCH_SIZE \
    --max_length $MAX_LENGTH \
    --output_filename generate_predictions
MODEL_PATH=ziqingyang/chinese-llama-2-7b
MODEL_TYPE=llama
DATATAG=medical-qa-zhcn
EPOCHS=5
LR=3e-4
BATCH_SIZE=16
MICRO_BATCH_SIZE=1
MAX_LENGTH=1024
SAVE_STEPS=100


WORLD_SIZE=2 CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 finetune.py --base_model $MODEL_PATH \
    --model_type $MODEL_TYPE \
    --train_file ../../instruction-datasets/$DATATAG/train.json \
    --val_file ../../instruction-datasets/$DATATAG/dev.json \
    --output_dir finetuned/$DATATAG-$MODEL_TYPE-$MAX_LENGTH-$LR \
    --num_epochs $EPOCHS \
    --learning_rate $LR \
    --batch_size $BATCH_SIZE \
    --micro_batch_size $MICRO_BATCH_SIZE \
    --cutoff_len $MAX_LENGTH \
    --save_steps $SAVE_STEPS \
    --lora_target_modules q_proj,v_proj

# CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=3 --master_port=29050 finetune.py --base_model $MODEL_PATH \
#     --model_type $MODEL_TYPE \
#     --train_file ../../instruction-datasets/$DATATAG/train.json \
#     --val_file ../../instruction-datasets/$DATATAG/dev.json \
#     --output_dir finetuned/$DATATAG-$MODEL_TYPE-$MAX_LENGTH-$LR \
#     --num_epochs $EPOCHS \
#     --learning_rate $LR \
#     --batch_size $BATCH_SIZE \
#     --micro_batch_size $MICRO_BATCH_SIZE \
#     --cutoff_len $MAX_LENGTH \
#     --save_steps $SAVE_STEPS \
#     --lora_target_modules q_proj,v_proj

# accelerate launch --config_file ../config/use_deepspeed.yaml finetune.py --base_model $MODEL_PATH \
#     --model_type $MODEL_TYPE \
#     --train_file ../../instruction-datasets/$DATATAG/train.json \
#     --val_file ../../instruction-datasets/$DATATAG/dev.json \
#     --output_dir finetuned/$DATATAG-$MODEL_TYPE-$MAX_LENGTH-$LR \
#     --num_epochs $EPOCHS \
#     --learning_rate $LR \
#     --batch_size $BATCH_SIZE \
#     --micro_batch_size $MICRO_BATCH_SIZE \
#     --cutoff_len $MAX_LENGTH \
#     --save_steps $SAVE_STEPS \
#     --lora_target_modules q_proj,v_proj

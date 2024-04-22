BASE_MODEL=hfl/chinese-llama-2-7b
OUTPUT_DIR=/home/abaoyang/LLM-Finetune-Guide/efficient-finetune/lora/finetuned
LORA_MODEL=${OUTPUT_DIR}/pt-dentist_zhcn-llama2_7b-lora-64-128-0.05-2e-4-1-8-512/pt_lora_model

python inference_hf.py --base_model ${BASE_MODEL} \
    --lora_model ${LORA_MODEL} \
    --with_prompt \
    --interactive
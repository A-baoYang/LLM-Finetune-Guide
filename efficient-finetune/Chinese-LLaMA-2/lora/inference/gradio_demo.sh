BASE_MODEL=hfl/chinese-llama-2-7b
OUTPUT_DIR=/home/abaoyang/LLM-Finetune-Guide/efficient-finetune/Chinese-LLaMA-2/lora/finetuned
LORA_MODEL=${OUTPUT_DIR}/pt-dentist_zhcn-llama2_7b-lora-64-128-0.05-2e-4-1-5-8-512/checkpoint-550/pt_lora_model

python gradio_demo.py --base_model ${BASE_MODEL} \
    --lora_model ${LORA_MODEL}
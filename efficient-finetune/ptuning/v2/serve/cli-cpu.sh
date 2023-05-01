python cli_demo.py \
    --pretrained_model_path /home/jovyan/ripple-dev/temp_gpt/ChatGLM-6B/model/THUDM/chatglm-6b \
    --ptuning_checkpoint /home/jovyan/ripple-dev/temp_gpt/ChatGLM-6B/ptuning/output/medical-qa-instruction-chatglm-6b-pt-512-2e-2/checkpoint-3000 \
    --quantization_bit 4 \
    --is_cuda False
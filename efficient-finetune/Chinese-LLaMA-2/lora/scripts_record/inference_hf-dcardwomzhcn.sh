BASE_MODEL=hfl/chinese-llama-2-7b
# BASE_MODEL=hfl/llama-3-chinese-8b-instruct
# BASE_MODEL=/workspace/Code/models/huggingface/llama-3-chinese-8b
OUTPUT_DIR=/workspace/Code/LLM-Finetune-Guide/efficient-finetune/Chinese-LLaMA-2/lora/finetuned
LORA_MODEL=${OUTPUT_DIR}/sft-dcard-wom-zhcn-llama2_7b-lora-llama2_7b-lora-64-128-0.05-1e-4-1-2048-10-8/sft_lora_model
cache_dir=/workspace/Code/models/huggingface

python inference_hf.py --base_model ${BASE_MODEL} \
    --lora_model ${LORA_MODEL} \
    --cache_dir ${cache_dir} \
    --gpus 1 \
    --system_prompt "你和我是一间拥有自由灵魂的行销团队。服务过多家广告公司、上市公司，许多成功的口碑案子背后都有我们团队的努力，因为我们致力于把行销赋予专属的生命力，让品牌走的每一步都具有意义，提供最全面、最有温度的服务给所有的客户，在这竞争的时代，我们努力打破行销的制式框架，创造新的局面。\n\n量身订做专属口碑专案，提供议题带入品牌和产品，由专业的写手撰文和风向监控，提升讨论度及曝光。\n\n达成方法\n利用时事议题创造网路讨论度，让消费者透过日常生活方式达到品牌接触。协助操作品牌和产品的相关话题性，主要分为三种文章格式包括不同话题的：「议题文」「分享 / 比较文」「情境文」借由话题在论坛创造品牌和产品的声量，提高消费者对品牌的认知度与好感度。\n议题文：透过询问、时事、烦恼，日常生活等问题带起论坛内的讨论度。\n\n分享 / 比较文：包含开箱文、竞品分析，透过这一系列的操作，加深消费者对品牌的认知。\n\n情境文：创意、感人的情境故事，自然的把品牌、产品融入其中，可达到高度的网路声量。" \
    --with_prompt \
    --interactive
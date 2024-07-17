BASE_MODEL=taide/Llama3-TAIDE-LX-8B-Chat-Alpha1
OUTPUT_DIR=/workspace/Code/LLM-Finetune-Guide/efficient-finetune/Chinese-LLaMA-2/lora/finetuned
LORA_MODEL=${OUTPUT_DIR}/sft-dcard-wom-zhtw-llama3_taide_8b-lora-llama3_taide_8b-lora-64-128-0.05-1e-4-4-100-8/sft_lora_model
cache_dir=/workspace/Code/models/huggingface

python inference_hf.py --base_model ${BASE_MODEL} \
    --lora_model ${LORA_MODEL} \
    --cache_dir ${cache_dir} \
    --gpus 1 \
    --system_prompt "你和我是一間擁有自由靈魂的行銷團隊。服務過多家廣告公司、上市公司，許多成功的口碑案子背後都有我們團隊的努力，因為我們致力於把行銷賦予專屬的生命力，讓品牌走的每一步都具有意義，提供最全面、最有溫度的服務給所有的客戶，在這競爭的時代，我們努力打破行銷的制式框架，創造新的局面。\n\n量身訂做專屬口碑專案，提供議題帶入品牌和產品，由專業的寫手撰文和風向監控，提昇討論度及曝光。\n\n達成方法\n利用時事議題創造網路討論度，讓消費者透過日常生活方式達到品牌接觸。協助操作品牌和產品的相關話題性，主要分為三種文章格式包括不同話題的：「議題文」「分享 / 比較文」「情境文」藉由話題在論壇創造品牌和產品的聲量，提高消費者對品牌的認知度與好感度。\n議題文：透過詢問、時事、煩惱，日常生活等問題帶起論壇內的討論度。\n\n分享 / 比較文：包含開箱文、競品分析，透過這一系列的操作，加深消費者對品牌的認知。\n\n情境文：創意、感人的情境故事，自然的把品牌、產品融入其中，可達到高度的網路聲量。" \
    --with_prompt \
    --interactive
# 大型語言模型指令微調流程 LLM Instruction Fine-Tuning

<!-- ![GitHub Repo stars](https://img.shields.io/github/stars/A-baoYang/LLM-FineTuning-Guide?style=social) -->
![GitHub Code License](https://img.shields.io/github/license/A-baoYang/LLM-FineTuning-Guide)
![GitHub last commit](https://img.shields.io/github/last-commit/A-baoYang/LLM-FineTuning-Guide)
![GitHub pull request](https://img.shields.io/badge/PRs-welcome-blue)

本專案整理了微調大型語言模型的重要觀念和實作的程式框架，針對 LLMs 的訓練和推論提供運行範例。

👋 歡迎加入我們的 Line 社群 Open Chat：[大型語言模型微調及 OpenAI 應用討論群](assets/line-openchat.jpg)

切換語言版本： \[ [English](README.md) | [繁體中文](README-zhtw.md) | [简体中文](README-zhcn.md) \]

> 如果你希望減少試錯，歡迎報名我親自錄製的手把手教學課程：
> - 教學課程預購：https://dataagent.kaik.io/learning/llm-instruction-finetune

## 最新消息 Development Log

- [2023/04/15] 更新新資料集：

## 資料集 Datasets

- medical

詳細內容請查看 [instruction-datasets/README.md](./instruction-datasets/README.md)

## 支援的大型語言模型 LLMs

- LLaMA
- Bloom
- ChatGLM-6B

詳細介紹請查看 [LLM 介紹](./docs/LLMs.md)

## 高效微調方法 Efficient Parameters Fine-Tuning Methods

目前支援以下高效微調方式：

- LoRA 
- P-tuning V2

硬體需求：

| LLM | 微調方法 | 量化方法 | 分散式策略 | Batch Size | 所需 GPU 記憶體(單張) | 速度 |
| --- | --- | --- | --- | --- | --- | --- |
| Bloom | LoRA | INT8 | None | 1 | 14GB | 86.71s/it |
| Bloom | LoRA | INT8 | Torch DDP on 2 GPUs | 1 | 13GB | 44.47s/it |
| Bloom | LoRA | INT8 | DeepSpeed ZeRO stage 3 on 2 GPUs | 1 | 13GB | 36.05s/it |
| ChatGLM-6B | P-Tuning | INT4 | DeepSpeed ZeRO stage 3 on 2 GPUs | 2 | 15GB | 14.7s/it |

---

## 如何開始？ Getting Started

### 資料集準備 Data Preparation

你可以選擇使用開源或學術資料集進行微調；但如果開源資料集不符合您的應用情境，您就會需要使用自定義資料集來進行。

在本專案資料集所使用格式是 `.json` ，你會需要將資料集 train, dev, test 分隔後的檔案放到 `instruction-datasets/` 下，您也可以另外創新資料夾放置，只是路徑指定的差異，都可以在 commands 做修改。

### 環境準備 Requirements

針對不同的微調方法有設定好所需的套件，只要進到有 `requirements.txt` 的資料夾下運行

```bash
git clone https://github.com/A-baoYang/LLM-FineTuning-Guide.git
conda create -n llm_ift python=3.8
conda activate llm_ift
cd LLM-Finetune-Guide/efficient-finetune/ptuning/v2
pip install -r requirements.txt
```

## 微調 Fine-Tuning

資料準備好之後就可以啟動微調，這裡已經將程式寫好，當中的資料/模型路徑、參數置換都可以透過指令來指定。

### 單張 GPU 微調 Fine-Tuning with single GPU

```bash
CUDA_VISIBLE_DEVICES=0 python finetune.py \
    --do_train \
    --train_file ../../../instruction-datasets/$DATATAG/train.json \
    --validation_file ../../../instruction-datasets/$DATATAG/dev.json \
    --prompt_column input \
    --response_column output \
    --overwrite_cache \
    --model_name_or_path $MODEL_PATH \
    --output_dir finetuned/$DATATAG-$MODEL_TYPE-pt-$PRE_SEQ_LEN-$LR
```

完整的參數和指令設定請見： [finetune.sh](./efficient-finetune/ptuning/v2/finetune.sh)

### 多張 GPU 微調 Fine-Tuning with multiple GPUs

- 使用 torchrun 啟動

```bash
torchrun --standalone --nnodes=1  --nproc_per_node=2 finetune.py --do_train \
    --train_file ../../../instruction-datasets/$DATATAG/train.json \
    --validation_file ../../../instruction-datasets/$DATATAG/dev.json \
    --prompt_column input \
    --response_column output \
    --overwrite_cache \
    --model_name_or_path $MODEL_PATH \
    --output_dir finetuned/$DATATAG-$MODEL_TYPE-pt-$PRE_SEQ_LEN-$LR \
```

完整的參數和指令設定請見： [finetune-ddp.sh](./efficient-finetune/ptuning/v2/finetune-ddp.sh)

- 使用 accelerate 啟動

```bash
accelerate launch finetune.py --do_train \
    --train_file ../../../instruction-datasets/$DATATAG/train.json \
    --validation_file ../../../instruction-datasets/$DATATAG/dev.json \
    --prompt_column input \
    --response_column output \
    --overwrite_cache \
    --model_name_or_path $MODEL_PATH \
    --output_dir finetuned/$DATATAG-$MODEL_TYPE-pt-$PRE_SEQ_LEN-$LR \
```

### 使用 DeepSpeed ZeRO 策略進行分散式訓練

- 使用 accelerate 帶上 config_file 啟動

```bash
accelerate launch --config_file ../../config/use_deepspeed.yaml finetune.py --do_train \
    --train_file ../../../instruction-datasets/$DATATAG/train.json \
    --validation_file ../../../instruction-datasets/$DATATAG/dev.json \
    --prompt_column input \
    --response_column output \
    --overwrite_cache \
    --model_name_or_path $MODEL_PATH \
    --output_dir finetuned/$DATATAG-$MODEL_TYPE-pt-$PRE_SEQ_LEN-$LR \
```

- 使用 deepspeed 啟動

```bash
deepspeed --num_nodes 1 --num_gpus 2 finetune.py \
    --deepspeed ../../config/zero_stage3_offload_config.json \
    --do_train \
    --train_file ../../../instruction-datasets/$DATATAG/train.json \
    --validation_file ../../../instruction-datasets/$DATATAG/dev.json \
    --prompt_column input \
    --response_column output \
    --overwrite_cache \
    --model_name_or_path $MODEL_PATH \
    --output_dir finetuned/$DATATAG-$MODEL_TYPE-pt-$PRE_SEQ_LEN-$LR \
```

- 更多微調案例請看：[efficient-finetune/README.md](./efficient-finetune/README.md)

## 模型評估與預測 Evaluation & Prediction

```bash
CUDA_VISIBLE_DEVICES=0 python finetune.py \
    --do_predict \
    --validation_file ../../../instruction-datasets/$DATATAG/dev.json \
    --test_file ../../../instruction-datasets/$DATATAG/test.json \
    --overwrite_cache \
    --prompt_column input \
    --response_column output \
    --model_name_or_path $MODEL_PATH \
    --ptuning_checkpoint finetuned/$DATATAG-$MODEL_TYPE-pt-$PRE_SEQ_LEN-$LR/checkpoint-$STEP \
    --output_dir finetuned/$DATATAG-$MODEL_TYPE-pt-$PRE_SEQ_LEN-$LR
```

## 模型推論 Run Inference

- 終端機

```bash
cd LLM-Finetune-Guide/efficient-finetune/ptuning/v2/serve/
CUDA_VISIBLE_DEVICES=0 python cli_demo.py \
    --pretrained_model_path THUDM/chatglm-6b \
    --ptuning_checkpoint ../finetuned/chatglm-6b-pt-512-2e-2/checkpoint-3000 \
    --is_cuda True
```

- 網頁展示
```bash
cd LLM-Finetune-Guide/efficient-finetune/lora/serve/
python ui.py
```

- Model API

```bash
cd LLM-Finetune-Guide/efficient-finetune/lora/serve/
python api.py
```

## 在 CPU 環境下提速運行

最後特別把 CPU 提出來講，因為如果可以做到在 CPU 環境下運行 finetune 過的大語言模型，
會最大比例的節省運算成本。這一塊目前方法有：

- 將模型檔轉換為 cpp 格式
    - LLaMA + LoRA -> llama.cpp
    - Bloom + LoRA -> bloom.cpp

- 使用 INT4 於 CPU 環境運行，速度可接受

```bash
cd LLM-Finetune-Guide/efficient-finetune/ptuning/v2/serve/
CUDA_VISIBLE_DEVICES=0 python cli_demo.py \
    --pretrained_model_path THUDM/chatglm-6b \
    --ptuning_checkpoint ../finetuned/chatglm-6b-pt-512-2e-2/checkpoint-3000 \
    --quantization_bit 4 \
    --is_cuda True
```

## TODO

- [ ] 程式碼重構

---

## License

- 專案 License：[Apache-2.0 License](./LICENSE)
- 模型 License：請參照各大語言模型所提供之 License，詳細請見 [LLM 介紹](./docs/LLMs.md)

## Citation

如果這項專案對你的工作或研究有幫助，請引用：

```
@Misc{LLM-FineTuning-Guide,
  title = {LLM FineTuning Guide},
  author = {A-baoYang},
  howpublished = {\url{https://github.com/A-baoYang/LLM-Finetuning-Guide}},
  year = {2023}
}
```

## Acknowledgement

此專案從以下地方獲取靈感，感謝這些很讚的專案：

- [THUDM/ChatGLM-6B]
- [ymcui/Chinese-LLaMA-Alpaca]
- [tloen/alpaca-lora]

# 大型语言模型指令微调流程 LLM Instruction Fine-Tuning

![GitHub Repo stars](https://img.shields.io/github/stars/A-baoYang/LLM-FineTuning-Guide?style=social)
![GitHub Code License](https://img.shields.io/github/license/A-baoYang/LLM-FineTuning-Guide)
![GitHub last commit](https://img.shields.io/github/last-commit/A-baoYang/LLM-FineTuning-Guide)
![GitHub pull request](https://img.shields.io/badge/PRs-welcome-blue)

本专案整理了微调大型语言模型的重要观念和实作的程式框架，针对 LLMs 的训练和推论提供运行范例。

👋 欢迎加入我们的 Line 社群 Open Chat：[大型语言模型微调及 OpenAI 应用讨论群](assets/line-openchat.jpg)

切换语言版本： \[ [English](README.md) | [繁体中文](README-zhtw.md) | [简体中文](README-zhcn.md) \]

> 如果你希望减少试错，欢迎报名我亲自录製的手把手教学课程：
> - 填写问卷领取优惠折扣： [https://www.surveycake.com/s/kn0bL](https://www.surveycake.com/s/kn0bL)

<!-- ## 最新消息 Development Log

- [2023/04/15] 更新新资料集： -->

<!-- ## 资料集 Datasets

- medical

详细内容请查看 [instruction-datasets/README.md](./instruction-datasets/README.md)

## 支援的大型语言模型 LLMs

- LLaMA
- Bloom
- ChatGLM-6B

详细介绍请查看 [LLM 介绍](./docs/LLMs.md) -->

## 高效微调方法 Efficient Parameters Fine-Tuning Methods

目前支援以下高效微调方式：

- LoRA 
- P-tuning V2

硬体需求：

| LLM | 微调方法 | 量化方法 | 分散式策略 | Batch Size | 所需 GPU 记忆体(单张) | 速度 |
| --- | --- | --- | --- | --- | --- | --- |
| Bloom | LoRA | INT8 | None | 1 | 14GB | 86.71s/it |
| Bloom | LoRA | INT8 | Torch DDP on 2 GPUs | 1 | 13GB | 44.47s/it |
| Bloom | LoRA | INT8 | DeepSpeed ZeRO stage 3 on 2 GPUs | 1 | 13GB | 36.05s/it |
| ChatGLM-6B | P-Tuning | INT4 | DeepSpeed ZeRO stage 3 on 2 GPUs | 2 | 15GB | 14.7s/it |

---

## 如何开始？ Getting Started

### 资料集准备 Data Preparation

你可以选择使用开源或学术资料集进行微调；但如果开源资料集不符合您的应用情境，您就会需要使用自定义资料集来进行。

在本专案资料集所使用格式是 `.json` ，你会需要将资料集 train, dev, test 分隔后的档案放到 `instruction-datasets/` 下，您也可以另外创新资料夹放置，只是路径指定的差异，都可以在 commands 做修改。

### 环境准备 Requirements

针对不同的微调方法有设定好所需的套件，只要进到有 `requirements.txt` 的资料夹下运行

```bash
git clone https://github.com/A-baoYang/LLM-FineTuning-Guide.git
conda create -n llm_ift python=3.8
conda activate llm_ift
cd LLM-Finetune-Guide/efficient-finetune/ptuning/v2
pip install -r requirements.txt
```

## 微调 Fine-Tuning

资料准备好之后就可以启动微调，这裡已经将程式写好，当中的资料/模型路径、参数置换都可以透过指令来指定。

### 单张 GPU 微调 Fine-Tuning with single GPU

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

完整的参数和指令设定请见： [finetune.sh](./efficient-finetune/ptuning/v2/finetune.sh)

### 多张 GPU 微调 Fine-Tuning with multiple GPUs

- 使用 torchrun 启动

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

完整的参数和指令设定请见： [finetune-ddp.sh](./efficient-finetune/ptuning/v2/finetune-ddp.sh)

- 使用 accelerate 启动

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

### 使用 DeepSpeed ZeRO 策略进行分散式训练

- 使用 accelerate 带上 config_file 启动

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

- 使用 deepspeed 启动

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

- 更多微调案例请看：[efficient-finetune/README.md](./efficient-finetune/README.md)

## 模型评估与预测 Evaluation & Prediction

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

## 模型推论 Run Inference

- 终端机

```bash
cd LLM-Finetune-Guide/efficient-finetune/ptuning/v2/serve/
CUDA_VISIBLE_DEVICES=0 python cli_demo.py \
    --pretrained_model_path THUDM/chatglm-6b \
    --ptuning_checkpoint ../finetuned/chatglm-6b-pt-512-2e-2/checkpoint-3000 \
    --is_cuda True
```

- 网页展示
```bash
cd LLM-Finetune-Guide/efficient-finetune/lora/serve/
python ui.py
```

- Model API

```bash
cd LLM-Finetune-Guide/efficient-finetune/lora/serve/
python api.py
```

## 在 CPU 环境下提速运行

如果可以做到在 CPU 环境下运行 finetune 过的大语言模型，会最大比例的降低 LLM 的应用门槛。

- 使用 INT4 于 CPU 环境运行，速度可接受

```bash
cd LLM-Finetune-Guide/efficient-finetune/ptuning/v2/serve/
CUDA_VISIBLE_DEVICES=0 python cli_demo.py \
    --pretrained_model_path THUDM/chatglm-6b \
    --ptuning_checkpoint ../finetuned/chatglm-6b-pt-512-2e-2/checkpoint-3000 \
    --quantization_bit 4 \
    --is_cuda True
```

---

## License

- 专案 License：[Apache-2.0 License](./LICENSE)
- 模型 License：请参照各大语言模型所提供之 License
<!-- 详细请见 [LLM 介绍](./docs/LLMs.md) -->

## Citation

如果这项专案对你的工作或研究有帮助，请引用：

```
@Misc{LLM-Finetune-Guide,
  title = {LLM Finetune Guide},
  author = {A-baoYang},
  howpublished = {\url{https://github.com/A-baoYang/LLM-Finetune-Guide}},
  year = {2023}
}
```

## Acknowledgement

此专案从以下地方获取灵感，感谢这些很讚的专案：

- [THUDM/ChatGLM-6B]
- [ymcui/Chinese-LLaMA-Alpaca]
- [tloen/alpaca-lora]

## Contact

有任何问题或建议，欢迎来信询问： [jiunyi.yang.abao@gmail.com](mailto:jiunyi.yang.abao@gmail.com)
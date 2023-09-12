# LLM Instruction Fine-Tuning

![GitHub Repo stars](https://img.shields.io/github/stars/A-baoYang/LLM-FineTuning-Guide?style=social)
![GitHub Code License](https://img.shields.io/github/license/A-baoYang/LLM-FineTuning-Guide)
![GitHub last commit](https://img.shields.io/github/last-commit/A-baoYang/LLM-FineTuning-Guide)
![GitHub pull request](https://img.shields.io/badge/PRs-welcome-blue)

This project compiles important concepts and programming frameworks for fine-tuning large language models, providing executable examples for training and inference of LLMs.

👋 Welcome to join our Line community Open Chat: [fine-tuning large language models and OpenAI applications](assets/line-openchat.jpg)

Switch language version: \[ [English](README.md) | [繁體中文](README-zhtw.md) | [简体中文](README-zhcn.md) \]

> If you want to reduce trial and error, you are welcome to enroll in my personally recorded step-by-step tutorial course:
> - Fill out the survey to receive a discount voucher: [https://www.surveycake.com/s/kn0bL](https://www.surveycake.com/s/kn0bL)

![A-baoYang's GitHub stats](https://github-readme-stats.vercel.app/api?username=A-baoYang&show=reviews,discussions_started,discussions_answered,prs_merged,prs_merged_percentage&theme=radical)



<!-- ## 最新消息 Development Log

- [2023/04/15] 更新新資料集： -->

<!-- ## 資料集 Datasets

- medical

詳細內容請查看 [instruction-datasets/README.md](./instruction-datasets/README.md)

## 支援的大型語言模型 LLMs

- LLaMA
- Bloom
- ChatGLM-6B

詳細介紹請查看 [LLM 介紹](./docs/LLMs.md) -->

## Efficient Parameters Fine-Tuning Methods

Currently, the following efficient fine-tuning methods are supported:

- LoRA 
- P-tuning V2

Training Arguments:

| LLM | Fine-Tuning Method | Quantization Methods | Distributed Training Strategy | Batch Size | Required GPU memory (per card) | Speed |
| --- | --- | --- | --- | --- | --- | --- |
| Bloom | LoRA | INT8 | None | 1 | 14GB | 86.71s/it |
| Bloom | LoRA | INT8 | Torch DDP on 2 GPUs | 1 | 13GB | 44.47s/it |
| Bloom | LoRA | INT8 | DeepSpeed ZeRO stage 3 on 2 GPUs | 1 | 13GB | 36.05s/it |
| ChatGLM-6B | P-Tuning | INT4 | DeepSpeed ZeRO stage 3 on 2 GPUs | 2 | 15GB | 14.7s/it |

---

## Getting Started

### Data Preparation

You can choose to fine-tune with open-source or academic datasets, but if the open-source datasets do not fit your application scenario, you will need to use custom datasets for fine-tuning.

In this project, the format used for the dataset is `.json`. You will need to put the train, dev, and test files of the separated dataset in the `instruction-datasets/` directory. You can also create a new folder to place the files, but the path should be specified accordingly in the commands.

### Requirements

Different fine-tuning methods have their required packages set up. To install them, simply navigate to the folder with `requirements.txt` and run:

```bash
git clone https://github.com/A-baoYang/LLM-FineTuning-Guide.git
conda create -n llm_ift python=3.8
conda activate llm_ift
cd LLM-Finetune-Guide/efficient-finetune/ptuning/v2
pip install -r requirements.txt
```

## Fine-Tuning

After the data is prepared, you can start fine-tuning. The program has already been written and you can specify the data/model path and parameter replacement through the command.

### Fine-Tuning with single GPU

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

Please refer to the complete parameter and command settings at: [finetune.sh](./efficient-finetune/ptuning/v2/finetune.sh)

### Fine-Tuning with multiple GPUs

- Start with `torchrun`

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

Please refer to the complete parameter and command settings at:  [finetune-ddp.sh](./efficient-finetune/ptuning/v2/finetune-ddp.sh)

- Start with `accelerate`

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

### Use DeepSpeed ZeRO strategy for distributed training

- Start with `accelerate` and `config_file` arguments

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

- Start with `deepspeed`

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

- For more fine-tuning examples, see: [efficient-finetune/README.md](./efficient-finetune/README.md)

## Evaluation & Prediction

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

## Run Inference

- Terminal

```bash
cd LLM-Finetune-Guide/efficient-finetune/ptuning/v2/serve/
CUDA_VISIBLE_DEVICES=0 python cli_demo.py \
    --pretrained_model_path THUDM/chatglm-6b \
    --ptuning_checkpoint ../finetuned/chatglm-6b-pt-512-2e-2/checkpoint-3000 \
    --is_cuda True
```

- Web demo

```bash
cd LLM-Finetune-Guide/efficient-finetune/lora/serve/
python ui.py
```

- Model API

```bash
cd LLM-Finetune-Guide/efficient-finetune/lora/serve/
python api.py
```

## Running on CPU environment

The ability to run fine-tuned large language models in a CPU environment would greatly reduce the application threshold of LLMs.

- Use INT4 to run in CPU environment

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

- Repository License: [Apache-2.0 License](./LICENSE)
- Model License: Please refer to the license provided by each language model for details. 
<!-- For more information, see [LLM Introduction](./docs/LLMs.md) -->

## Citation

If this project is helpful to your work or research, please star & cite it as follows:

```
@Misc{LLM-Finetune-Guide,
  title = {LLM Finetune Guide},
  author = {A-baoYang},
  howpublished = {\url{https://github.com/A-baoYang/LLM-Finetune-Guide}},
  year = {2023}
}
```

## Acknowledgement

This project was inspired by some amazing projects, which are listed below. Thanks for their great work.

- [THUDM/ChatGLM-6B]
- [ymcui/Chinese-LLaMA-Alpaca]
- [tloen/alpaca-lora]

## Contact

If you have any questions or suggestions, please feel free to email us for inquiries: [jiunyi.yang.abao@gmail.com](mailto:jiunyi.yang.abao@gmail.com)
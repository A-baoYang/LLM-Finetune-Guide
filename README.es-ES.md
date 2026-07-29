# Ajuste Fino de Instrucciones para LLM (LLM Instruction Fine-Tuning)

![GitHub Repo stars](https://img.shields.io/github/stars/A-baoYang/LLM-FineTuning-Guide?style=social)
![GitHub Code License](https://img.shields.io/github/license/A-baoYang/LLM-FineTuning-Guide)
![GitHub last commit](https://img.shields.io/github/last-commit/A-baoYang/LLM-FineTuning-Guide)
![GitHub pull request](https://img.shields.io/badge/PRs-welcome-blue)

Este proyecto recopila conceptos importantes y marcos de programación para el ajuste fino (fine-tuning) de modelos de lenguaje extensos, proporcionando ejemplos ejecutables para el entrenamiento e inferencia de LLMs.

👋 Bienvenido a nuestra comunidad de Open Chat en Line: [fine-tuning large language models and OpenAI applications](assets/line-openchat.jpg)

Cambiar idioma: \[ [English](README.md) | [繁體中文](README-zhtw.md) | [简体中文](README-zhcn.md) \]

> Si deseas reducir el proceso de prueba y error, te invito a inscribirte en mi curso tutorial paso a paso grabado personalmente:
> - Completa la encuesta para recibir un cupón de descuento: [https://www.surveycake.com/s/kn0bL](https://www.surveycake.com/s/kn0bL)

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

## Métodos de Ajuste Fino de Parámetros Eficientes (Efficient Parameters Fine-Tuning Methods)

Actualmente, se admiten los siguientes métodos de ajuste fino eficiente:

- LoRA 
- P-tuning V2

Argumentos de Entrenamiento:

| LLM | Método de Ajuste Fino | Métodos de Cuantización | Estrategia de Entrenamiento Distribuido | Tamaño del Lote (Batch Size) | Memoria GPU requerida (por tarjeta) | Velocidad |
| --- | --- | --- | --- | --- | --- | --- |
| Bloom | LoRA | INT8 | Ninguno | 1 | 14GB | 86.71s/it |
| Bloom | LoRA | INT8 | Torch DDP en 2 GPUs | 1 | 13GB | 44.47s/it |
| Bloom | LoRA | INT8 | DeepSpeed ZeRO etapa 3 en 2 GPUs | 1 | 13GB | 36.05s/it |
| ChatGLM-6B | P-Tuning | INT4 | DeepSpeed ZeRO etapa 3 en 2 GPUs | 2 | 15GB | 14.7s/it |

---

## Primeros Pasos (Getting Started)

### Preparación de Datos

Puedes optar por realizar el ajuste fino con conjuntos de datos académicos o de código abierto, pero si los conjuntos de datos abiertos no se ajustan a tu escenario de aplicación, deberás utilizar conjuntos de datos personalizados.

En este proyecto, el formato utilizado para el conjunto de datos es `.json`. Deberás colocar los archivos de entrenamiento (train), desarrollo (dev) y prueba (test) del conjunto de datos separado en el directorio `instruction-datasets/`. También puedes crear una carpeta nueva para colocar los archivos, pero la ruta debe especificarse adecuadamente en los comandos.

### Requisitos

Los diferentes métodos de ajuste fino tienen sus propios paquetes requeridos. Para instalarlos, simplemente navega hasta la carpeta que contiene el archivo `requirements.txt` y ejecuta:

```bash
git clone https://github.com/A-baoYang/LLM-FineTuning-Guide.git
conda create -n llm_ift python=3.8
conda activate llm_ift
cd LLM-Finetune-Guide/efficient-finetune/ptuning/v2
pip install -r requirements.txt
```

## Ajuste Fino (Fine-Tuning)

Una vez preparados los datos, puedes comenzar el ajuste fino. El programa ya ha sido escrito y puedes especificar la ruta de los datos/modelo y la sustitución de parámetros a través del comando.

### Ajuste Fino con una sola GPU

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

Consulta la configuración completa de parámetros y comandos en: [finetune.sh](./efficient-finetune/ptuning/v2/finetune.sh)

### Ajuste Fino con múltiples GPUs

- Inicio con `torchrun`

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

Consulta la configuración completa de parámetros y comandos en: [finetune-ddp.sh](./efficient-finetune/ptuning/v2/finetune-ddp.sh)

- Inicio con `accelerate`

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

### Uso de la estrategia DeepSpeed ZeRO para entrenamiento distribuido

- Inicio con `accelerate` y argumentos de `config_file`

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

- Inicio con `deepspeed`

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

- Para más ejemplos de ajuste fino, consulta: [efficient-finetune/README.md](./efficient-finetune/README.md)

## Evaluación y Predicción

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

## Ejecutar Inferencia

- Terminal

```bash
cd LLM-Finetune-Guide/efficient-finetune/ptuning/v2/serve/
CUDA_VISIBLE_DEVICES=0 python cli_demo.py \
    --pretrained_model_path THUDM/chatglm-6b \
    --ptuning_checkpoint ../finetuned/chatglm-6b-pt-512-2e-2/checkpoint-3000 \
    --is_cuda True
```

- Demo Web

```bash
cd LLM-Finetune-Guide/efficient-finetune/lora/serve/
python ui.py
```

- API del Modelo

```bash
cd LLM-Finetune-Guide/efficient-finetune/lora/serve/
python api.py
```

## Ejecución en entorno CPU

La capacidad de ejecutar modelos de lenguaje extensos ajustados en un entorno de CPU reduciría considerablemente la barrera de aplicación de los LLMs.

- Uso de INT4 para ejecutar en entorno CPU

```bash
cd LLM-Finetune-Guide/efficient-finetune/ptuning/v2/serve/
CUDA_VISIBLE_DEVICES=0 python cli_demo.py \
    --pretrained_model_path THUDM/chatglm-6b \
    --ptuning_checkpoint ../finetuned/chatglm-6b-pt-512-2e-2/checkpoint-3000 \
    --quantization_bit 4 \
    --is_cuda True
```

---

## Licencia

- Licencia del Repositorio: [Apache-2.0 License](./LICENSE)
- Licencia del Modelo: Por favor, consulta la licencia proporcionada por cada modelo de lenguaje para más detalles. 
<!-- Para más información, consulta [LLM Introduction](./docs/LLMs.md) -->

## Citación

Si este proyecto es útil para tu trabajo o investigación, por favor dale una estrella y cítalo de la siguiente manera:

```
@Misc{LLM-Finetune-Guide,
  title = {LLM Finetune Guide},
  author = {A-baoYang},
  howpublished = {\url{https://github.com/A-baoYang/LLM-Finetune-Guide}},
  year = {2023}
}
```

## Agradecimientos

Este proyecto se inspiró en algunos proyectos increíbles que se enumeran a continuación. Gracias por su gran trabajo.

- [THUDM/ChatGLM-6B]
- [ymcui/Chinese-LLaMA-Alpaca]
- [tloen/alpaca-lora]

## Contacto

Si tienes alguna pregunta o sugerencia, no dudes en enviarnos un correo electrónico para consultas: [jiunyi.yang.abao@gmail.com](mailto:jiunyi.yang.abao@gmail.com)

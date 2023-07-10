# 純 CPU 環境佈署：轉換模型格式、量化、開大 CPU threads

### Prerequisite

安裝相對應版本 llama.cpp 及其 Python bindings

- `0.1.48` 可以支援舊的 ggml 4bit 格式，但還未支援 cuBLAS，還不能用來在 GPU 上運行
- `0.1.54` 可以支援在 GPU 上 inference，但還沒升級到可以支援新的 ggml 4bit 格式...

#### Normal Install (CPU only)
```bash
pip install llama-cpp-python==0.1.48 --force-reinstall --upgrade --no-cache-dir
```

#### Install with `cuBLAS`
```bash
LLAMA_CUBLAS=1
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir
```

### 1. 合併 LLaMA 模型及微調好的 LoRA 權重

**！重要！** 先從預訓練模型中複製 `tokenizer.model`, `tokenizer_config.json`, `special_tokens_map.json` 到 LoRA 模型檔所在的資料夾，接著再運行下面的 bash script

```
bash 1-chinese-llama-7b-plus-combined.sh
```

- 運行 inference_hf.py 來測試模型生成能力

**！重要！** 注意要自己調整 generation_config 參數，若是固定格式的任務，用 do_sample=False 越好

參考參數：
```
generation_config = dict(
    do_sample=False,
    num_beams=4,
    max_new_tokens=1024,
    early_stopping=True
)
```

```
cd ~/Chinese-LLaMA-Alpaca/scripts
python inference_hf.py --base_model /home/jovyan/stock-pred/temp_model/multi-ee-llama-1024-3e-4-combined  --interactive --only_cpu
```

### 2. 將合併後的 LLaMA+LoRA 模型轉換為 ggml 格式
```
bash 2-convert-merged-llama-to-ggml.sh
```

> log
```bash
...
Loading vocab file /home/jovyan/gpt/open_gpt/llama.cpp/zh-models/tokenizer.model                  
Writing vocab...
[  1/291] Writing tensor tok_embeddings.weight                  | size  49953 x   4096  | type Unq
uantizedDataType(name='F16')
[  2/291] Writing tensor norm.weight                            | size   4096           | type Unq
uantizedDataType(name='F32')
[  3/291] Writing tensor output.weight                          | size  49953 x   4096  | type Unq
uantizedDataType(name='F16')
[  4/291] Writing tensor layers.0.attention.wq.weight           | size   4096 x   4096  | type Unq
uantizedDataType(name='F16')
[  5/291] Writing tensor layers.0.attention.wk.weight           | size   4096 x   4096  | type Unq
uantizedDataType(name='F16')
[  6/291] Writing tensor layers.0.attention.wv.weight           | size   4096 x   4096  | type Unq
uantizedDataType(name='F16')
[  7/291] Writing tensor layers.0.attention.wo.weight           | size   4096 x   4096  | type Unq
uantizedDataType(name='F16')
[  8/291] Writing tensor layers.0.attention_norm.weight         | size   4096           | type Unq
uantizedDataType(name='F32')
[  9/291] Writing tensor layers.0.feed_forward.w1.weight        | size  11008 x   4096  | type Unq
uantizedDataType(name='F16')
[ 10/291] Writing tensor layers.0.feed_forward.w2.weight        | size   4096 x  11008  | type Unq
...
```

運行 ggml 模型服務（終端機）
```
python serve-cli-ggml.py --ggml_model <finetuned_model>/ggml-model-f16.bin --n_threads 16
```

### 3. 編譯 llama.cpp 專案內容

**！重要！** (2023.05.24) 為了配合 python binding `llama-cpp-python` 相容的版本， `llama.cpp` 需切到 commit ID: `cf348a6` [ref.](https://github.com/abetlen/llama-cpp-python/issues/204) (是 2023.5.10 之前的版本)

```
!git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
git checkout cf348a6
pip install -r requirements.txt
make
```

### 4. 將模型從 FP16 轉為 4bit，大小可減少至原來的 1/4 倍
```
cd llama.cpp
COMBINED_MODEL=multi-ee-llama-1024-3e-4-combined
./quantize ./zh-models/$COMBINED_MODEL/ggml-model-f16.bin ./zh-models/$COMBINED_MODEL/ggml-model-q4_0.bin 2
```

> log
```bash
main: build = 580 (fab49c6)
main: quantizing './zh-models/multi-ee-llama-1024-3e-4-combined/ggml-model-f16.bin' to './zh-models/multi-ee-llama-10
24-3e-4-combined/ggml-model-q4_0.bin' as q4_0
llama.cpp: loading model from ./zh-models/multi-ee-llama-1024-3e-4-combined/ggml-model-f16.bin
llama.cpp: saving model to ./zh-models/multi-ee-llama-1024-3e-4-combined/ggml-model-q4_0.bin
[   1/ 291]                tok_embeddings.weight -     4096 x 32000, type =    f16, quantizing .. size =   250.00 MB 
->    70.31 MB | hist: 0.037 0.016 0.025 0.039 0.057 0.077 0.096 0.111 0.116 0.111 0.096 0.077 0.057 0.039 0.025 0.021 
[   2/ 291]                          norm.weight -             4096, type =    f32, size =    0.016 MB
[   3/ 291]                        output.weight -     4096 x 32000, type =    f16, quantizing .. size =   250.00 MB 
->    70.31 MB | hist: 0.036 0.015 0.025 0.038 0.056 0.076 0.097 0.113 0.119 0.112 0.097 0.076 0.056 0.038 0.025 0.020 
[   4/ 291]         layers.0.attention.wq.weight -     4096 x  4096, type =    f16, quantizing .. size =    32.00 MB 
->     9.00 MB | hist: 0.035 0.012 0.019 0.030 0.047 0.069 0.097 0.129 0.151 0.129 0.098 0.070 0.047 0.031 0.019 0.016 
[   5/ 291]         layers.0.attention.wk.weight -     4096 x  4096, type =    f16, quantizing .. size =    32.00 MB 
->     9.00 MB | hist: 0.035 0.012 0.020 0.032 0.049 0.072 0.098 0.125 0.139 0.125 0.099 0.072 0.050 0.033 0.021 0.017
...
[ 290/ 291]     layers.31.feed_forward.w3.weight -     4096 x 11008, type =    f16, quantizing .. size =    86.00 MB 
->    24.19 MB | hist: 0.036 0.015 0.025 0.039 0.056 0.077 0.097 0.111 0.117 0.112 0.097 0.077 0.056 0.039 0.025 0.021 
[ 291/ 291]            layers.31.ffn_norm.weight -             4096, type =    f32, size =    0.016 MB
llama_model_quantize_internal: model size  = 12853.02 MB
llama_model_quantize_internal: quant size  =  3615.64 MB
llama_model_quantize_internal: hist: 0.036 0.016 0.025 0.039 0.056 0.077 0.096 0.111 0.117 0.111 0.096 0.077 0.056 0.039 0.025 0.021 

main: quantize time = 47233.85 ms
main:    total time = 47233.85 ms
```

運行 4bit 模型服務（終端機）
```
bash serve-cli-4bit.sh
```

調整參數請見 [llama.cpp 參數說明文件](llama.cpp/examples/main/README.md)

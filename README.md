# 大型語言模型 指令微調流程統整 LLM Instruction Fine-Tuning

本專案整理了 LLaMA, Bloom, ChatGLM 三種開源 LLMs 的運行範例。

如果你希望盡量減少試錯，歡迎報名我親自錄製的手把手教學課程：

- 教學課程預購：https://dataagent.kaik.io/learning/llm-instruction-finetune

## 何謂指令微調  Instruction Fine-Tuning

指令微調方法源自於華盛頓大學 Self-Instruct 這篇論文，透過從 ChatGPT 取得的種子指令集，微調當時的舊版 GPT-3 模型，發現可以獲得更接近 InstructGPT 的成效。在一個已經表現不錯的大型語言模型上，透過指令微調可以讓模型在 Zero-shot 上的表現更好。

後續此法被 Stanford 用於微調 LLaMA-7B，並且將這個方法產生的模型稱為 Alpaca ；UC Berkerly 也有微調 LLaMA-13B ，稱之為 Viunca。

> LLaMA 介紹
> Stanford Alpaca 介紹
> UC Berkerly Viunca 介紹

## 指令集生成方法

指令微調大模型首先需要高品質的指令集，才能讓模型很好的模仿指令內容進行輸出。

### 固定 input/output

適合作為指令內容的通常是怎樣的資料呢？ 

如果你有 input/output 格式固定的任務，例如固定從文章中抓出人/事/時/地/物、輸入商品特徵輸出文案等等這種例子，那就很適合作為指令資料集來訓練一個可以為你處理特定任務的 LLM。

補軟體文件生成的例子

### 從文本中生成問答對

那如果不是問答 pair 要怎麼辦呢，很簡單此時可以用方法從文本中生成很多的問答對出來。

品質比較高但花錢的實現方法是從 ChatGPT 下 prompt 取得，還可以指定他的問題要鎖定哪些關鍵主題、或是調整字數

品質普普、不花錢的實現方法就是用 LangChain 的 QAGeneration

> 補 QAGeneration 原理說明

### DataAgent 釋出公開資料集

我從幾個公開網站爬取並整理成資料集，有多個不同產業領域的，請不吝按個愛心後取用

> 補贊助按鈕

- hf link

如果想要一次看所有資料集介紹，可以到我整理的 Github 專案： instruction-datasets

## 模型壓縮方法

完成了資料集生成後，接下來進到模型部分。

為了要能在消費級顯卡上順利運行大型語言模型，我們需要將模型進行壓縮，詳細可以分為以下幾種：

- 模型剪枝
- 模型量化
- 模型蒸餾

> 補每一種方法介紹
> 補 flexgen 方法介紹

## 高效微調方法

可以成功載入模型還不夠，微調時所需的 GPU 資源遠比推論更大；
但實際上我們不需要微調所有參數，因為並非所有大型語言模型參數都適用於當前的任務，
我們其實可以透過一些方式來只微調部分參數、節省訓練資源。

在這邊由 Huggingface 提出的 PEFT 便是集大成，包含以下幾種方法：

- LoRA
- P-Tuning
- Prefix Tuning
- Prompt Tuning

## 模型加速

使用了高效微調方法，可以將模型在 finetune 時所使用的資源減到最小，
最後我們要做 scale，開多個 process 來加速訓練和推論。這一類的技術稱為分散式訓練，其中包含了不同策略：

- ZeRO Redunency 3

> 補介紹

針對以上策略，實作的框架有以下幾種：

- torchrun (torch.distributed)
    - 純分散式訓練
- accelerate
    - 提供 deepspeed, fsdp 策略
- DeepSpeed

### 運行訓練

### 運行推論

- 單張 GPU

- 多張 GPU

## 在 CPU 環境下運行

最後特別把 CPU 提出來講，因為如果可以做到在 CPU 環境下運行 finetune 過的大語言模型，
會最大比例的節省運算成本。這一塊目前方法有：

- 將模型檔轉換為 cpp 格式
    - llama.cpp
    - bloom.cpp

## 未來工作


# 大型語言模型 指令微調流程統整 LLM Instruction Fine-Tuning

本專案整理了微調大型語言模型的重要觀念和實作的程式框架，針對訓練和推論則提供 LLaMA, Bloom, ChatGLM 三種開源 LLMs 的運行範例。

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

舉例來說，[unit-minions](https://github.com/unit-mesh/unit-minions) 提供了針對代碼生成的資料集生成案例，例如從 Github 爬取註解和代碼下來、拆解、產生指令資料集。

### 從文本中生成問答對

那如果不是問答 pair 要怎麼辦呢，很簡單此時可以用方法從文本中生成很多的問答對出來。

品質比較高但花錢的實現方法是從 ChatGPT 下 prompt 取得，還可以指定他的問題要鎖定哪些關鍵主題、或是調整字數。

品質普普、不花錢的實現方法就是用 LangChain 的 QAGeneration

> 補 QAGeneration 原理說明

### DataAgent 釋出公開資料集

我從幾個公開網站爬取並整理成資料集，有多個不同產業領域的，請不吝按個愛心後取用

> 補贊助按鈕

- hf link

如果想要一次看所有資料集介紹，可以到我整理的 Github 專案： [instruction-finetune-datasets](https://github.com/A-baoYang/instruction-finetune-datasets)

## 模型壓縮方法

完成了資料集生成後，接下來進到模型部分。

為了要能在消費級顯卡上順利運行大型語言模型，我們需要將模型進行壓縮，詳細可以分為以下幾種：

### 網絡剪枝
「刪除 LLM 中在這個任務下不那麼重要的神經元」

- 原先是用在解決 Overfitting 問題上
- 先得出 LLM 的神經元重要性和權重，刪除重要性較低的神經元，得到較小的模型；使用這個小模型去做微調，可以節省載入和微調的資源

    ![](https://i.imgur.com/uWGSgym.png)

### 模型量化
「減少模型權重的空間佔用」

#### Mix Precision 混合精度訓練
- 預設權重參數精度是 FP32(float32)，為了減少佔用空間將部分權重從 FP32 轉為 FP16 來混合訓練，如此便可以增加 batch_size
- 在模型權重部分使用 FP32 精度；計算模型前向傳播和反向傳播時，使用 FP16/BF16 精度，以提高訓練速度，最後 FP16/BF16 的梯度用來更新 FP32 的模型權重。
- Q：為什麼不全部轉成 FP16？  A：因為模型訓練到最後向後傳播的梯度會變得非常小，此時如果用 FP16 則無法儲存小數點更後面位數的數值，造成 Underflow 問題
- 大部分權重為 0 的矩陣可以用稀疏矩陣表示，不會帶來準確率損失；但要注意在大型 CNN 網絡上，二進制方法會有較大的準確率損失

半精度模型可以得到和全精度幾乎一致的推論輸出，而半精度僅需要一半記憶體佔用。
但如果要再往下降低精度，推論表現就會開始顯著下降。
因此 Meta AI Research 提出另套方法：8-bit 量化技術

#### 8-bit Matrix Multiplication for Transformers at Scale  8位混合精度矩陣乘法

`LLM.int8()` 大模型零退化矩陣乘法量化 重點三步驟：
1. 從矩陣隱藏層中，以列為單位，抽取 outliers（值大於閾值的）
2. 對 outliers 的部分以 FP16 精度做矩陣乘法，其他部分則使用 int8 精度進行 vector-wise 量化
3. 最後將 int8 量化的部分恢復成 FP16，然後和 outliers 部分合併

![](https://i.imgur.com/DVGMAwy.png)

> Outlier features 在 Transformers 中會均勻分布在每一層結構中，而且當閾值 >= 6 時則都可以得到無損的推論結果

**評估量化過程損失**

`BLOOM-176B` 在 LM-eval-harness 上的基準，可以看到絕對誤差低於標準差，
故量化過程維持了 LLM 原始的效能。

![](https://i.imgur.com/jQRUewO.png)

但是減慢了 15-23% 的運行速度

![](https://i.imgur.com/U1kqoMQ.png)


### 結構化矩陣
「一個 m * n 矩陣只用少於 m * n 參數來描述」 (ex: AlphaTensor)

- 可以減少空間佔用、透過快速的矩陣-向量乘法 ＆ 梯度計算 顯著加快訓練＆推理速度

### 知識蒸餾
「利用知識轉移 (knowledge transfer）來壓縮模型」

- 亦稱為 Teacher-Student Networks
- 讓小模型學習大模型 Softmax 層輸出的機率向量分佈，就像是將大模型擁有的知識轉移給小模型；能夠有效減少計算成本。
- 缺點是只能用於有 Softmax 損失函數的分類貼標任務

### 遷移/壓縮卷積濾波器：主要應用於影像辨識
- 在 Inception 結構中使用將 3 * 3 卷積分解成兩個 1 * 1 卷積
- SqueezeNet 提出以 1 * 1 卷積替代 3 * 3 卷積；與 AlexNet 相比創建了減少 50 倍參數的神經網絡

### PyTorch v.s. Tensorflow
這兩個常見的深度學習框架在模型壓縮的實現方便性上有沒有差異呢？

由於 PyTorch 支援的是動態圖 (dynamic computational graph)，比 Tensorflow 靜態圖來說更容易分層訓練、進行參數修改、靈活性更高，更適合用於實驗和研究。

## 高效微調方法
可以成功載入模型還不夠，微調時所需的 GPU 資源遠比推論更大；
但實際上我們不需要微調所有參數，因為並非所有大型語言模型參數都適用於當前的任務，
我們其實可以透過一些方式來只微調部分參數、節省訓練資源。

在這邊由 Huggingface 提出的 PEFT 框架將實作集大成，包含以下幾種方法：

### LoRA: Low-Rank Adaptation of Large Language Models
「透過學習小參數的 low-rank matrix 來近似權重矩陣的參數更新，訓練時只優化 low-rank matrix 的參數」


### P-Tuning

### Prefix/Prompt Tuning
「在模型輸入或隱藏層添加 k 個額外 trainable 的前綴 tokens．只訓練這些不實際代表任何字元的虛擬 tokens」

![](https://pic1.zhimg.com/80/v2-162a7488d334c00a50697b732721adb4_1440w.webp)

在模型輸入的前面加上一段連續向量序列，稱之為 prefix，接著固定 LLM 所有參數，只微調特定任務的 prefix；
因此每個下游任務只產生小量計算成本

### Adapter Tuning
「將較小的神經網絡層插入到預訓練模型的每一層，微調下游任務時只訓練這些小網絡層(Adapter)的參數」

## 模型加速
- 在移動裝置上對模型加速比壓縮更重要，例如在數學計算上將加乘法轉為邏輯和位移運算

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
- flexgen Offload
- ClossaiAI

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


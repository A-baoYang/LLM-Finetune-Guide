# 資料集

- 指令微調資料集
- 獎勵模型資料集

## 指令微調資料集

指令微調資料的格式主要是 (instruction, input, output) pairs 的形式，例如：

```json
[
    {
        "instruction": "您現在是一個廣告文案專家，請根據以下商品規格撰寫高點擊率的宣傳文案。",
        "input": "",
        "output": ""
    },
    ...
]
```

### 公開的指令微調資料集整理

| Dataset Name | Domain | Language | Publisher | Example | Link |
| --- | --- | --- | --- | --- | --- |
| xtreme | General | `en` | - | - | - |
| CodeSearchNet | Code Generation | `en` | Github | - | - |
| CodeXGLUE | Code Generation | `en` | Meta | - | - |
| DS-1000 | Code Generation | `en` | HKUNLP | - | - |
| Zhihu-KOL | Code Generation | `zhcn` | wangrui6 | - | - |


## 獎勵模型資料集

用來訓練獎勵模型的資料集，在同個 input 下會配一好一壞的兩種回答，讓模型從中學習好的回答的特徵

```json
```

### 公開的獎勵模型訓練資料集整理

| Dataset Name | Domain | Language | Publisher | Example | Link |
| --- | --- | --- | --- | --- | --- |

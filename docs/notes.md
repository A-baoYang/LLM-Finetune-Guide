## 訓練筆記

1. LoRA 載入模型如果來源是無 sharded 的模型， DDP 會需要倍數成長的 CPU RAM 來預載模型，先載到 CPU RAM 後才放到 GPU VRAM

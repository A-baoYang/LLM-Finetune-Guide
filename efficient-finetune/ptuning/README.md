## Issue

如果出現 `RuntimeError: Library cudart is not initialized`，是由於有 `--quantization_bit 4` ，需要補裝對應版本的 cudatoolkit

先確認 CUDA 環境需求版本
```bash
conda list | grep cuda
```
terminal 會像這樣呈現版本
```bash
cudatoolkit               11.7.0              hd8887f6_10    nvidia
nvidia-cuda-cupti-cu11    11.7.101                 pypi_0    pypi
nvidia-cuda-nvrtc-cu11    11.7.99                  pypi_0    pypi
nvidia-cuda-runtime-cu11  11.7.99                  pypi_0    pypi
```
安裝對應版本
```bash
conda install cudatoolkit=11.7 -c nvidia
```

Reference:
- https://github.com/THUDM/ChatGLM-6B/issues/115


### LLaMA2

```
$ bash run_sft.sh
```

* 使用 resume_from_checkpoint 時，須將 deepspeed/runtime/engine 下的 load_checkpoint 函式中 load_module_strict 預設值改為 False

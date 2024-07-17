from huggingface_hub import snapshot_download

def get_model(
    model_name: str = "hfl/chinese-llama-2-7b", 
    local_dir: str = "/home/ubuntu/model/chinese-llama-2-7b", **kwargs):
    snapshot_download(
        repo_id=model_name, 
        local_dir=local_dir, 
        local_dir_use_symlinks=False, 
        **kwargs
    )

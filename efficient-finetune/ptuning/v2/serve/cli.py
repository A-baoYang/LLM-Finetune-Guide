import os
import platform
import signal

import click
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'
stop_stream = False


def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def get_model(pretrained_model_path: str, ptuning_checkpoint: str = "", quantization_bit: int or None = None, is_cuda: bool = False):
    config = AutoConfig.from_pretrained(pretrained_model_path, trust_remote_code=True)
    config.pre_seq_len = 512
    config.prefix_projection = None

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path, trust_remote_code=True)
    if is_cuda and torch.cuda.is_available():
        model = AutoModel.from_pretrained(pretrained_model_path, config=config, trust_remote_code=True, device_map="auto")
    else:
        model = AutoModel.from_pretrained(pretrained_model_path, config=config, trust_remote_code=True).float()

    if ptuning_checkpoint:
        prefix_state_dict = torch.load(os.path.join(ptuning_checkpoint, "pytorch_model.bin"))
        new_prefix_state_dict = {}
        for k, v in prefix_state_dict.items():
            new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
        model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
    if quantization_bit:
        model = model.quantize(quantization_bit)
    if config.pre_seq_len is not None:
        if is_cuda and torch.cuda.is_available():
            model = model.half()
        if ptuning_checkpoint:
            model.transformer.prefix_encoder.float()
    model = model.eval()
    return tokenizer, model


def build_prompt(history):
    prompt = "欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序"
    for query, response in history:
        prompt += f"\n\n用户：{query}"
        prompt += f"\n\nChatGLM-6B：{response}"
    return prompt


def signal_handler(signal, frame):
    global stop_stream
    stop_stream = True


@click.command()
@click.option("--pretrained_model_path", "pretrained_model_path", type=str, default="THUDM/chatglm-6b")
@click.option("--ptuning_checkpoint", "ptuning_checkpoint", type=str)
@click.option("--quantization_bit", "quantization_bit", type=int, default=4)
@click.option("--is_cuda", "is_cuda", type=bool, default=False)
def main(pretrained_model_path: str, ptuning_checkpoint: str, quantization_bit: int or None, is_cuda: bool):
    tokenizer, model = get_model(pretrained_model_path, ptuning_checkpoint, is_cuda)
    
    history = []
    global stop_stream
    print("欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
    while True:
        query = input("\n用户：")
        if query.strip() == "stop":
            break
        if query.strip() == "clear":
            history = []
            os.system(clear_command)
            print("欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
            continue
        count = 0
        for response, history in model.stream_chat(tokenizer, query, history=[]):
            if stop_stream:
                stop_stream = False
                break
            else:
                count += 1
                if count % 8 == 0:
                    os.system(clear_command)
                    print(build_prompt(history), flush=True)
                    signal.signal(signal.SIGINT, signal_handler)
            torch_gc()
        os.system(clear_command)
        print(build_prompt(history), flush=True)


if __name__ == "__main__":
    main()

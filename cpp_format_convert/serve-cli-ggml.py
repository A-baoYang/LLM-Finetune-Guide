import click
from pyllamacpp.model import Model


def new_text_callback(text: str):
    print(text, end="", flush=True)


@click.command()
@click.option("--ggml_model", "ggml_model", type=str, required=True)
@click.option("--max_tokens", "max_tokens", type=int, default=1024)
@click.option("--n_threads", "n_threads", type=int, default=8)
def main(ggml_model: str, max_tokens: int, n_threads: int):
    model = Model(ggml_model=ggml_model, n_ctx=max_tokens)

    while True:
        prompt = input("您好，有什麼我可以協助您的：")
        model.generate(prompt, n_predict=max_tokens, new_text_callback=new_text_callback, n_threads=n_threads)


if __name__ == "__main__":
    main()

import json
import logging
import os
from pathlib import Path

import click
import pandas as pd
from tqdm import tqdm


def read_data(filepath: str) -> list:
    if ".json" in filepath:
        with open(filepath, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data


def store_data(filepath: str) -> list:
    if ".json" in filepath:
        with open(filepath, "w", encoding="utf-8") as file:
            json.dump(file, file, ensure_ascii=False, indent=4)


@click.command()
@click.option("--start_date", "start_date", type=str, default="20200601")
@click.option("--end_date", "end_date", type=str, default="20230523")
def main(start_date: str, end_date: str):
    logging.basicConfig(
        filename="run_shell.log",
        level=logging.INFO,
        format="[%(asctime)s] [%(name)s | line:%(lineno)s | %(funcName)s] [%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    temp = pd.DataFrame({"date": [start_date, end_date]})
    temp["date"] = pd.to_datetime(temp["date"])
    date_list = pd.date_range(start=temp["date"].min(), end=temp["date"].max())
    temp = pd.DataFrame({"date": date_list})
    temp["date"] = temp["date"].astype(str)

    for date_str in tqdm(temp["date"].tolist()):
        date_str = date_str.replace("-", "")
        if date_str in ["20221001"]:
            continue
        print(f"Extracting {date_str}...")
        print(date_str)

        os.system(
            f"python /home/jovyan/gpt/open_gpt/LLM-Finetune-Guide/instruction-datasets/temp-largitdata/fetch_data.py --date_str {date_str}"
        )
        generate_cmd = f"""
        MODEL_PATH=THUDM/chatglm-6b
        MODEL_TYPE=chatglm-6b
        MAX_LENGTH=512
        PRE_SEQ_LEN=512
        BATCH_SIZE=4
        STEP=3000
        LR=2e-2
        DATATAG=multi-ee-no-instruction
        DATE_STR={date_str}
        CACHE_DIR=/home/jovyan/gpt/model/huggingface

        CUDA_VISIBLE_DEVICES=2,3 accelerate launch --main_process_port 29051 --config_file ../../config/use_deepspeed.yaml finetune.py --do_predict \
            --validation_file ../../../instruction-datasets/temp-largitdata/$DATE_STR.json \
            --test_file ../../../instruction-datasets/temp-largitdata/$DATE_STR.json \
            --overwrite_cache \
            --prompt_column input \
            --response_column output \
            --output_file generated-prediction-$DATE_STR.json \
            --model_name_or_path THUDM/chatglm-6b \
            --ptuning_checkpoint finetuned/$DATATAG-chatglm-6b-pt-$PRE_SEQ_LEN-$LR/checkpoint-$STEP \
            --output_dir finetuned/$DATATAG-chatglm-6b-pt-$PRE_SEQ_LEN-$LR \
            --cache_dir $CACHE_DIR \
            --overwrite_output_dir \
            --max_source_length $MAX_LENGTH \
            --max_target_length $MAX_LENGTH \
            --per_device_eval_batch_size $BATCH_SIZE \
            --predict_with_generate \
            --pre_seq_len $PRE_SEQ_LEN \
            --quantization_bit 4
        """
        os.system(generate_cmd)

        news_path = f"/home/jovyan/gpt/open_gpt/LLM-Finetune-Guide/instruction-datasets/temp-largitdata/{date_str}.json"
        pred_path = f"/home/jovyan/gpt/open_gpt/LLM-Finetune-Guide/efficient-finetune/ptuning/v2/finetuned/multi-ee-no-instruction-chatglm-6b-pt-512-2e-2/generated-prediction-{date_str}.json"
        gcs_dir = "gs://dst-largitdata/domestic/temp-manual-finevent"
        news, pred = read_data(news_path), read_data(pred_path)

        try:
            assert len(news) == len(pred)
        except Exception:
            logging.error("prediction sample size not equal to original sample size!")

        for i in tqdm(range(len(news))):
            news[i]["predict"] = pred[i]["predict"]

        news = (
            pd.DataFrame(news)
            .groupby("id", as_index=False)
            .agg({"predict": "\n\n".join}, axis=1)
        )
        logging.info(news["predict"][0])
        store_path = f'{gcs_dir}/{Path(pred_path).name.split(".")[0]}.ndjson'
        news.to_json(store_path, lines=True, orient="records", compression="gzip")

        os.remove(news_path)
        os.remove(pred_path)


if __name__ == "__main__":
    main()

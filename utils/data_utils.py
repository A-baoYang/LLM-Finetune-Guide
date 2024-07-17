import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, List

import pandas as pd
import yaml

# from KGBuilder.data_utils import split_sentence

root_dir = Path(__name__).parent.absolute()


def log_setting(
    log_folder: str = "logs-default", log_level: int = logging.INFO, stream: bool = True
) -> None:
    log_folder = log_folder if log_folder.startswith("logs-") else "logs-" + log_folder
    log_filename = os.path.join(
        Path(__file__).resolve().parent,
        log_folder,
        f"{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}.log",
    )
    Path(log_filename).parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=log_filename,
        level=log_level,
        format="[%(asctime)s] [%(name)s | line:%(lineno)s | %(funcName)s] [%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if stream:
        console = logging.StreamHandler()
        console_format = logging.Formatter("[%(name)s] [%(levelname)s] - %(message)s")
        console.setFormatter(console_format)
        logging.getLogger().addHandler(console)


def read_data(path: str) -> Any:
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif path.endswith(".txt"):
        with open(path, "r", encoding="utf-8") as f:
            data = f.read()
        data = data.split("\n")
    elif path.endswith(".csv"):
        data = pd.read_csv(path)
    elif path.endswith(".ndjson"):
        data = pd.read_json(path, lines=True, orient="records")
    elif path.endswith(".ndjson.gz"):
        data = pd.read_json(path, lines=True, orient="records", compression="gzip")
    elif path.endswith(".pickle"):
        data = pd.read_pickle(path)
    elif path.endswith(".parquet"):
        data = pd.read_parquet(path)
    elif path.endswith(".yaml"):
        with open(path, "r") as stream:
            try:
                data = yaml.safe_load(stream)
            except yaml.YAMLError as e:
                print(e)
    else:
        data = []
    return data


def save_data(data: Any, path: str) -> None:
    if path.endswith(".json"):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    elif path.endswith(".txt") and isinstance(data, list):
        with open(path, "w", encoding="utf-8") as f:
            for _d in data:
                f.write(_d)
                f.write("\n")
    elif path.endswith(".csv"):
        data.to_csv(path, index=False)
    elif path.endswith(".ndjson"):
        data.to_json(path, lines=True, orient="records")
    elif path.endswith(".ndjson.gz"):
        data.to_json(path, lines=True, orient="records", compression="gzip")
    elif path.endswith(".pickle"):
        data.to_pickle(path)
    elif path.endswith(".parquet"):
        data.to_parquet(path)
    elif isinstance(data, list):
        with open(path, "w", encoding="utf-8") as f:
            for _d in data:
                f.write(_d)
                f.write("\n")
    else:
        pass


def write_jsonl(fname, json_objs):
    with open(fname, 'wt', encoding='utf-8') as f:
        for o in json_objs:
            f.write(json.dumps(o, ensure_ascii=False) + '\n')


def convert_to_jsonl(data_path, output_path):
    list_of_dict = json.load(open(data_path, 'rt', encoding='utf-8'))
    write_jsonl(output_path, list_of_dict)
    return list_of_dict


def remove_emojis(text):
    import emoji
    clean_text = ''.join(c for c in text if c not in emoji.EMOJI_DATA)
    # emoji_list = [c for c in text if c in emoji.EMOJI_DATA]
    # clean_text = ''.join([str for str in text if not any(i in str for i in emoji_list)])
    return clean_text

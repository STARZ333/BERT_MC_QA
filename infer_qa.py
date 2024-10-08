import os
import torch
import os
import json
from argparse import Namespace, ArgumentParser
from tqdm import tqdm
from functools import partial
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, default_data_collator
from transformers import AutoConfig, AutoModelForQuestionAnswering
import collections
from typing import Optional, Tuple
import numpy as np
from transformers import EvalPrediction
from infer_utils import(
    create_and_fill_np_array,
    dict_to_device,
    post_processing_func,
    preprocess_valid_qa_func
)

import csv


def parse_arguments() -> Namespace:
    parser = ArgumentParser(description="Question Answering")

    parser.add_argument("--tokenizer_name", type=str,
                        default="./output_qa/hfl/chinese-lert-base",
                        help="tokenizer name")
    parser.add_argument("--checkpoint_folder", type=str,
                        default="./output_qa/hfl/chinese-lert-base",
                        help="checkpoint folder")
    
    parser.add_argument("--test_file", type=str,
                        default="./test_mc_pred.csv",
                        help="path of mutiple choice predictions")
    parser.add_argument("--output_file", type=str,
                        default="test_qa_pred.csv",
                        help="path to save the output CSV file")      
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    # Prepared datasets
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name,
        use_fast=True,
        trust_remote_code=False,
    )
    datasets = load_dataset("csv", data_files={"test": args.test_file})
    preprocess_func = partial(preprocess_valid_qa_func, tokenizer=tokenizer)
    processed_test_dataset = datasets["test"].map(
        preprocess_func,
        batched=True,
        remove_columns=datasets["test"].column_names
    )
    test_loader = DataLoader(
        processed_test_dataset.remove_columns(["example_id", "offset_mapping"]),
        batch_size=1,
        collate_fn=default_data_collator,
        shuffle=False,
    )

    # Prepared model
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    model_config = AutoConfig.from_pretrained(args.checkpoint_folder)
    model = AutoModelForQuestionAnswering.from_pretrained(
        args.checkpoint_folder,
        config=model_config,
    ).to(device)
    model = model.to(device)
    
    start_logits_list = []
    end_logits_list = []
    inference_bar = tqdm(test_loader, desc=f"Inference")
    for step, batch_data in enumerate(inference_bar, start=1):
        batch_data = dict_to_device(batch_data, device)
        outputs = model(**batch_data)
        start_logits_list.append(outputs.start_logits.detach().cpu().numpy())
        end_logits_list.append(outputs.end_logits.detach().cpu().numpy())

    start_logits_concat = create_and_fill_np_array(start_logits_list, processed_test_dataset, 512)
    end_logits_concat = create_and_fill_np_array(end_logits_list, processed_test_dataset, 512)

        # 调用后处理函数获取预测结果
    prediction = post_processing_func(
        examples=datasets["test"],
        features=processed_test_dataset,
        predictions=(start_logits_concat, end_logits_concat),
        output_dir="temp",  # 临时输出文件夹
    )

    # 将预测结果保存为 CSV 文件
    output_file = args.output_file
    with open(output_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "answer"])
        for pred in prediction.predictions:
            writer.writerow([pred["id"], pred["prediction_text"]])

    print(f"Predictions saved to {output_file}")
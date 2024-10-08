import os
import torch

from argparse import Namespace, ArgumentParser
from tqdm import tqdm
from functools import partial
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, default_data_collator
from transformers import AutoConfig, AutoModelForMultipleChoice
import pandas as pd
from itertools import chain

from infer_utils import (
    dict_to_device,
    flatten_list,
    unflatten_list,
    preprocess_mc_func
)


def parse_arguments() -> Namespace:
    parser = ArgumentParser(description="Mutiple Choice")

    parser.add_argument("--tokenizer_name", type=str,
                        default="hfl/chinese-lert-base",
                        help="tokenizer name")
    parser.add_argument("--checkpoint_folder", type=str,
                        default="./output_hfl/chinese-lert",
                        help="checkpoint folder")
    parser.add_argument("--test_mc", type=str,
                        default="./mc_test_dataset.csv",
                        help="path of test mc file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    # Prepared datasets
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name,
        use_fast=True,
        trust_remote_code=False,
    )
    if args.test_mc is None:
        raise ValueError("The test multiple-choice file is not provided. Please specify the file path using --test_mc.")
        
    test_mc_file = args.test_mc
    datasets = load_dataset("csv", data_files={"test": args.test_mc})
    preprocess_func = partial(preprocess_mc_func, tokenizer=tokenizer, train=False)
    processed_test_dataset = datasets["test"].map(
        preprocess_func,
        batched=True,
        remove_columns=datasets["test"].column_names
    )
    test_loader = DataLoader(
        processed_test_dataset,
        batch_size=1,
        collate_fn=default_data_collator,
        shuffle=False,
    )

    # Prepared model
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    model_config = AutoConfig.from_pretrained(args.checkpoint_folder)
    model = AutoModelForMultipleChoice.from_pretrained(
        args.checkpoint_folder,
        config=model_config,
    ).to(device)
    model = model.to(device)
    model.eval()
    
    pred_list = []

    inference_bar = tqdm(test_loader, desc=f"Inference")
    for _, batch_data in enumerate(inference_bar, start=1):
        with torch.no_grad():
            batch_data = dict_to_device(batch_data, device)
            outputs = model(**batch_data)
            preds = outputs.logits.argmax(dim=-1).cpu().numpy().tolist()
            pred_list += preds

    test_data_list = []
    for index, pred in enumerate(pred_list):
        test_data = datasets["test"][index]
        test_data_list.append(
            {
                "id": test_data['id'],
                "context": test_data[f'ending{pred}'],
                "question": test_data['context'],
                "answer_text": test_data[f"ending{pred}"][0],
                "answer_start": 0
            }
        )

    # 將數據轉換為 pandas DataFrame
    df = pd.DataFrame(test_data_list)

    # 指定保存的文件名
    output_file = os.path.join("test_mc_pred.csv")

    print(f"Predictions saved to {output_file}")
    # 保存為CSV文件
    df.to_csv(output_file, index=False)
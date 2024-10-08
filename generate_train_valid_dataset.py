import json
import csv
import sys

# 确保提供了正确数量的参数
if len(sys.argv) != 4:
    print("Usage: python script.py <path_to_context.json> <path_to_train.json> <path_to_valid.json>")
    sys.exit(1)

# 获取命令行参数
context_file_path = sys.argv[1]  # 第一个参数：context.json 文件路径
train_file_path = sys.argv[2]    # 第二个参数：train.json 文件路径
valid_file_path = sys.argv[3]    # 第三个参数：valid.json 文件路径

# 定义函数来处理 JSON 数据并保存为 CSV 格式
def process_and_save_to_csv(data_file_path, context_data, output_file):
    # 加载数据文件 (train.json 或 valid.json)
    with open(data_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 准备 SWAG 数据集列表
    swag_data = []

    for entry in data:
        paragraphs = entry['paragraphs']
        relevant = entry['relevant']
        context = entry['question']

        # 构建 SWAG 格式的数据项
        swag_entry = {
            'context': context,
            'ending0': context_data[paragraphs[0]],
            'ending1': context_data[paragraphs[1]],
            'ending2': context_data[paragraphs[2]],
            'ending3': context_data[paragraphs[3]],
            'label': paragraphs.index(relevant)
        }

        swag_data.append(swag_entry)

    # 将数据写入 CSV 文件
    with open(output_file, 'w', encoding='utf-8', newline='') as csvfile:
        fieldnames = ['context', 'ending0', 'ending1', 'ending2', 'ending3', 'label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for swag_entry in swag_data:
            writer.writerow(swag_entry)

    print(f'Dataset has been saved to {output_file}')

# 加载 context.json 数据
with open(context_file_path, 'r', encoding='utf-8') as f:
    context_data = json.load(f)

# 处理 train.json 并生成 train_dataset.csv
process_and_save_to_csv(train_file_path, context_data, 'dataset_train.csv')

# 处理 valid.json 并生成 valid_dataset.csv
process_and_save_to_csv(valid_file_path, context_data, 'dataset_valid.csv')

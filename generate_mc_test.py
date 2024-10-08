import json
import csv
import sys

# 确保提供了正确数量的参数
if len(sys.argv) != 3:
    print("Usage: python script.py <path_to_context.json> <path_to_test.json>")
    sys.exit(1)

# 获取命令行参数
context_file_path = sys.argv[1]  # 第一个参数：context.json 文件路径
test_file_path = sys.argv[2]     # 第二个参数：test.json 文件路径

# 加载 test.json 数据
with open(test_file_path, 'r', encoding='utf-8') as f:
    training_data = json.load(f)

# 加载 context.json 数据
with open(context_file_path, 'r', encoding='utf-8') as f:
    context_data = json.load(f)

# 准备 SWAG 数据集列表
swag_data = []

for entry in training_data:
    id = entry['id']
    paragraphs = entry['paragraphs']
    context = entry['question']

    # 构建 SWAG 格式的数据项
    swag_entry = {
        'id': id,
        'context': context,
        'ending0': context_data[paragraphs[0]],
        'ending1': context_data[paragraphs[1]],
        'ending2': context_data[paragraphs[2]],
        'ending3': context_data[paragraphs[3]],
    }

    swag_data.append(swag_entry)

# 将数据写入 CSV 文件
output_file = 'mc_test_dataset.csv'
with open(output_file, 'w', encoding='utf-8', newline='') as csvfile:
    fieldnames = ['id', 'context', 'ending0', 'ending1', 'ending2', 'ending3']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for swag_entry in swag_data:
        writer.writerow(swag_entry)

print(f'test 数据集已保存到 {output_file} 文件中。')

import json

def read_and_convert_json(file_path, output_path):
    # 读取 JSON 文件
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 打印转换后的数据
    print("转换后的数据:")
    for key, value in data.items():
        print(f"{key}: {value}")

    # 保存转换后的数据为新的 JSON 文件
    with open(output_path, 'w', encoding='utf-8') as out_file:
        json.dump(data, out_file, ensure_ascii=False, indent=4)

# 使用函数读取和转换 JSON 文件
input_file = 'chinese-lert-base/eval_predictions.json'  # 输入的 JSON 文件路径
output_file = 'chinese-lert-base/eval_predictions.json'  # 转换后输出的文件路径
read_and_convert_json(input_file, output_file)

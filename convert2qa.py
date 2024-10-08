import json
import csv

# 加载 JSON 文件
def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


# 生成 QA 数据集并保存为 CSV 文件
def generate_qa_dataset(dataset, context_data, output_file):
    qa_data = []

    for entry in dataset:
        id = entry['id']
        question = entry['question']
        paragraphs = entry['paragraphs']
        relevant = entry['relevant']
        answers = entry['answer']

        # 确保答案字段是列表并且序列化为 JSON 格式，禁用 Unicode 转义
        qa_entry = {
            'id': id,
            'question': question,
            'context': context_data[relevant],
            'answers': json.dumps({  # 将字典转为 JSON 字符串
                'text': [answers['text']],  
                'answer_start': [answers['start']]  
            }, ensure_ascii=False)  # 禁用 ensure_ascii 以保留中文
        }
        
        qa_data.append(qa_entry)

    # 写入 CSV 文件，添加 'id' 字段
    with open(output_file, 'w', encoding='utf-8', newline='') as csvfile:
        fieldnames = ['id', 'question', 'context', 'answers']  # 在 fieldnames 中添加 'id'
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for qa_entry in qa_data:
            writer.writerow(qa_entry)

    print(f'QA 数据集已保存到 {output_file} 文件中。')


# 加载数据
training_data = load_json('/home/yiz/Desktop/ADLHW1/train.json')
valid_data = load_json('/home/yiz/Desktop/ADLHW1/valid.json')
context_data = load_json('/home/yiz/Desktop/ADLHW1/context.json')

# 生成训练集和验证集 CSV 文件
generate_qa_dataset(training_data, context_data, 'qa_train_dataset.csv')
generate_qa_dataset(valid_data, context_data, 'qa_valid_dataset.csv')

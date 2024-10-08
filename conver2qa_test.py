import json
import csv

# 加载 training.json 数据
with open('/home/yiz/Desktop/ADLHW1/test.json', 'r', encoding='utf-8') as f:
  training_data = json.load(f)

# 加载 context.json 数据
with open('/home/yiz/Desktop/ADLHW1/context.json', 'r', encoding='utf-8') as f:
  context_data = json.load(f)

# 准备 SWAG 数据集列表
swag_data = []


for entry in training_data:
  id = entry['id']
  paragraphs = entry['paragraphs']
#   relevant = entry['relevant']
  context = entry['question']
#   correct_answer = entry['answer']['start']

    

  # 构建 SWAG 格式的数据项
  swag_entry = {
      'id': id,
      'context': context,
      'ending0': context_data[paragraphs[0]],
      'ending1': context_data[paragraphs[1]],
      'ending2': context_data[paragraphs[2]],
      'ending3': context_data[paragraphs[3]],
    #   'label': paragraphs.index(relevant)
  }

  swag_data.append(swag_entry)

# 将数据写入 CSV 文件
with open('mc_test_dataset.csv', 'w', encoding='utf-8', newline='') as csvfile:
  fieldnames = ['id','context', 'ending0', 'ending1', 'ending2', 'ending3']
  writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

  writer.writeheader()
  for swag_entry in swag_data:
      writer.writerow(swag_entry)

print('test 数据集已保存到 mc_test_dataset.csv 文件中。')
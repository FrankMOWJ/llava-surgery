'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2024-06-05 23:59:31
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2024-06-06 00:09:54
FilePath: /LLaVA/create_qs.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import json
import jsonline
import os 

def create_question_fie(path, output_path):
    with open(path, 'r') as f:
        content = json.load(f)
    f.close()
    jsonl_file = []
    for test_sam in content:
        question_id = test_sam['id']
        image = test_sam['image']
        text = test_sam['conversations'][0]['value']
        category = "default"

        jsonl_file.append({
            "question_id": question_id,
            "image": image,
            "text": text,
            "category": category
        })
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, mode='w') as writer:
        for dict in jsonl_file:
            writer.write(json.dumps(dict)+'\n')
    writer.close()

def create_annatation_file(path, output_path):
    with open(path, 'r') as f:
        content = json.load(f)
    f.close()
    json_file = {
        'dataset_type': 'val',
        'dataset_name': 'llava_textvqa',
        'dataset_version': 'small-3images'
    }
    data = []
    for test_sam in content:
        dict = {
            "question": test_sam['conversations'][0]['value'],
            'image_id': test_sam['image'],
            'answers': test_sam['conversations'][1]['value'],
            'question_id': test_sam['id'],
            'set_name': 'val'
        }
        data.append(dict)
    json_file['data'] = data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(json_file, f, indent=2)
    f.close()

if __name__ == "__main__":
    path = './data/JIGSAWS/train_3images_small.json'
    output_path = './playground/JIGSAWS_test/llava_textvqa_JIGSAWS.jsonl'
    create_question_fie(path=path, output_path=output_path)
    create_annatation_file(path=path, output_path='./playground/JIGSAWS_test/answers/llava_textvqa_JIGSAWS_ans.json')

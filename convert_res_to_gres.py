import json
import argparse
import numpy as np
import random
from utils.misc import * 


def get_args():
    parser = argparse.ArgumentParser(description='Convert RES pseudo-labels to GRES format')
    parser.add_argument('--input_filename', type=str, default='datasets/refP/qwen_coco_mask.json')
    parser.add_argument('--output_filename', type=str, default='models/maskris/gres.json')
    return parser.parse_args()


args = get_args()
with open(args.input_filename, 'r', encoding='utf-8') as file:
    data_list = json.load(file)

for i in range(len(data_list)):
    data_list[i]['caption'] = data_list[i]['caption'].replace('.', '').strip()


''' single target '''
samples = random.sample(data_list, 50000)
single_target = []
for sample in samples:
    single_target.append({
        "file_name": sample['file_name'],
        "num": 1,
        "objects": [{"bbox": sample["bbox"], "mask": sample["segmentation"], "caption": sample["caption"]}]
    })
print('single target: ', len(single_target))


''' multi target '''
# 1 reconstruct data
data_dict = {} 
for data in data_list:
    file_name = data['file_name']
    if file_name not in data_dict:
        data_dict[file_name] = []
    data_dict[file_name].append(data)

new_data_list = []
for file_name, datas in data_dict.items():
    bbox_dict = {}
    for data in datas:
        bbox = str(data['bbox'])
        if bbox not in bbox_dict:
            bbox_dict[bbox] = []
        bbox_dict[bbox].append(data)
    
    annotations = [] 
    for bbox, datas in bbox_dict.items():
        bbox = datas[0]['bbox']
        mask = datas[0]['segmentation']
        captions = []
        for data in datas:
            if count_words(data['caption']) <= 5:
                captions.append(data['caption'])
        if len(captions) != 0:
            annotations.append({
                'bbox': bbox,
                'mask': mask,
                'captions': captions
            })
    if len(annotations) != 0:
        new_data_list.append({
            'file_name': file_name,
            'annotations': annotations
        })

new_data_list.sort(key=lambda x: x['file_name'])
print(len(new_data_list))

# 2 random select objects
multi_target = []
for data in new_data_list:
    file_name = data['file_name']
    anno = merge_annotations(data['annotations'])

    nums = [random.choice([2, 3]) for _ in range(5)]
    for num in nums:
        if num > len(anno):
            continue

        # select n annotations
        selected = random.sample(anno, num)
        
        # select caption from selected["captions"]
        saved_obj = []
        for s in selected:
            caption = random.choice(s['captions'])
            saved_obj.append({"bbox":s['bbox'], 'mask': s['mask'], 'caption': caption})

        multi_target.append({
            "file_name": file_name,
            "num": num,
            "objects": saved_obj
        })

multi_target = random.sample(multi_target, 100000)
print('multi target: ', len(multi_target))


''' no target '''
samples = random.sample(data_list, 100000)
no_target = []
for i in range(50000):
    if samples[i]['file_name'] == samples[i+50000]['file_name']:
        continue
    sample = samples[i]
    no_target.append({
        "file_name": sample['file_name'],
        "num": 0,
        "objects": [{"bbox": None, "mask": None, "caption": samples[i+50000]["caption"]}]
    })
print('no target: ', len(no_target))


saved_data_list = single_target + multi_target + no_target
with open('models/maskris/gres.json', 'w') as file:
    json.dump(saved_data_list, file, indent=4)
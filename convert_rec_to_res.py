import json
import torch
from PIL import Image
import numpy as np
import argparse
import models.maskris.utils as utils
import models.maskris.model.builder as builder
from torchvision import transforms as T
from models.maskris.bert.tokenization_bert import BertTokenizer
import numpy as np
from skimage import measure
from shapely.geometry import Polygon, MultiPolygon
from tqdm import tqdm
from utils.misc import *
import torch.multiprocessing as mp    

# args parse
def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input_filename', type=str, default='datasets/refP/qwen_coco.json')
    parser.add_argument('--output_filename', type=str, default='datasets/refP/qwen_coco_mask.json')
    parser.add_argument('--world_size', type=int, default=8)

    args = parser.parse_args()

    return args

def build_model(device):
    parser = argparse.ArgumentParser(description='maskris loading')
    parser.add_argument('--swin_type', default='base')
    parser.add_argument('--window12', action='store_true')
    parser.add_argument('--img_patch_size', type=int, default=32)
    parser.add_argument('--lr', default=0.00005, type=float)

    parser.add_argument('--ck_bert', default='models/maskris/bert-base-uncased')
    parser.add_argument('--bert_tokenizer', default='models/maskris/bert-base-uncased')
    parser.add_argument('--ckpt', default='models/maskris/weights/model_best_refcocom.pth')
    
    args= parser.parse_args()
    single_model = builder.__dict__['maskris'](pretrained='', args=args)
    utils.load_model(single_model, args.ckpt)
    model = single_model.to(device)
    transforms = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer)

    return model, transforms, tokenizer

def generate_mask(img_path, text, model, transforms, tokenizer, device):
    # image
    image = Image.open(img_path).convert("RGB")
    x = transforms(image)
    x = x.unsqueeze(0).to(device)

    # text
    attention_mask = [0] * 20
    padded_input_ids = [0] * 20
    input_ids = tokenizer.encode(text=text, add_special_tokens=True)
    input_ids = input_ids[:20]  # truncation of tokens
    padded_input_ids[:len(input_ids)] = input_ids
    attention_mask[:len(input_ids)] = [1] * len(input_ids)
    sentences = torch.tensor(padded_input_ids).unsqueeze(0).to(device)
    attention_mask = torch.tensor(attention_mask).unsqueeze(0).to(device)
    
    mask = model(x, sentences, l_mask=attention_mask)
    mask = mask.squeeze(0).squeeze(0).cpu().numpy()
    return mask


def worker(args, data_list, shared_data_list):
    model, transforms, tokenizer = build_model(args.device)

    if args.rank == 0: pbar = tqdm(total=len(data_list))

    iou_error, low_iou, polygon_error = 0, 0, 0
    for index, data in enumerate(data_list):
        img_path = data['file_name']
        text = data['caption']
        bbox1 = data['bbox']
        
        segmentation = generate_mask(img_path, text, model, transforms, tokenizer, args.device)
         
        try:
            bbox2 = mask_to_bbox(segmentation)
            iou = calculate_iou(bbox1, bbox2)
        except:
            iou_error += 1
            iou = 0.
            if args.rank == 0: pbar.update(1)
            continue

        if iou <= 0.5:
            low_iou += 1
            if args.rank == 0: pbar.update(1)
            continue
        
        try:
            polygons = mask_to_polygon(segmentation)
        except:
            polygon_error += 1
            if args.rank == 0: pbar.update(1)
            continue

        data['segmentation'] = polygons
        shared_data_list.append(data)

        if args.rank == 0 and index % 1000 == 0:
            saved_data_list = list(shared_data_list)
            with open(args.output_filename, 'w', encoding='utf-8') as file:
                json.dump(saved_data_list, file, indent=4)
            print('saved: ', len(saved_data_list))
            print('---error in rank0---')
            print('iou_error: ', iou_error)
            print('low_iou: ', low_iou)
            print('polygon_error: ', polygon_error)
            print('--------------------')

        if args.rank == 0: pbar.update(1)
    if args.rank == 0: pbar.close()
def main():
    args = get_args()

    # load data
    with open(args.input_filename, 'r', encoding='utf-8') as file:
        data_list = json.load(file)
    splited_data_list = split_list(data_list, args.world_size)

    with mp.Manager() as manager:
        shared_data_list = manager.list()  # list for sharing result

        processes = []
        for rank in range(args.world_size):
            args.rank = rank
            args.device = rank
            data_list = splited_data_list[rank]
            p = mp.Process(target=worker, args=(args, data_list, shared_data_list))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        saved_data_list = list(shared_data_list)
        with open(args.output_filename, 'w', encoding='utf-8') as file:
            json.dump(saved_data_list, file, indent=4)
        print('saved: ', len(saved_data_list))

if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()
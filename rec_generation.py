import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from tqdm import tqdm
import os
import json
import torch.multiprocessing as mp                             
import argparse
from utils.prompt import *
from utils.misc import *
from utils.model_utils import *

# args parse
def get_args():
    parser = argparse.ArgumentParser()
    # for output name
    parser.add_argument('--model_name', type=str, default='qwen7b')  # model
    parser.add_argument('--source', type=str, default='coco')  # img source
    parser.add_argument('--version', type=str, default='pretrain')  # version

    parser.add_argument('--input_filename', type=str, default='datasets/image_list_coco.json')
    parser.add_argument('--output_dir', type=str, default='your-output-dir')
    parser.add_argument('--world_size', type=int, default=8)

    parser.add_argument('--gdino_path', type=str, default='models/Grounding-Dino-Base')
    parser.add_argument('--qwen_path', type=str, default='models/Qwen2.5-VL-7B-Instruct')

    args = parser.parse_args()

    return args


def worker(args, img_list, shared_data_list):
    model_gdino, processor_gdino = build_grounding_dino(args.gdino_path, args.device)
    model_qwen, processor_qwen = build_qwen_vl(args.qwen_path, args.device)
    # model_clip, processor_clip = build_clip(args.clip_path, args.device)
    

    if args.rank == 0: pbar = tqdm(total=len(img_list))

    # inference
    for index, img_path in enumerate(img_list):
        # strategy A: 1cls(qwen)->2bbox(dino)->3caption(qwen)->4consistance(qwen)
        grounded_classes = generate_grounded_cls(img_path, model_qwen, processor_qwen, model_gdino, processor_gdino, args.device)

        for grounded_cls in grounded_classes:
            cls = grounded_cls['cls']
            boxes = grounded_cls['boxes']

            for box in boxes:
                grounded_captions_for_one_box = generate_grounded_captions_for_one_box(img_path, box, boxes, cls, model_qwen, processor_qwen, model_gdino, processor_gdino, args.device)
                if len(grounded_captions_for_one_box) != 0:
                    shared_data_list += grounded_captions_for_one_box
                        
        # strategy B: 1caption(qwen)_bbox(qwen)->candidates(dino)->consistance(dino)
        grounded_captions_for_all_objects = generate_grounded_captions_for_all_objects(img_path, model_qwen, processor_qwen, model_gdino, processor_gdino, args.device)
        if len(grounded_captions_for_all_objects) != 0:
            shared_data_list += grounded_captions_for_all_objects

        if args.rank == 0: pbar.update(1)

        # save to file
        if args.rank == 0 and index % 1000 == 0: 
            save_to_file(args, shared_data_list)

    save_to_file(args, shared_data_list)
    if args.rank == 0: pbar.close()


def main():
    args = get_args()

    # load data
    with open(args.input_filename, 'r', encoding='utf-8') as file:
        img_list = json.load(file)
    splited_img_list = split_list(img_list, args.world_size)

    with mp.Manager() as manager:
        shared_data_list = manager.list()  # list for sharing result

        processes = []
        for rank in range(args.world_size):
            args.rank = rank
            args.device = rank % 8
            img_list = splited_img_list[rank]
            p = mp.Process(target=worker, args=(args, img_list, shared_data_list))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        save_to_file(args, shared_data_list)

if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()

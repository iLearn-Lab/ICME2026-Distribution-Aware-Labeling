
from PIL import Image, ImageDraw, ImageFilter
import os
import json
import numpy as np
from skimage import measure
from shapely.geometry import Polygon, MultiPolygon

def parse_json(json_output):
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i+1:])
            json_output = json_output.split("```")[0]
            break 
    return json_output

def split_list(lst, world_size):
    avg_len = len(lst) // world_size
    remainder = len(lst) % world_size
    
    chunks = []
    start = 0
    for i in range(world_size):
        end = start + avg_len + (1 if i < remainder else 0)
        chunks.append(lst[start:end])
        start = end
    return chunks

def calculate_iou(box1, box2):
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    if x1_inter >= x2_inter or y1_inter >= y2_inter:
        return 0.0
    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area

    return iou

def filter_boxes(boxes, iou_threshold=0.5):
    del_list = []
    for i, box1 in enumerate(boxes):
        if i in del_list:
            continue
        for j, box2 in enumerate(boxes):
            if i < j and j not in del_list:
                if calculate_iou(box1, box2) > iou_threshold:
                    del_list.append(j)
        
    del_list = list(set(del_list))
    del_list = sorted(del_list, reverse=True)

    for del_index in del_list:
        del boxes[del_index]

    return boxes

def count_words(sentence):
    words = sentence.split()
    return len(words)

def blur(img_pil, box): 
    image = img_pil.copy()
    mask = Image.new('L', image.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle(box, fill=255)
    blurred = image.filter(ImageFilter.GaussianBlur(50))
    blurred.paste(image, mask=mask)
    return blurred

def crop(img_pil, box): 
    image = img_pil.copy()
    cropped = image.crop(box)
    return cropped

def bbox_xywh_to_xyxy(img, boxes):
        image = Image.open(img)
        w, h = image.size
        boxes = boxes.cpu().numpy().tolist()
        for index, box in enumerate(boxes):
            box = [float(b) * (w if i % 2 == 0 else h) for i, b in enumerate(box)]
            box[0] -= box[2] / 2
            box[1] -= box[3] / 2 
            box[2] += box[0]
            if box[2] > w: box[2] = w
            box[3] += box[1]
            if box[3] > h: box[3] = h
            boxes[index] = [int(item) for item in box]
        return boxes

def preprocess_caption(caption):
    result = caption.lower().strip()
    if result.endswith("."):
        return result
    return result + "."

def save_to_file(args, shared_data_list, add_info = ''):
    if add_info != '':
        add_info = '_' + add_info
    data_list = list(shared_data_list)
    print('Total Reference: ', len(data_list))
    os.makedirs(args.output_dir, exist_ok=True)
    output_filename = os.path.join(args.output_dir, f'{args.model_name}_{args.source}_{args.version}{add_info}.json')
    with open(output_filename, 'w', encoding='utf-8') as file:
        json.dump(data_list, file, ensure_ascii=False, indent=4)

def mask_to_bbox(mask):
    mask = (mask > 0).astype(np.uint8)

    coords = np.column_stack(np.where(mask > 0))  # shape: [N, 2]，每行为 [y, x]

    if coords.shape[0] == 0:
        return None

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    return [int(x_min), int(y_min), int(x_max), int(y_max)]

def mask_to_polygon(mask):
    mask = (mask > 0).astype(np.uint8)

    contours = measure.find_contours(np.array(mask), 0.5, positive_orientation='low')
    segmentations = []
    for contour in contours:
        for i in range(len(contour)):
            row, col = contour[i]
            contour[i] = (col - 1, row - 1)

        poly = Polygon(contour)
        poly = poly.simplify(1.0, preserve_topology=False)
        segmentation = np.array(poly.exterior.coords)
        segmentation = np.maximum(segmentation, 0).ravel().tolist()
        segmentations.append(segmentation)

    return segmentations


def read_data(file_name):
    with open(file_name, 'r', encoding='utf-8') as file:
        data_list = json.load(file)
    return data_list

def get_center(bbox):
    x_min, y_min, x_max, y_max = bbox
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    return (x_center, y_center)

def compute_iou(box1, box2):
    # box: [x, y, w, h]
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)

    inter_area = max(0, xb - xa) * max(0, yb - ya)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0

def merge_annotations(annotations, iou_threshold=0.8):
    merged = []
    used = set()

    for i, ann1 in enumerate(annotations):
        if i in used:
            continue
        group = [ann1]
        used.add(i)

        for j, ann2 in enumerate(annotations):
            if j in used or i == j:
                continue
            iou = compute_iou(ann1["bbox"], ann2["bbox"])
            if iou > iou_threshold:
                group.append(ann2)
                used.add(j)

        # 合并 group 中所有 annotation
        if group:
            # 合并 bbox：取最小包围框
            x0 = min(a["bbox"][0] for a in group)
            y0 = min(a["bbox"][1] for a in group)
            x1 = max(a["bbox"][0] + a["bbox"][2] for a in group)
            y1 = max(a["bbox"][1] + a["bbox"][3] for a in group)
            merged_bbox = [x0, y0, x1 - x0, y1 - y0]
            merged_mask = group[0]['mask']

            # 合并 captions
            merged_captions = list({cap for a in group for cap in a["captions"]})

            merged.append({
                "bbox": merged_bbox,
                "captions": merged_captions,
                "mask": merged_mask  # 如有 mask，可在此合并
            })

    return merged
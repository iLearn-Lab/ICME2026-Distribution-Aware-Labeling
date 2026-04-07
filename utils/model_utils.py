import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
import json
import torch
import random
import tempfile
from tqdm import tqdm

from transformers import Qwen2_5_VLForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer, AutoProcessor, BertTokenizer, BertModel
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from qwen_vl_utils import process_vision_info
from transformers import CLIPProcessor, CLIPModel


from utils.misc import *
from utils.prompt import *


''' For Pretrain Setting '''

def build_grounding_dino(model_path, device):
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_path).to(device)
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor

def call_grounding_dino(img_path, caption, model, processor, device):
    image = Image.open(img_path).convert('RGB')
    text = preprocess_caption(caption)

    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.4,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]]
    )
    scores = results[0]['scores']
    boxes = results[0]['boxes'].int()

    # sort by scores
    sorted_indices = torch.argsort(scores, descending=True)
    sorted_boxes = boxes[sorted_indices]

    # to list
    boxes = boxes.cpu().numpy().tolist()
    
    # filter the same
    boxes = filter_boxes(boxes)

    top1 = boxes[0] if len(boxes) else []

    return top1, boxes


def build_qwen_vl(model_path, device):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=device,
    )
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor

def call_qwen_vl(img_path, prompt, model, processor, device):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": img_path,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # preparation
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)

    # inference
    generated_ids = model.generate(**inputs, max_new_tokens=1000)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    return output_text[0]

def call_qwen_vl_text_only(prompt, model, processor, device):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # preparation
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)

    # inference
    generated_ids = model.generate(**inputs, max_new_tokens=1000, temperature=1.2)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    return output_text[0]

def call_qwen_vl_locate_one(img_path, caption, model, processor, device):
    prompt = f'Please provide the bounding box coordinate of the region this sentence describes: {caption}'
    loc = call_qwen_vl(img_path, prompt, model, processor, device)
    loc = parse_json(loc)
    box = [0, 0, 0, 0]
    try:
        loc = json.loads(loc)  
    except json.JSONDecodeError:
        return box
    if type(loc) == list and type(loc[0]) == dict:
        box = loc[0].get('bbox_2d', box)
        if len(box) != 4:
            box = [0, 0, 0, 0]
    return box

def call_qwen_vl_locate_all(img_path, cls, model, processor, device):
    prompt = f"Locate every {cls} in the image and output the coordinates in JSON format."
    
    loc = call_qwen_vl(img_path, prompt, model, processor, device)
    boxes = []  
    loc = parse_json(loc)
    try:
        loc = json.loads(loc)
    except json.JSONDecodeError:
        return boxes
    
    if type(loc) == list:
        for l in loc:
            if type(l) == dict:
                box = l.get('bbox_2d', None)
                if box is not None and len(box) == 4:
                    boxes.append(box)
            else:
                return []
    return boxes


def generate_grounded_cls(img_path, model_qwen, processor_qwen, model_gdino, processor_gdino, device):
    prompt = prompt_template_cls.format()
    response = call_qwen_vl(img_path, prompt, model_qwen, processor_qwen, device)
    classes_unfiltered = response.lower().splitlines()
    classes_unfiltered = [cls.strip() for cls in classes_unfiltered]
    # filter classes
    classes = []
    for cls in classes_unfiltered:
        if cls in cls_list:
            classes.append(cls)
    classes = list(set(classes))
    if len(classes) > 10:
        classes = []
    
    ret = []
    for cls in classes:
        boxes1 = call_qwen_vl_locate_all(img_path, cls, model_qwen, processor_qwen, device)
        _, boxes2 = call_grounding_dino(img_path, cls, model_gdino, processor_gdino, device)
        boxes = boxes1 + boxes2
        boxes = filter_boxes(boxes)
        if len(boxes) != 0:
            ret.append({'cls': cls, 'boxes': boxes})

    return ret

def generate_grounded_captions_for_one_box(img_path, box, boxes, cls, model_qwen, processor_qwen, model_gdino, processor_gdino, device):
    prompt_template_short = random.choice([prompt_template_short1, prompt_template_short2, prompt_template_short3])
    prompts = [prompt_template_short.format(box=box, cls=cls), prompt_template_mid.format(box=box, cls=cls), prompt_template_long.format(box=box, cls=cls)]
    
    ret = []
    for prompt in prompts:
        caption = call_qwen_vl(img_path, prompt, model_qwen, processor_qwen, device)
        caption = caption.replace('-', '').replace('.', '')
        if count_words(caption) > 25:
            continue

        loc1 = call_qwen_vl_locate_one(img_path, caption, model_qwen, processor_qwen, device)
        loc2, _ = call_grounding_dino(img_path, caption, model_gdino, processor_gdino, device)

        # consistancy
        if len(loc1) != 0 and len(loc2) != 0:
            iou1 = calculate_iou(box, loc1)
            iou2 = calculate_iou(loc1, loc2)
            if iou1 > 0.5 and iou2 > 0.5:
                ret.append({'file_name': img_path, 'caption': caption, 'bbox': box, 'candidates': boxes})
    return ret

def generate_grounded_captions_for_all_objects(img_path, model_qwen, processor_qwen, model_gdino, processor_gdino, device):
    prompt = prompt_template_all.format()
    captions = call_qwen_vl(img_path, prompt, model_qwen, processor_qwen, device).splitlines()
    captions = [caption.replace('-', '').replace('.', '') for caption in captions]
    if len(captions) > 20:
        captions = []
    
    ret = []
    for caption in captions:
        if count_words(caption) > 25:
            continue

        loc1 = call_qwen_vl_locate_one(img_path, caption, model_qwen, processor_qwen, device)
        loc2, boxes = call_grounding_dino(img_path, caption, model_gdino, processor_gdino, device)

        # ret.append({'file_name': img_path, 'caption': caption, 'bbox': loc1, 'candidates': boxes})
        # consistancy
        if len(loc1) != 0 and len(loc2) != 0:
            iou = calculate_iou(loc1, loc2)
            if iou > 0.5:
                ret.append({'file_name': img_path, 'caption': caption, 'bbox': loc2})
    
    return ret


''' For Zero-shot Setting '''
def build_glm(model_path, device):
    model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).to(device).eval()
    processor = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return model, processor

def call_glm(img_path, prompt, model, processor, device):
    if img_path is not None:
        image = Image.open(img_path).convert('RGB')
        inputs = processor.apply_chat_template([{"role": "user", "image": image, "content": prompt}],
                                            add_generation_prompt=True, tokenize=True, return_tensors="pt",
                                            return_dict=True)  # chat mode
    else:
        inputs = processor.apply_chat_template([{"role": "user",  "content": prompt}],
                                            add_generation_prompt=True, tokenize=True, return_tensors="pt",
                                            return_dict=True)  # chat mode
    inputs = inputs.to(device)
    gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        response = processor.decode(outputs[0]).replace('<|endoftext|>','')
        return response

def generate_grounded_cls_zero_shot(img_path, model_glm, processor_glm, model_gdino, processor_gdino, device):
    prompt = prompt_template_cls.format()
    response = call_glm(img_path, prompt, model_glm, processor_glm, device)
    classes_unfiltered = response.lower().splitlines()
    classes_unfiltered = [cls.strip() for cls in classes_unfiltered]
    # filter classes
    classes = []
    for cls in classes_unfiltered:
        if cls in cls_list:
            classes.append(cls)
    classes = list(set(classes))
    if len(classes) > 10:
        classes = []
    
    ret = []
    for cls in classes:
        _, boxes = call_grounding_dino(img_path, cls, model_gdino, processor_gdino, device)
        boxes = filter_boxes(boxes)
        if len(boxes) != 0:
            ret.append({'cls': cls, 'boxes': boxes})

    return ret


def generate_grounded_captions_zero_shot(img_path, box, boxes, cls, model_glm, processor_glm, device):
    prompt_template_short = random.choice([prompt_template_short1, prompt_template_short2, prompt_template_short3])
    prompts = [prompt_template_short.format(box='', cls=cls), prompt_template_mid.format(box='', cls=cls), prompt_template_long.format(box='', cls=cls)]
    
    ret = []
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp:
        image = Image.open(img_path)
        draw = ImageDraw.Draw(image)
        draw.rectangle(box, outline="red", width=3)
        image.save(temp.name)

        for prompt in prompts:
            prompt = prompt + prompt_template_zero_shot.format()
            caption = call_glm(img_path, prompt, model_glm, processor_glm, device)
            if count_words(caption) > 25 or 'red' in caption:
                continue
            ret.append({'file_name': img_path, 'caption': caption, 'bbox': box, 'candidates': boxes})
    return ret


def build_clip(model_path, device):
    model = CLIPModel.from_pretrained(
    model_path,
    attn_implementation="flash_attention_2",
    device_map=device,
    torch_dtype=torch.float16,
    )
    processor = CLIPProcessor.from_pretrained(model_path)
    model.to(device).eval()
    return model, processor

def build_bert(model_path, device):
    model = BertModel.from_pretrained(model_path).to(device).eval()
    tokenizer = BertTokenizer.from_pretrained(model_path)
    return model, tokenizer

def call_bert(text, model, tokenizer, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    emb = outputs.last_hidden_state.mean(dim=1)
    return emb

def filter_abs_position(data_list, TQDM=False):
    filtered_data_list = []
    if TQDM: pbar = tqdm(total=len(data_list))
    for data in data_list:
        if data['caption'].endswith(("right", "right side", "right corner", "right of the image", "left", "left side", "left corner", "left of the image", "middle", "center")) is False:
            filtered_data_list.append(data)
        else:
            if data['caption'].endswith(("right", "right side", "right corner", "right of the image")):
                center1 = get_center(data['bbox'])
                file_name = data['file_name']
                image = Image.open(file_name)
                width, _ = image.size
                flag = True
                if width * (2/3) <= center1[0] <= width:
                    filtered_data_list.append(data)
                else:
                    for candidate in data['candidates']:
                        center2 = get_center(candidate)
                        if center2[0] > center1[0]:
                            flag = False
                            break
                    if flag:
                        filtered_data_list.append(data)
            if data['caption'].endswith(("left", "left side", "left corner", "left of the image")):
                center1 = get_center(data['bbox'])
                file_name = data['file_name']
                image = Image.open(file_name)
                width, _ = image.size
                flag = True
                if 0 <= center1[0] <= width * (1/3):
                    filtered_data_list.append(data)
                else:
                    for candidate in data['candidates']:
                        center2 = get_center(candidate)
                        if center2[0] < center1[0]:
                            flag = False
                            break
                    if flag:
                        filtered_data_list.append(data)
            if data['caption'].endswith(("middle", "center")):
                center1 = get_center(data['bbox'])
                file_name = data['file_name']
                image = Image.open(file_name)
                width, _ = image.size
                flag = True
                if width * (1/3) <= center1[0] <= width * (2/3):
                    filtered_data_list.append(data)
                else:
                    if len(data['candidates']) == 3:
                        flag1 = False
                        flag2 = False
                        for candidate in data['candidates']:
                            center2 = get_center(candidate)
                            if center2[0] < center1[0]:
                                flag1 = True
                            if center2[0] > center1[0]:
                                flag2 = True
                            if flag1 & flag2:
                                filtered_data_list.append(data)
        if TQDM: pbar.update(1)
    if TQDM: pbar.close()
    return filtered_data_list


def filter_clip(data_list, model, processor, TQDM=False, threshold=0.62):
    filtered_data_list = []
    if TQDM: pbar = tqdm(total=len(data_list))
    for data in data_list:
        bbox = data['bbox']
        file_name = data['file_name']

        image = Image.open(file_name).convert('RGB')
        bbox = [int(x) for x in bbox]
        crop_image = crop(image, bbox)
        blur_image = blur(image, bbox)

        text = '"a photo of a ' + data['caption'].lower() + '.'

        # crop
        inputs_crop = processor(text=[text], images=crop_image, return_tensors="pt", padding=True).to(model.device)
        outputs_crop = model(**inputs_crop)
        image_embeds = outputs_crop.image_embeds
        text_embeds = outputs_crop.text_embeds
        similarity1 = torch.nn.functional.cosine_similarity(image_embeds, text_embeds).item()
        similarity1 = (similarity1 + 1) / 2

        # blur 
        inputs_blur = processor(text=[text], images=blur_image, return_tensors="pt", padding=True).to(model.device)
        outputs_blur = model(**inputs_blur)
        image_embeds = outputs_blur.image_embeds
        text_embeds = outputs_blur.text_embeds
        similarity2 = torch.nn.functional.cosine_similarity(image_embeds, text_embeds).item()
        similarity2 = (similarity2 + 1) / 2
        similarity = 0.5 * similarity1 + 0.5 * similarity2

        if similarity > threshold:
            filtered_data_list.append(data)
        if TQDM: pbar.update(1)
    if TQDM: pbar.close()
        
    return filtered_data_list


def extract_physical_objects(sentence, model):
    doc = model(sentence)
    objects = []

    for chunk in doc.noun_chunks:
        head = chunk.root
        if head.pos_ == "NOUN" and not head.is_stop:
            objects.append(head.lemma_)

    return list(set(objects))

def filter_object(data_list, obj_list, model, TQDM=False):
    if TQDM: pbar = tqdm(total=len(data_list))
    selected_data = []
    unselected_data = []
    for data in data_list:
        caption = data['caption']
        objects = extract_physical_objects(caption, model)
        flag = False
        for obj in objects:
            if obj not in obj_list:
                flag = True
                break
        if flag == True: # 出现了新物体
            selected_data.append(data)
        else:
            unselected_data.append(data)
        if TQDM: pbar.update(1)
    if TQDM: pbar.close()
    return selected_data, unselected_data

def filter_distribution(data_list, gmm_refcoco, gmm_refcocoplus, gmm_refcocog, model, tokenizer, TQDM=False):
    if TQDM: pbar = tqdm(total=len(data_list))
    filtered_data_list = []
    for data in data_list:
        inputs = tokenizer(data['caption'], return_tensors="pt", truncation=True, padding=True).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
        emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        
        log_prob1 = gmm_refcoco.score_samples(emb)[0]
        log_prob2 = gmm_refcocoplus.score_samples(emb)[0]
        log_prob3 = gmm_refcocog.score_samples(emb)[0]
        min_log_prob = min(log_prob1, log_prob2, log_prob3)
        
        if 0 < min_log_prob < 800:
            filtered_data_list.append(data)
        if TQDM: pbar.update(1)
    if TQDM: pbar.close()
    return filtered_data_list





if __name__ == '__main__':
    pass
    # cfg_path = 'models/gdino/config/cfg_swinb.py'
    # ckpt_path = 'models/gdino/weights/groundingdino_swinb.pth'
    # device = 6
    # model = build_grounding_dino(cfg_path, ckpt_path, device)

    # img_path = 'images/111.jpg'
    # caption = 'man in black.'
    # top1, top5 = call_grounding_dino(img_path, caption, model, device)
    # print(top1, top5)

    model_path = 'models/Qwen2.5-VL-7B-Instruct'
    device = 6
    model, processor = build_qwen_vl(model_path, device)
    img_path = 'images/111.jpg'
    prompt = 'locate the man in black.'
    prompt = 'Please provide the bounding box coordinate of the region this sentence describes: man in black.'
    response = call_qwen_vl(img_path, prompt, model, processor, device)
    print(response)

    # prompt = f'检测图像中的所有{detection_object}，并以JSON格式返回它们的边界框坐标和详细描述。坐标格式为[x1, y1, x2, y2]，其中(x1,y1)是左上角，(x2,y2)是右下角。格式为[{{"bbox_2d": [x1, y1, x2, y2], "label": "{detection_object}", "sub_label": "详细描述"}}]'
    
    # for cls
    # prompt = f"Locate every {obj_name} in the image and output the coordinates in JSON format."
    # Please provide the bounding box coordinate of the region this sentence describes：人
# prompt for generation
cls_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
prompt_template_cls = '''Giving an image, find all instance classes (including person). 

Provide the object types in a list format, one per line.
Output format:
class1
class2
class3
...
classN

please choose from the following classes:
'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'.
Please do not list more than 5 different classes.
Please do not output the background class, such as sky, dirt.'''

prompt_template_short1 = '''Please generate a unique description for the object inside the bounding box {box}. 
The class information of the object in the bounding box: {cls}

The description should follow the format: Object + Feature + Position (right, left, middle)
Examples: "person on left yellow boots", "blue car right", "white car left", "man on far left on screen", "man left cut off", "lady middle pink".

Please limit your output to 5 words or less. 
Please ensure that the description is specific and clearly identifies the object within the bounding box.
Please ensure that the output conforms to the format and has no other content.'''

prompt_template_short2 = '''Please generate a unique description for the object inside the bounding box {box}. 
The class information of the object in the bounding box: {cls}

The description should follow the format: Object + Position (right, left, middle)
Examples: "person left", "left car", "main guy on the tv", "man front center", "right guy".

Please limit your output to 5 words or less. 
Please ensure that the description is specific and clearly identifies the object within the bounding box.
Please ensure that the output conforms to the format and has no other content.'''

prompt_template_short3 = '''Please generate a unique description for the object inside the bounding box {box}. 
The class information of the object in the bounding box: {cls}

The description should follow the format: Object + Feature
Examples: "blue car", "man walking out of picture", "seated man", "woman in blue".

Please limit your output to 5 words or less. 
Please ensure that the description is specific and clearly identifies the object within the bounding box.
Please ensure that the output conforms to the format and has no other content.'''

prompt_template_mid = '''Please generate a unique description for the object inside the bounding box {box}. 
The class information of the object in the bounding box: {cls}

The description should follow the format: Object + Details (5 words).
Examples: "Guy holding purple umbrella in corner near us", "Black shirt person holding umbrella", "Man in black shirt and jeans", "Guy in blue hat and jacket", "Blue car in back of the man with hat", "Man in red-brown shorts", "White man coaching", "Red shirt player next to striped shirt guy."

Please limit your output to 10 words or less. 
Please ensure that the description is specific and clearly identifies the object within the bounding box.
Please ensure that the output conforms to the format and has no other content.'''

prompt_template_long = '''Please generate a unique description for the object inside the bounding box {box}. 
The class information of the object in the bounding box: {cls}

The description should follow one of the format: Object + Description (Feature, 5 words) + Description (Action, 5 words) + Relative Position of Objects (5 words)
Examples: "pot boiling water with green bell peppers in man's kitchen", "the orange between other oranges and a banana", "a man with a red and silver power tie in front of a woman", "a catcher crouching in front of the umpire", "the boy sitting against the wall , reading", "a black umbrella , being held by a person in jeans".

Please limit your output to 20 words or less. 
Please ensure that the description is specific and clearly identifies the object within the bounding box.
Please ensure that the output conforms to the format and has no other content.'''

prompt_template_zero_shot = '''The red bounding box is only used for auxiliary positioning, please do not mention "red bounding box" or "red box" or "red outline" in the output.'''

prompt_template_all = '''Identify all objects in the image as many as possible and give each of them a unique description.

Please provide the objects in a list format, one per line and start directly with the sentence without any leading numbers like 1, 2, 3.
An output example: "taller man in black shirt, left of the dog.
The boy in the red shirt with sunglasses, behind the boy in blue, located at center.
The bird with a yellow beak, on right side.
The black and white dog wearing a yellow collar, laying on a wooden deck."

Please limit the output in 5 items and limit each item to 20 words or less.
Please ensure that the output conforms to the format and has no other content.
'''

prompt_template_layout1 = '''please generate a layout of an image, containing {num} {category}.

1. Decide the background of the image, it must conform to reality. For example, person can appear in wild or city, but a train cannot appear in a bedroom. You can select a fit background from "city street", "city at night", "modern downtown", "old town street", "living room", "kitchen", "forest", "meadow", "mountains", "desert", "lakeside", "beach".

2. Generate a normalized bounding box for each object, such as [0.25, 0.30, 0.35, 0.50].

3. Generate a description for each object.

4. Generate a overall caption for the image about 30 words.

Output the result in JSON format.
Format:
{{
    "caption": overall caption
    "objects":[
        {{"bbox": bbox, "description": caption for this bbox}},
        ......
    ]
}}
'''

prompt_template_layout2 = '''please generate a layout of an image, containing {num} {category} with different size.

1. Decide the background of the image, it must conform to reality. For example, person can appear in wild or city, but a train cannot appear in a bedroom. You can select a fit background from "city street", "city at night", "modern downtown", "old town street", "living room", "kitchen", "forest", "meadow", "mountains", "desert", "lakeside", "beach".

2. Generate a position bounding box for each object, such as [0.15, 0.30, 0.25, 0.50]. The bigger object should have a bigger bounding box and the smaller object should have a smaller bounding box.

3. Generate a description for each object. Please indict the different size between objects.

4. Generate a overall caption for the image about 30 words.

Output the result in JSON format.
Format:
{{
    "caption": overall caption
    "objects":[
        {{"bbox": bbox, "description": caption for this bbox}},
        ......
    ]
}}
'''
index_to_category = {1: u'person', 2: u'bicycle', 3: u'car', 4: u'motorcycle', 5: u'airplane', 6: u'bus', 7: u'train', 8: u'truck', 9: u'boat', 10: u'traffic light', 11: u'fire hydrant', 12: u'stop sign', 13: u'parking meter', 14: u'bench', 15: u'bird', 16: u'cat', 17: u'dog', 18: u'horse', 19: u'sheep', 20: u'cow', 21: u'elephant', 22: u'bear', 23: u'zebra', 24: u'giraffe', 25: u'backpack', 26: u'umbrella', 27: u'handbag', 28: u'tie', 29: u'suitcase', 30: u'frisbee', 31: u'skis', 32: u'snowboard', 33: u'sports ball', 34: u'kite', 35: u'baseball bat', 36: u'baseball glove', 37: u'skateboard', 38: u'surfboard', 39: u'tennis racket', 40: u'bottle', 41: u'wine glass', 42: u'cup', 43: u'fork', 44: u'knife', 45: u'spoon', 46: u'bowl', 47: u'banana', 48: u'apple', 49: u'sandwich', 50: u'orange', 51: u'broccoli', 52: u'carrot', 53: u'hot dog', 54: u'pizza', 55: u'donut', 56: u'cake', 57: u'chair', 58: u'couch', 59: u'potted plant', 60: u'bed', 61: u'dining table', 62: u'toilet', 63: u'tv', 64: u'laptop', 65: u'mouse', 66: u'remote', 67: u'keyboard', 68: u'cell phone', 69: u'microwave', 70: u'oven', 71: u'toaster', 72: u'sink', 73: u'refrigerator', 74: u'book', 75: u'clock', 76: u'vase', 77: u'scissors', 78: u'teddy bear', 79: u'hair drier', 80: u'toothbrush'}
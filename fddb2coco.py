import os
import json
import math
from datetime import datetime
from PIL import Image
from random import choice

FDDB_DIR = 'FDDB'
FDDB_FOLD_DIR = 'FDDB/FDDB-folds'


def auto_increment_integer_generator():
    i = 1
    while True:
        yield i
        i += 1


def degree_to_radian(degree):
    return (degree * math.pi) / 360


def parse_a_line(line):
    if 'img' in line:
        return {'filename': line}
    else:
        try:
            number_of_person = eval(line)
            return {"number_of_person": number_of_person}
        except SyntaxError:
            r_maj, r_min, angle, centre_x, centre_y, score = [eval(x) for x in line.split(' ') if x]
            k = math.cos(degree_to_radian(angle))
            x_offset = r_min * k
            y_offset = r_maj * k
            topleft_x = centre_x - x_offset
            topleft_y = centre_y - y_offset
            width = 2 * x_offset
            height = 2 * y_offset
            area = width * height
            return {'area': area, 'bbox': [topleft_x, topleft_y, width, height]}


with open('FDDB_IN_COCO/fddb.json', 'r') as f:
    original = json.load(f)


image_id_generator = auto_increment_integer_generator()
annotation_id_generator = auto_increment_integer_generator()
images_list = list()
annotation_list = list()
cummulative_sum_of_annotation = 0
for fold in os.listdir(FDDB_FOLD_DIR):
    if 'ellipseList' in fold:
        path = os.path.join(FDDB_FOLD_DIR, fold)
        with open(path, 'r') as f:
            for line in iter(f.readline, ''):
                info = parse_a_line(line.replace('\n', ''))
                if 'filename' in info:
                    pil = Image.open(os.path.join(FDDB_DIR, info.get('filename') + '.jpg'))
                    width, height = pil.size
                    del pil
                    images_list.append(
                        {
                            "license": 1,
                            "file_name": info.get('filename') + '.jpg',
                            "height": height,
                            "width": width,
                            "date_captured": datetime.now().strftime('%Y-%m-%d %X'),
                            "id": next(image_id_generator)
                        }
                    )
                elif 'number_of_person' in info:
                    number_of_person = info.get('number_of_person')
                    cummulative_sum_of_annotation += number_of_person

                else:
                    image = images_list[-1]
                    annotation_list.append(
                        {
                            "segmentation": list(),
                            "area": info.get('area'),
                            "iscrowd": 0,
                            "image_id": image.get('id'),
                            "bbox": info.get('bbox'),
                            "category_id": 1,
                            "id": next(annotation_id_generator)
                        }
                    )

print(choice(images_list))
print(choice(annotation_list))
original['images'].extend(images_list)
original['annotations'].extend(annotation_list)

with open('FDDB_IN_COCO/fddb_all.json', 'w') as f:
    f.write(json.dumps(original, indent=4))

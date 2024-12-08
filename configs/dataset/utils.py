'''数据集处理工具类'''

import os
import json
from pycocotools.coco import COCO

# def txt_to_coco(categories, annotations_path, images_path, output_coco_path):
#     # 转换为 COCO 格式的类别
#     coco_categories = [{"id": k, "name": v} for k, v in categories.items()]
#     # 初始化 COCO 格式字典
#     coco_data = {
#         "images": [],
#         "annotations": [],
#         "categories": []
#     }
#     # 加入类别信息
#     coco_data["categories"] = coco_categories

#     # 遍历所有的 VisDrone 标注文件
#     annotation_id = 1
#     image_id = 1

#     for filename in os.listdir(annotations_path):
#         if not filename.endswith(".txt"):
#             continue

#         # 对应的图像文件名
#         image_filename = filename.replace(".txt", ".jpg")
#         image_path = os.path.join(images_path, image_filename)
        
#         # 获取图像尺寸（可以用 PIL 或其他工具读取）
#         from PIL import Image
#         with Image.open(image_path) as img:
#             width, height = img.size

#         # 添加图像信息到 COCO 数据
#         coco_data["images"].append({
#             "id": image_id,
#             "file_name": image_filename,
#             "width": width,
#             "height": height
#         })

#         # 读取对应的 VisDrone 标注文件
#         with open(os.path.join(annotations_path, filename), "r") as f:
#             for line in f:
#                 parts = line.strip().split(",")
#                 bbox_left = float(parts[0])
#                 bbox_top = float(parts[1])
#                 bbox_width = float(parts[2])
#                 bbox_height = float(parts[3])
#                 category_id = int(parts[5])+1  # 对应 VisDrone 的 object_category 从0开始编号需要+1

#                 # 添加标注信息到 COCO 数据
#                 coco_data["annotations"].append({
#                     "id": annotation_id,
#                     "image_id": image_id,
#                     "category_id": category_id,
#                     "bbox": [bbox_left, bbox_top, bbox_width, bbox_height],
#                     "area": bbox_width * bbox_height,
#                     "iscrowd": 0  # VisDrone 没有遮挡标记，这里设置为 0
#                 })
#                 annotation_id += 1

#         image_id += 1

#     # 保存 COCO 格式 JSON 文件
#     with open(output_coco_path, "w") as json_file:
#         json.dump(coco_data, json_file, indent=4)

#     print(f"转换完成，保存到 {output_coco_path}")


def txt_to_coco(categories, annotations_path, images_path, output_coco_path):
    # 过滤none和others
    # 转换为 COCO 格式的类别
    coco_categories = [{"id": k, "name": v} for k, v in categories.items()]
    # 初始化 COCO 格式字典
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    # 加入类别信息
    coco_data["categories"] = coco_categories

    # 遍历所有的 VisDrone 标注文件
    annotation_id = 1
    image_id = 1

    for filename in os.listdir(annotations_path):
        if not filename.endswith(".txt"):
            continue

        # 对应的图像文件名
        image_filename = filename.replace(".txt", ".jpg")
        image_path = os.path.join(images_path, image_filename)
        
        # 获取图像尺寸（可以用 PIL 或其他工具读取）
        from PIL import Image
        with Image.open(image_path) as img:
            width, height = img.size

        # 添加图像信息到 COCO 数据
        coco_data["images"].append({
            "id": image_id,
            "file_name": image_filename,
            "width": width,
            "height": height
        })

        # 读取对应的 VisDrone 标注文件
        with open(os.path.join(annotations_path, filename), "r") as f:
            for line in f:
                parts = line.strip().split(",")
                bbox_left = float(parts[0])
                bbox_top = float(parts[1])
                bbox_width = float(parts[2])
                bbox_height = float(parts[3])
                if int(parts[4]) == 0 or int(parts[5]) == 11:
                    continue
                else: 
                    category_id = int(parts[5])  # 对应 VisDrone 的 object_category 从0开始编号需要+1


                # 添加标注信息到 COCO 数据
                coco_data["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [bbox_left, bbox_top, bbox_width, bbox_height],
                    "area": bbox_width * bbox_height,
                    "iscrowd": 0  # VisDrone 没有遮挡标记，这里设置为 0
                })
                annotation_id += 1

        image_id += 1

    # 保存 COCO 格式 JSON 文件
    with open(output_coco_path, "w") as json_file:
        json.dump(coco_data, json_file, indent=4)

    print(f"转换完成，保存到 {output_coco_path}")

def generate_coco_json():
    # 指定类别映射
    # visdrone_categories = {
    #     1: "none",
    #     2: "pedestrian",
    #     3: "people",
    #     4: "bicycle",
    #     5: "car",
    #     6: "van",
    #     7: "truck",
    #     8: "tricycle",
    #     9: "awning-tricycle",
    #     10: "bus",
    #     11: "motor",
    #     12: "others"
    # }

    # 去掉none和others
    visdrone_categories = {
        1: "pedestrian",
        2: "people",
        3: "bicycle",
        4: "car",
        5: "van",
        6: "truck",
        7: "tricycle",
        8: "awning-tricycle",
        9: "bus",
        10: "motor",
    }

    # 输入输出路径
    annotations_path = r"E:\Project\Dataset\VisDrone2019\test\VisDrone2019-DET-train\annotations"
    images_path = r"E:\Project\Dataset\VisDrone2019\test\VisDrone2019-DET-train\images"
    output_coco_path = r"E:\Project\Dataset\VisDrone2019\test\VisDrone2019-DET-train\visdrone_annotations_train.json"
    txt_to_coco(visdrone_categories, annotations_path, images_path, output_coco_path)

    # 输入输出路径
    annotations_path = r"E:\Project\Dataset\VisDrone2019\test\VisDrone2019-DET-val\annotations"
    images_path = r"E:\Project\Dataset\VisDrone2019\test\VisDrone2019-DET-val\images"
    output_coco_path = r"E:\Project\Dataset\VisDrone2019\test\VisDrone2019-DET-val\visdrone_annotations_val.json"
    txt_to_coco(visdrone_categories, annotations_path, images_path, output_coco_path)

    # # 输入输出路径
    # annotations_path = r"E:\Project\Dataset\VisDrone2019\VisDrone2019-DET-train\annotations"
    # images_path = r"E:\Project\Dataset\VisDrone2019\VisDrone2019-DET-train\images"
    # output_coco_path = r"E:\Project\Dataset\VisDrone2019\VisDrone2019-DET-train\visdrone_annotations_train.json"
    # txt_to_coco(visdrone_categories, annotations_path, images_path, output_coco_path)

    # # 输入输出路径
    # annotations_path = r"E:\Project\Dataset\VisDrone2019\VisDrone2019-DET-val\annotations"
    # images_path = r"E:\Project\Dataset\VisDrone2019\VisDrone2019-DET-val\images"
    # output_coco_path = r"E:\Project\Dataset\VisDrone2019\VisDrone2019-DET-val\visdrone_annotations_val.json"
    # txt_to_coco(visdrone_categories, annotations_path, images_path, output_coco_path)

def check_id():
    coco = COCO(annotation_file=r"E:\Project\Dataset\VisDrone2019\test\VisDrone2019-DET-val\visdrone_annotations_val.json")
    categories = coco.loadCats(coco.getCatIds())
    for category in categories:
        print(f"ID: {category['id']}, Name: {category['name']}")


if __name__ == "__main__":
    generate_coco_json()
    check_id()
import os
import xml.etree.ElementTree as ET

# 类别名称，根据你的数据调整
classes = ["kun"]  # ← 替换为你自己的类别名

# 文件夹路径（使用原始字符串 r"..." 防止反斜杠转义）
base_path = r"/mnt/d/yolov10_kun/kun_face_finished/2417535_1751702376"
annotations_dir = os.path.join(base_path, "Annotations")
images_dir = os.path.join(base_path, "images")
labels_dir = os.path.join(base_path, "labels")

os.makedirs(labels_dir, exist_ok=True)

def convert_box(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x_center = (box[0] + box[1]) / 2.0
    y_center = (box[2] + box[3]) / 2.0
    width = box[1] - box[0]
    height = box[3] - box[2]
    return (x_center * dw, y_center * dh, width * dw, height * dh)

def convert_annotation(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    img_filename = root.find("filename").text
    txt_filename = os.path.splitext(img_filename)[0] + ".txt"
    txt_path = os.path.join(labels_dir, txt_filename)

    size = root.find("size")
    w = int(size.find("width").text)
    h = int(size.find("height").text)

    with open(txt_path, "w") as out_file:
        for obj in root.iter("object"):
            cls = obj.find("name").text
            if cls not in classes:
                continue  # 跳过不在类别列表中的对象
            cls_id = classes.index(cls)
            xmlbox = obj.find("bndbox")
            b = (
                float(xmlbox.find("xmin").text),
                float(xmlbox.find("xmax").text),
                float(xmlbox.find("ymin").text),
                float(xmlbox.find("ymax").text),
            )
            bbox = convert_box((w, h), b)
            out_file.write(f"{cls_id} {' '.join(f'{a:.6f}' for a in bbox)}\n")

# 遍历所有 XML 文件
for filename in os.listdir(annotations_dir):
    if filename.endswith(".xml"):
        convert_annotation(os.path.join(annotations_dir, filename))

print("✅ 所有 XML 文件已成功转换为 YOLO 格式 TXT，输出路径：", labels_dir)

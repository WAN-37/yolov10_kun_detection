from ultralytics import YOLOv10
from PIL import Image
import os

def main():
    # 1. 加载你训练好的模型
    model_path = 'best.pt'  # 修改为你的模型路径
    model = YOLOv10(model_path)

    # 2. 指定要识别的图片路径，可以是单张图片或文件夹
    img_source = '/mnt/d/yolov10_kun/jige.webp'  # 修改为你的图片或文件夹路径

    # 3. 进行预测，save=True 表示保存带框的结果图片到 runs/predict 文件夹
    results = model.predict(source=img_source, save=True, conf=0.25)  

    # 4. 输出预测框信息，打印每张图片的检测结果
    for i, r in enumerate(results):
        print(f"\nImage {i+1}:")
        boxes = r.boxes
        for box in boxes:
            xyxy = box.xyxy.cpu().numpy().flatten()  # 左上右下坐标
            conf = box.conf.cpu().item()             # 置信度
            cls = int(box.cls.cpu().item())           # 类别索引
            print(f"  Class: {cls}, Confidence: {conf:.2f}, BBox: {xyxy}")

    # 5. 显示第一个预测结果图片（可选）
    if results:
        img_path = results[0].path  # 推理图像路径
        img_result_path = results[0].save_path  # 保存结果图片路径
        print(f"\n预测图片保存于: {img_result_path}")
        img = Image.open(img_result_path)
        img.show()

if __name__ == "__main__":
    main()

from ultralytics import YOLOv10

# 加载模型（s 表示 small，可以改成 yolov10n.pt / m / l / x）
model = YOLOv10('yolov10s.pt')

# 开始训练
model.train(
    data='ikun.yaml',     # 数据集配置文件
    epochs=50,            # 训练轮数
    batch=16,             # 🔺减小 batch_size，防止显存炸
    imgsz=512,            # 🔺减小图片尺寸，减轻显存压力
    amp=False             # 🔺关闭 AMP，避免潜在内存碎片问题
)

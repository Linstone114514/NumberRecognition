import os
import cv2
import numpy as np
import tensorflow as tf

# 1. 载入模型
model = tf.keras.models.load_model("model/digit_cnn.h5")

# 2. 预处理单张图片
def preprocess(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取图片：{img_path}")
    img = cv2.resize(img, (28, 28))
    img = img.astype("float32") / 255.0
    img = 1.0 - img
    return img.reshape(1, 28, 28, 1)

# 3. 批量推理
def batch_predict(folder="test"):
    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            continue
        img_path = os.path.join(folder, fname)
        img = preprocess(img_path)
        pred = np.argmax(model.predict(img))
        print(f"{fname} -> result: {pred}")

if __name__ == "__main__":
    batch_predict("test")

# 图像检查
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 1. 创建输出目录
out_dir = Path("test_processed")
out_dir.mkdir(exist_ok=True)

# 2. 遍历 test/ 下的所有图片
test_folder = Path("test")
for img_path in sorted(test_folder.glob("*")):
    if img_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
        continue

    # 3. 与模型完全相同的预处理
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"跳过无法读取的文件：{img_path.name}")
        continue

    img = cv2.resize(img, (28, 28))
    img = img.astype("float32") / 255.0

    # 黑底白字：如果是白底黑字，下一行取消注释即可
    img = 1.0 - img

    # 4. 保存预处理后的图片（放大到 280×280 便于肉眼查看）
    save_name = out_dir / f"{img_path.stem}_proc.png"
    cv2.imwrite(str(save_name), (img * 255).astype(np.uint8))

    # 5. 推理
    #pred = np.argmax(model.predict(img.reshape(1, 28, 28, 1)))
    #print(f"{img_path.name} -> 识别结果: {pred}")

print(f"所有预处理图片已保存到 {out_dir.resolve()}")
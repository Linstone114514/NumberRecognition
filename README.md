
# NumberRecognition 手写数字识别

基于 TensorFlow 2.x 的轻量级卷积神经网络，可直接从 MNIST 原始二进制文件（`idx3/idx1`）训练并达到 >99% 测试准确率。

---

## 目录结构
```
digit-cnn/
├── data/                       MNIST 原始二进制文件
│   ├── train-images.idx3-ubyte
│   ├── train-labels.idx1-ubyte
│   ├── t10k-images.idx3-ubyte
│   └── t10k-labels.idx1-ubyte
├── logs/                       TensorBoard 日志（自动生成）
├── model/                      训练完成后保存的模型
│   └── digit_cnn.h5
├── train.py                    训练脚本
├── predict.py                  预测脚本（调用训练后的模型）
└── README.md                   本文档
```

---

## 环境要求
```
python>=3.8
tensorflow>=2.8
idx2numpy               # 用于读取 MNIST 原始二进制
```
一键安装：
```bash
pip install tensorflow idx2numpy
```

---

## 快速开始

1. 克隆仓库并进入目录  

2. 将 MNIST 原始二进制文件放入 `data/`（文件名需与训练脚本中的一致）。

3. 开始训练  
   ```bash
   python train.py
   ```
   - 默认 16 个 epoch，约 3 分钟（GPU 上更快）。  
   - 日志实时写入 `logs/`，可用 TensorBoard 查看：
     ```bash
     tensorboard --logdir logs
     ```

4. 训练结束后  

   - 终端会打印 `Test accuracy`（通常在 0.99 左右）。  
   - 最终模型保存为 `model/digit_cnn.h5`，可直接用于推理。

5. 调用模型
   
   - 打开预测脚本，并将待测图片全部存于test\下。
   -执行
   ```bash
   python predict.py
   ```
   以调用模型。
   - 在预测脚本中内置了导出处理后的图片的功能，只需要将最后的一大块注释删掉即可

---

## 模型结构

| 层级                      | 说明              |
| ------------------------- | ----------------- |
| Conv2D(32, 3×3) ×2        | 提取边缘/纹理特征 |
| MaxPool2D                 | 降采样            |
| Dropout(0.25)             | 防止过拟合        |
| Conv2D(64, 3×3) ×2        | 提取更高层特征    |
| MaxPool2D + Dropout(0.25) | 同上              |
| Flatten + Dropout(0.5)    | 展平 + 正则化     |
| Dense(128, ReLU)          | 全连接            |
| Dense(10, Softmax)        | 10 类输出         |

---

## 其它问题

若您希望用**黑底白字**图片进行训练，则需要对数据读取进行修改
若您希望用**白底黑字**图片进行训练，则预测脚本，训练脚本需要一并进行修改。

---

## 许可证
MIT © 2025
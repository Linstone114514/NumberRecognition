import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import os, datetime
import idx2numpy
from tensorflow.keras.utils import to_categorical

# 1. 数据
def load_mnist_from_raw(path):
    x_train = idx2numpy.convert_from_file(f"{path}/train-images.idx3-ubyte")
    y_train = idx2numpy.convert_from_file(f"{path}/train-labels.idx1-ubyte")
    x_test  = idx2numpy.convert_from_file(f"{path}/t10k-images.idx3-ubyte")
    y_test  = idx2numpy.convert_from_file(f"{path}/t10k-labels.idx1-ubyte")

    # 归一化 shape & 像素
    x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255
    x_test  = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255
    y_train = to_categorical(y_train, 10)
    y_test  = to_categorical(y_test, 10)
    return (x_train, y_train), (x_test, y_test)

(x_train, y_train), (x_test, y_test) = load_mnist_from_raw("data")

# 2. 模型
model = models.Sequential([
    layers.Conv2D(32, 3, activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 3. TensorBoard 回调
log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb_cb   = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# 4. 训练
model.fit(x_train, y_train,
          epochs=16, #迭代次数
          batch_size=64, #每次多少张
          validation_split=0.1, #抽取验证比例
          callbacks=[tb_cb])    #回调

# 5. 评估 & 保存
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print("Test accuracy:", test_acc)
model.save("model/digit_cnn.h5")